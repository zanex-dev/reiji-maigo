"""End-to-end integration tests for SwarmCoordinator, telemetry pulse, `reiji swarm` CLI, and Shinobu bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

import reiji_maigo.swarm as swarm_pkg
from reiji_maigo.bridges.shinobu import ReijiShinobuPlugin, create_plugin
from reiji_maigo.cli import main as cli_main
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm import (
    ARCHIVIST_L0_PREFIX,
    SCOUT_L0_PREFIX,
    SCRIBE_L0_PREFIX,
    ArchivistAgent,
    ArchivistReport,
    AsyncScribeWorker,
    CallableFlashRunner,
    ENV_FLASH_API_KEY,
    ENV_FLASH_BASE_URL,
    ScoutAgent,
    ScoutBriefing,
    ScribeAgent,
    ScribeDigestReport,
    SwarmCoordinator,
    SwarmRunnerError,
)


def _make_tri_worker_runner() -> CallableFlashRunner:
    """Offline deterministic runner serving Scribe, Scout, and Archivist shadow calls."""
    calls: list[str] = []

    def _dispatch(role: str, locked_system_prefix: str, user_payload: str) -> dict[str, Any]:
        calls.append(role)
        payload = json.loads(user_payload)

        if role == "scribe":
            turns = payload.get("undigested_turns") or []
            if not turns:
                return {"skipped_banter": 0, "operations": []}
            first_tid = int(turns[0]["turn_id"])
            last_tid = int(turns[-1]["turn_id"])
            ep = str(turns[0].get("episode_id", "ep_int"))
            return {
                "skipped_banter": 1,
                "operations": [
                    {
                        "op": "create_event",
                        "title": "Deploy Reiji Swarm Pipeline",
                        "scope": "quest",
                        "status": "done",
                        "step": "Configured SwarmCoordinator telemetry and Shinobu bridge hooks",
                        "span": {
                            "episode_id": ep,
                            "turn_start": first_tid,
                            "turn_end": last_tid,
                        },
                        "detail": {
                            "summary": "SwarmCoordinator and bridge closed-loop wired with port 8787",
                            "facts": ["Telemetry banner locks L0 prefixes", "Port 8787 reserved"],
                            "entities": ["SwarmCoordinator", "ReijiShinobuPlugin"],
                            "open_questions": [],
                        },
                    }
                ],
            }

        if role == "scout":
            l1_cards = payload.get("level1_cards") or []
            card_ids = [str(c["card_id"]) for c in l1_cards if isinstance(c, dict) and c.get("card_id")]
            return {
                "status": "final",
                "final_briefing": {
                    "found": True,
                    "abstained": False,
                    "briefing_text": "SwarmCoordinator locks L0 cache prefixes and reserves port 8787.",
                    "cited_card_ids": card_ids,
                    "cited_spans": [{"episode_id": "ep_int", "turn_start": 1, "turn_end": 2}],
                },
            }

        if role == "archivist":
            task = payload.get("task", "archive")
            if task == "compact":
                return {"checkpoint_summary": "Lossless checkpoint covering early swarm setup steps."}
            return {
                "domain": "infra",
                "card_type": "architecture",
                "title": "Reiji Swarm Pipeline Architecture",
                "tags": ["swarm", "telemetry"],
                "state": {
                    "summary": "Headless three-worker swarm with L0 prefix locking on port 8787.",
                    "detail": ["Scribe, Scout, Archivist physically isolated", "Port 8787"],
                    "open_questions": [],
                },
                "foresight_note": "",
                "foresight_until": None,
                "supersede_card_id": None,
                "revoke_card_ids": [],
                "related_card_ids": [],
                "skip": False,
            }

        raise AssertionError(f"Unexpected role: {role!r}")

    runner = CallableFlashRunner(_dispatch)
    runner.calls = calls  # type: ignore[attr-defined]
    return runner


class _FakeHookBus:
    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event_name: str, handler: Any) -> tuple[str, Any]:
        self.handlers.setdefault(event_name, []).append(handler)
        return (event_name, handler)

    def emit(self, event_name: str, payload: Any = None, **kwargs: Any) -> list[Any]:
        results = []
        for fn in self.handlers.get(event_name, []):
            results.append(fn(payload, **kwargs))
        return results


class _FakeKernel:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.tools: list[Any] = []

    def register_tools(self, tools: list[Any]) -> None:
        self.tools.extend(tools)


class TestSwarmCoordinatorAndPulse(unittest.TestCase):
    def test_swarm_package_exports_complete(self) -> None:
        for sym in (
            "SwarmCoordinator",
            "ScribeAgent",
            "AsyncScribeWorker",
            "ScribeDigestReport",
            "SCRIBE_L0_PREFIX",
            "ScoutAgent",
            "ScoutBriefing",
            "SCOUT_L0_PREFIX",
            "ArchivistAgent",
            "ArchivistReport",
            "ARCHIVIST_L0_PREFIX",
        ):
            self.assertTrue(hasattr(swarm_pkg, sym), f"Missing swarm export: {sym}")

    def test_unconfigured_runner_pulse_zero_error(self) -> None:
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in (ENV_FLASH_BASE_URL, ENV_FLASH_API_KEY)
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, clean_env, clear=True):
                with ReijiMemoryEngine(home=tmp, load_plugins=False) as engine:
                    engine.append_turn("ep_u", "user", "hello unconfigured swarm")
                    with SwarmCoordinator(engine) as coord:
                        self.assertFalse(coord.is_configured)
                        self.assertEqual(coord.runner_source, "unconfigured")

                        summary = coord.pulse_summary()
                        self.assertEqual(summary["runner"]["status"], "unconfigured")
                        self.assertFalse(summary["runner"]["configured"])
                        self.assertEqual(summary["l0_cache"]["status"], "100% LOCKED")
                        self.assertEqual(summary["storage"]["raw_turns"], 1)
                        self.assertEqual(summary["active_pool"]["undigested_turns"], 1)

                        banner = coord.render_pulse_banner()
                        self.assertIn("L0 Cache: 100% LOCKED", banner)
                        self.assertIn("runner=unconfigured", banner)
                        self.assertIn("Active Pool", banner)
                        self.assertIn("Scribe", banner)
                        self.assertIn("Scout", banner)
                        self.assertIn("Archivist", banner)

                        with self.assertRaises(SwarmRunnerError):
                            coord.run_scribe_now()
                        with self.assertRaises(SwarmRunnerError):
                            coord.dispatch_scout("anything")
                        with self.assertRaises(SwarmRunnerError):
                            coord.run_midnight_archivist()

    def test_full_closed_loop_scribe_scout_archivist_and_pulse(self) -> None:
        runner = _make_tri_worker_runner()
        with tempfile.TemporaryDirectory() as tmp:
            with ReijiMemoryEngine(home=tmp, load_plugins=False) as engine:
                with SwarmCoordinator(engine, runner=runner, async_scribe=True, debounce_sec=0.02) as coord:
                    self.assertTrue(coord.is_configured)
                    t1 = coord.on_turn_completed("ep_int", "user", "haha morning!")
                    t2 = coord.on_turn_completed(
                        "ep_int",
                        "assistant",
                        "Wired SwarmCoordinator telemetry and reserved port 8787.",
                    )
                    self.assertGreater(t2, t1)
                    self.assertTrue(coord.flush_scribe(timeout=5.0))

                    events = engine.pool.list_events(include_inactive=True)
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0].title, "Deploy Reiji Swarm Pipeline")

                    arch_report = coord.run_midnight_archivist()
                    self.assertIsInstance(arch_report, ArchivistReport)
                    self.assertEqual(len(arch_report.archived_cards), 1)

                    briefing = coord.dispatch_scout(
                        "Which port is reserved by SwarmCoordinator?",
                        domain="infra",
                        hint_tags=["swarm"],
                    )
                    self.assertIsInstance(briefing, ScoutBriefing)
                    self.assertTrue(briefing.found)
                    self.assertFalse(briefing.abstained)
                    self.assertIn("8787", briefing.briefing_text)
                    self.assertTrue(briefing.to_tail_block().startswith("[Scout Briefing"))

                    summary = coord.pulse_summary()
                    self.assertEqual(summary["scribe"]["runs"], 1)
                    self.assertEqual(summary["scribe"]["turns_digested"], 2)
                    self.assertEqual(summary["scribe"]["skipped_banter"], 1)
                    self.assertEqual(summary["scout"]["runs"], 1)
                    self.assertEqual(summary["archivist"]["runs"], 1)
                    self.assertEqual(summary["archivist"]["archived_cards"], 1)

                    banner = coord.render_pulse_banner()
                    self.assertIn("L0 Cache: 100% LOCKED", banner)
                    self.assertIn("raw turns ->", banner)
                    self.assertIn("distilled tok", banner)
                    self.assertIn("revoked=0/archived=1/compacted=0", banner)

    def test_kernel_extractor_provider_fallback(self) -> None:
        runner = _make_tri_worker_runner()
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in (ENV_FLASH_BASE_URL, ENV_FLASH_API_KEY)
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, clean_env, clear=True):
                with ReijiMemoryEngine(home=tmp, load_plugins=False) as engine:
                    engine.kernel.register_providers("extractor", "mock_flash", runner)
                    with SwarmCoordinator(engine) as coord:
                        self.assertTrue(coord.is_configured)
                        self.assertEqual(coord.runner_source, "kernel_extractor")
                        coord.on_turn_completed("ep_int", "user", "Use port 8787 for swarm")
                        report = coord.run_scribe_now()
                        self.assertIsInstance(report, ScribeDigestReport)
                        self.assertEqual(report.turns_digested, 1)


class TestSwarmCLIAndSubprocess(unittest.TestCase):
    def test_bare_subprocess_reiji_swarm_status_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in (ENV_FLASH_BASE_URL, ENV_FLASH_API_KEY)
            }
            env["REIJI_MAIGO_DISABLE_PLUGINS"] = "1"
            repo_root = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = (
                f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
                if env.get("PYTHONPATH")
                else repo_root
            )

            proc = subprocess.run(
                [sys.executable, "-m", "reiji_maigo.cli", "--home", tmp, "swarm", "status"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("L0 Cache: 100% LOCKED", proc.stdout)
            self.assertIn("runner=unconfigured", proc.stdout)

            proc_json = subprocess.run(
                [sys.executable, "-m", "reiji_maigo.cli", "--home", tmp, "swarm", "status", "--json"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc_json.returncode, 0, msg=proc_json.stderr)
            data = json.loads(proc_json.stdout)
            self.assertEqual(data["runner"]["status"], "unconfigured")
            self.assertEqual(data["l0_cache"]["status"], "100% LOCKED")

            proc_scribe_err = subprocess.run(
                [sys.executable, "-m", "reiji_maigo.cli", "--home", tmp, "swarm", "scribe"],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc_scribe_err.returncode, 2)
            self.assertIn("unconfigured", proc_scribe_err.stderr)

    def test_cli_swarm_scribe_scout_midnight_with_mock_coordinator_runner(self) -> None:
        runner = _make_tri_worker_runner()
        with tempfile.TemporaryDirectory() as tmp:
            with ReijiMemoryEngine(home=tmp, load_plugins=False) as engine:
                engine.append_turn("ep_int", "user", "Set up swarm telemetry on port 8787")

            with mock.patch("reiji_maigo.swarm.coordinator._resolve_runner", return_value=(runner, "explicit")):
                with mock.patch("sys.stdout") as fake_out:
                    chunks: list[str] = []
                    fake_out.write.side_effect = chunks.append

                    rc1 = cli_main(["--home", tmp, "swarm", "scribe", "--json"])
                    self.assertEqual(rc1, 0)
                    scribe_data = json.loads("".join(chunks))
                    self.assertEqual(scribe_data["turns_digested"], 1)
                    chunks.clear()

                    rc2 = cli_main(["--home", tmp, "swarm", "midnight", "--json"])
                    self.assertEqual(rc2, 0)
                    arch_data = json.loads("".join(chunks))
                    self.assertEqual(len(arch_data["archived_cards"]), 1)
                    chunks.clear()

                    rc3 = cli_main(
                        ["--home", tmp, "swarm", "scout", "--query", "port 8787", "--domain", "infra", "--json"]
                    )
                    self.assertEqual(rc3, 0)
                    scout_data = json.loads("".join(chunks))
                    self.assertTrue(scout_data["found"])
                    self.assertIn("8787", scout_data["briefing_text"])


class TestShinobuSwarmBridge(unittest.TestCase):
    def test_shinobu_tools_and_closed_loop_hooks(self) -> None:
        runner = _make_tri_worker_runner()
        with tempfile.TemporaryDirectory() as tmp:
            kernel = _FakeKernel(Path(tmp))
            bus = _FakeHookBus()
            plugin = create_plugin(home=Path(tmp) / "reiji_home", runner=runner)
            try:
                tools = plugin.register_tools(kernel)
                tool_map = {t.name: t for t in tools}
                self.assertIn("reiji_scout_deep", tool_map)
                self.assertIn("reiji_swarm_pulse", tool_map)
                self.assertIn("reiji_security_audit", tool_map)
                self.assertEqual(len(tools), 7)

                plugin.register_hooks(bus)
                self.assertIn("done", bus.handlers)
                self.assertIn("episode_consolidated", bus.handlers)

                done_results = bus.emit(
                    "done",
                    {
                        "episode_id": "ep_int",
                        "user_message": "Hey, set up SwarmCoordinator on port 8787.",
                        "assistant_message": "Done, telemetry banner and bridge hooks wired.",
                    },
                )
                self.assertEqual(len(done_results), 1)
                self.assertEqual(len(done_results[0]["turn_ids"]), 2)
                self.assertIsNotNone(done_results[0]["scribe_report"])
                self.assertEqual(done_results[0]["scribe_report"].turns_digested, 2)

                arch_results = bus.emit("episode_consolidated", {})
                self.assertEqual(len(arch_results), 1)
                self.assertIsInstance(arch_results[0], ArchivistReport)
                self.assertEqual(len(arch_results[0].archived_cards), 1)

                scout_raw = tool_map["reiji_scout_deep"].handler(
                    query="Which port is reserved?",
                    domain="infra",
                    hint_tags=["swarm"],
                )
                scout_payload = json.loads(scout_raw)
                self.assertTrue(scout_payload["ok"])
                self.assertTrue(scout_payload["found"])
                self.assertIn("8787", scout_payload["tail_block"])

                pulse_raw = tool_map["reiji_swarm_pulse"].handler(format="both")
                pulse_payload = json.loads(pulse_raw)
                self.assertTrue(pulse_payload["ok"])
                self.assertIn("L0 Cache: 100% LOCKED", pulse_payload["banner"])
                self.assertEqual(pulse_payload["summary"]["scribe"]["runs"], 1)
                self.assertEqual(pulse_payload["summary"]["scout"]["runs"], 1)
                self.assertEqual(pulse_payload["summary"]["archivist"]["runs"], 1)
            finally:
                plugin.close()

    def test_shinobu_bridge_graceful_when_unconfigured(self) -> None:
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in (ENV_FLASH_BASE_URL, ENV_FLASH_API_KEY)
        }
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, clean_env, clear=True):
                plugin = ReijiShinobuPlugin(home=Path(tmp))
                try:
                    tools = {t.name: t for t in plugin.build_tools()}
                    pulse_payload = json.loads(tools["reiji_swarm_pulse"].handler())
                    self.assertTrue(pulse_payload["ok"])
                    self.assertEqual(pulse_payload["summary"]["runner"]["status"], "unconfigured")
                    self.assertIn("runner=unconfigured", pulse_payload["banner"])

                    scout_payload = json.loads(tools["reiji_scout_deep"].handler(query="anything"))
                    self.assertFalse(scout_payload["ok"])
                    self.assertTrue(scout_payload["abstained"])

                    bus = _FakeHookBus()
                    plugin.register_hooks(bus)
                    done_res = bus.emit("done", {"episode_id": "ep_x", "role": "user", "content": "hi"})
                    self.assertEqual(len(done_res[0]["turn_ids"]), 1)
                    self.assertIsNone(done_res[0]["scribe_report"])
                    arch_res = bus.emit("episode_consolidated", {})
                    self.assertEqual(arch_res, [None])
                finally:
                    plugin.close()


if __name__ == "__main__":
    unittest.main()
