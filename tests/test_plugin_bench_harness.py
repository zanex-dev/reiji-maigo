"""Unit tests for the six-dimensional adversarial benchmark harness plugin."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.bench_harness import (
    BENCH_DIMENSIONS,
    PLUGIN_NAME,
    BenchHarnessPlugin,
    ReijiBenchHarness,
    _DeterministicBenchRunner,
    create_plugin,
)
from reiji_maigo.swarm.runner import (
    ENV_FLASH_API_KEY,
    ENV_FLASH_BASE_URL,
    SwarmCallResult,
    SwarmRunnerError,
    compute_prefix_sha256,
)

_SEVEN_DEC_RE = re.compile(r"^-?\d+\.\d{7}$")


class _StubKernel:
    """Minimal microkernel stub capturing CLI registrations."""

    def __init__(self) -> None:
        self.cli_commands: dict[str, tuple[object, str, str]] = {}

    def register_cli(
        self,
        command: str,
        handler: object,
        *,
        help_text: str = "",
        plugin_name: str = "",
    ) -> None:
        self.cli_commands[command] = (handler, help_text, plugin_name)


class _HallucinatingRunner:
    """Adversarial mock runner that hallucinates on abstention queries to verify test rigor."""

    def __init__(self) -> None:
        self._base = _DeterministicBenchRunner()

    def run_shadow_json(
        self,
        *,
        role: str,
        locked_system_prefix: str,
        user_payload: str,
    ) -> SwarmCallResult:
        if role == "scout" and "zephyr-q" in user_payload.lower():
            data = {
                "status": "final",
                "final_briefing": {
                    "found": True,
                    "abstained": False,
                    "briefing_text": "Fabricated P-521 key approved by Alice.",
                    "cited_card_ids": ["card_fake"],
                    "cited_spans": [],
                },
            }
            raw = json.dumps(data)
            return SwarmCallResult(
                data=data,
                raw_text=raw,
                prompt_tokens=120,
                completion_tokens=40,
                elapsed_ms=2,
                prefix_sha256=compute_prefix_sha256(locked_system_prefix),
            )
        return self._base.run_shadow_json(
            role=role,
            locked_system_prefix=locked_system_prefix,
            user_payload=user_payload,
        )


class TestBenchHarnessPlugin(unittest.TestCase):
    def test_dimensions_and_list_cases(self) -> None:
        harness = ReijiBenchHarness()
        self.assertEqual(
            BENCH_DIMENSIONS,
            (
                "single_hop",
                "multi_hop",
                "temporal_foresight",
                "tepa_reversal",
                "abstention",
                "anti_poison_banter",
            ),
        )
        all_cases = harness.list_cases()
        self.assertEqual(len(all_cases), 6)
        self.assertEqual({c["dimension"] for c in all_cases}, set(BENCH_DIMENSIONS))

        tepa_cases = harness.list_cases(dimension="tepa_reversal")
        self.assertEqual(len(tepa_cases), 1)
        self.assertEqual(tepa_cases[0]["dimension"], "tepa_reversal")

        with self.assertRaises(ValueError):
            harness.list_cases(dimension="nonexistent_dim")

    def test_full_six_dim_deterministic_suite_and_economics(self) -> None:
        harness = ReijiBenchHarness()
        report = harness.run_suite(mode="deterministic")

        self.assertEqual(report["mode"], "deterministic")
        self.assertTrue(report["sandbox_isolated"])
        self.assertEqual(report["total_cases"], 6)
        self.assertEqual(report["passed_cases"], 6)
        self.assertEqual(report["failed_cases"], 0)
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertGreaterEqual(report["latency_p50_ms"], 0.0)
        self.assertGreaterEqual(report["latency_p95_ms"], report["latency_p50_ms"])
        self.assertGreater(report["avg_compression_ratio"], 0.0)

        # Verify every dimension achieved 100% pass rate.
        by_dim = report["by_dimension"]
        self.assertEqual(set(by_dim.keys()), set(BENCH_DIMENSIONS))
        for dim in BENCH_DIMENSIONS:
            self.assertEqual(by_dim[dim]["total"], 1)
            self.assertEqual(by_dim[dim]["passed"], 1)
            self.assertEqual(by_dim[dim]["pass_rate"], 1.0)

        # Verify granular per-dimension adversarial assertions.
        cases_by_dim = {c["dimension"]: c for c in report["cases"]}

        # 1. single_hop
        sh = cases_by_dim["single_hop"]
        self.assertTrue(sh["passed"], sh)
        self.assertGreater(sh["compression_ratio"], 0.0)
        self.assertGreaterEqual(sh["rounds_used"], 2)

        # 2. multi_hop (dual-card graph link + cross-session raw evidence)
        mh = cases_by_dim["multi_hop"]
        self.assertTrue(mh["passed"], mh)
        self.assertGreaterEqual(mh["metrics"]["graph_links_count"], 1)
        self.assertIn("ep_mh_alpha", mh["metrics"]["evidence_episodes"])
        self.assertIn("ep_mh_beta", mh["metrics"]["evidence_episodes"])

        # 3. temporal_foresight
        tf = cases_by_dim["temporal_foresight"]
        self.assertTrue(tf["passed"], tf)
        self.assertGreaterEqual(tf["metrics"]["expired_cards_count"], 1)
        self.assertGreaterEqual(tf["metrics"]["active_window_cards_count"], 1)

        # 4. tepa_reversal (A -> B -> revoke B & restore A, zero ghost recall of B)
        tr = cases_by_dim["tepa_reversal"]
        self.assertTrue(tr["passed"], tr)
        self.assertEqual(tr["metrics"]["card_a_status"], "superseded")
        self.assertEqual(tr["metrics"]["card_b_status"], "revoked")
        self.assertEqual(tr["metrics"]["ghost_b_recalled"], 0)
        self.assertEqual(len(tr["metrics"]["recalled_active_ids"]), 1)

        # 5. abstention
        ab = cases_by_dim["abstention"]
        self.assertTrue(ab["passed"], ab)
        self.assertTrue(ab["metrics"]["abstained"])
        self.assertFalse(ab["metrics"]["found"])

        # 6. anti_poison_banter
        ap = cases_by_dim["anti_poison_banter"]
        self.assertTrue(ap["passed"], ap)
        self.assertGreaterEqual(ap["metrics"]["skipped_banter"], 3)
        self.assertEqual(ap["metrics"]["pool_events_count"], 0)
        self.assertEqual(ap["metrics"]["cards_count"], 0)
        self.assertGreaterEqual(ap["metrics"]["poison_detected"], 1)
        self.assertGreaterEqual(ap["metrics"]["poison_blocked"], 1)
        self.assertGreaterEqual(ap["metrics"]["redacted_secrets"], 2)

        # Verify 7-decimal ROI economics and two-way reconciliation.
        econ = report["economics"]
        for key in ("flash_cost_usd", "gross_saved_usd", "net_roi_usd", "roi_multiple"):
            self.assertRegex(
                str(econ[key]),
                _SEVEN_DEC_RE,
                f"Expected 7-decimal format for {key}, got {econ[key]!r}",
            )
        self.assertTrue(econ["reconciled_ok"])
        self.assertGreater(econ["entries_count"], 0)
        self.assertGreater(econ["prompt_tokens"], 0)

        # Verify Markdown scorecard rendering.
        md = harness.render_markdown_scorecard(report)
        self.assertIn("# Reiji-Maigo Six-Dimensional Adversarial Benchmark Scorecard", md)
        self.assertIn("6/6 (100.00%)", md)
        for dim in BENCH_DIMENSIONS:
            self.assertIn(f"`{dim}`", md)
        self.assertIn(econ["net_roi_usd"], md)

    def test_sandbox_isolation_never_pollutes_user_home(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reiji_user_home_") as user_home:
            engine = ReijiMemoryEngine(home=user_home, load_plugins=False)
            try:
                engine.append_turn("ep_user_primary", "user", "Primary user memory stays untouched.")
            finally:
                engine.close()

            home_path = Path(user_home)
            before_files = sorted(p.name for p in home_path.iterdir())
            db_candidates = [p for p in home_path.iterdir() if p.suffix == ".db"]
            self.assertEqual(len(db_candidates), 1)
            db_file = db_candidates[0]
            before_hash = hashlib.sha256(db_file.read_bytes()).hexdigest()

            harness = ReijiBenchHarness()
            report = harness.run_suite(
                mode="deterministic",
                dimensions=["single_hop", "tepa_reversal"],
                home=user_home,
            )
            self.assertEqual(report["passed_cases"], 2)

            after_files = sorted(p.name for p in home_path.iterdir())
            after_hash = hashlib.sha256(db_file.read_bytes()).hexdigest()
            self.assertEqual(before_files, after_files)
            self.assertEqual(before_hash, after_hash)

    def test_live_mode_requires_runner_and_supports_custom_runner(self) -> None:
        harness = ReijiBenchHarness()

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in (ENV_FLASH_BASE_URL, ENV_FLASH_API_KEY)
        }
        with patch.dict(os.environ, clean_env, clear=True):
            with self.assertRaises(SwarmRunnerError):
                harness.run_suite(mode="live", dimensions=["single_hop"])

        # Custom live runner passes cleanly.
        live_ok = harness.run_suite(
            mode="live",
            dimensions=["single_hop", "abstention"],
            runner=_DeterministicBenchRunner(),
        )
        self.assertEqual(live_ok["mode"], "live")
        self.assertEqual(live_ok["passed_cases"], 2)
        self.assertEqual(live_ok["pass_rate"], 1.0)

        # Hallucinating runner must fail abstention check.
        live_fail = harness.run_suite(
            mode="live",
            dimensions=["abstention"],
            runner=_HallucinatingRunner(),
        )
        self.assertEqual(live_fail["passed_cases"], 0)
        self.assertEqual(live_fail["failed_cases"], 1)
        self.assertFalse(live_fail["cases"][0]["checks"]["abstained_true"])

    def test_plugin_registration_and_cli_subcommands(self) -> None:
        plugin = create_plugin()
        self.assertIsInstance(plugin, BenchHarnessPlugin)
        self.assertEqual(plugin.name, PLUGIN_NAME)

        kernel = _StubKernel()
        plugin.register_cli(kernel)
        self.assertIn("bench", kernel.cli_commands)
        handler, _help_text, registered_plugin = kernel.cli_commands["bench"]
        self.assertEqual(registered_plugin, "bench_harness")

        parser = argparse.ArgumentParser(prog="bench")
        handler.configure_parser(parser)  # type: ignore[attr-defined]

        # 1. list-cases --json
        ns_list = parser.parse_args(["list-cases", "--json"])
        buf_list = io.StringIO()
        with redirect_stdout(buf_list):
            rc_list = handler(args=ns_list)  # type: ignore[operator]
        self.assertEqual(rc_list, 0)
        parsed_list = json.loads(buf_list.getvalue())
        self.assertEqual(len(parsed_list["cases"]), 6)

        # 2. run --mode deterministic --dimension single_hop --markdown --output <file>
        with tempfile.TemporaryDirectory(prefix="reiji_bench_out_") as out_dir:
            out_md = Path(out_dir) / "scorecard.md"
            ns_run = parser.parse_args(
                [
                    "run",
                    "--mode",
                    "deterministic",
                    "--dimension",
                    "single_hop,abstention",
                    "--markdown",
                    "--output",
                    str(out_md),
                ]
            )
            buf_run = io.StringIO()
            with redirect_stdout(buf_run):
                rc_run = handler(args=ns_run)  # type: ignore[operator]
            self.assertEqual(rc_run, 0)
            self.assertTrue(out_md.is_file())
            md_content = out_md.read_text(encoding="utf-8")
            self.assertIn("2/2 (100.00%)", md_content)
            self.assertIn("`single_hop`", md_content)
            self.assertIn("`abstention`", md_content)

        # 3. extra_args fallback compatibility
        fallback_ns = argparse.Namespace(
            extra_args=["run", "--mode", "deterministic", "--dimension", "abstention"],
            json=True,
        )
        buf_fb = io.StringIO()
        with redirect_stdout(buf_fb):
            rc_fb = handler(args=fallback_ns)  # type: ignore[operator]
        self.assertEqual(rc_fb, 0)
        fb_json = json.loads(buf_fb.getvalue())
        self.assertEqual(fb_json["total_cases"], 1)
        self.assertEqual(fb_json["passed_cases"], 1)

    def test_source_hygiene_no_forbidden_literals(self) -> None:
        src_path = (
            Path(__file__).resolve().parents[1]
            / "reiji_maigo"
            / "plugins"
            / "bench_harness.py"
        )
        source = src_path.read_text(encoding="utf-8")
        forbidden_tokens = (
            "AppData",
            "acerola" + "_orion",
            "C:\\Users\\",
            "/Users/",
        )
        for token in forbidden_tokens:
            self.assertNotIn(token, source, f"Forbidden literal found in source: {token!r}")


if __name__ == "__main__":
    unittest.main()
