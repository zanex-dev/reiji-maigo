"""Integration, bare-kernel CLI, Shinobu bridge, and clean-room gate tests for reiji-maigo."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

from reiji_maigo.cli import build_parser, main as cli_main
from reiji_maigo.core.contracts import SourceSpan
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.engine import HIBERNATED_CARD_TYPE, ReijiMemoryEngine

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "reiji_maigo"
SHINOBU_PUBLIC_ROOT = REPO_ROOT.parent / "shinobu-public"


class TestReijiMemoryEngineFlow(unittest.TestCase):
    """Verify two-tier ActivePool <-> CardStore lifecycle transitions."""

    def test_archive_completed_event_preserves_all_spans_and_removes_from_pool(self) -> None:
        events_emitted: list[dict[str, object]] = []
        with ReijiMemoryEngine(db_path=":memory:", load_plugins=False) as engine:
            engine.kernel.register_hooks("pool.archived", lambda p: events_emitted.append(dict(p)))
            t1 = engine.append_turn("ep_arch", "user", "设计零时迷子双层归档流程")
            t2 = engine.append_turn("ep_arch", "assistant", "已确认保留全部原文坐标")
            t3 = engine.append_turn("ep_arch", "user", "任务完成，归档到长期记忆卡片")

            s1 = SourceSpan(episode_id="ep_arch", turn_start=t1, turn_end=t2)
            s2 = SourceSpan(episode_id="ep_arch", turn_start=t3, turn_end=t3)

            ev = engine.pool.create_event("双层归档协议", "梳理归档契约", s1, now_iso="2026-11-27T10:00:00Z")
            engine.pool.append_step(ev.event_id, "完成验收闭环", s2, status="done", now_iso="2026-11-27T11:00:00Z")

            card = engine.archive_pool_event(
                ev.event_id,
                domain="proj",
                tags=["reiji-maigo", "archive"],
                state={"outcome": "shipped"},
                hibernate=False,
                now_iso="2026-11-27T12:00:00Z",
            )

            self.assertEqual(card.domain, "proj")
            self.assertEqual(card.lifecycle_status, "active")
            self.assertEqual(card.spans, (s1, s2))
            self.assertFalse(card.state["hibernated"])
            self.assertEqual(card.state["pool_state"], "archived")
            self.assertEqual(card.state["outcome"], "shipped")

            # Removed from visible ActivePool and snapshot.
            self.assertEqual(engine.pool.list_events(), [])
            self.assertNotIn(ev.event_id, engine.pool_snapshot(include_undigested=False))

            # Recallable via Level-1 and verifiable via Level-2 raw turns.
            recalled = engine.recall_cards("proj", ["archive"])
            self.assertEqual([c.card_id for c in recalled], [card.card_id])
            evidence = engine.read_evidence(card.card_id, pad=0)
            self.assertEqual([row["turn_id"] for row in evidence], [t1, t2, t3])
            self.assertEqual(len(events_emitted), 1)
            self.assertEqual(events_emitted[0]["card_id"], card.card_id)

    def test_hibernate_and_hot_wake_round_trip(self) -> None:
        woken_hooks: list[dict[str, object]] = []
        with ReijiMemoryEngine(db_path=":memory:", load_plugins=False) as engine:
            engine.kernel.register_hooks("card.woken", lambda p: woken_hooks.append(dict(p)))
            t1 = engine.append_turn("ep_hib", "user", "调研 CJK Bigram 分词性能")
            t2 = engine.append_turn("ep_hib", "assistant", "先挂起等待后续基准数据")
            t3 = engine.append_turn("ep_hib", "user", "基准数据已齐，重新唤醒继续推进")

            s1 = SourceSpan(episode_id="ep_hib", turn_start=t1, turn_end=t1)
            s2 = SourceSpan(episode_id="ep_hib", turn_start=t2, turn_end=t2)
            s3 = SourceSpan(episode_id="ep_hib", turn_start=t3, turn_end=t3)

            ev = engine.pool.create_event("CJK 分词压测", "搭建初始基线", s1, now_iso="2026-11-20T08:00:00Z")
            engine.pool.append_step(ev.event_id, "暂停等待数据集", s2, now_iso="2026-11-20T09:00:00Z")
            engine.pool.increment_dormant_nights([ev.event_id], delta=7)

            hib_card = engine.archive_pool_event(
                ev.event_id,
                domain="research",
                tags=["fts5", "cjk"],
                state={"blocker": "dataset_pending"},
                hibernate=True,
                now_iso="2026-11-27T00:00:00Z",
            )

            self.assertEqual(hib_card.card_type, HIBERNATED_CARD_TYPE)
            self.assertTrue(hib_card.state["hibernated"])
            self.assertEqual(hib_card.state["pool_state"], "hibernated")
            self.assertEqual(hib_card.spans, (s1, s2))
            self.assertEqual(engine.pool.list_events(), [])

            # Level-1 recall surfaces the hibernated card before wake-up.
            recalled = engine.recall_cards("research", ["fts5"])
            self.assertEqual([c.card_id for c in recalled], [hib_card.card_id])

            # Hot-wake back into ActivePool with new step and span.
            woken = engine.wake_hibernated_card(
                hib_card.card_id,
                "数据集到位，恢复执行压测",
                s3,
                now_iso="2026-11-27T14:00:00Z",
            )
            self.assertEqual(woken.event_id, ev.event_id)
            self.assertEqual(woken.status, "active")
            self.assertEqual(woken.dormant_nights, 0)
            self.assertEqual(woken.spans, (s1, s2, s3))
            self.assertEqual(
                woken.steps,
                ("搭建初始基线", "暂停等待数据集", "数据集到位，恢复执行压测"),
            )
            self.assertEqual([e.event_id for e in engine.pool.list_events()], [ev.event_id])
            self.assertIn("数据集到位，恢复执行压测", engine.pool_snapshot(include_undigested=False))
            self.assertEqual(len(woken_hooks), 1)
            self.assertEqual(woken_hooks[0]["event_id"], ev.event_id)

            # Deactivated hibernated snapshot no longer leaks into active Level-1 recall.
            self.assertEqual(engine.recall_cards("research", ["fts5"]), [])

    def test_wake_standalone_hibernated_card_and_guards(self) -> None:
        with ReijiMemoryEngine(db_path=":memory:", load_plugins=False) as engine:
            s1 = SourceSpan(episode_id="ep_sa", turn_start=1, turn_end=2)
            s2 = SourceSpan(episode_id="ep_sa", turn_start=3, turn_end=3)

            # Non-hibernated card cannot be woken.
            normal = engine.cards.create_card(
                domain="infra",
                card_type="fact",
                title="普通配置事实",
                state={"k": "v"},
                tags=["config"],
                spans=[s1],
            )
            with self.assertRaises(ValueError):
                engine.wake_hibernated_card(normal.card_id, "try wake", s2)

            # Missing IDs raise KeyError.
            with self.assertRaises(KeyError):
                engine.archive_pool_event("ev_missing", domain="proj", tags=["x"], state={})
            with self.assertRaises(KeyError):
                engine.wake_hibernated_card("card_missing", "step", s2)

            # Standalone hibernated card without prior pool_events row can still wake cleanly.
            hib = engine.cards.create_card(
                domain="infra",
                card_type=HIBERNATED_CARD_TYPE,
                title="独立休眠巡检任务",
                state={
                    "hibernated": True,
                    "pool_state": "hibernated",
                    "steps": ["初始排查"],
                    "checkpoint_summary": "已完成初检",
                },
                tags=["ops"],
                spans=[s1],
            )
            woken = engine.wake_hibernated_card(hib.card_id, "继续深度巡检", s2)
            self.assertEqual(woken.status, "active")
            self.assertEqual(woken.checkpoint_summary, "已完成初检")
            self.assertEqual(woken.steps, ("初始排查", "继续深度巡检"))
            self.assertEqual(woken.spans, (s1, s2))


class TestBareKernelCLI(unittest.TestCase):
    """Verify bare-kernel CLI execution via subprocess with plugins disabled."""

    def _run_cli(self, args: list[str], home: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["REIJI_MAIGO_DISABLE_PLUGINS"] = "1"
        env["REIJI_MAIGO_HOME"] = str(home)
        env["PYTHONUTF8"] = "1"
        existing_py = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{REPO_ROOT}{os.pathsep}{existing_py}" if existing_py else str(REPO_ROOT)
        )
        return subprocess.run(
            [sys.executable, "-m", "reiji_maigo", *args],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def test_bare_kernel_cli_help_status_pool_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "reiji_home"

            res_help = self._run_cli(["--help"], home)
            self.assertEqual(res_help.returncode, 0, res_help.stderr)
            for cmd in ("status", "pool", "recall"):
                self.assertIn(cmd, res_help.stdout)

            res_status = self._run_cli(["status", "--json"], home)
            self.assertEqual(res_status.returncode, 0, res_status.stderr)
            status_data = json.loads(res_status.stdout)
            self.assertTrue(status_data["plugins_disabled"])
            self.assertEqual(status_data["counts"]["raw_turns"], 0)

            res_pool = self._run_cli(["pool"], home)
            self.assertEqual(res_pool.returncode, 0, res_pool.stderr)
            self.assertIn("[ActivePool]", res_pool.stdout)

            res_recall_empty = self._run_cli(
                ["recall", "--domain", "proj", "--tag", "alpha", "--json"],
                home,
            )
            self.assertEqual(res_recall_empty.returncode, 0, res_recall_empty.stderr)
            empty_data = json.loads(res_recall_empty.stdout)
            self.assertEqual(empty_data["cards"], [])

            # Populate data and verify end-to-end CLI turn/pool/recall/evidence.
            res_turn = self._run_cli(
                ["turn", "--episode", "ep_cli", "--role", "user", "--content", "验证零时迷子中文全文检索", "--json"],
                home,
            )
            self.assertEqual(res_turn.returncode, 0, res_turn.stderr)
            turn_id = json.loads(res_turn.stdout)["turn_id"]

            with ReijiMemoryEngine(home=home, disable_plugins=True, load_plugins=False) as engine:
                ev = engine.pool.create_event(
                    "CLI 冒烟任务",
                    "写入测试步骤",
                    SourceSpan("ep_cli", turn_id, turn_id),
                )
                card = engine.cards.create_card(
                    domain="proj",
                    card_type="decision",
                    title="采用纯标准库架构",
                    state={"zero_deps": True},
                    tags=["architecture", "stdlib"],
                    spans=[SourceSpan("ep_cli", turn_id, turn_id)],
                )
                event_id = ev.event_id
                card_id = card.card_id

            res_pool_list = self._run_cli(["pool", "--list", "--json"], home)
            self.assertEqual(res_pool_list.returncode, 0, res_pool_list.stderr)
            listed = json.loads(res_pool_list.stdout)
            self.assertEqual([item["event_id"] for item in listed], [event_id])

            res_recall = self._run_cli(
                ["recall", "--domain", "proj", "--tag", "stdlib", "--json"],
                home,
            )
            self.assertEqual(res_recall.returncode, 0, res_recall.stderr)
            recalled = json.loads(res_recall.stdout)
            self.assertEqual([c["card_id"] for c in recalled["cards"]], [card_id])

            res_evidence = self._run_cli(["recall", "--evidence", card_id, "--json"], home)
            self.assertEqual(res_evidence.returncode, 0, res_evidence.stderr)
            ev_data = json.loads(res_evidence.stdout)
            self.assertEqual(ev_data["turns"][0]["turn_id"], turn_id)

            res_deep = self._run_cli(
                ["recall", "--query", "中文全文检索", "--json"],
                home,
            )
            self.assertEqual(res_deep.returncode, 0, res_deep.stderr)
            deep_data = json.loads(res_deep.stdout)
            self.assertEqual(len(deep_data["deep_hits"]), 1)

    def test_plugin_cli_slot_registered_in_parser(self) -> None:
        kernel = ReijiKernel(disable_plugins=True)
        kernel.register_cli("ping", lambda _engine, _args: 0, help_text="Plugin ping command")
        parser = build_parser(plugin_commands=kernel.cli_commands)
        parsed = parser.parse_args(["ping"])
        self.assertEqual(parsed.command, "ping")
        self.assertEqual(parsed._plugin_spec.name, "ping")

    def test_in_process_cli_main_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = cli_main(["--home", tmp, "status"])
            self.assertEqual(code, 0)


class TestShinobuBridge(unittest.TestCase):
    """Verify standalone zero-ImportError Shinobu bridge and live ShinobuKernel attachment."""

    def test_standalone_import_without_shinobu_core(self) -> None:
        # Run in an isolated subprocess that blocks shinobu_core completely.
        script = (
            "import json, sys, tempfile\n"
            "sys.modules['shinobu_core'] = None\n"
            "sys.modules['shinobu_core.protocol'] = None\n"
            "sys.modules['shinobu_core.protocol.engine'] = None\n"
            "from reiji_maigo.bridges.shinobu import create_plugin\n"
            "with tempfile.TemporaryDirectory() as td:\n"
            "    plugin = create_plugin(home=td)\n"
            "    try:\n"
            "        tools = plugin.build_tools()\n"
            "        names = [t.name for t in tools]\n"
            "        snap = json.loads(tools[0].handler())\n"
            "        print(json.dumps({'names': names, 'ok': snap['ok']}))\n"
            "    finally:\n"
            "        plugin.close()\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertEqual(
            payload["names"],
            [
                "reiji_pool_snapshot",
                "reiji_update_pool",
                "reiji_recall_cards",
                "reiji_read_evidence",
                "reiji_scout_deep",
                "reiji_swarm_pulse",
                "reiji_security_audit",
            ],
        )
        self.assertTrue(payload["ok"])

    def test_attach_to_real_shinobu_kernel_in_memory(self) -> None:
        if not SHINOBU_PUBLIC_ROOT.is_dir():
            self.skipTest("shinobu-public sibling repository not present")

        added_path = False
        shinobu_str = str(SHINOBU_PUBLIC_ROOT)
        if shinobu_str not in sys.path:
            sys.path.insert(0, shinobu_str)
            added_path = True

        try:
            shinobu_kernel_mod = importlib.import_module("shinobu_core.core.kernel")
            shinobu_engine_mod = importlib.import_module("shinobu_core.protocol.engine")
            ShinobuKernel = shinobu_kernel_mod.ShinobuKernel
            ToolSpec = shinobu_engine_mod.ToolSpec

            from reiji_maigo.bridges.shinobu import create_plugin

            with tempfile.TemporaryDirectory() as tmp:
                shinobu_home = Path(tmp) / "shinobu_home"
                reiji_home = Path(tmp) / "reiji_home"
                kernel = ShinobuKernel(home=shinobu_home)
                plugin = create_plugin(home=reiji_home)
                try:
                    record = kernel.attach_plugin("reiji_maigo", plugin)
                    self.assertEqual(record.status, "loaded", record.error)
                    self.assertIsNone(record.error)

                    expected_tools = (
                        "reiji_pool_snapshot",
                        "reiji_update_pool",
                        "reiji_recall_cards",
                        "reiji_read_evidence",
                        "reiji_scout_deep",
                        "reiji_swarm_pulse",
                        "reiji_security_audit",
                    )
                    for tool_name in expected_tools:
                        tool = kernel.get_tool(tool_name)
                        self.assertIsNotNone(tool, f"Missing tool: {tool_name}")
                        self.assertIsInstance(tool, ToolSpec)

                    # Exercise hook -> update_pool -> hibernate -> recall -> wake -> archive -> read_evidence.
                    turn_results = kernel.hooks.emit(
                        "reiji.turn.recorded",
                        {
                            "episode_id": "ep_bridge",
                            "role": "user",
                            "content": "联调忍野忍微内核挂载零时迷子记忆插件",
                        },
                    )
                    turn_id = turn_results[0]
                    self.assertIsInstance(turn_id, int)

                    t_update = kernel.get_tool("reiji_update_pool")
                    t_snap = kernel.get_tool("reiji_pool_snapshot")
                    t_recall = kernel.get_tool("reiji_recall_cards")
                    t_evidence = kernel.get_tool("reiji_read_evidence")
                    assert t_update and t_snap and t_recall and t_evidence

                    created = json.loads(
                        t_update.handler(
                            action="create",
                            title="Shinobu 联调任务",
                            step="挂接四个尾部工具",
                            span={"episode_id": "ep_bridge", "turn_start": turn_id, "turn_end": turn_id},
                        )
                    )
                    ev_id = created["event"]["event_id"]

                    snap_res = json.loads(t_snap.handler())
                    self.assertIn("Shinobu 联调任务", snap_res["snapshot"])

                    hib_res = json.loads(
                        t_update.handler(
                            action="hibernate",
                            event_id=ev_id,
                            domain="proj",
                            tags=["shinobu", "bridge"],
                            state={"note": "waiting"},
                        )
                    )
                    hib_card_id = hib_res["card"]["card_id"]

                    rec_res = json.loads(
                        t_recall.handler(domain="proj", query_tags=["bridge"])
                    )
                    self.assertEqual([c["card_id"] for c in rec_res["cards"]], [hib_card_id])

                    evid_res = json.loads(t_evidence.handler(card_id=hib_card_id, pad=0))
                    self.assertEqual(evid_res["turns"][0]["turn_id"], turn_id)

                    woken_res = json.loads(
                        t_update.handler(
                            action="wake",
                            card_id=hib_card_id,
                            step="联调通过，恢复活跃",
                            span={"episode_id": "ep_bridge", "turn_start": turn_id, "turn_end": turn_id},
                        )
                    )
                    self.assertEqual(woken_res["event"]["status"], "active")

                    # Verify CLI registrar applies cleanly without error.
                    root_parser = argparse.ArgumentParser(prog="shinobu")
                    subparsers = root_parser.add_subparsers(dest="cmd")
                    failed_cli = kernel.apply_cli(subparsers)
                    self.assertEqual(failed_cli, [])
                finally:
                    plugin.close()
        finally:
            if added_path and shinobu_str in sys.path:
                sys.path.remove(shinobu_str)


class TestCleanRoomGate(unittest.TestCase):
    """Static clean-room scan across the entire reiji_maigo package."""

    def test_clean_room_zero_leaks_and_stdlib_only_top_level_imports(self) -> None:
        py_files = sorted(PKG_ROOT.rglob("*.py"))
        self.assertGreater(len(py_files), 0)

        # Build forbidden tokens dynamically so this test file stays clean too.
        forbidden_path_patterns = [
            re.compile(r"[A-Za-z]:\\\\Users\\\\", re.IGNORECASE),
            re.compile(r"[A-Za-z]:/Users/", re.IGNORECASE),
            re.compile(r"/Users/[A-Za-z0-9._-]+"),
            re.compile(r"/home/[A-Za-z0-9._-]+"),
            re.compile(r"AppData", re.IGNORECASE),
        ]
        forbidden_private_tokens = (
            "shinobu" + "_private",
            "orion" + "_private",
            "acerola" + "_orion",
        )
        allowed_top_modules = set(sys.stdlib_module_names) | {"__future__", "reiji_maigo"}

        for py_file in py_files:
            source = py_file.read_text(encoding="utf-8")
            rel = py_file.relative_to(REPO_ROOT)

            for pattern in forbidden_path_patterns:
                self.assertIsNone(
                    pattern.search(source),
                    f"Host path leak detected in {rel}: pattern={pattern.pattern}",
                )

            lower_src = source.lower()
            for token in forbidden_private_tokens:
                self.assertNotIn(
                    token,
                    lower_src,
                    f"Private package token {token!r} leaked in {rel}",
                )

            tree = ast.parse(source, filename=str(py_file))
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        self.assertIn(
                            top,
                            allowed_top_modules,
                            f"Non-stdlib top-level import {alias.name!r} in {rel}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        continue
                    module_name = node.module or ""
                    top = module_name.split(".", 1)[0]
                    self.assertIn(
                        top,
                        allowed_top_modules,
                        f"Non-stdlib top-level from-import {module_name!r} in {rel}",
                    )


if __name__ == "__main__":
    unittest.main()
