"""Unit tests for reiji_maigo.swarm.scribe (ScribeAgent and AsyncScribeWorker)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from reiji_maigo import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm.guards import (
    build_role_capabilities,
    get_pool_event_note,
    list_swarm_telemetry,
)
from reiji_maigo.swarm.runner import (
    CallableFlashRunner,
    SwarmJSONError,
    SwarmRunnerError,
    compute_prefix_sha256,
)
from reiji_maigo.swarm.scribe import (
    SCRIBE_L0_PREFIX,
    AsyncScribeWorker,
    ScribeAgent,
    ScribeDigestReport,
)


def _append_turn(engine: ReijiMemoryEngine, episode_id: str, role: str, content: str) -> int:
    """Append a raw conversation turn through the engine's database layer."""
    if hasattr(engine, "append_turn"):
        res = engine.append_turn(episode_id=episode_id, role=role, content=content)
        if isinstance(res, int):
            return res
        if isinstance(res, dict):
            return int(res.get("turn_id", res.get("id", 0)))
        return int(getattr(res, "turn_id", getattr(res, "id", 0)))
    res = engine.db.append_turn(episode_id=episode_id, role=role, content=content)
    if isinstance(res, int):
        return res
    if isinstance(res, dict):
        return int(res.get("turn_id", res.get("id", 0)))
    return int(getattr(res, "turn_id", getattr(res, "id", 0)))


class TestScribeAgent(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = ReijiMemoryEngine(
            home=Path(self._tmp.name),
            db_path=":memory:",
            load_plugins=False,
            disable_plugins=True,
        )
        self.caps = build_role_capabilities(self.engine, "scribe")

    def tearDown(self) -> None:
        self.engine.close()
        self._tmp.cleanup()

    def test_no_undigested_turns_makes_zero_model_calls(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_model(system: str, prompt: str) -> dict[str, Any]:
            calls.append((system, prompt))
            return {"skipped_banter": 0, "operations": []}

        scribe = ScribeAgent(self.caps, CallableFlashRunner(fake_model))
        report = scribe.digest_once(now_iso="2026-11-27T12:00:00+00:00")

        self.assertIsInstance(report, ScribeDigestReport)
        self.assertEqual(report.turns_digested, 0)
        self.assertEqual(report.watermark_after, 0)
        self.assertEqual(report.created_ids, ())
        self.assertEqual(report.updated_ids, ())
        self.assertEqual(report.skipped_banter, 0)
        self.assertGreaterEqual(report.elapsed_ms, 0)
        self.assertEqual(len(calls), 0)

    def test_pure_banter_advances_watermark_without_polluting_pool(self) -> None:
        t1 = _append_turn(self.engine, "ep_chat", "user", "哈哈哈哈早安呀，今天天气不错")
        t2 = _append_turn(self.engine, "ep_chat", "assistant", "早安！摸鱼一时爽，一直摸鱼一直爽")
        t3 = _append_turn(self.engine, "ep_chat", "user", "随便想想以后要不要养只猫，算了先不聊这个")

        seen_prefixes: list[str] = []

        def fake_model(system: str, prompt: str) -> dict[str, Any]:
            seen_prefixes.append(system)
            payload = json.loads(prompt)
            self.assertEqual(len(payload["undigested_turns"]), 3)
            return {
                "skipped_banter": 3,
                "operations": [],
            }

        scribe = ScribeAgent(self.caps, CallableFlashRunner(fake_model))
        report = scribe.digest_once(now_iso="2026-11-27T12:05:00+00:00")

        self.assertEqual(report.turns_digested, 3)
        self.assertEqual(report.watermark_after, max(t1, t2, t3))
        self.assertEqual(report.created_ids, ())
        self.assertEqual(report.updated_ids, ())
        self.assertEqual(report.skipped_banter, 3)

        # ActivePool remains clean (0 events)
        self.assertEqual(len(self.caps.list_events()), 0)
        self.assertEqual(self.caps.get_watermark("pool_digested"), max(t1, t2, t3))

        # Byte-locked L0 prefix verified
        self.assertEqual(seen_prefixes, [SCRIBE_L0_PREFIX])

        # Telemetry recorded with exact prefix sha256
        telemetry = list_swarm_telemetry(self.engine.db, role="scribe")
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["action"], "digest_once")
        self.assertEqual(telemetry[0]["metrics"]["skipped_banter"], 3)
        self.assertEqual(
            telemetry[0]["metrics"]["prefix_sha256"],
            compute_prefix_sha256(SCRIBE_L0_PREFIX),
        )

        # Second call with no new turns triggers 0 model calls
        report2 = scribe.digest_once(now_iso="2026-11-27T12:06:00+00:00")
        self.assertEqual(report2.turns_digested, 0)
        self.assertEqual(len(seen_prefixes), 1)

    def test_cross_turn_create_append_and_level2_detail_merge(self) -> None:
        # Batch 1: establish task
        t1 = _append_turn(
            self.engine,
            "ep_work",
            "user",
            "我们要把 PaymentGateway 的超时重试从 3 次改到 5 次，并加上指数退避。",
        )
        t2 = _append_turn(
            self.engine,
            "ep_work",
            "assistant",
            "好的，先确认配置文件在 billing/retry.yaml，初始退避 200ms。",
        )

        created_event_holder: list[str] = []

        def batch1_model(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, SCRIBE_L0_PREFIX)
            return {
                "skipped_banter": 0,
                "operations": [
                    {
                        "op": "create_event",
                        "title": "PaymentGateway 重试策略升级",
                        "scope": "quest",
                        "step": "确认将 PaymentGateway 重试次数调整为 5 次并启用指数退避",
                        "span": {"episode_id": "ep_work", "turn_start": t1, "turn_end": t2},
                        "detail": {
                            "summary": "PaymentGateway 重试改造启动",
                            "facts": [
                                "配置文件: billing/retry.yaml",
                                "初始退避: 200ms",
                            ],
                            "entities": ["PaymentGateway", "billing/retry.yaml"],
                        },
                    }
                ],
            }

        scribe1 = ScribeAgent(self.caps, CallableFlashRunner(batch1_model))
        rep1 = scribe1.digest_once(now_iso="2026-11-27T13:00:00+00:00")
        self.assertEqual(rep1.turns_digested, 2)
        self.assertEqual(rep1.watermark_after, t2)
        self.assertEqual(len(rep1.created_ids), 1)
        self.assertEqual(rep1.updated_ids, ())
        ev_id = rep1.created_ids[0]
        created_event_holder.append(ev_id)

        ev_after_1 = self.caps.get_event(ev_id)
        self.assertEqual(ev_after_1.title, "PaymentGateway 重试策略升级")
        self.assertEqual(len(ev_after_1.steps), 1)
        self.assertEqual(
            ev_after_1.spans,
            (SourceSpan(episode_id="ep_work", turn_start=t1, turn_end=t2),),
        )
        note1 = get_pool_event_note(self.engine.db, ev_id)
        self.assertIsNotNone(note1)
        assert note1 is not None
        self.assertEqual(note1["facts"], ["配置文件: billing/retry.yaml", "初始退避: 200ms"])

        # Batch 2: cross-turn continuation with pronouns + one banter turn
        t3 = _append_turn(self.engine, "ep_work", "user", "顺便喝口水哈哈")
        t4 = _append_turn(
            self.engine,
            "ep_work",
            "user",
            "刚才那个改完了，最大退避上限设成 3200ms，单测也全绿了。",
        )

        def batch2_model(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, SCRIBE_L0_PREFIX)
            payload = json.loads(prompt)
            self.assertIn(ev_id, payload["active_pool_snapshot"])
            return {
                "skipped_banter": 1,
                "operations": [
                    {
                        "op": "append_step",
                        "event_id": ev_id,
                        "step": "PaymentGateway 重试策略实现完成（最大退避 3200ms）且单测通过",
                        "status": "done",
                        "span": {"episode_id": "ep_work", "turn_start": t4, "turn_end": t4},
                        "detail": {
                            "summary": "PaymentGateway 5 次指数退避改造已验收通过",
                            "facts": [
                                "初始退避: 200ms",  # duplicate fact must be deduplicated
                                "最大退避上限: 3200ms",
                            ],
                            "entities": ["PaymentGateway", "retry_tests"],
                        },
                    }
                ],
            }

        scribe2 = ScribeAgent(self.caps, CallableFlashRunner(batch2_model))
        rep2 = scribe2.digest_once(now_iso="2026-11-27T13:10:00+00:00")
        self.assertEqual(rep2.turns_digested, 2)
        self.assertEqual(rep2.watermark_after, t4)
        self.assertEqual(rep2.created_ids, ())
        self.assertEqual(rep2.updated_ids, (ev_id,))
        self.assertEqual(rep2.skipped_banter, 1)

        ev_after_2 = self.caps.get_event(ev_id)
        self.assertEqual(ev_after_2.status, "done")
        self.assertEqual(len(ev_after_2.steps), 2)
        self.assertEqual(len(ev_after_2.spans), 2)
        self.assertEqual(
            ev_after_2.spans[-1],
            SourceSpan(episode_id="ep_work", turn_start=t4, turn_end=t4),
        )

        # Batch 3: touch_event and update_status (with alias normalization)
        t5 = _append_turn(self.engine, "ep_work", "user", "顺手复查了下 PaymentGateway 指标，暂时卡在压测环境")

        def batch3_model(system: str, prompt: str) -> dict[str, Any]:
            return {
                "skipped_banter": 0,
                "operations": [
                    {
                        "op": "touch_event",
                        "event_id": ev_id,
                    },
                    {
                        "op": "update_status",
                        "event_id": ev_id,
                        "status": "stalled",  # alias -> blocked
                        "detail": {"open_questions": ["等待压测环境配额"]},
                    },
                ],
            }

        scribe3 = ScribeAgent(self.caps, CallableFlashRunner(batch3_model))
        rep3 = scribe3.digest_once(now_iso="2026-11-27T13:20:00+00:00")
        self.assertEqual(rep3.watermark_after, t5)
        self.assertEqual(rep3.updated_ids, (ev_id,))
        ev_after_3 = self.caps.get_event(ev_id)
        self.assertEqual(ev_after_3.status, "blocked")
        self.assertGreaterEqual(ev_after_3.hit_count, 2)

        # Level-2 drawer detail merged cleanly without losing Batch 1/2 facts
        note3 = get_pool_event_note(self.engine.db, ev_id)
        self.assertIsNotNone(note3)
        assert note3 is not None
        self.assertEqual(note3["summary"], "PaymentGateway 5 次指数退避改造已验收通过")
        self.assertEqual(
            note3["facts"],
            [
                "配置文件: billing/retry.yaml",
                "初始退避: 200ms",
                "最大退避上限: 3200ms",
            ],
        )
        self.assertEqual(
            note3["entities"],
            ["PaymentGateway", "billing/retry.yaml", "retry_tests"],
        )
        self.assertEqual(note3["open_questions"], ["等待压测环境配额"])

    def test_out_of_bounds_span_clamped_and_error_preserves_watermark(self) -> None:
        t1 = _append_turn(self.engine, "ep_real", "user", "部署 Redis 缓存到 staging 节点")
        t2 = _append_turn(self.engine, "ep_real", "assistant", "已确认端口 6379 与密码策略")

        # 1. Model error must NOT advance watermark
        def failing_model(system: str, prompt: str) -> dict[str, Any]:
            raise SwarmRunnerError("upstream 503")

        failing_scribe = ScribeAgent(self.caps, CallableFlashRunner(failing_model))
        with self.assertRaises(SwarmRunnerError):
            failing_scribe.digest_once()
        self.assertEqual(self.caps.get_watermark("pool_digested"), 0)

        # 2. Malformed JSON must NOT advance watermark
        bad_json_scribe = ScribeAgent(self.caps, CallableFlashRunner(lambda s, p: "not-json"))
        with self.assertRaises(SwarmJSONError):
            bad_json_scribe.digest_once()
        self.assertEqual(self.caps.get_watermark("pool_digested"), 0)

        # 3. Out-of-bounds episode_id, negative/huge/inverted turn_ids get clamped strictly
        def hallucinating_model(system: str, prompt: str) -> dict[str, Any]:
            return {
                "skipped_banter": 0,
                "operations": [
                    {
                        "op": "create_event",
                        "title": "Staging Redis 部署",
                        "scope": "invalid_scope_name",
                        "step": "确认 staging Redis 端口 6379 与认证配置",
                        "span": {
                            "episode_id": "ep_hallucinated",
                            "start_turn": 9999,
                            "end_turn": -5,
                        },
                        "detail": {"port": 6379},
                    },
                    {
                        # Non-existent event_id gracefully falls back to create_event
                        "op": "append_step",
                        "event_id": "ev_does_not_exist",
                        "title": "Staging 监控告警补齐",
                        "step": "补充 Redis 内存水位告警",
                        "span": {
                            "episode_id": "ep_real",
                            "start_turn": t2 + 50,
                            "end_turn": t2 + 100,
                        },
                    },
                ],
            }

        scribe = ScribeAgent(self.caps, CallableFlashRunner(hallucinating_model))
        rep = scribe.digest_once(now_iso="2026-11-27T14:00:00+00:00")
        self.assertEqual(len(rep.created_ids), 2)
        self.assertEqual(rep.watermark_after, t2)

        ev1 = self.caps.get_event(rep.created_ids[0])
        self.assertEqual(ev1.scope, "quest")
        self.assertEqual(
            ev1.spans,
            (SourceSpan(episode_id="ep_real", turn_start=t1, turn_end=t2),),
        )
        self.assertEqual(get_pool_event_note(self.engine.db, ev1.event_id), {"port": 6379})

        ev2 = self.caps.get_event(rep.created_ids[1])
        self.assertEqual(
            ev2.spans,
            (SourceSpan(episode_id="ep_real", turn_start=t2, turn_end=t2),),
        )

        telemetry = list_swarm_telemetry(self.engine.db, role="scribe")
        self.assertGreaterEqual(telemetry[0]["metrics"]["clamped_spans"], 2)


class TestAsyncScribeWorker(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = ReijiMemoryEngine(
            home=Path(self._tmp.name),
            db_path=str(Path(self._tmp.name) / "reiji.db"),
            load_plugins=False,
            disable_plugins=True,
        )
        self.caps = build_role_capabilities(self.engine, "scribe")

    def tearDown(self) -> None:
        self.engine.close()
        self._tmp.cleanup()

    def test_notify_is_non_blocking_and_respects_min_turns_and_flush(self) -> None:
        call_count = 0
        call_lock = threading.Lock()

        def slow_model(system: str, prompt: str) -> dict[str, Any]:
            nonlocal call_count
            time.sleep(0.03)
            with call_lock:
                call_count += 1
            payload = json.loads(prompt)
            turns = payload["undigested_turns"]
            return {
                "skipped_banter": len(turns),
                "operations": [],
            }

        scribe = ScribeAgent(self.caps, CallableFlashRunner(slow_model), batch_limit=12)
        with AsyncScribeWorker(scribe, min_turns=3, debounce_sec=0.02) as worker:
            # Two turns (< min_turns=3) must not auto-trigger before flush
            t1 = _append_turn(self.engine, "ep_async", "user", "hi 1")
            t0 = time.perf_counter()
            ok1 = worker.notify("ep_async", t1)
            dt1 = time.perf_counter() - t0
            self.assertTrue(ok1)
            self.assertLess(dt1, 0.01)  # far below model latency (30ms)

            t2 = _append_turn(self.engine, "ep_async", "user", "hi 2")
            worker.notify("ep_async", t2)
            time.sleep(0.06)
            self.assertEqual(call_count, 0)

            # Explicit flush drains even when pending < min_turns
            self.assertTrue(worker.flush(timeout=5.0))
            self.assertEqual(call_count, 1)
            self.assertEqual(self.caps.get_watermark("pool_digested"), t2)

            # Burst of 3 turns (>= min_turns) coalesces into a single debounced call
            t3 = _append_turn(self.engine, "ep_async", "user", "hi 3")
            t4 = _append_turn(self.engine, "ep_async", "user", "hi 4")
            t5 = _append_turn(self.engine, "ep_async", "user", "hi 5")
            worker.notify("ep_async", t3)
            worker.notify("ep_async", t4)
            worker.notify("ep_async", t5)

            # Wait for debounced auto-digest without calling flush
            deadline = time.monotonic() + 3.0
            while self.caps.get_watermark("pool_digested") < t5 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(self.caps.get_watermark("pool_digested"), t5)
            self.assertEqual(call_count, 2)
            self.assertIsNone(worker.last_error)

    def test_concurrent_notify_serializes_persistence_without_races(self) -> None:
        in_flight = 0
        max_concurrency = 0
        digested_turn_ids: list[int] = []
        state_lock = threading.Lock()
        write_lock = threading.Lock()

        def model_with_overlap_detector(system: str, prompt: str) -> dict[str, Any]:
            nonlocal in_flight, max_concurrency
            with state_lock:
                in_flight += 1
                if in_flight > max_concurrency:
                    max_concurrency = in_flight
            try:
                time.sleep(0.015)
                payload = json.loads(prompt)
                turns = payload["undigested_turns"]
                tids = [int(t["turn_id"]) for t in turns]
                with state_lock:
                    digested_turn_ids.extend(tids)
                ep = turns[0]["episode_id"]
                return {
                    "skipped_banter": 0,
                    "operations": [
                        {
                            "op": "create_event",
                            "title": f"并发批次 {min(tids)}-{max(tids)}",
                            "scope": "quest",
                            "step": f"消化轮次 {min(tids)} 到 {max(tids)}",
                            "span": {
                                "episode_id": ep,
                                "start_turn": min(tids),
                                "end_turn": max(tids),
                            },
                            "detail": {"batch_turn_ids": tids},
                        }
                    ],
                }
            finally:
                with state_lock:
                    in_flight -= 1

        scribe = ScribeAgent(
            self.caps,
            CallableFlashRunner(model_with_overlap_detector),
            batch_limit=6,
        )
        worker = AsyncScribeWorker(scribe, min_turns=2, debounce_sec=0.01)

        total_threads = 6
        turns_per_thread = 4
        all_written: list[int] = []
        notify_durations: list[float] = []

        def producer(idx: int) -> None:
            for j in range(turns_per_thread):
                with write_lock:
                    tid = _append_turn(
                        self.engine,
                        "ep_concurrent",
                        "user",
                        f"worker-{idx} task-{j} concrete progress update",
                    )
                    all_written.append(tid)
                t0 = time.perf_counter()
                worker.notify("ep_concurrent", tid)
                notify_durations.append(time.perf_counter() - t0)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(total_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertTrue(worker.flush(timeout=10.0))
        worker.close(timeout=5.0)
        self.assertFalse(worker.is_alive)
        self.assertIsNone(worker.last_error)

        expected_total = total_threads * turns_per_thread
        self.assertEqual(len(all_written), expected_total)
        # Strict serialization: never > 1 concurrent digest call
        self.assertEqual(max_concurrency, 1)
        # Zero duplicate or lost turns across all batches
        self.assertEqual(sorted(digested_turn_ids), sorted(all_written))
        self.assertEqual(self.caps.get_watermark("pool_digested"), max(all_written))

        # Every created event has a staged Level-2 detail note
        events = self.caps.list_events()
        self.assertGreaterEqual(len(events), 1)
        for ev in events:
            note = get_pool_event_note(self.engine.db, ev.event_id)
            self.assertIsNotNone(note)

        # notify() overhead stayed sub-millisecond on average
        avg_notify_ms = (sum(notify_durations) / len(notify_durations)) * 1000.0
        self.assertLess(avg_notify_ms, 1.0)


if __name__ == "__main__":
    unittest.main()
