"""Unit tests for reiji_maigo.swarm.scout (ScoutAgent / ScoutBriefing)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from reiji_maigo import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm import (
    CallableFlashRunner,
    build_role_capabilities,
    compute_prefix_sha256,
    list_swarm_telemetry,
    save_pool_event_note,
)
from reiji_maigo.swarm.scout import (
    SCOUT_L0_PREFIX,
    ScoutAgent,
    ScoutBriefing,
    estimate_tokens,
)


class TestScoutAgent(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = ReijiMemoryEngine(
            home=Path(self._tmp.name),
            db_path=":memory:",
            load_plugins=False,
            disable_plugins=True,
        )
        self.scout_caps = build_role_capabilities(self.engine, "scout")

    def tearDown(self) -> None:
        self.engine.close()
        self._tmp.cleanup()

    def test_layer2_detail_direct_hit_single_round(self) -> None:
        """Case 1: Layer-2 card detail directly answers the query in 1 round."""
        t1 = self.engine.db.append_turn(
            "ep_db_cfg",
            "user",
            "We standardized PostgreSQL pool sizing on production.",
        )
        t2 = self.engine.db.append_turn(
            "ep_db_cfg",
            "assistant",
            "Noted: max_pool_size=32, idle_timeout_ms=5000, sslmode=require.",
        )
        span = SourceSpan(episode_id="ep_db_cfg", turn_start=t1, turn_end=t2)
        card = self.engine.cards.create_card(
            domain="infra",
            card_type="decision",
            title="Production PostgreSQL Pool Policy",
            state={
                "summary": "Production PostgreSQL connection pool limits",
                "detail": (
                    "Standardized production PostgreSQL pool parameters: "
                    "max_pool_size=32, idle_timeout_ms=5000, sslmode=require, "
                    "statement_timeout_ms=15000."
                ),
            },
            tags=["postgresql", "pool", "production"],
            spans=[span],
        )

        captured_calls: list[tuple[str, dict[str, Any]]] = []

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            payload = json.loads(prompt)
            captured_calls.append((system, payload))
            details = payload["observations"]["prefetched_card_details"]
            self.assertIn(card.card_id, details)
            self.assertIn("max_pool_size=32", details[card.card_id])
            return {
                "status": "final",
                "final_briefing": {
                    "found": True,
                    "abstained": False,
                    "briefing_text": (
                        "Production PostgreSQL pool policy: max_pool_size=32, "
                        "idle_timeout_ms=5000, sslmode=require."
                    ),
                    "cited_card_ids": [card.card_id],
                    "cited_spans": [span.to_dict()],
                },
            }

        agent = ScoutAgent(
            self.scout_caps,
            CallableFlashRunner(fake_flash),
            max_rounds=3,
            max_briefing_tokens=240,
        )
        briefing = agent.investigate(
            "What is our production postgresql pool configuration?",
            domain="infra",
            hint_tags=["postgresql"],
        )

        self.assertIsInstance(briefing, ScoutBriefing)
        self.assertTrue(briefing.found)
        self.assertFalse(briefing.abstained)
        self.assertEqual(briefing.rounds_used, 1)
        self.assertEqual(briefing.raw_turns_scanned, 0)
        self.assertEqual(briefing.cited_card_ids, (card.card_id,))
        self.assertEqual(briefing.cited_spans, (span,))
        self.assertGreater(briefing.distilled_tokens, 0)
        self.assertLessEqual(briefing.distilled_tokens, 240)
        self.assertGreater(briefing.compression_ratio, 0.0)
        self.assertIn("max_pool_size=32", briefing.briefing_text)

        tail = briefing.to_tail_block()
        self.assertIn("[Scout Briefing |", tail)
        self.assertIn(card.card_id, tail)
        self.assertIn(f"ep_db_cfg:{t1}-{t2}", tail)
        self.assertIn("max_pool_size=32", tail)

        self.assertEqual(len(captured_calls), 1)
        self.assertEqual(captured_calls[0][0], SCOUT_L0_PREFIX)

        telemetry = list_swarm_telemetry(self.engine.db, role="scout", limit=5)
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["action"], "investigate")
        self.assertEqual(
            telemetry[0]["metrics"]["prefix_sha256"],
            compute_prefix_sha256(SCOUT_L0_PREFIX),
        )
        self.assertEqual(telemetry[0]["metrics"]["rounds_used"], 1)

    def test_drilldown_round2_raw_turns_and_compression_ratio(self) -> None:
        """Case 2: Coarse Layer-2 detail triggers Round-2 raw turn slicing over 22 turns."""
        episode_id = "ep_vllm_bench"
        turn_ids: list[int] = []
        for idx in range(1, 23):
            if idx == 14:
                content = (
                    "Final verified inference benchmark command that eliminated OOM spikes "
                    "while keeping p99 latency under 180ms across 64 concurrent streams: "
                    "--tensor-parallel-size 4 --kv-cache-dtype fp8_e4m3 "
                    "--max-model-len 32768 --gpu-memory-utilization 0.92. "
                    "We double-checked throughput at 4,820 tok/s with zero swap faults "
                    "over a full 45-minute soak test on the 4xH100 node."
                )
            else:
                content = (
                    f"Benchmark iteration #{idx}: tested intermediate batch scheduling, "
                    f"chunked prefill window size {512 + idx * 32}, warmup telemetry logs, "
                    f"memory fragmentation checkpoints, CUDA graph capture timings, "
                    f"and intermediate throughput samples before locking final launch flags. "
                    f"Observed transient queue jitter during step {idx} which required further tuning."
                )
            tid = self.engine.db.append_turn(
                episode_id,
                "user" if idx % 2 == 1 else "assistant",
                content,
            )
            turn_ids.append(tid)

        full_span = SourceSpan(
            episode_id=episode_id,
            turn_start=turn_ids[0],
            turn_end=turn_ids[-1],
        )
        card = self.engine.cards.create_card(
            domain="infra",
            card_type="decision",
            title="Inference Engine Soak-Test Outcome",
            state={
                "summary": "Inference engine tuning increased throughput by 2.3x.",
                "detail": "Completed 4xH100 soak test; exact CLI flags are in the raw transcript.",
            },
            tags=["vllm", "h100", "benchmark"],
            spans=[full_span],
        )

        seen_prefixes: list[str] = []

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            seen_prefixes.append(system)
            payload = json.loads(prompt)
            round_no = payload["round"]
            if round_no == 1:
                self.assertFalse(payload["must_finalize"])
                return {
                    "status": "need_more",
                    "reasoning": "Layer-2 detail lacks exact CLI flags; slice raw turns.",
                    "next_actions": [
                        {
                            "action": "slice_card_evidence",
                            "card_id": card.card_id,
                            "pad": 0,
                        }
                    ],
                }

            self.assertEqual(round_no, 2)
            action_results = payload["observations"]["action_results"]
            self.assertEqual(len(action_results), 1)
            self.assertEqual(action_results[0]["action"], "slice_card_evidence")
            self.assertEqual(action_results[0]["turn_count"], 22)
            # Confirm Round-2 shadow payload actually received the buried turn #14 text
            all_text = "\n".join(t["content"] for t in action_results[0]["turns"])
            self.assertIn("fp8_e4m3", all_text)
            self.assertIn("32768", all_text)

            return {
                "status": "final",
                "final_briefing": {
                    "found": True,
                    "abstained": False,
                    "briefing_text": (
                        "Verified 4xH100 inference flags: --tensor-parallel-size 4, "
                        "--kv-cache-dtype fp8_e4m3, --max-model-len 32768, "
                        "--gpu-memory-utilization 0.92 (4,820 tok/s, p99 < 180ms)."
                    ),
                    "cited_card_ids": [card.card_id],
                    "cited_spans": [
                        {
                            "episode_id": episode_id,
                            "turn_start": turn_ids[13],
                            "turn_end": turn_ids[13],
                        }
                    ],
                },
            }

        agent = ScoutAgent(
            self.scout_caps,
            CallableFlashRunner(fake_flash),
            max_rounds=3,
            max_briefing_tokens=240,
        )
        briefing = agent.investigate(
            "What exact vllm launch flags passed the H100 benchmark?",
            domain="infra",
            hint_tags=["vllm"],
        )

        self.assertTrue(briefing.found)
        self.assertFalse(briefing.abstained)
        self.assertEqual(briefing.rounds_used, 2)
        self.assertEqual(briefing.raw_turns_scanned, 22)
        self.assertLessEqual(briefing.distilled_tokens, 240)
        self.assertGreater(briefing.compression_ratio, 5.0)
        self.assertIn("fp8_e4m3", briefing.briefing_text)
        self.assertIn("32768", briefing.briefing_text)
        self.assertEqual(
            briefing.cited_spans,
            (SourceSpan(episode_id=episode_id, turn_start=turn_ids[13], turn_end=turn_ids[13]),),
        )
        self.assertEqual(seen_prefixes, [SCOUT_L0_PREFIX, SCOUT_L0_PREFIX])

        tail = briefing.to_tail_block()
        self.assertIn("scanned_turns=22", tail)
        self.assertIn("ratio=", tail)
        self.assertIn("fp8_e4m3", tail)

    def test_abstain_when_no_matching_fact(self) -> None:
        """Case 3: Honest abstention when no relevant memory exists (LongMemEval law)."""
        t1 = self.engine.db.append_turn(
            "ep_coffee",
            "user",
            "I prefer single-origin Ethiopian pour-over coffee in the morning.",
        )
        self.engine.cards.create_card(
            domain="life",
            card_type="preference",
            title="Morning Coffee Preference",
            state={"detail": "Prefers Ethiopian pour-over coffee."},
            tags=["coffee", "morning"],
            spans=[SourceSpan(episode_id="ep_coffee", turn_start=t1, turn_end=t1)],
        )

        def fake_flash_abstain(system: str, prompt: str) -> dict[str, Any]:
            payload = json.loads(prompt)
            self.assertEqual(payload["observations"]["level1_cards"], [])
            return {
                "status": "final",
                "final_briefing": {
                    "found": False,
                    "abstained": True,
                    "briefing_text": "",
                    "cited_card_ids": [],
                    "cited_spans": [],
                },
            }

        agent = ScoutAgent(
            self.scout_caps,
            CallableFlashRunner(fake_flash_abstain),
            max_rounds=3,
            max_briefing_tokens=240,
        )
        briefing = agent.investigate("What is the Mars relay satellite rotation key?")

        self.assertFalse(briefing.found)
        self.assertTrue(briefing.abstained)
        self.assertEqual(briefing.briefing_text, "")
        self.assertEqual(briefing.to_tail_block(), "")
        self.assertEqual(briefing.cited_card_ids, ())
        self.assertEqual(briefing.cited_spans, ())
        self.assertEqual(briefing.distilled_tokens, 0)
        self.assertEqual(briefing.compression_ratio, 0.0)
        self.assertEqual(briefing.rounds_used, 1)

        # Empty query short-circuits cleanly without calling model
        empty_briefing = agent.investigate("   ")
        self.assertTrue(empty_briefing.abstained)
        self.assertFalse(empty_briefing.found)
        self.assertEqual(empty_briefing.rounds_used, 0)
        self.assertEqual(empty_briefing.to_tail_block(), "")

        # Exhausted max_rounds without final_briefing defaults safely to abstained
        def fake_flash_never_finishes(system: str, prompt: str) -> dict[str, Any]:
            return {
                "status": "need_more",
                "next_actions": [{"action": "deep_fts_search", "query": "nonexistent_xyz"}],
            }

        exhausted_agent = ScoutAgent(
            self.scout_caps,
            CallableFlashRunner(fake_flash_never_finishes),
            max_rounds=2,
        )
        exhausted = exhausted_agent.investigate("Unknown quantum router serial?")
        self.assertTrue(exhausted.abstained)
        self.assertFalse(exhausted.found)
        self.assertEqual(exhausted.to_tail_block(), "")

    def test_deep_fts_expand_span_event_note_and_budget_cap(self) -> None:
        """Case 4: FTS deep recon, span expansion, event note inspection, and token cap."""
        t1 = self.engine.db.append_turn(
            "ep_legacy", "user", "Discussing legacy archive retention script."
        )
        t2 = self.engine.db.append_turn(
            "ep_legacy", "assistant", "Found archaic token QUARTZ_ROTATION_991 in cron."
        )
        t3 = self.engine.db.append_turn(
            "ep_legacy", "user", "Confirmed QUARTZ_ROTATION_991 runs every Sunday 03:00 UTC."
        )
        save_pool_event_note(
            self.engine.db,
            "ev_cron_1",
            {"note": "Sunday maintenance window verified."},
        )

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            payload = json.loads(prompt)
            r = payload["round"]
            if r == 1:
                return {
                    "status": "need_more",
                    "next_actions": [
                        {
                            "action": "deep_fts_search",
                            "query": "QUARTZ_ROTATION_991",
                            "limit": 2,
                            "pad": 1,
                        },
                        {
                            "action": "expand_span",
                            "episode_id": "ep_legacy",
                            "turn_start": t1,
                            "turn_end": t3,
                            "pad": 0,
                        },
                        {
                            "action": "inspect_event_note",
                            "event_id": "ev_cron_1",
                        },
                    ],
                }
            results = payload["observations"]["action_results"]
            actions = [item["action"] for item in results]
            self.assertEqual(
                actions,
                ["deep_fts_search", "expand_span", "inspect_event_note"],
            )
            self.assertEqual(
                results[2]["note"],
                {"note": "Sunday maintenance window verified."},
            )
            # Emit deliberately verbose text to exercise hard token budget truncation
            verbose = "QUARTZ_ROTATION_991 runs every Sunday 03:00 UTC. " * 40
            return {
                "status": "final",
                "final_briefing": {
                    "found": True,
                    "abstained": False,
                    "briefing_text": verbose,
                    "cited_card_ids": [],
                    "cited_spans": [f"ep_legacy:{t2}-{t3}"],
                },
            }

        agent = ScoutAgent(
            self.scout_caps,
            CallableFlashRunner(fake_flash),
            max_rounds=3,
            max_briefing_tokens=40,
        )
        briefing = agent.investigate("When does QUARTZ_ROTATION_991 execute?")

        self.assertTrue(briefing.found)
        self.assertFalse(briefing.abstained)
        self.assertEqual(briefing.rounds_used, 2)
        self.assertEqual(briefing.raw_turns_scanned, 3)
        self.assertLessEqual(briefing.distilled_tokens, 40)
        self.assertLessEqual(estimate_tokens(briefing.briefing_text), 40)
        self.assertTrue(briefing.briefing_text.endswith("..."))
        self.assertEqual(
            briefing.cited_spans,
            (SourceSpan(episode_id="ep_legacy", turn_start=t2, turn_end=t3),),
        )
        # Level-3 deep_fts_search should also have enqueued a self-healing backfill
        backfills = self.engine.cards.list_pending_backfills(status="pending")
        self.assertGreaterEqual(len(backfills), 1)

    def test_constructor_validation_and_role_isolation(self) -> None:
        """ScoutAgent refuses non-Scout capabilities and invalid parameters."""
        scribe_caps = build_role_capabilities(self.engine, "scribe")
        dummy_runner = CallableFlashRunner(lambda s, p: {"abstained": True})

        with self.assertRaises(TypeError):
            ScoutAgent(scribe_caps, dummy_runner)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ScoutAgent(self.scout_caps, object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ScoutAgent(self.scout_caps, dummy_runner, max_rounds=0)
        with self.assertRaises(ValueError):
            ScoutAgent(self.scout_caps, dummy_runner, max_briefing_tokens=0)


if __name__ == "__main__":
    unittest.main()
