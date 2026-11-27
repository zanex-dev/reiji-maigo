"""Unit tests for Midnight Archivist (ArchivistAgent) curation workflows."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from reiji_maigo.core.contracts import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm.archivist import (
    ARCHIVIST_L0_PREFIX,
    ArchivistAgent,
    ArchivistReport,
)
from reiji_maigo.swarm.guards import (
    ArchivistCapabilities,
    ScribeCapabilities,
    build_role_capabilities,
    list_swarm_telemetry,
)
from reiji_maigo.swarm.runner import CallableFlashRunner, compute_prefix_sha256


class TestArchivistAgent(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.engine = ReijiMemoryEngine(
            home=Path(self._tmpdir.name),
            load_plugins=False,
            disable_plugins=True,
        )
        self.scribe = ScribeCapabilities(self.engine)
        self.archivist_caps = build_role_capabilities(self.engine, "archivist")
        assert isinstance(self.archivist_caps, ArchivistCapabilities)

    def tearDown(self) -> None:
        self.engine.close()
        self._tmpdir.cleanup()

    def test_constructor_validation(self) -> None:
        runner = CallableFlashRunner(lambda s, p: {})
        with self.assertRaises(TypeError):
            ArchivistAgent(self.scribe, runner)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ArchivistAgent(self.archivist_caps, object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ArchivistAgent(self.archivist_caps, runner, max_steps=0)
        with self.assertRaises(ValueError):
            ArchivistAgent(self.archivist_caps, runner, max_dormant_nights=0)
        with self.assertRaises(ValueError):
            ArchivistAgent(self.archivist_caps, runner, backfill_limit=-1)

    def test_long_chain_compaction_shortens_steps_and_preserves_spans_losslessly(self) -> None:
        base_ts = "2026-11-27T10:00:00+00:00"
        spans: list[SourceSpan] = []
        for idx in range(1, 8):
            tid = self.engine.append_turn("ep_compact", "user", f"Step {idx} progress note.")
            spans.append(SourceSpan(episode_id="ep_compact", turn_start=tid, turn_end=tid))

        ev = self.scribe.create_event(
            "Build memory swarm",
            "Step 1: scaffold runner",
            spans[0],
            scope="quest",
            now_iso=base_ts,
        )
        for idx in range(1, 7):
            ev = self.scribe.append_step(
                ev.event_id,
                f"Step {idx + 1}: milestone {idx + 1}",
                spans[idx],
                now_iso=base_ts,
            )
        self.assertEqual(len(ev.steps), 7)
        self.assertEqual(ev.spans, tuple(spans))

        seen_systems: list[str] = []

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            seen_systems.append(system)
            body = json.loads(prompt)
            self.assertEqual(body["task"], "compact")
            self.assertEqual(body["event"]["event_id"], ev.event_id)
            return {
                "checkpoint_summary": "Completed milestones 1-5 for memory swarm foundation."
            }

        agent = ArchivistAgent(
            self.archivist_caps,
            CallableFlashRunner(fake_flash),
            max_steps=5,
        )
        report = agent.curate_midnight(now_iso="2026-11-27T23:55:00+00:00")

        self.assertIsInstance(report, ArchivistReport)
        self.assertEqual(report.compacted_events, (ev.event_id,))
        self.assertEqual(report.archived_cards, ())
        self.assertEqual(seen_systems, [ARCHIVIST_L0_PREFIX])

        compacted = self.engine.pool.get_event(ev.event_id)
        self.assertIsNotNone(compacted)
        assert compacted is not None
        self.assertLess(len(compacted.steps), 7)
        self.assertEqual(len(compacted.steps), 2)
        self.assertEqual(
            compacted.steps,
            ("Step 6: milestone 6", "Step 7: milestone 7"),
        )
        self.assertEqual(
            compacted.checkpoint_summary,
            "Completed milestones 1-5 for memory swarm foundation.",
        )
        # 100% lossless span preservation
        self.assertEqual(compacted.spans, tuple(spans))

    def test_archive_absorbs_event_notes_hibernates_dormant_and_drops_micro(self) -> None:
        # Pre-register controlled L2 tag with alias to verify vocabulary reuse
        self.engine.cards.register_or_resolve_tag("proj", "swarm", aliases=("bee-pack",))

        t1 = self.engine.append_turn("ep_done", "user", "Finished archivist design and tests.")
        t2 = self.engine.append_turn("ep_done", "assistant", "All checks passed.")
        done_span = SourceSpan(episode_id="ep_done", turn_start=t1, turn_end=t2)
        done_ev = self.scribe.create_event(
            "Ship ArchivistAgent",
            "Implemented curation pipeline",
            done_span,
            scope="quest",
            now_iso="2026-11-27T12:00:00+00:00",
        )
        self.scribe.update_status(done_ev.event_id, "done", now_iso="2026-11-27T14:00:00+00:00")
        self.scribe.save_event_note(
            done_ev.event_id,
            {
                "files": ["reiji_maigo/swarm/archivist.py", "tests/test_swarm_archivist.py"],
                "decisions": ["strict stdlib only", "lossless span compaction"],
            },
        )

        # Dormant quest (> 7 days untouched -> hibernate)
        t3 = self.engine.append_turn("ep_dormant", "user", "Draft benchmark suite later.")
        dormant_span = SourceSpan(episode_id="ep_dormant", turn_start=t3, turn_end=t3)
        dormant_ev = self.scribe.create_event(
            "Benchmark suite",
            "Outline latency scenarios",
            dormant_span,
            scope="quest",
            now_iso="2026-11-10T09:00:00+00:00",
        )

        # Expired micro task (> 24h untouched -> drop deterministically)
        t4 = self.engine.append_turn("ep_micro", "user", "Check typo in readme.")
        micro_span = SourceSpan(episode_id="ep_micro", turn_start=t4, turn_end=t4)
        micro_ev = self.scribe.create_event(
            "Fix quick typo",
            "Update line 12",
            micro_span,
            scope="micro",
            now_iso="2026-11-25T08:00:00+00:00",
        )
        self.scribe.save_event_note(micro_ev.event_id, {"scratch": "disposable"})

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, ARCHIVIST_L0_PREFIX)
            body = json.loads(prompt)
            self.assertEqual(body["task"], "archive")
            self.assertIn("proj", body["controlled_tags"])
            ev_id = body["event"]["event_id"]
            if ev_id == done_ev.event_id:
                self.assertFalse(body["hibernate"])
                self.assertIn("files", body["event_note"])
                return {
                    "domain": "proj",
                    "title": "ArchivistAgent Shipped",
                    "tags": ["bee-pack", "curation"],
                    "state": {
                        "summary": "Completed midnight archivist curation worker.",
                        "detail": ["Verified lossless compaction and TEPA lifecycle."],
                        "open_questions": ["Should midnight run automatically on idle?"],
                    },
                }
            if ev_id == dormant_ev.event_id:
                self.assertTrue(body["hibernate"])
                return {
                    "domain": "research",
                    "title": "Latency Benchmark Suite",
                    "tags": ["benchmark"],
                    "state": {
                        "summary": "Paused benchmark outline awaiting swarm completion.",
                        "detail": ["Outline latency scenarios"],
                        "open_questions": ["Pick target corpus size"],
                    },
                }
            raise AssertionError(f"Unexpected event archived: {ev_id}")

        agent = ArchivistAgent(self.archivist_caps, CallableFlashRunner(fake_flash))
        report = agent.curate_midnight(now_iso="2026-11-27T23:59:00+00:00")

        self.assertEqual(report.dropped_micro, (micro_ev.event_id,))
        self.assertEqual(len(report.archived_cards), 1)
        self.assertEqual(len(report.hibernated_cards), 1)
        self.assertGreaterEqual(report.elapsed_ms, 0)

        # Micro event dropped & its staging note cleaned
        self.assertEqual(self.engine.pool.get_event(micro_ev.event_id).status, "dropped")
        self.assertIsNone(self.archivist_caps.get_event_note(micro_ev.event_id))

        # Archived card absorbed pool_event_notes into state["detail"] and reused canonical tag
        archived_card = self.engine.cards.get_card(report.archived_cards[0])
        assert archived_card is not None
        self.assertEqual(archived_card.domain, "proj")
        self.assertIn("swarm", archived_card.tags)
        self.assertNotIn("bee-pack", archived_card.tags)
        self.assertEqual(archived_card.spans, (done_span,))
        self.assertEqual(
            archived_card.state["summary"],
            "Completed midnight archivist curation worker.",
        )
        self.assertIsInstance(archived_card.state["detail"], dict)
        self.assertEqual(
            archived_card.state["detail"]["files"],
            ["reiji_maigo/swarm/archivist.py", "tests/test_swarm_archivist.py"],
        )
        self.assertEqual(
            archived_card.state["detail"]["decisions"],
            ["strict stdlib only", "lossless span compaction"],
        )
        self.assertEqual(
            archived_card.state["detail"]["notes"],
            ["Verified lossless compaction and TEPA lifecycle."],
        )
        self.assertEqual(
            archived_card.state["open_questions"],
            ["Should midnight run automatically on idle?"],
        )
        self.assertIsNone(self.archivist_caps.get_event_note(done_ev.event_id))
        self.assertEqual(self.engine.pool.get_event(done_ev.event_id).status, "archived")

        # Hibernated card preserves hibernation metadata + three-layer state
        hib_card = self.engine.cards.get_card(report.hibernated_cards[0])
        assert hib_card is not None
        self.assertTrue(hib_card.state["hibernated"])
        self.assertEqual(hib_card.state["pool_state"], "hibernated")
        self.assertEqual(hib_card.state["open_questions"], ["Pick target corpus size"])
        self.assertEqual(self.engine.pool.get_event(dormant_ev.event_id).status, "hibernated")

    def test_tepa_conflict_triggers_supersede_and_revoke_for_same_domain_cards(self) -> None:
        t_old = self.engine.append_turn("ep_gpu_old", "user", "Workstation runs RTX 3090 on CUDA 11.8.")
        old_span = SourceSpan(episode_id="ep_gpu_old", turn_start=t_old, turn_end=t_old)

        old_gpu = self.engine.cards.create_card(
            domain="infra",
            card_type="fact",
            title="Primary Workstation GPU",
            state={
                "summary": "Primary GPU is RTX 3090 (24GB).",
                "detail": ["RTX 3090"],
                "open_questions": [],
            },
            tags=["gpu", "workstation"],
            spans=[old_span],
        )
        old_workaround = self.engine.cards.create_card(
            domain="infra",
            card_type="fact",
            title="CUDA 11.8 Compatibility Workaround",
            state={
                "summary": "Pin torch to cu118 for legacy driver.",
                "detail": ["Requires cu118 wheel"],
                "open_questions": [],
            },
            tags=["gpu", "cuda"],
            spans=[old_span],
        )

        t_new = self.engine.append_turn(
            "ep_gpu_new",
            "user",
            "Swapped GPU to RTX 4090 and upgraded to CUDA 12.6; old cu118 workaround is obsolete.",
        )
        new_span = SourceSpan(episode_id="ep_gpu_new", turn_start=t_new, turn_end=t_new)
        ev = self.scribe.create_event(
            "Upgrade workstation GPU",
            "Installed RTX 4090 + CUDA 12.6",
            new_span,
            scope="quest",
            now_iso="2026-11-27T18:00:00+00:00",
        )
        self.scribe.update_status(ev.event_id, "done", now_iso="2026-11-27T19:00:00+00:00")

        captured_candidates: list[str] = []

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, ARCHIVIST_L0_PREFIX)
            body = json.loads(prompt)
            captured_candidates.extend(c["card_id"] for c in body["existing_active_cards"])
            return {
                "domain": "infra",
                "card_type": "fact",
                "title": "Primary Workstation GPU",
                "tags": ["gpu", "workstation"],
                "state": {
                    "summary": "Primary GPU upgraded to RTX 4090 with CUDA 12.6.",
                    "detail": ["RTX 4090", "CUDA 12.6 native wheels"],
                    "open_questions": [],
                },
                "supersede_card_id": old_gpu.card_id,
                "revoke_card_ids": [old_workaround.card_id],
            }

        now_iso = "2026-11-27T23:55:00+00:00"
        agent = ArchivistAgent(self.archivist_caps, CallableFlashRunner(fake_flash))
        report = agent.curate_midnight(now_iso=now_iso)

        self.assertIn(old_gpu.card_id, captured_candidates)
        self.assertIn(old_workaround.card_id, captured_candidates)
        self.assertEqual(len(report.archived_cards), 1)
        new_card_id = report.archived_cards[0]
        self.assertEqual(report.superseded_cards, (old_gpu.card_id,))
        self.assertEqual(report.revoked_cards, (old_workaround.card_id,))

        sup_card = self.engine.cards.get_card(old_gpu.card_id)
        assert sup_card is not None
        self.assertEqual(sup_card.lifecycle_status, "superseded")
        self.assertEqual(sup_card.superseded_by, new_card_id)
        self.assertEqual(sup_card.valid_to, now_iso)

        rev_card = self.engine.cards.get_card(old_workaround.card_id)
        assert rev_card is not None
        self.assertEqual(rev_card.lifecycle_status, "revoked")
        self.assertEqual(rev_card.valid_to, now_iso)

        # Zero leakage: Level-1 recall returns only the new active card
        recalled = self.engine.cards.recall_level1("infra", ["gpu"], top_k=5)
        self.assertEqual([c.card_id for c in recalled], [new_card_id])

    def test_level3_pending_backfill_curated_into_formal_card_and_marked_done(self) -> None:
        t1 = self.engine.append_turn(
            "ep_legacy",
            "user",
            "Production PostgreSQL replica listens on port 5433 with SSL required.",
        )
        t2 = self.engine.append_turn(
            "ep_legacy",
            "assistant",
            "Noted: PostgreSQL replica port 5433 + SSL.",
        )

        # Trigger Level-3 page-fault search to enqueue pending_backfills entry
        hits = self.engine.cards.deep_slice_level3("PostgreSQL replica", limit=2, pad=1)
        self.assertGreaterEqual(len(hits), 1)
        backfill_id = hits[0]["backfill_id"]
        self.assertEqual(len(self.engine.cards.list_pending_backfills(status="pending")), 1)

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, ARCHIVIST_L0_PREFIX)
            body = json.loads(prompt)
            self.assertEqual(body["task"], "backfill")
            self.assertEqual(body["backfill_id"], backfill_id)
            self.assertGreaterEqual(len(body["turns"]), 1)
            return {
                "domain": "infra",
                "card_type": "fact",
                "title": "PostgreSQL Replica Connection",
                "tags": ["postgres", "database"],
                "state": {
                    "summary": "Production PostgreSQL replica runs on port 5433 with SSL required.",
                    "detail": ["port=5433", "ssl=required"],
                    "open_questions": [],
                },
            }

        agent = ArchivistAgent(self.archivist_caps, CallableFlashRunner(fake_flash))
        report = agent.curate_midnight(now_iso="2026-11-27T23:58:00+00:00")

        self.assertEqual(len(report.backfilled_cards), 1)
        card = self.engine.cards.get_card(report.backfilled_cards[0])
        assert card is not None
        self.assertEqual(card.domain, "infra")
        self.assertEqual(card.title, "PostgreSQL Replica Connection")
        self.assertEqual(card.tags, ("database", "postgres"))
        self.assertEqual(card.spans, (SourceSpan(episode_id="ep_legacy", turn_start=t1, turn_end=t2),))
        self.assertEqual(card.state["detail"], ["port=5433", "ssl=required"])

        # Queue entry transitioned from pending -> done
        self.assertEqual(self.engine.cards.list_pending_backfills(status="pending"), [])
        done_rows = self.engine.cards.list_pending_backfills(status="done")
        self.assertEqual(len(done_rows), 1)
        self.assertEqual(done_rows[0]["backfill_id"], backfill_id)
        self.assertEqual(done_rows[0]["attempts"], 1)

    def test_cross_domain_tag_collision_guard_and_telemetry(self) -> None:
        # "gpu" already owned by infra; Flash mistakenly proposes "gpu" under proj
        self.engine.cards.register_or_resolve_tag("infra", "gpu")

        t1 = self.engine.append_turn("ep_col", "user", "Finished shader demo project.")
        span = SourceSpan(episode_id="ep_col", turn_start=t1, turn_end=t1)
        ev = self.scribe.create_event(
            "Shader demo",
            "Wrapped up demo",
            span,
            scope="quest",
            now_iso="2026-11-27T15:00:00+00:00",
        )
        self.scribe.update_status(ev.event_id, "done", now_iso="2026-11-27T16:00:00+00:00")

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            return {
                "domain": "proj",
                "title": "Shader Demo Complete",
                "tags": ["gpu", "shader"],
                "state": {
                    "summary": "Wrapped up shader demo.",
                    "detail": ["Demo passes locally"],
                    "open_questions": [],
                },
            }

        agent = ArchivistAgent(self.archivist_caps, CallableFlashRunner(fake_flash))
        report = agent.curate_midnight(now_iso="2026-11-27T23:59:00+00:00")

        self.assertEqual(len(report.archived_cards), 1)
        card = self.engine.cards.get_card(report.archived_cards[0])
        assert card is not None
        self.assertEqual(card.domain, "proj")
        self.assertIn("proj:gpu", card.tags)
        self.assertIn("shader", card.tags)

        telemetry = list_swarm_telemetry(self.engine.db, role="archivist")
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["action"], "curate_midnight")
        self.assertEqual(telemetry[0]["metrics"]["archived_cards"], 1)
        self.assertEqual(telemetry[0]["metrics"]["llm_calls"], 1)
        self.assertEqual(
            telemetry[0]["metrics"]["prefix_sha256"],
            compute_prefix_sha256(ARCHIVIST_L0_PREFIX),
        )

    def test_event_note_list_absorption_and_second_pass_conflict_check(self) -> None:
        # Seed >8 unrelated active cards so target old card is outside initial top_k=8 scan
        # when its title/tags do not textually match the event text.
        filler_turn = self.engine.append_turn("ep_fill", "user", "Filler memories.")
        filler_span = SourceSpan(episode_id="ep_fill", turn_start=filler_turn, turn_end=filler_turn)
        target_old = self.engine.cards.create_card(
            domain="life",
            card_type="preference",
            title="Beverage Routine",
            state={"summary": "Drinks espresso daily.", "detail": ["espresso"], "open_questions": []},
            tags=["caffeine"],
            spans=[filler_span],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        for i in range(9):
            self.engine.cards.create_card(
                domain="proj",
                card_type="fact",
                title=f"Project Milestone {i}",
                state={"summary": f"Milestone {i}", "detail": [f"m{i}"], "open_questions": []},
                tags=[f"tag{i}", "shared_recent"],
                spans=[filler_span],
                created_at=f"2026-11-2{min(i, 7)}T00:00:00+00:00",
                updated_at=f"2026-11-2{min(i, 7)}T00:00:00+00:00",
            )

        t_ev = self.engine.append_turn("ep_tea", "user", "Switched morning drink to oolong.")
        ev_span = SourceSpan(episode_id="ep_tea", turn_start=t_ev, turn_end=t_ev)
        ev = self.scribe.create_event(
            "Morning drink switch",
            "Switched to oolong",
            ev_span,
            scope="quest",
            now_iso="2026-11-27T09:00:00+00:00",
        )
        self.scribe.update_status(ev.event_id, "done", now_iso="2026-11-27T10:00:00+00:00")
        self.scribe.save_event_note(ev.event_id, {"detail": ["stopped espresso", "prefers oolong"]})

        tasks_seen: list[str] = []

        def fake_flash(system: str, prompt: str) -> dict[str, Any]:
            self.assertEqual(system, ARCHIVIST_L0_PREFIX)
            body = json.loads(prompt)
            tasks_seen.append(body["task"])
            if body["task"] == "archive":
                return {
                    "domain": "life",
                    "card_type": "preference",
                    "title": "Morning Beverage Routine",
                    "tags": ["caffeine"],
                    "state": {
                        "summary": "Switched morning drink to oolong tea.",
                        "detail": ["prefers oolong", "brew at 90C"],
                        "open_questions": [],
                    },
                }
            if body["task"] == "conflict_check":
                candidate_ids = [c["card_id"] for c in body["existing_active_cards"]]
                self.assertIn(target_old.card_id, candidate_ids)
                return {
                    "supersede_card_id": target_old.card_id,
                    "revoke_card_ids": [],
                    "related_card_ids": [],
                }
            raise AssertionError(f"Unexpected task: {body['task']}")

        agent = ArchivistAgent(self.archivist_caps, CallableFlashRunner(fake_flash))
        report = agent.curate_midnight(now_iso="2026-11-27T23:59:00+00:00")

        self.assertEqual(tasks_seen, ["archive", "conflict_check"])
        self.assertEqual(report.superseded_cards, (target_old.card_id,))
        archived = self.engine.cards.get_card(report.archived_cards[0])
        assert archived is not None
        self.assertEqual(
            archived.state["detail"],
            ["stopped espresso", "prefers oolong", "brew at 90C"],
        )
        self.assertEqual(self.engine.cards.get_card(target_old.card_id).lifecycle_status, "superseded")


if __name__ == "__main__":
    unittest.main()
