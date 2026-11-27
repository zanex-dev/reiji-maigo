"""Unit tests for reiji_maigo.swarm runner, telemetry/staging tables, and physical role guards."""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any

from reiji_maigo import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.swarm import (
    ArchivistCapabilities,
    CallableFlashRunner,
    HttpFlashRunner,
    ScoutCapabilities,
    ScribeCapabilities,
    SwarmCallResult,
    SwarmJSONError,
    SwarmModelRunner,
    SwarmRunnerError,
    build_role_capabilities,
    compute_prefix_sha256,
    ensure_swarm_tables,
    extract_json_object,
    get_pool_event_note,
    list_swarm_telemetry,
    pop_pool_event_note,
    record_swarm_telemetry,
    save_pool_event_note,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


class TestSwarmRunner(unittest.TestCase):
    def test_extract_json_object_variants(self) -> None:
        self.assertEqual(extract_json_object('{"ok": true, "n": 1}'), {"ok": True, "n": 1})

        fenced = 'Sure, here is the JSON:\n```json\n{"action": "noop", "events": []}\n```\nDone.'
        self.assertEqual(extract_json_object(fenced), {"action": "noop", "events": []})

        plain_fence = "```\n{\"k\": \"v\"}\n```"
        self.assertEqual(extract_json_object(plain_fence), {"k": "v"})

        prose_wrapped = 'Prefix commentary {"nested": {"a": 1}, "flag": false} trailing notes.'
        self.assertEqual(extract_json_object(prose_wrapped), {"nested": {"a": 1}, "flag": False})

        for bad in ("", "   ", "[1, 2, 3]", '"just a string"', "not json at all"):
            with self.assertRaises(SwarmJSONError):
                extract_json_object(bad)

    def test_callable_flash_runner_dict_str_and_duck_adapter(self) -> None:
        recorded: list[tuple[str, str]] = []

        def fn_dict(system: str, prompt: str) -> dict[str, Any]:
            recorded.append((system, prompt))
            return {"ops": ["append"], "len": len(prompt)}

        runner = CallableFlashRunner(fn_dict)
        self.assertIsInstance(runner, SwarmModelRunner)

        prefix = "L0_LOCKED_SCRIBE_PREFIX_V1"
        res1 = runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="turn 1")
        res2 = runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="turn 2")

        self.assertEqual(res1.data, {"ops": ["append"], "len": 6})
        self.assertEqual(res2.data, {"ops": ["append"], "len": 6})
        self.assertEqual(res1.prefix_sha256, compute_prefix_sha256(prefix))
        self.assertEqual(res1.prefix_sha256, res2.prefix_sha256)
        self.assertGreaterEqual(res1.elapsed_ms, 0)
        # Verify strictly stateless Turn-1 calls with no history accumulation
        self.assertEqual(recorded, [(prefix, "turn 1"), (prefix, "turn 2")])

        # String return with fenced JSON
        str_runner = CallableFlashRunner(lambda s, p: f"```json\n{{\"echo\": {json.dumps(p)}}}\n```")
        res_str = str_runner.run_shadow_json(role="scout", locked_system_prefix=prefix, user_payload="hi")
        self.assertEqual(res_str.data, {"echo": "hi"})
        self.assertIn("```json", res_str.raw_text)

        # Duck-typed EngineAdapter.complete_json
        class _FakeEngineAdapter:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def complete_json(self, *, system: str, prompt: str) -> dict[str, Any]:
                self.calls.append((system, prompt))
                return {"adapter": "ok", "prompt": prompt}

        adapter = _FakeEngineAdapter()
        adapter_runner = CallableFlashRunner(adapter)
        res_adp = adapter_runner.run_shadow_json(
            role="archivist",
            locked_system_prefix=prefix,
            user_payload="triage",
        )
        self.assertEqual(res_adp.data, {"adapter": "ok", "prompt": "triage"})
        self.assertEqual(adapter.calls, [(prefix, "triage")])

        # Non-dict return raises SwarmJSONError
        bad_runner = CallableFlashRunner(lambda s, p: [1, 2, 3])
        with self.assertRaises(SwarmJSONError):
            bad_runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="x")

        with self.assertRaises(TypeError):
            CallableFlashRunner("not_callable")

    def test_http_flash_runner_env_headers_stateless_and_errors(self) -> None:
        captured: list[dict[str, Any]] = []

        def fake_opener(req: Any, timeout: float = 0.0) -> _FakeHTTPResponse:
            body = json.loads(req.data.decode("utf-8"))
            captured.append(
                {
                    "url": req.full_url,
                    "headers": {k.lower(): v for k, v in req.header_items()},
                    "body": body,
                    "timeout": timeout,
                }
            )
            return _FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '```json\n{"status": "ok", "hits": 2}\n```',
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 123, "completion_tokens": 45},
                }
            )

        env = {
            "REIJI_FLASH_BASE_URL": "https://example.invalid/proxy",
            "REIJI_FLASH_API_KEY": "sk-test-token",
            "REIJI_FLASH_MODEL": "flash-lite-test",
        }
        runner = HttpFlashRunner(env=env, timeout=12.5, opener=fake_opener)
        self.assertIsInstance(runner, SwarmModelRunner)

        prefix = "L0_SCOUT_IMMUTABLE_PREFIX"
        res = runner.run_shadow_json(
            role="scout",
            locked_system_prefix=prefix,
            user_payload="query=gpu",
        )

        self.assertIsInstance(res, SwarmCallResult)
        self.assertEqual(res.data, {"status": "ok", "hits": 2})
        self.assertEqual(res.prompt_tokens, 123)
        self.assertEqual(res.completion_tokens, 45)
        self.assertEqual(res.prefix_sha256, compute_prefix_sha256(prefix))
        self.assertGreaterEqual(res.elapsed_ms, 0)

        self.assertEqual(len(captured), 1)
        req_info = captured[0]
        self.assertEqual(req_info["url"], "https://example.invalid/proxy/v1/chat/completions")
        self.assertEqual(req_info["timeout"], 12.5)
        self.assertEqual(req_info["headers"]["authorization"], "Bearer sk-test-token")
        self.assertEqual(req_info["headers"]["x-reiji-swarm-role"], "scout")
        self.assertEqual(req_info["headers"]["x-reiji-prefix-sha256"], res.prefix_sha256)
        self.assertEqual(req_info["body"]["model"], "flash-lite-test")
        # Strictly 2 messages: Turn-1 shadow call without accumulated history
        self.assertEqual(
            req_info["body"]["messages"],
            [
                {"role": "system", "content": prefix},
                {"role": "user", "content": "query=gpu"},
            ],
        )

        # Verify /v1 suffix normalization
        v1_runner = HttpFlashRunner(
            base_url="https://example.invalid/v1",
            opener=fake_opener,
        )
        v1_runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="p2")
        self.assertEqual(captured[-1]["url"], "https://example.invalid/v1/chat/completions")

        # HTTP error handling
        def fail_http(req: Any, timeout: float = 0.0) -> Any:
            raise urllib.error.HTTPError(
                req.full_url,
                503,
                "Service Unavailable",
                hdrs={},  # type: ignore[arg-type]
                fp=io.BytesIO(b"overloaded"),
            )

        err_runner = HttpFlashRunner(base_url="https://example.invalid", opener=fail_http)
        with self.assertRaises(SwarmRunnerError) as ctx:
            err_runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="p")
        self.assertIn("503", str(ctx.exception))

        # Missing base_url
        missing_runner = HttpFlashRunner(env={})
        with self.assertRaises(ValueError):
            missing_runner.run_shadow_json(role="scribe", locked_system_prefix=prefix, user_payload="p")


class TestSwarmTablesAndRoleGuards(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = ReijiMemoryEngine(
            home=Path(self._tmp.name),
            db_path=":memory:",
            load_plugins=False,
            disable_plugins=True,
        )

    def tearDown(self) -> None:
        self.engine.close()
        self._tmp.cleanup()

    def test_swarm_tables_idempotent_notes_and_telemetry(self) -> None:
        ensure_swarm_tables(self.engine.db)
        ensure_swarm_tables(self.engine.db)

        with self.engine.db.read_connection() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        self.assertIn("pool_event_notes", tables)
        self.assertIn("swarm_telemetry", tables)

        self.assertIsNone(get_pool_event_note(self.engine.db, "ev_1"))
        save_pool_event_note(self.engine.db, "ev_1", {"summary": "v1", "files": ["a.py"]})
        self.assertEqual(
            get_pool_event_note(self.engine.db, "ev_1"),
            {"summary": "v1", "files": ["a.py"]},
        )

        # Upsert overwrites cleanly
        save_pool_event_note(self.engine.db, "ev_1", {"summary": "v2", "files": ["a.py", "b.py"]})
        self.assertEqual(
            get_pool_event_note(self.engine.db, "ev_1"),
            {"summary": "v2", "files": ["a.py", "b.py"]},
        )

        # Pop returns payload and deletes row
        popped = pop_pool_event_note(self.engine.db, "ev_1")
        self.assertEqual(popped, {"summary": "v2", "files": ["a.py", "b.py"]})
        self.assertIsNone(get_pool_event_note(self.engine.db, "ev_1"))
        self.assertIsNone(pop_pool_event_note(self.engine.db, "ev_1"))

        # Telemetry ledger
        row1 = record_swarm_telemetry(
            self.engine.db,
            role="scribe",
            action="digest",
            metrics={"turns": 3, "elapsed_ms": 42},
        )
        row2 = record_swarm_telemetry(
            self.engine.db,
            role="scout",
            action="recon",
            metrics={"hits": 2, "elapsed_ms": 18},
        )
        self.assertGreater(row2, row1)

        all_rows = list_swarm_telemetry(self.engine.db)
        self.assertEqual(len(all_rows), 2)
        self.assertEqual(all_rows[0]["role"], "scout")
        self.assertEqual(all_rows[1]["role"], "scribe")

        scribe_only = list_swarm_telemetry(self.engine.db, role="scribe")
        self.assertEqual(len(scribe_only), 1)
        self.assertEqual(scribe_only[0]["action"], "digest")
        self.assertEqual(scribe_only[0]["metrics"], {"turns": 3, "elapsed_ms": 42})

    def test_scribe_capabilities_allowed_and_forbidden(self) -> None:
        t1 = self.engine.append_turn("ep_1", "user", "We started setting up the swarm runner.")
        t2 = self.engine.append_turn("ep_1", "assistant", "Runner and guards are underway.")

        scribe = build_role_capabilities(self.engine, "scribe")
        self.assertIsInstance(scribe, ScribeCapabilities)

        # Allowed: undigested turns & watermark
        undigested = scribe.get_undigested_turns(watermark_key="pool_digested", limit=10)
        self.assertEqual([t["turn_id"] for t in undigested], [t1, t2])
        self.assertEqual(scribe.get_watermark("pool_digested"), 0)
        scribe.set_watermark("pool_digested", t2)
        self.assertEqual(scribe.get_watermark("pool_digested"), t2)
        self.assertEqual(scribe.get_undigested_turns(watermark_key="pool_digested", limit=10), [])

        # Allowed: ActivePool read/write + event note staging
        span1 = SourceSpan(episode_id="ep_1", turn_start=t1, turn_end=t1)
        span2 = SourceSpan(episode_id="ep_1", turn_start=t2, turn_end=t2)
        ev = scribe.create_event("Swarm setup", "Init runner", span1, scope="quest")
        self.assertEqual(scribe.get_event(ev.event_id).event_id, ev.event_id)
        updated = scribe.append_step(ev.event_id, "Add guards", span2)
        self.assertEqual(len(updated.steps), 2)
        scribe.touch_event(ev.event_id)
        scribe.update_status(ev.event_id, "blocked")
        self.assertEqual(scribe.get_event(ev.event_id).status, "blocked")
        self.assertIn("Swarm setup", scribe.pool_snapshot())
        self.assertGreaterEqual(len(scribe.list_events()), 1)

        scribe.save_event_note(ev.event_id, {"detail": "staged by scribe"})
        self.assertEqual(scribe.get_event_note(ev.event_id), {"detail": "staged by scribe"})
        scribe.record_telemetry("shadow_digest", {"events_touched": 1})
        self.assertEqual(list_swarm_telemetry(self.engine.db, role="scribe")[0]["action"], "shadow_digest")

        # Physically forbidden: CardStore writes, archival, midnight triage, internal handles, mutation
        for forbidden_attr in (
            "create_card",
            "supersede_card",
            "revoke_card",
            "link_cards",
            "resolve_backfill",
            "archive_pool_event",
            "wake_hibernated_card",
            "plan_midnight_triage",
            "compact_chain",
            "pop_event_note",
            "get_or_pop_event_note",
            "engine",
            "db",
            "cards",
            "pool",
            "kernel",
        ):
            with self.assertRaises(PermissionError, msg=f"Scribe must block {forbidden_attr}"):
                getattr(scribe, forbidden_attr)

        with self.assertRaises(PermissionError):
            setattr(scribe, "_engine", None)

    def test_scout_capabilities_strictly_read_only(self) -> None:
        t1 = self.engine.append_turn("ep_s", "user", "My graphics card is RTX 4090 with 24GB VRAM.")
        t2 = self.engine.append_turn("ep_s", "assistant", "Noted your RTX 4090 hardware setup.")
        span = SourceSpan(episode_id="ep_s", turn_start=t1, turn_end=t2)

        ev = ScribeCapabilities(self.engine).create_event(
            "GPU tuning", "Check VRAM", span, scope="quest"
        )
        ScribeCapabilities(self.engine).save_event_note(ev.event_id, {"vram": "24GB"})

        card = self.engine.cards.create_card(
            domain="infra",
            card_type="fact",
            title="Primary GPU",
            state={"gpu": "RTX 4090", "vram_gb": 24},
            tags=["gpu", "rtx4090", "hardware"],
            spans=[span],
        )

        scout = build_role_capabilities(self.engine, "scout")
        self.assertIsInstance(scout, ScoutCapabilities)

        # Allowed read-only operations
        self.assertIn("GPU tuning", scout.pool_snapshot())
        self.assertEqual(scout.get_event(ev.event_id).event_id, ev.event_id)
        self.assertEqual(scout.get_event_note(ev.event_id), {"vram": "24GB"})
        tags = scout.list_tags()
        self.assertTrue(any("gpu" in str(item).lower() for item in tags))
        recalled = scout.recall_level1("infra", ["gpu"], top_k=3)
        self.assertEqual([c.card_id for c in recalled], [card.card_id])
        self.assertEqual(scout.get_card(card.card_id).card_id, card.card_id)
        evidence = scout.read_evidence_level2(card.card_id, pad=0)
        self.assertGreaterEqual(len(evidence), 1)
        deep = scout.deep_slice_level3("RTX 4090", limit=2, pad=1)
        self.assertGreaterEqual(len(deep), 1)
        turns = scout.read_span(span, pad=0)
        self.assertEqual([t["turn_id"] for t in turns], [t1, t2])
        fts_hits = scout.search_turns_fts("RTX 4090", limit=5)
        self.assertGreaterEqual(len(fts_hits), 1)

        scout.record_telemetry("recon", {"recalled": 1})
        self.assertEqual(list_swarm_telemetry(self.engine.db, role="scout")[0]["role"], "scout")

        # Physically forbidden: ANY write action or internal handle access
        for forbidden_attr in (
            "append_turn",
            "set_watermark",
            "create_event",
            "open_event",
            "append_step",
            "update_status",
            "save_event_note",
            "pop_event_note",
            "get_or_pop_event_note",
            "archive_pool_event",
            "wake_hibernated_card",
            "create_card",
            "supersede_card",
            "revoke_card",
            "link_cards",
            "resolve_backfill",
            "plan_midnight_triage",
            "compact_chain",
            "engine",
            "db",
            "cards",
            "pool",
            "kernel",
        ):
            with self.assertRaises(PermissionError, msg=f"Scout must block {forbidden_attr}"):
                getattr(scout, forbidden_attr)

    def test_archivist_capabilities_curation_and_boundaries(self) -> None:
        t1 = self.engine.append_turn("ep_a", "user", "Finished writing module alpha.")
        t2 = self.engine.append_turn("ep_a", "assistant", "Module alpha verified and ready.")
        t3 = self.engine.append_turn("ep_a", "user", "Step 3 completed.")
        t4 = self.engine.append_turn("ep_a", "assistant", "Step 4 completed.")
        span1 = SourceSpan(episode_id="ep_a", turn_start=t1, turn_end=t1)
        span2 = SourceSpan(episode_id="ep_a", turn_start=t2, turn_end=t2)
        span3 = SourceSpan(episode_id="ep_a", turn_start=t3, turn_end=t3)
        span4 = SourceSpan(episode_id="ep_a", turn_start=t4, turn_end=t4)

        scribe = ScribeCapabilities(self.engine)
        ev = scribe.create_event("Module Alpha", "Drafted", span1, scope="quest")
        scribe.append_step(ev.event_id, "Tested", span2)
        scribe.append_step(ev.event_id, "Reviewed", span3)
        scribe.append_step(ev.event_id, "Polished", span4)
        scribe.update_status(ev.event_id, "done")
        scribe.save_event_note(ev.event_id, {"deliverable": "alpha.py"})

        archivist = build_role_capabilities(self.engine, "archivist")
        self.assertIsInstance(archivist, ArchivistCapabilities)

        # Allowed: midnight triage & chain compaction
        triage = archivist.plan_midnight_triage()
        self.assertIsNotNone(triage)
        compacted = archivist.compact_chain(ev.event_id, "All four stages complete.")
        self.assertEqual(compacted.checkpoint_summary, "All four stages complete.")

        # Allowed: get_or_pop_event_note (peek then pop)
        self.assertEqual(
            archivist.get_or_pop_event_note(ev.event_id, pop=False),
            {"deliverable": "alpha.py"},
        )
        self.assertEqual(
            archivist.get_or_pop_event_note(ev.event_id, pop=True),
            {"deliverable": "alpha.py"},
        )
        self.assertIsNone(archivist.get_or_pop_event_note(ev.event_id, pop=False))

        # Allowed: archive_pool_event
        archived_card = archivist.archive_pool_event(
            ev.event_id,
            domain="proj",
            tags=["alpha", "milestone"],
            state={"deliverable": "alpha.py"},
        )
        self.assertEqual(archived_card.domain, "proj")

        # Allowed: CardStore create / supersede / link / revoke
        card_v1 = archivist.create_card(
            domain="proj",
            card_type="fact",
            title="Alpha Status",
            state={"version": 1},
            tags=["alpha"],
            spans=[span1],
        )
        replacement = dataclasses.replace(
            card_v1,
            card_id="card_alpha_v2",
            state={"version": 2},
            spans=(span2,),
        )
        card_v2 = archivist.supersede_card(card_v1.card_id, replacement)
        self.assertEqual(archivist.get_card(card_v1.card_id).lifecycle_status, "superseded")
        self.assertEqual(card_v2.state["version"], 2)
        archivist.link_cards(archived_card.card_id, card_v2.card_id, relation="relates_to")
        archivist.revoke_card(card_v2.card_id)
        self.assertEqual(archivist.get_card(card_v2.card_id).lifecycle_status, "revoked")

        # Allowed: backfill resolution
        with self.engine.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO pending_backfills (episode_id, turn_start, turn_end, reason, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', '2026-11-27T00:00:00+00:00', '2026-11-27T00:00:00+00:00')
                """,
                ("ep_a", t1, t2, "missed digest"),
            )
            bf_id = int(cur.lastrowid)
        self.assertEqual(len(archivist.list_pending_backfills()), 1)
        self.assertTrue(archivist.resolve_backfill(bf_id, increment_attempts=True))
        self.assertEqual(len(archivist.list_pending_backfills()), 0)
        self.assertFalse(archivist.resolve_backfill(999999))

        archivist.record_telemetry("midnight_curate", {"archived": 1})
        self.assertEqual(
            list_swarm_telemetry(self.engine.db, role="archivist")[0]["action"],
            "midnight_curate",
        )

        # Physically forbidden: Scribe watermarks, raw turns, event note writes, internal handles
        for forbidden_attr in (
            "set_watermark",
            "append_turn",
            "save_event_note",
            "engine",
            "db",
            "cards",
            "pool",
            "kernel",
        ):
            with self.assertRaises(PermissionError, msg=f"Archivist must block {forbidden_attr}"):
                getattr(archivist, forbidden_attr)

        with self.assertRaises(ValueError):
            build_role_capabilities(self.engine, "unknown_role")


if __name__ == "__main__":
    unittest.main()
