"""Unit tests for AuditChainLedger and MutationApprovalGate (T03)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from reiji_maigo.cards.card_store import CardStore
from reiji_maigo.core.contracts import MemoryCard, SourceSpan
from reiji_maigo.security.audit_gate import (
    GENESIS_PREV_HASH,
    PROTECTED_CARD_TYPES,
    AuditChainLedger,
    MutationApprovalGate,
    MutationTier,
    ensure_security_tables,
)
from reiji_maigo.storage.sqlite_engine import ReijiDatabase


def _make_span(idx: int = 1) -> SourceSpan:
    return SourceSpan(
        episode_id=f"ep_{idx}",
        turn_start=1,
        turn_end=2,
    )


class AuditChainLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "reiji_test.db"
        self.db = ReijiDatabase(self.db_path)
        self.addCleanup(self.db.close)

    def test_ensure_security_tables_is_idempotent(self) -> None:
        ensure_security_tables(self.db)
        ensure_security_tables(self.db)

        with self.db.read_connection() as conn:
            tables = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
        self.assertTrue(
            {"audit_chain", "mutation_reviews", "card_seals"}.issubset(tables)
        )

    def test_append_and_verify_chain_happy_path_and_hmac(self) -> None:
        ledger = AuditChainLedger(self.db)
        empty_res = ledger.verify_chain()
        self.assertEqual(
            empty_res,
            {"ok": True, "checked": 0, "broken_seq": None, "reason": "empty chain"},
        )

        e1 = ledger.append(
            "curator_bee",
            "card_create",
            target_id="card_1",
            tier="L0",
            detail={"note": "initial entry"},
            now_iso="2026-11-27T08:00:00+00:00",
        )
        e2 = ledger.append(
            "curator_bee",
            "mutation_queued:revoke",
            target_id="card_1",
            tier=MutationTier.L1_APPROVAL,
            detail={"review_id": "rev_001"},
            now_iso="2026-11-27T08:01:00+00:00",
        )
        e3 = ledger.append(
            "human_admin",
            "mutation_approved:revoke",
            target_id="card_1",
            tier="L1",
            detail={"review_id": "rev_001"},
            now_iso="2026-11-27T08:02:00+00:00",
        )

        self.assertEqual(e1["seq"], 1)
        self.assertEqual(e1["prev_hash"], GENESIS_PREV_HASH)
        self.assertEqual(e2["seq"], 2)
        self.assertEqual(e2["prev_hash"], e1["entry_hash"])
        self.assertEqual(e3["seq"], 3)
        self.assertEqual(e3["prev_hash"], e2["entry_hash"])

        res = ledger.verify_chain()
        self.assertTrue(res["ok"])
        self.assertEqual(res["checked"], 3)
        self.assertIsNone(res["broken_seq"])

        # Separate DB verifying HMAC signing key enforcement.
        hmac_db = ReijiDatabase(Path(self._tmpdir.name) / "hmac_test.db")
        try:
            keyed_ledger = AuditChainLedger(hmac_db, signing_key=b"test-secret-key-32bytes")
            keyed_ledger.append("human_admin", "seal_init", target_id="card_9")
            self.assertTrue(keyed_ledger.verify_chain()["ok"])

            wrong_key_ledger = AuditChainLedger(hmac_db, signing_key=b"wrong-key")
            wrong_res = wrong_key_ledger.verify_chain()
            self.assertFalse(wrong_res["ok"])
            self.assertEqual(wrong_res["broken_seq"], 1)

            unsigned_ledger = AuditChainLedger(hmac_db)
            unsigned_res = unsigned_ledger.verify_chain()
            self.assertFalse(unsigned_res["ok"])
            self.assertEqual(unsigned_res["broken_seq"], 1)
        finally:
            hmac_db.close()

    def test_verify_chain_detects_sql_tampering_in_middle_row(self) -> None:
        ledger = AuditChainLedger(self.db)
        ledger.append("curator_bee", "op_1", target_id="c1", detail={"v": 1})
        ledger.append("curator_bee", "op_2", target_id="c2", detail={"v": 2})
        ledger.append("curator_bee", "op_3", target_id="c3", detail={"v": 3})

        # Directly tamper row seq=2 in SQLite.
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE audit_chain SET detail_json = ? WHERE seq = 2",
                ('{"v":999}',),
            )

        res = ledger.verify_chain()
        self.assertFalse(res["ok"])
        self.assertEqual(res["broken_seq"], 2)
        self.assertIn("entry_hash mismatch", res["reason"])

    def test_verify_chain_detects_sql_row_deletion(self) -> None:
        ledger = AuditChainLedger(self.db)
        ledger.append("curator_bee", "op_1", target_id="c1")
        ledger.append("curator_bee", "op_2", target_id="c2")
        ledger.append("curator_bee", "op_3", target_id="c3")

        # Directly delete middle row seq=2 via SQL.
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM audit_chain WHERE seq = 2")

        res = ledger.verify_chain()
        self.assertFalse(res["ok"])
        self.assertEqual(res["broken_seq"], 2)
        self.assertIn("sequence gap", res["reason"])

    def test_verify_chain_detects_first_row_deletion(self) -> None:
        ledger = AuditChainLedger(self.db)
        ledger.append("curator_bee", "op_1", target_id="c1")
        ledger.append("curator_bee", "op_2", target_id="c2")

        with self.db.transaction() as conn:
            conn.execute("DELETE FROM audit_chain WHERE seq = 1")

        res = ledger.verify_chain()
        self.assertFalse(res["ok"])
        self.assertEqual(res["broken_seq"], 1)


class MutationApprovalGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.db_path = Path(self._tmpdir.name) / "reiji_gate.db"
        self.db = ReijiDatabase(self.db_path)
        self.addCleanup(self.db.close)
        self.cards = CardStore(self.db)
        self.gate = MutationApprovalGate(self.db)

    def _create_sample_card(
        self,
        *,
        card_id: str,
        domain: str = "proj",
        card_type: str = "fact",
        state: dict | None = None,
    ) -> MemoryCard:
        return self.cards.create_card(
            domain=domain,
            card_type=card_type,
            title=f"Title for {card_id}",
            state=state if state is not None else {"value": "v1"},
            tags=[f"baseline_{domain}"],
            spans=[_make_span()],
            card_id=card_id,
            valid_from="2026-11-27T08:00:00+00:00",
        )

    def test_is_protected_card_rules(self) -> None:
        companion_card = self._create_sample_card(
            card_id="c_comp", domain="companion", card_type="note"
        )
        self.assertTrue(self.gate.is_protected_card(companion_card))

        for idx, ctype in enumerate(sorted(PROTECTED_CARD_TYPES)):
            card = self._create_sample_card(
                card_id=f"c_prot_{idx}", domain="proj", card_type=ctype
            )
            self.assertTrue(self.gate.is_protected_card(card))

        flagged_card = self._create_sample_card(
            card_id="c_flag",
            domain="proj",
            card_type="fact",
            state={"protected": True},
        )
        self.assertTrue(self.gate.is_protected_card(flagged_card))

        normal_card = self._create_sample_card(
            card_id="c_norm", domain="proj", card_type="fact", state={}
        )
        self.assertFalse(self.gate.is_protected_card(normal_card))

    def test_l1_staging_and_approve_revoke_on_companion_card(self) -> None:
        comp = self._create_sample_card(
            card_id="card_comp_1", domain="companion", card_type="bond_note"
        )

        # Curator bee tries to revoke companion card -> L1 staged, original stays active.
        outcome = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="revoke",
            target_card_id=comp.card_id,
            now_iso="2026-11-27T09:00:00+00:00",
        )
        self.assertEqual(outcome["status"], "pending_approval")
        self.assertFalse(outcome["allowed"])
        self.assertEqual(outcome["tier"], "L1")
        review_id = outcome["review_id"]
        self.assertIsNotNone(review_id)

        still_active = self.cards.get_card(comp.card_id)
        self.assertIsNotNone(still_active)
        assert still_active is not None
        self.assertEqual(still_active.lifecycle_status, "active")

        pending = self.gate.list_reviews(status="pending")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["review_id"], review_id)

        # Human admin approves -> card is now revoked atomically and logged.
        approved = self.gate.approve_review(
            review_id,
            self.cards,
            now_iso="2026-11-27T09:05:00+00:00",
        )
        self.assertEqual(approved["status"], "approved")
        self.assertTrue(approved["revoked"])

        after = self.cards.get_card(comp.card_id)
        assert after is not None
        self.assertEqual(after.lifecycle_status, "revoked")

        # Double approval must fail.
        with self.assertRaises(ValueError):
            self.gate.approve_review(review_id, self.cards)

        chain = self.gate.ledger.verify_chain()
        self.assertTrue(chain["ok"])
        self.assertEqual(chain["checked"], 2)

    def test_l1_staging_and_approve_supersede_on_core_rule_card(self) -> None:
        old_rule = self._create_sample_card(
            card_id="card_rule_v1", domain="proj", card_type="core_rule"
        )
        replacement = MemoryCard(
            card_id="card_rule_v2",
            domain="proj",
            card_type="core_rule",
            title="Updated core rule",
            state={"value": "v2"},
            lifecycle_status="active",
            valid_from="2026-11-27T09:10:00+00:00",
            valid_to=None,
            superseded_by=None,
            tags=("baseline_proj",),
            spans=(_make_span(2),),
        )

        outcome = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="supersede",
            target_card_id=old_rule.card_id,
            replacement_card=replacement,
            now_iso="2026-11-27T09:10:00+00:00",
        )
        self.assertEqual(outcome["status"], "pending_approval")
        self.assertEqual(outcome["tier"], "L1")
        review_id = outcome["review_id"]
        self.assertIsNotNone(review_id)

        # Old card remains active and replacement is not committed yet.
        old_before = self.cards.get_card(old_rule.card_id)
        assert old_before is not None
        self.assertEqual(old_before.lifecycle_status, "active")
        self.assertIsNone(self.cards.get_card(replacement.card_id))

        # Approve review -> old card becomes superseded, new card persists.
        approved = self.gate.approve_review(
            review_id,
            self.cards,
            now_iso="2026-11-27T09:15:00+00:00",
        )
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["result_card"].card_id, replacement.card_id)

        old_after = self.cards.get_card(old_rule.card_id)
        assert old_after is not None
        self.assertEqual(old_after.lifecycle_status, "superseded")
        self.assertEqual(old_after.superseded_by, replacement.card_id)

        new_after = self.cards.get_card(replacement.card_id)
        self.assertIsNotNone(new_after)
        self.assertTrue(self.gate.ledger.verify_chain()["ok"])

    def test_l1_reject_review_keeps_protected_card_active(self) -> None:
        protected = self._create_sample_card(
            card_id="card_prot_reject",
            domain="proj",
            card_type="persona_anchor",
        )

        staged = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="revoke",
            target_card_id=protected.card_id,
        )
        review_id = staged["review_id"]

        rejected = self.gate.reject_review(
            review_id,
            reason="anchor must remain active",
            now_iso="2026-11-27T09:20:00+00:00",
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["reason"], "anchor must remain active")

        card_after = self.cards.get_card(protected.card_id)
        assert card_after is not None
        self.assertEqual(card_after.lifecycle_status, "active")

        # Cannot approve a rejected review.
        with self.assertRaises(ValueError):
            self.gate.approve_review(review_id, self.cards)

        self.assertTrue(self.gate.ledger.verify_chain()["ok"])

    def test_l0_auto_executes_for_non_protected_card_and_human_admin(self) -> None:
        normal = self._create_sample_card(
            card_id="card_normal_1", domain="proj", card_type="fact"
        )
        res_normal = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="revoke",
            target_card_id=normal.card_id,
        )
        self.assertEqual(res_normal["status"], "executed")
        self.assertTrue(res_normal["allowed"])
        self.assertEqual(res_normal["tier"], "L0")
        after_normal = self.cards.get_card(normal.card_id)
        assert after_normal is not None
        self.assertEqual(after_normal.lifecycle_status, "revoked")

        # human_admin bypasses L1 queue on protected card.
        comp = self._create_sample_card(
            card_id="card_comp_admin", domain="companion", card_type="companion_bond"
        )
        res_admin = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="human_admin",
            op_kind="revoke",
            target_card_id=comp.card_id,
        )
        self.assertEqual(res_admin["status"], "executed")
        self.assertEqual(res_admin["tier"], "L0")
        after_comp = self.cards.get_card(comp.card_id)
        assert after_comp is not None
        self.assertEqual(after_comp.lifecycle_status, "revoked")
        self.assertTrue(self.gate.ledger.verify_chain()["ok"])

    def test_l2_hard_rejects_poison_and_untrusted_external(self) -> None:
        normal = self._create_sample_card(
            card_id="card_l2_target", domain="proj", card_type="fact"
        )

        # Poison detected -> L2 hard reject even for human_admin.
        r1 = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="human_admin",
            op_kind="revoke",
            target_card_id=normal.card_id,
            poison_detected=True,
        )
        self.assertEqual(r1["status"], "rejected")
        self.assertFalse(r1["allowed"])
        self.assertEqual(r1["tier"], "L2")
        self.assertIsNone(r1["review_id"])

        # Untrusted external origin -> L2 hard reject.
        r2 = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="revoke",
            target_card_id=normal.card_id,
            origin_trust="untrusted_external",
        )
        self.assertEqual(r2["status"], "rejected")
        self.assertFalse(r2["allowed"])
        self.assertEqual(r2["tier"], "L2")
        self.assertIsNone(r2["review_id"])

        # Target card remains untouched and no pending reviews created.
        card_after = self.cards.get_card(normal.card_id)
        assert card_after is not None
        self.assertEqual(card_after.lifecycle_status, "active")
        self.assertEqual(self.gate.list_reviews(status=None), [])

        chain = self.gate.ledger.verify_chain()
        self.assertTrue(chain["ok"])
        self.assertEqual(chain["checked"], 2)

    def test_approve_review_restores_pending_if_target_card_missing(self) -> None:
        comp = self._create_sample_card(
            card_id="card_comp_vanish", domain="companion", card_type="bond_note"
        )
        staged = self.gate.guard_supersede_or_revoke(
            self.cards,
            actor="curator_bee",
            op_kind="revoke",
            target_card_id=comp.card_id,
        )
        review_id = staged["review_id"]

        # Simulate external deletion of the card row before approval.
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM memory_cards WHERE card_id = ?", (comp.card_id,))

        with self.assertRaises(KeyError):
            self.gate.approve_review(review_id, self.cards)

        review = self.gate.get_review(review_id)
        assert review is not None
        self.assertEqual(review["status"], "pending")


if __name__ == "__main__":
    unittest.main()
