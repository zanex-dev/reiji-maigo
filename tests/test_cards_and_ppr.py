"""Unit tests for reiji_maigo.cards (PPR, controlled L2 tags, bitemporal CardStore, 3-level recall)."""

from __future__ import annotations

import unittest

from reiji_maigo.cards import CardStore, personalized_pagerank
from reiji_maigo.core.contracts import MemoryCard, SourceSpan
from reiji_maigo.storage.sqlite_engine import ReijiDatabase


class TestPersonalizedPageRank(unittest.TestCase):
    def test_two_hop_diffusion_and_mass_conservation(self) -> None:
        # Directed chain A -> B -> C (dangling C teleports back to seed A):
        # stationary ratios satisfy r(B) = damping * r(A), r(C) = damping^2 * r(A).
        adj = {
            "A": {"B": 1.0},
            "B": {"C": 1.0},
            "C": {},
            "D": {"E": 1.0},  # disconnected component
        }
        scores = personalized_pagerank(adj, {"A": 1.0}, damping=0.85, iters=12)

        self.assertIn("A", scores)
        self.assertIn("B", scores)
        self.assertIn("C", scores)
        self.assertNotIn("D", scores)
        self.assertNotIn("E", scores)
        self.assertGreater(scores["A"], scores["B"])
        self.assertGreater(scores["B"], scores["C"])
        self.assertGreater(scores["C"], 0.0)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_dangling_node_and_weight_preference(self) -> None:
        adj = {
            "seed": {"heavy": 3.0, "light": 1.0},
            "heavy": {},  # dangling node
            "light": {},  # dangling node
        }
        scores = personalized_pagerank(adj, {"seed": 2.0}, damping=0.85, iters=12)

        self.assertGreater(scores["heavy"], scores["light"])
        self.assertAlmostEqual(scores["heavy"] / scores["light"], 3.0, places=5)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_edge_cases_and_validation(self) -> None:
        self.assertEqual(personalized_pagerank({"A": {"B": 1.0}}, {}), {})
        self.assertEqual(personalized_pagerank({"A": {"B": 1.0}}, {"A": 0.0}), {})

        with self.assertRaises(ValueError):
            personalized_pagerank({}, {"A": 1.0}, damping=1.5)
        with self.assertRaises(ValueError):
            personalized_pagerank({}, {"A": 1.0}, iters=-1)
        with self.assertRaises(ValueError):
            personalized_pagerank({"A": {"B": -0.1}}, {"A": 1.0})
        with self.assertRaises(ValueError):
            personalized_pagerank({}, {"A": -1.0})


class TestCardStoreAndThreeLevelRecall(unittest.TestCase):
    def setUp(self) -> None:
        self.db = ReijiDatabase(":memory:")
        self.store = CardStore(self.db)

    def tearDown(self) -> None:
        self.db.close()

    def test_controlled_tag_normalization_and_domain_lock(self) -> None:
        with self.assertRaises(ValueError):
            self.store.register_or_resolve_tag("invalid_domain", "sqlite")

        canonical = self.store.register_or_resolve_tag(
            "infra",
            "  SQLite-FTS  ",
            aliases=("FTS5", "全文检索"),
        )
        self.assertEqual(canonical, "sqlite-fts")

        # Alias and case variations resolve back to canonical_name.
        self.assertEqual(self.store.register_or_resolve_tag("infra", "fts5"), "sqlite-fts")
        self.assertEqual(self.store.register_or_resolve_tag("infra", "全文检索"), "sqlite-fts")

        # Merging a new alias onto existing canonical tag preserves existing aliases.
        self.assertEqual(
            self.store.register_or_resolve_tag("infra", "sqlite-fts", aliases=("bigram",)),
            "sqlite-fts",
        )
        tags = self.store.list_tags("infra")
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0]["canonical_name"], "sqlite-fts")
        self.assertEqual(set(tags[0]["aliases"]), {"fts5", "全文检索", "bigram"})

        # Cross-domain canonical collision is rejected.
        with self.assertRaises(ValueError):
            self.store.register_or_resolve_tag("proj", "sqlite-fts")

    def test_create_card_requires_non_empty_spans(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_card(
                domain="proj",
                card_type="decision",
                title="Missing provenance",
                state={"choice": "A"},
                tags=["architecture"],
                spans=[],
            )

        span = SourceSpan("ep1", 1, 2)
        self.store.register_or_resolve_tag("proj", "reiji-core", aliases=("核心引擎",))
        card = self.store.create_card(
            domain="proj",
            card_type="decision",
            title="Zero-dep microkernel",
            state={"runtime": "stdlib"},
            tags=["核心引擎", "REIJi-CORE"],  # duplicates via alias collapse to single canonical tag
            spans=[span],
            foresight_note="Revisit after T05 integration",
            foresight_until="2026-12-01T00:00:00+00:00",
        )
        self.assertEqual(card.tags, ("reiji-core",))
        self.assertEqual(card.spans, (span,))
        self.assertEqual(card.lifecycle_status, "active")
        self.assertEqual(card.foresight_note, "Revisit after T05 integration")

        tag_info = self.store.list_tags("proj")[0]
        self.assertEqual(tag_info["usage_count"], 1)

    def test_bitemporal_supersede_and_revoke_zero_leakage(self) -> None:
        span_v1 = SourceSpan("ep_cfg", 1, 2)
        span_v2 = SourceSpan("ep_cfg", 3, 4)
        span_rev = SourceSpan("ep_cfg", 5, 6)

        old_card = self.store.create_card(
            domain="infra",
            card_type="config",
            title="Old port 8080",
            state={"port": 8080},
            tags=["gateway", "port"],
            spans=[span_v1],
            valid_from="2026-11-01T00:00:00+00:00",
        )
        # Unpersisted replacement MemoryCard passed directly to supersede_card.
        replacement = MemoryCard(
            card_id="card_new_port",
            domain="infra",
            card_type="config",
            title="New port 9090",
            state={"port": 9090},
            lifecycle_status="active",
            valid_from="2026-11-27T12:00:00+00:00",
            valid_to=None,
            superseded_by=None,
            tags=("gateway", "port"),
            spans=(span_v2,),
        )
        persisted_new = self.store.supersede_card(
            old_card.card_id,
            replacement,
            now_iso="2026-11-27T12:00:00+00:00",
        )
        self.assertEqual(persisted_new.card_id, "card_new_port")

        # Audit record on old card is preserved with bitemporal closure.
        audited_old = self.store.get_card(old_card.card_id)
        self.assertIsNotNone(audited_old)
        assert audited_old is not None
        self.assertEqual(audited_old.lifecycle_status, "superseded")
        self.assertEqual(audited_old.valid_to, "2026-11-27T12:00:00+00:00")
        self.assertEqual(audited_old.superseded_by, "card_new_port")

        # Create another card sharing exact tags and a link, then revoke it explicitly (TEPA).
        revoked_card = self.store.create_card(
            domain="infra",
            card_type="config",
            title="Retracted proxy rule",
            state={"proxy": "invalid"},
            tags=["gateway", "port"],
            spans=[span_rev],
        )
        self.store.link_cards(persisted_new.card_id, revoked_card.card_id, weight=5.0)
        self.store.link_cards(persisted_new.card_id, old_card.card_id, weight=5.0)

        self.assertTrue(self.store.revoke_card(revoked_card.card_id, now_iso="2026-11-27T13:00:00+00:00"))
        self.assertFalse(self.store.revoke_card("non_existent_card"))

        audited_rev = self.store.get_card(revoked_card.card_id)
        self.assertIsNotNone(audited_rev)
        assert audited_rev is not None
        self.assertEqual(audited_rev.lifecycle_status, "revoked")
        self.assertEqual(audited_rev.valid_to, "2026-11-27T13:00:00+00:00")

        # FAMA/TEPA red line: neither superseded nor revoked cards may leak into recall_level1.
        recalled = self.store.recall_level1("infra", ["gateway", "port"], top_k=10)
        recalled_ids = [c.card_id for c in recalled]
        self.assertEqual(recalled_ids, ["card_new_port"])
        self.assertTrue(all(c.lifecycle_status == "active" for c in recalled))

    def test_ppr_recalls_two_hop_active_card_without_literal_tag_overlap(self) -> None:
        self.store.register_or_resolve_tag("proj", "alpha-seed", aliases=("起点别名",))

        card_a = self.store.create_card(
            domain="proj",
            card_type="fact",
            title="Seed card A",
            state={"role": "seed"},
            tags=["alpha-seed"],
            spans=[SourceSpan("ep_ppr", 1, 1)],
        )
        card_b = self.store.create_card(
            domain="proj",
            card_type="fact",
            title="One-hop bridge B",
            state={"role": "hop1"},
            tags=["bridge-tag"],
            spans=[SourceSpan("ep_ppr", 2, 2)],
        )
        card_c = self.store.create_card(
            domain="proj",
            card_type="fact",
            title="Two-hop target C",
            state={"role": "hop2"},
            tags=["distant-tag"],  # zero literal overlap with query tag
            spans=[SourceSpan("ep_ppr", 3, 3)],
        )
        card_d = self.store.create_card(
            domain="proj",
            card_type="fact",
            title="Unrelated isolated card D",
            state={"role": "isolated"},
            tags=["unrelated-tag"],
            spans=[SourceSpan("ep_ppr", 4, 4)],
        )
        card_rev = self.store.create_card(
            domain="proj",
            card_type="fact",
            title="Revoked neighbor R",
            state={"role": "revoked"},
            tags=["revoked-tag"],
            spans=[SourceSpan("ep_ppr", 5, 5)],
        )

        # Link graph: A <-> B <-> C, plus B <-> R (revoked).
        self.store.link_cards(card_a.card_id, card_b.card_id, relation="depends_on", weight=1.0)
        self.store.link_cards(card_b.card_id, card_c.card_id, relation="enables", weight=1.0)
        self.store.link_cards(card_b.card_id, card_rev.card_id, relation="obsolete", weight=3.0)
        self.store.revoke_card(card_rev.card_id)

        # Query via alias of A's tag; top_k=5 should activate A -> B -> C (2-hop) and exclude D & R.
        recalled = self.store.recall_level1("proj", ["起点别名"], min_tag_hits=1, top_k=5)
        recalled_ids = [c.card_id for c in recalled]

        self.assertEqual(recalled_ids[0], card_a.card_id)
        self.assertIn(card_b.card_id, recalled_ids)
        self.assertIn(card_c.card_id, recalled_ids)
        self.assertNotIn(card_d.card_id, recalled_ids)
        self.assertNotIn(card_rev.card_id, recalled_ids)

        # Domain-agnostic recall (domain=None) also works.
        cross_recalled = self.store.recall_level1(None, ["起点别名"], top_k=5)
        self.assertEqual([c.card_id for c in cross_recalled], recalled_ids)

    def test_level2_evidence_and_level3_deep_slice_backfill(self) -> None:
        t1 = self.db.append_turn("ep_hist", "user", "先初始化工作区和配置文件")
        t2 = self.db.append_turn("ep_hist", "assistant", "已确认零时迷子记忆内核采用 SQLite WAL")
        t3 = self.db.append_turn("ep_hist", "user", "同时加上双时态卡片撤销机制")
        t4 = self.db.append_turn("ep_hist", "assistant", "好的，旧卡会标记 revoked 并留档")
        t5 = self.db.append_turn("ep_hist", "user", "最后一步记录验收日志")

        card = self.store.create_card(
            domain="research",
            card_type="note",
            title="WAL choice",
            state={"engine": "sqlite-wal"},
            tags=["storage"],
            spans=[SourceSpan("ep_hist", t2, t2)],
        )

        # Level-2 evidence slicing with pad=1 returns [t1, t2, t3].
        evidence = self.store.read_evidence_level2(card.card_id, pad=1)
        self.assertEqual([row["turn_id"] for row in evidence], [t1, t2, t3])

        with self.assertRaises(KeyError):
            self.store.read_evidence_level2("missing_card_id")

        # Level-3 deep slice via CJK FTS5 hits old turn and enqueues pending_backfills.
        slices = self.store.deep_slice_level3("零时迷子", limit=2, pad=1)
        self.assertEqual(len(slices), 1)
        hit_slice = slices[0]
        self.assertEqual(hit_slice["hit"]["turn_id"], t2)
        self.assertEqual([t["turn_id"] for t in hit_slice["turns"]], [t1, t2, t3])
        self.assertEqual(hit_slice["span"], {"episode_id": "ep_hist", "turn_start": t1, "turn_end": t3})

        backfills = self.store.list_pending_backfills()
        self.assertEqual(len(backfills), 1)
        self.assertEqual(backfills[0]["backfill_id"], hit_slice["backfill_id"])
        self.assertEqual(backfills[0]["episode_id"], "ep_hist")
        self.assertEqual(backfills[0]["turn_start"], t1)
        self.assertEqual(backfills[0]["turn_end"], t3)
        self.assertEqual(backfills[0]["status"], "pending")
        self.assertIn("零时迷子", backfills[0]["reason"])

        # Repeating same Level-3 query reuses existing pending backfill instead of duplicating rows.
        slices_repeat = self.store.deep_slice_level3("零时迷子", limit=2, pad=1)
        self.assertEqual(slices_repeat[0]["backfill_id"], hit_slice["backfill_id"])
        self.assertEqual(len(self.store.list_pending_backfills()), 1)
        self.assertNotEqual(t4, t5)  # keep all turns referenced cleanly


if __name__ == "__main__":
    unittest.main()
