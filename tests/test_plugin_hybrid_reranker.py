"""Unit tests for reiji_maigo.plugins.hybrid_reranker (MeMo + RRF + Foresight + MemDream)."""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from typing import Any

from reiji_maigo.cli import build_parser
from reiji_maigo.core.contracts import SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.hybrid_reranker import (
    EXPIRED_FORESIGHT_WARNING,
    HybridRerankerPlugin,
    MEMO_PROJECTION_L0_PREFIX,
    MEM_DREAM_L0_PREFIX,
    MemoRrfReranker,
    PLUGIN_NAME,
    PROVIDER_NAME,
    create_plugin,
    inspect_foresight,
    sweep_expired_foresight,
)
from reiji_maigo.swarm.runner import SwarmCallResult, compute_prefix_sha256


class _StubShadowRunner:
    """Deterministic stub implementing SwarmModelRunner for prefix-lock tests."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def run_shadow_json(
        self,
        *,
        role: str,
        locked_system_prefix: str,
        user_payload: str,
    ) -> SwarmCallResult:
        self.calls.append(
            {
                "role": role,
                "locked_system_prefix": locked_system_prefix,
                "user_payload": user_payload,
            }
        )
        data = self._responses.pop(0) if self._responses else {}
        raw = json.dumps(data, ensure_ascii=False)
        return SwarmCallResult(
            data=data,
            raw_text=raw,
            prompt_tokens=32,
            completion_tokens=16,
            elapsed_ms=2,
            prefix_sha256=compute_prefix_sha256(locked_system_prefix),
        )


class HybridRerankerPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReijiMemoryEngine(db_path=":memory:", load_plugins=False)
        self.plugin = create_plugin()
        self.record = self.engine.kernel.register_plugin(self.plugin)
        self.reranker = self.engine.kernel.get_provider("reranker")

    def tearDown(self) -> None:
        self.engine.close()

    def test_plugin_registers_default_reranker_provider_and_cli(self) -> None:
        self.assertIsInstance(self.plugin, HybridRerankerPlugin)
        self.assertEqual(self.plugin.name, PLUGIN_NAME)
        self.assertEqual(self.record.status, "loaded")
        self.assertIn("register_providers", self.record.slots)
        self.assertIn("register_cli", self.record.slots)

        self.assertIsInstance(self.reranker, MemoRrfReranker)
        self.assertIs(self.engine.kernel.get_provider("reranker", PROVIDER_NAME), self.reranker)

        spec = self.engine.kernel.get_cli_command("rerank")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertTrue(callable(getattr(spec.handler, "configure_parser", None)))

        parser = build_parser(plugin_commands=self.engine.kernel.cli_commands)
        ns_recall = parser.parse_args(["rerank", "recall", "--query", "测试", "--top-k", "5", "--json"])
        self.assertEqual(ns_recall.command, "rerank")
        self.assertEqual(ns_recall.rerank_action, "recall")
        self.assertEqual(ns_recall.top_k, 5)
        self.assertTrue(ns_recall.json)

        ns_foresight = parser.parse_args(["rerank", "foresight", "--domain", "infra"])
        self.assertEqual(ns_foresight.rerank_action, "foresight")
        self.assertEqual(ns_foresight.domain, "infra")

        ns_dream = parser.parse_args(["rerank", "dream", "--card-id", "c1", "--top-k", "2"])
        self.assertEqual(ns_dream.rerank_action, "dream")
        self.assertEqual(ns_dream.card_id, ["c1"])
        self.assertEqual(ns_dream.top_k, 2)

    def test_zero_lexical_overlap_alias_recall_and_bigram_projection(self) -> None:
        t1 = self.engine.append_turn("ep_gpu", "user", "记录一下集群显卡配额阈值")
        self.engine.cards.register_or_resolve_tag(
            "infra",
            "gpu_quota",
            aliases=["油箱见底", "预算上限"],
        )
        card = self.engine.cards.create_card(
            domain="infra",
            card_type="policy",
            title="GPU 算力池配额规则",
            state={"limit_gb": 512},
            tags=["gpu_quota"],
            spans=[SourceSpan("ep_gpu", t1, t1)],
        )

        # Query has zero character overlap with title, state, canonical tag, or raw turn.
        query = "油箱见底了怎么办"
        projected = self.reranker.project_query_tags(self.engine, query, domain="infra")
        self.assertEqual(projected, ["gpu_quota"])

        hits = self.reranker.hybrid_recall(self.engine, query, domain="infra", top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["card_id"], card.card_id)
        self.assertIn("ppr", hits[0]["ranks"])
        self.assertNotIn("lexical", hits[0]["ranks"])
        self.assertNotIn("fts", hits[0]["ranks"])
        self.assertFalse(hits[0]["expired_foresight"])
        self.assertNotIn("foresight_warning", hits[0])

        # CJK-Bigram partial overlap against alias ("预算上限" <-> "查预算")
        projected_bigram = self.reranker.project_query_tags(
            self.engine,
            "怎么查预算",
            domain="infra",
        )
        self.assertIn("gpu_quota", projected_bigram)

    def test_project_query_tags_with_locked_l0_runner_rejects_hallucinated_tags(self) -> None:
        self.engine.cards.register_or_resolve_tag("proj", "release_gate")
        self.engine.cards.register_or_resolve_tag("proj", "db_migration")

        stub = _StubShadowRunner(
            [{"tags": ["release_gate", "invented_hallucination_tag"]}]
        )
        projected = self.reranker.project_query_tags(
            self.engine,
            "发版前检查清单在哪里",
            domain="proj",
            runner=stub,
        )
        self.assertEqual(projected, ["release_gate"])
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(stub.calls[0]["role"], "memo_projector")
        self.assertEqual(stub.calls[0]["locked_system_prefix"], MEMO_PROJECTION_L0_PREFIX)

    def test_rrf_fuses_ppr_lexical_and_fts_channels_and_excludes_inactive_cards(self) -> None:
        t1 = self.engine.append_turn(
            "ep_arch",
            "user",
            "星核网关采用令牌桶限流策略保护上游并发",
        )
        t2 = self.engine.append_turn(
            "ep_arch",
            "user",
            "旧版轮询限流方案已被废弃",
        )
        self.engine.cards.register_or_resolve_tag("infra", "令牌桶", aliases=["限流"])

        # Card A: matches all 3 channels (PPR + lexical title + FTS bound turn).
        card_all = self.engine.cards.create_card(
            domain="infra",
            card_type="architecture",
            title="星核网关令牌桶限流设计",
            state={"algorithm": "token_bucket"},
            tags=["令牌桶"],
            spans=[SourceSpan("ep_arch", t1, t1)],
        )
        # Card B: matches only lexical state (no tag hit, different turn).
        card_lex_only = self.engine.cards.create_card(
            domain="infra",
            card_type="note",
            title="日常备忘",
            state={"memo": "提到过令牌桶"},
            tags=["misc"],
            spans=[SourceSpan("ep_arch", t2, t2)],
        )
        # Card C: revoked card must never leak into hybrid_recall.
        card_revoked = self.engine.cards.create_card(
            domain="infra",
            card_type="architecture",
            title="星核网关令牌桶限流旧版",
            state={"algorithm": "legacy"},
            tags=["令牌桶"],
            spans=[SourceSpan("ep_arch", t1, t1)],
        )
        self.engine.cards.revoke_card(card_revoked.card_id)

        results = self.reranker.hybrid_recall(
            self.engine,
            "令牌桶限流",
            domain="infra",
            top_k=5,
            rrf_k=60,
        )
        returned_ids = [r["card_id"] for r in results]
        self.assertNotIn(card_revoked.card_id, returned_ids)
        self.assertEqual(returned_ids[0], card_all.card_id)
        self.assertIn(card_lex_only.card_id, returned_ids)

        top = results[0]
        self.assertEqual(set(top["ranks"].keys()), {"ppr", "lexical", "fts"})
        expected_rrf = sum(1.0 / (60 + rank) for rank in top["ranks"].values())
        self.assertAlmostEqual(top["base_rrf_score"], expected_rrf, places=9)
        self.assertAlmostEqual(top["score"], expected_rrf, places=9)

    def test_foresight_inspection_sweep_and_expired_penalty_demotion(self) -> None:
        t1 = self.engine.append_turn("ep_plan", "user", "部署窗口排期记录")
        self.engine.cards.register_or_resolve_tag("proj", "部署窗口")

        expired_card = self.engine.cards.create_card(
            card_id="card_01_expired",
            domain="proj",
            card_type="plan",
            title="部署窗口排期方案",
            state={"window": "past"},
            tags=["部署窗口"],
            spans=[SourceSpan("ep_plan", t1, t1)],
            foresight_note="仅本周有效",
            foresight_until="2026-01-10T00:00:00Z",
        )
        fresh_card = self.engine.cards.create_card(
            card_id="card_02_fresh",
            domain="proj",
            card_type="plan",
            title="部署窗口排期方案",
            state={"window": "future"},
            tags=["部署窗口"],
            spans=[SourceSpan("ep_plan", t1, t1)],
            foresight_note="下月前有效",
            foresight_until="2026-12-31T23:59:59Z",
        )
        revoked_expired = self.engine.cards.create_card(
            card_id="card_03_revoked",
            domain="proj",
            card_type="plan",
            title="已撤销的过期排期",
            state={"window": "revoked"},
            tags=["部署窗口"],
            spans=[SourceSpan("ep_plan", t1, t1)],
            foresight_until="2025-01-01T00:00:00Z",
        )
        self.engine.cards.revoke_card(revoked_expired.card_id)

        now_iso = "2026-06-01T12:00:00Z"
        info_exp = inspect_foresight(expired_card, now_iso=now_iso)
        info_fresh = inspect_foresight(fresh_card, now_iso=now_iso)
        self.assertTrue(info_exp["expired"])
        self.assertEqual(info_exp["warning"], EXPIRED_FORESIGHT_WARNING)
        self.assertFalse(info_fresh["expired"])
        self.assertEqual(info_fresh["warning"], "")

        swept = sweep_expired_foresight(self.engine, domain="proj", now_iso=now_iso)
        self.assertEqual([item["card_id"] for item in swept], [expired_card.card_id])

        results = self.reranker.hybrid_recall(
            self.engine,
            "部署窗口排期方案",
            domain="proj",
            top_k=5,
            now_iso=now_iso,
            expired_penalty=0.5,
        )
        self.assertEqual(len(results), 2)
        # Fresh card must outrank the expired card after 0.5x penalty.
        self.assertEqual(results[0]["card_id"], fresh_card.card_id)
        self.assertFalse(results[0]["expired_foresight"])
        self.assertNotIn("foresight_warning", results[0])

        self.assertEqual(results[1]["card_id"], expired_card.card_id)
        self.assertTrue(results[1]["expired_foresight"])
        self.assertEqual(results[1]["foresight_warning"], EXPIRED_FORESIGHT_WARNING)
        self.assertAlmostEqual(
            results[1]["score"],
            results[1]["base_rrf_score"] * 0.5,
            places=9,
        )

    def test_mem_dream_auto_completes_aliases_and_heals_missed_colloquial_probe(self) -> None:
        t1 = self.engine.append_turn("ep_ops", "user", "记录容器镜像仓库定时清理规则")
        card = self.engine.cards.create_card(
            domain="infra",
            card_type="runbook",
            title="容器镜像仓库清理策略",
            state={"cron": "weekly"},
            tags=["registry_gc"],
            spans=[SourceSpan("ep_ops", t1, t1)],
        )

        colloquial_probe = "磁盘爆红拉不下新包怎么办？"
        # Before MemDream: zero lexical/tag overlap -> miss.
        before = self.reranker.hybrid_recall(
            self.engine,
            colloquial_probe,
            domain="infra",
            top_k=3,
        )
        self.assertEqual(before, [])

        stub = _StubShadowRunner([{"probes": [colloquial_probe]}])
        report = self.reranker.run_mem_dream(
            self.engine,
            card_ids=[card.card_id],
            runner=stub,
            top_k=3,
        )

        self.assertEqual(report["cards_scanned"], 1)
        self.assertEqual(report["probes_tested"], 1)
        self.assertEqual(report["missed_cards"], [card.card_id])
        self.assertGreater(report["aliases_added"], 0)
        self.assertEqual(stub.calls[0]["role"], "mem_dream")
        self.assertEqual(stub.calls[0]["locked_system_prefix"], MEM_DREAM_L0_PREFIX)

        # Verify aliases were persisted on primary tag in memory_tags.
        tags = self.engine.cards.list_tags(domain="infra")
        gc_entry = next(t for t in tags if t["canonical_name"] == "registry_gc")
        self.assertTrue(len(gc_entry["aliases"]) > 0)

        # After MemDream: the exact same colloquial probe now recalls the card!
        after = self.reranker.hybrid_recall(
            self.engine,
            colloquial_probe,
            domain="infra",
            top_k=3,
        )
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0]["card_id"], card.card_id)
        self.assertIn("registry_gc", after[0]["projected_tags"])

        # Card cryptographic seals remain intact after tag alias updates.
        self.assertTrue(self.engine.verify_card_seals()["ok"])

    def test_cli_rerank_subcommands_end_to_end(self) -> None:
        t1 = self.engine.append_turn("ep_cli", "user", "缓存键过期策略说明")
        self.engine.cards.register_or_resolve_tag("infra", "cache_ttl", aliases=["短命缓存"])
        card = self.engine.cards.create_card(
            domain="infra",
            card_type="policy",
            title="缓存键过期策略",
            state={"ttl_sec": 300},
            tags=["cache_ttl"],
            spans=[SourceSpan("ep_cli", t1, t1)],
            foresight_until="2026-01-01T00:00:00Z",
        )

        parser = build_parser(plugin_commands=self.engine.kernel.cli_commands)
        spec = self.engine.kernel.get_cli_command("rerank")
        assert spec is not None

        # 1. recall --json
        buf = io.StringIO()
        ns = parser.parse_args(
            [
                "rerank",
                "recall",
                "短命缓存",
                "--domain",
                "infra",
                "--now-iso",
                "2026-06-01T00:00:00Z",
                "--json",
            ]
        )
        with contextlib.redirect_stdout(buf):
            rc = spec.handler(self.engine, ns)
        self.assertEqual(rc, 0)
        recall_payload = json.loads(buf.getvalue())
        self.assertEqual(recall_payload["count"], 1)
        self.assertEqual(recall_payload["results"][0]["card_id"], card.card_id)
        self.assertTrue(recall_payload["results"][0]["expired_foresight"])
        self.assertEqual(
            recall_payload["results"][0]["foresight_warning"],
            EXPIRED_FORESIGHT_WARNING,
        )

        # 2. foresight --json
        buf = io.StringIO()
        ns = parser.parse_args(
            [
                "rerank",
                "foresight",
                "--domain",
                "infra",
                "--now-iso",
                "2026-06-01T00:00:00Z",
                "--json",
            ]
        )
        with contextlib.redirect_stdout(buf):
            rc = spec.handler(self.engine, ns)
        self.assertEqual(rc, 0)
        fs_payload = json.loads(buf.getvalue())
        self.assertEqual(fs_payload["expired_count"], 1)
        self.assertEqual(fs_payload["expired"][0]["card_id"], card.card_id)

        # 3. dream --json (offline mode without runner)
        buf = io.StringIO()
        ns = parser.parse_args(["rerank", "dream", "--card-id", card.card_id, "--json"])
        with contextlib.redirect_stdout(buf):
            rc = spec.handler(self.engine, ns)
        self.assertEqual(rc, 0)
        dream_payload = json.loads(buf.getvalue())
        self.assertEqual(dream_payload["cards_scanned"], 1)
        self.assertGreaterEqual(dream_payload["probes_tested"], 1)

    def test_source_hygiene_no_local_paths_or_private_literals(self) -> None:
        plugin_path = (
            Path(__file__).resolve().parents[1]
            / "reiji_maigo"
            / "plugins"
            / "hybrid_reranker.py"
        )
        source = plugin_path.read_text(encoding="utf-8")
        for forbidden in ("C:\\Users\\", "/Users/", "AppData", "acerola" + "_orion"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
