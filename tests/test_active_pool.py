"""Unit tests for ActivePool, 4D Zeigarnik decay, lossless compaction, and snapshot rendering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import unittest

from reiji_maigo.core.contracts import PoolEvent, SourceSpan
from reiji_maigo.pool import (
    ActivePool,
    compute_retention_score,
    estimate_event_tokens,
    estimate_tokens,
)
from reiji_maigo.storage import ReijiDatabase


class ActivePoolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = ReijiDatabase(":memory:")
        self.pool = ActivePool(self.db, max_events=8, max_tokens=900)
        self.base_dt = datetime(2026, 11, 27, 12, 0, 0, tzinfo=timezone.utc)
        self.base_iso = self.base_dt.isoformat()

    def tearDown(self) -> None:
        self.db.close()

    def _iso_after(self, *, hours: float = 0.0, days: float = 0.0) -> str:
        return (self.base_dt + timedelta(days=days, hours=hours)).isoformat()

    def test_create_and_append_step_renews_touch_hit_dormant_and_spans(self) -> None:
        span1 = SourceSpan(episode_id="ep_01", turn_start=1, turn_end=2)
        ev = self.pool.create_event(
            title="重构记忆池引擎",
            step="创建基础表结构与契约",
            span=span1,
            scope="quest",
            now_iso=self.base_iso,
        )
        self.assertEqual(ev.status, "active")
        self.assertEqual(ev.scope, "quest")
        self.assertEqual(ev.steps, ("创建基础表结构与契约",))
        self.assertEqual(ev.spans, (span1,))
        self.assertEqual(ev.hit_count, 0)
        self.assertEqual(ev.dormant_nights, 0)

        # Simulate 3 dormant nights before touching via append_step.
        updated_count = self.pool.increment_dormant_nights([ev.event_id], delta=3)
        self.assertEqual(updated_count, 1)
        dormant_ev = self.pool.get_event(ev.event_id)
        self.assertIsNotNone(dormant_ev)
        assert dormant_ev is not None
        self.assertEqual(dormant_ev.dormant_nights, 3)

        span2 = SourceSpan(episode_id="ep_01", turn_start=3, turn_end=5)
        t2 = self._iso_after(hours=6)
        ev2 = self.pool.append_step(
            ev.event_id,
            "接入四维衰减打分公式",
            span2,
            status="blocked",
            now_iso=t2,
        )
        self.assertEqual(ev2.status, "blocked")
        self.assertEqual(ev2.steps, ("创建基础表结构与契约", "接入四维衰减打分公式"))
        self.assertEqual(ev2.spans, (span1, span2))
        self.assertEqual(ev2.hit_count, 1)
        self.assertEqual(ev2.last_touched_at, t2)
        self.assertEqual(ev2.dormant_nights, 0)

        # Verify persistence round-trip.
        reloaded = self.pool.get_event(ev.event_id)
        self.assertEqual(reloaded, ev2)

    def test_retention_score_four_dimensions(self) -> None:
        span = SourceSpan(episode_id="ep_02", turn_start=1, turn_end=1)

        def make_ev(
            *,
            status: str = "active",
            scope: str = "quest",
            steps: tuple[str, ...] = ("简短步骤",),
            hit_count: int = 0,
            last_touched_at: str | None = None,
        ) -> PoolEvent:
            ts = last_touched_at or self.base_iso
            return PoolEvent(
                event_id="ev_score_test",
                title="评分基准任务",
                scope=scope,
                status=status,
                steps=steps,
                checkpoint_summary="",
                spans=(span,),
                hit_count=hit_count,
                created_at=self.base_iso,
                last_touched_at=ts,
                dormant_nights=0,
            )

        # 1. Zeigarnik status weight: active/blocked (1.5) vs done/aborted (0.35).
        ev_active = make_ev(status="active")
        ev_blocked = make_ev(status="blocked")
        ev_done = make_ev(status="done")
        ev_aborted = make_ev(status="aborted")

        s_active = compute_retention_score(ev_active, self.base_iso)
        s_blocked = compute_retention_score(ev_blocked, self.base_iso)
        s_done = compute_retention_score(ev_done, self.base_iso)
        s_aborted = compute_retention_score(ev_aborted, self.base_iso)

        self.assertAlmostEqual(s_active, s_blocked, places=9)
        self.assertAlmostEqual(s_done, s_aborted, places=9)
        self.assertAlmostEqual(s_active / s_done, 1.5 / 0.35, places=9)

        # 2. Scope weight & half-life decay (quest=168h, micro=12h).
        ev_micro = make_ev(scope="micro")
        s_quest_0 = compute_retention_score(ev_active, self.base_iso)
        s_quest_168h = compute_retention_score(ev_active, self._iso_after(hours=168))
        self.assertAlmostEqual(s_quest_168h / s_quest_0, 0.5, places=9)

        s_micro_0 = compute_retention_score(ev_micro, self.base_iso)
        s_micro_12h = compute_retention_score(ev_micro, self._iso_after(hours=12))
        self.assertAlmostEqual(s_micro_12h / s_micro_0, 0.5, places=9)
        self.assertAlmostEqual(s_quest_0 / s_micro_0, 2.0 / 0.8, places=9)

        # 3. Hit frequency boost: 1 + 0.3 * ln(1 + hit_count).
        ev_hit10 = make_ev(hit_count=10)
        s_hit10 = compute_retention_score(ev_hit10, self.base_iso)
        expected_boost = 1.0 + 0.3 * math.log(11.0)
        self.assertAlmostEqual(s_hit10 / s_quest_0, expected_boost, places=9)

        # 4. Token volume penalty: max(20, est_tokens).
        long_steps = tuple(f"第{i}步：详细排查异步调度与数据库锁竞争问题并记录日志" for i in range(12))
        ev_bloated = make_ev(steps=long_steps)
        self.assertGreater(estimate_event_tokens(ev_bloated), 20)
        s_bloated = compute_retention_score(ev_bloated, self.base_iso)
        self.assertLess(s_bloated, s_quest_0)
        self.assertAlmostEqual(
            s_quest_0 / s_bloated,
            max(20, estimate_event_tokens(ev_bloated))
            / max(20, estimate_event_tokens(ev_active)),
            places=9,
        )

        # 5. ISO Z-suffix and clock-skew clamp (now < last_touched -> dt=0).
        s_past = compute_retention_score(ev_active, self._iso_after(hours=-5))
        self.assertAlmostEqual(s_past, s_quest_0, places=9)
        s_z = compute_retention_score(ev_active, "2026-11-27T12:00:00Z")
        self.assertAlmostEqual(s_z, s_quest_0, places=9)

    def test_compact_chain_lossless_spans(self) -> None:
        spans = [
            SourceSpan(episode_id="ep_chain", turn_start=i * 2 + 1, turn_end=i * 2 + 2)
            for i in range(6)
        ]
        ev = self.pool.create_event(
            title="长链重构任务",
            step="步骤 0：初始化工作空间并梳理边界",
            span=spans[0],
            scope="quest",
            now_iso=self.base_iso,
        )
        for i in range(1, 6):
            ev = self.pool.append_step(
                ev.event_id,
                f"步骤 {i}：完成第 {i} 阶段详细变更与跨模块接口联调校验",
                spans[i],
                now_iso=self._iso_after(hours=i),
            )

        self.assertEqual(len(ev.steps), 6)
        self.assertEqual(ev.spans, tuple(spans))
        score_before = compute_retention_score(ev, self._iso_after(hours=6))

        compacted = self.pool.compact_chain(
            ev.event_id,
            "已完成阶段 0-3 核心架构搭建，进入最后联调。",
            keep_last=2,
        )
        self.assertEqual(len(compacted.steps), 2)
        self.assertEqual(compacted.steps, ev.steps[-2:])
        self.assertEqual(
            compacted.checkpoint_summary,
            "已完成阶段 0-3 核心架构搭建，进入最后联调。",
        )
        # Critical invariant: 100% lossless spans preservation!
        self.assertEqual(compacted.spans, tuple(spans))
        self.assertEqual(len(compacted.spans), 6)

        reloaded = self.pool.get_event(ev.event_id)
        self.assertEqual(reloaded, compacted)

        # Compacting reduces token footprint and improves retention score.
        score_after = compute_retention_score(compacted, self._iso_after(hours=6))
        self.assertGreater(score_after, score_before)

        # keep_last=0 clears step text while still keeping 100% spans.
        compacted_zero = self.pool.compact_chain(
            ev.event_id,
            "全部历史步骤已折叠为摘要。",
            keep_last=0,
        )
        self.assertEqual(compacted_zero.steps, ())
        self.assertEqual(compacted_zero.spans, tuple(spans))

    def test_plan_midnight_triage_buckets_and_capacity_eviction(self) -> None:
        now = self._iso_after(days=10)

        # 1. Done quest -> to_archive_done
        ev_done = self.pool.create_event(
            title="已完成的主线",
            step="上线验证通过",
            span=SourceSpan("ep_t", 1, 2),
            scope="quest",
            now_iso=self._iso_after(days=9),
        )
        ev_done = self.pool.append_step(
            ev_done.event_id,
            "收尾结项",
            SourceSpan("ep_t", 3, 4),
            status="done",
            now_iso=self._iso_after(days=9, hours=2),
        )

        # 2. Long active quest (6 steps > max_steps=5) -> to_compact
        ev_long = self.pool.create_event(
            title="步骤过长主线",
            step="s0",
            span=SourceSpan("ep_t", 10, 10),
            scope="quest",
            now_iso=self._iso_after(days=9, hours=18),
        )
        for idx in range(1, 6):
            ev_long = self.pool.append_step(
                ev_long.event_id,
                f"s{idx}",
                SourceSpan("ep_t", 10 + idx, 10 + idx),
                now_iso=self._iso_after(days=9, hours=18 + idx),
            )

        # 3. Dormant quest (touched 8 days ago -> >= 7 nights) -> to_hibernate
        ev_dormant = self.pool.create_event(
            title="沉睡主线",
            step="等待上游反馈",
            span=SourceSpan("ep_t", 20, 21),
            scope="quest",
            now_iso=self._iso_after(days=1),
        )

        # 4. Expired micro (> 24h old) and finished micro -> to_drop_micro
        ev_micro_old = self.pool.create_event(
            title="昨天的临时小事",
            step="顺手查个端口",
            span=SourceSpan("ep_t", 30, 30),
            scope="micro",
            now_iso=self._iso_after(days=8),
        )
        ev_micro_done = self.pool.create_event(
            title="刚做完的小事",
            step="改了个错别字",
            span=SourceSpan("ep_t", 31, 31),
            scope="micro",
            now_iso=self._iso_after(days=9, hours=22),
        )
        ev_micro_done = self.pool.update_status(ev_micro_done.event_id, "done")

        # 5. Fresh healthy quest & fresh healthy micro -> stay untouched
        ev_fresh_quest = self.pool.create_event(
            title="正在推进的新主线",
            step="起草接口",
            span=SourceSpan("ep_t", 40, 41),
            scope="quest",
            now_iso=self._iso_after(days=9, hours=23),
        )
        ev_fresh_micro = self.pool.create_event(
            title="刚提的微任务",
            step="格式化日志",
            span=SourceSpan("ep_t", 42, 42),
            scope="micro",
            now_iso=self._iso_after(days=9, hours=23),
        )

        plan = self.pool.plan_midnight_triage(
            now_iso=now,
            max_steps=5,
            max_dormant_nights=7,
        )
        self.assertEqual(
            set(plan.keys()),
            {"to_archive_done", "to_compact", "to_hibernate", "to_drop_micro"},
        )
        self.assertEqual([e.event_id for e in plan["to_archive_done"]], [ev_done.event_id])
        self.assertEqual([e.event_id for e in plan["to_compact"]], [ev_long.event_id])
        self.assertEqual([e.event_id for e in plan["to_hibernate"]], [ev_dormant.event_id])
        self.assertEqual(
            {e.event_id for e in plan["to_drop_micro"]},
            {ev_micro_old.event_id, ev_micro_done.event_id},
        )
        all_triaged = {
            e.event_id for bucket in plan.values() for e in bucket
        }
        self.assertNotIn(ev_fresh_quest.event_id, all_triaged)
        self.assertNotIn(ev_fresh_micro.event_id, all_triaged)

        # Capacity overflow check with a smaller max_events=2 pool.
        small_pool = ActivePool(self.db, max_events=2, max_tokens=500)
        small_plan = small_pool.plan_midnight_triage(
            now_iso=now, max_steps=5, max_dormant_nights=7
        )
        # Surviving active events were 3 (ev_long, ev_fresh_quest, ev_fresh_micro),
        # so the lowest-scoring 1 (ev_fresh_micro) must overflow into to_drop_micro.
        self.assertIn(
            ev_fresh_micro.event_id,
            {e.event_id for e in small_plan["to_drop_micro"]},
        )

    def test_render_snapshot_budget_and_undigested_tail(self) -> None:
        # Append 4 raw turns; mark first 2 as digested.
        t1 = self.db.append_turn("ep_snap", "user", "早期已经消化过的对话轮次一", self.base_iso)
        t2 = self.db.append_turn("ep_snap", "assistant", "早期已经消化过的对话轮次二", self._iso_after(hours=1))
        self.db.set_watermark("pool_digested", t2)
        t3 = self.db.append_turn("ep_snap", "user", "最新未消化请求：请顺便核对缓存淘汰策略", self._iso_after(hours=2))
        t4 = self.db.append_turn("ep_snap", "assistant", "收到，即将把缓存淘汰策略纳入核查清单", self._iso_after(hours=2.5))
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)
        self.assertLess(t3, t4)

        # Create high-priority active quest, lower-priority done quest, and off-pool archived quest.
        ev_high = self.pool.create_event(
            title="核心高优任务",
            step="定位并发写入瓶颈",
            span=SourceSpan("ep_snap", t1, t2),
            scope="quest",
            now_iso=self._iso_after(hours=2),
        )
        ev_high = self.pool.append_step(
            ev_high.event_id,
            "补充压测基准与指标验证",
            SourceSpan("ep_snap", t2, t2),
            now_iso=self._iso_after(hours=2.5),
        )
        ev_low = self.pool.create_event(
            title="低优收尾事项",
            step="整理归档文档",
            span=SourceSpan("ep_snap", t1, t1),
            scope="micro",
            now_iso=self._iso_after(hours=0),
        )
        self.pool.update_status(ev_low.event_id, "done")

        ev_archived = self.pool.create_event(
            title="已归档历史事件不应出现",
            step="历史记录",
            span=SourceSpan("ep_snap", t1, t1),
            scope="quest",
            now_iso=self._iso_after(hours=2),
        )
        self.pool.update_status(ev_archived.event_id, "archived")

        snapshot = self.pool.render_snapshot(
            now_iso=self._iso_after(hours=3),
            include_undigested=True,
        )
        self.assertLessEqual(estimate_tokens(snapshot), self.pool.max_tokens)
        self.assertIn("核心高优任务", snapshot)
        self.assertIn("低优收尾事项", snapshot)
        self.assertLess(snapshot.index("核心高优任务"), snapshot.index("低优收尾事项"))
        self.assertNotIn("已归档历史事件不应出现", snapshot)

        # Undigested tail fallback must include t3 & t4, and exclude digested t1 & t2.
        self.assertIn("[Undigested Tail]", snapshot)
        self.assertIn(f"#{t3}", snapshot)
        self.assertIn(f"#{t4}", snapshot)
        self.assertNotIn("早期已经消化过的对话轮次一", snapshot)
        self.assertNotIn("早期已经消化过的对话轮次二", snapshot)

        # When include_undigested=False, tail is omitted.
        no_tail = self.pool.render_snapshot(
            now_iso=self._iso_after(hours=3),
            include_undigested=False,
        )
        self.assertNotIn("[Undigested Tail]", no_tail)

        # Strict max_tokens enforcement under heavy load and tight budget.
        for i in range(10):
            self.pool.create_event(
                title=f"超长压测事件_{i}_" + ("测试文本" * 12),
                step="详细步骤描述：" + ("系统性能指标与上下文压缩验证" * 10),
                span=SourceSpan("ep_snap", 1, 2),
                scope="quest",
                now_iso=self._iso_after(hours=2.8),
            )
        tight_pool = ActivePool(self.db, max_events=8, max_tokens=140)
        tight_snapshot = tight_pool.render_snapshot(
            now_iso=self._iso_after(hours=3),
            include_undigested=True,
        )
        self.assertLessEqual(estimate_tokens(tight_snapshot), 140)
        self.assertIn("[ActivePool]", tight_snapshot)
        self.assertIn("[Undigested Tail]", tight_snapshot)

    def test_validation_and_error_handling(self) -> None:
        span = SourceSpan("ep_err", 1, 1)
        with self.assertRaises(ValueError):
            self.pool.create_event("", "step", span)
        with self.assertRaises(ValueError):
            self.pool.create_event("title", "   ", span)
        with self.assertRaises(ValueError):
            self.pool.create_event("title", "step", span, scope="invalid")
        with self.assertRaises(KeyError):
            self.pool.append_step("ev_nonexistent", "step", span)
        with self.assertRaises(KeyError):
            self.pool.compact_chain("ev_nonexistent", "summary")

        ev = self.pool.create_event("合法事件", "第一步", span, now_iso=self.base_iso)
        with self.assertRaises(ValueError):
            self.pool.compact_chain(ev.event_id, "  ")
        with self.assertRaises(ValueError):
            self.pool.compact_chain(ev.event_id, "summary", keep_last=-1)


if __name__ == "__main__":
    unittest.main()
