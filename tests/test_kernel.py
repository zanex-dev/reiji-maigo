"""Unit tests for reiji_maigo core contracts and fault-isolated microkernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from reiji_maigo import (
    LOCKED_DOMAINS,
    PLUGIN_ENTRY_POINT_GROUP,
    PROVIDER_KINDS,
    HookBus,
    MemoryCard,
    PoolEvent,
    ReijiKernel,
    SourceSpan,
    get_disabled_plugin_names,
    is_plugins_disabled,
    resolve_reiji_home,
    validate_domain,
)


class _FakeEntryPoint:
    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class _HealthyPlugin:
    name = "healthy_ext"

    def register_cli(self, kernel: ReijiKernel) -> None:
        kernel.register_cli("ping", lambda: "pong", help_text="Health check")

    def register_hooks(self, kernel: ReijiKernel) -> None:
        kernel.register_hooks("memory.saved", lambda payload: f"seen:{payload}")

    def register_providers(self, kernel: ReijiKernel) -> None:
        kernel.register_providers("reranker", "mock_rerank", {"kind": "rerank"}, default=True)
        kernel.register_providers(extractor={"kind": "extract"})


class _PartialCrashPlugin:
    name = "partial_crash"

    def register_cli(self, kernel: ReijiKernel) -> None:
        kernel.register_cli("ghost_cmd", lambda: "should-rollback")

    def register_hooks(self, kernel: ReijiKernel) -> None:
        kernel.register_hooks("memory.saved", lambda payload: "should-rollback")

    def register_providers(self, kernel: ReijiKernel) -> None:
        kernel.register_providers("reranker", "ghost_reranker", object())
        raise RuntimeError("boom during provider registration")


class TestCoreContracts(unittest.TestCase):
    def test_locked_domains_and_validation(self) -> None:
        self.assertEqual(LOCKED_DOMAINS, ("proj", "infra", "research", "companion", "life"))
        for domain in LOCKED_DOMAINS:
            self.assertEqual(validate_domain(domain), domain)
        with self.assertRaises(ValueError):
            validate_domain("finance")

    def test_source_span_immutability_slots_and_validation(self) -> None:
        span = SourceSpan(episode_id="ep-001", turn_start=1, turn_end=4)
        self.assertFalse(hasattr(span, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            span.turn_end = 5  # type: ignore[misc]
        self.assertEqual(SourceSpan.from_dict(span.to_dict()), span)

        with self.assertRaises(ValueError):
            SourceSpan(episode_id="", turn_start=0, turn_end=1)
        with self.assertRaises(ValueError):
            SourceSpan(episode_id="ep-001", turn_start=-1, turn_end=1)
        with self.assertRaises(ValueError):
            SourceSpan(episode_id="ep-001", turn_start=5, turn_end=4)

    def test_pool_event_contract_and_roundtrip(self) -> None:
        span = SourceSpan("ep-10", 2, 3)
        event = PoolEvent(
            event_id="evt-1",
            title="Investigate WAL lock",
            scope="proj",
            status="active",
            steps=("repro", "fix"),
            checkpoint_summary="Narrowed down to busy_timeout",
            spans=(span,),
            hit_count=2,
            created_at="2026-11-27T06:00:00Z",
            last_touched_at="2026-11-27T06:30:00Z",
        )
        self.assertEqual(event.dormant_nights, 0)
        self.assertFalse(hasattr(event, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            event.status = "closed"  # type: ignore[misc]
        self.assertEqual(PoolEvent.from_dict(event.to_dict()), event)

        with self.assertRaises(ValueError):
            PoolEvent(
                event_id="evt-2",
                title="Bad hit count",
                scope="proj",
                status="active",
                steps=(),
                checkpoint_summary="",
                spans=(),
                hit_count=-1,
                created_at="2026-11-27T06:00:00Z",
                last_touched_at="2026-11-27T06:00:00Z",
            )

    def test_memory_card_contract_domain_lock_and_roundtrip(self) -> None:
        span = SourceSpan("ep-20", 0, 2)
        card = MemoryCard(
            card_id="card-1",
            domain="research",
            card_type="fact",
            title="PPR damping factor",
            state={"alpha": 0.85},
            lifecycle_status="active",
            valid_from="2026-11-27T00:00:00Z",
            valid_to=None,
            superseded_by=None,
            tags=("graph", "retrieval"),
            spans=(span,),
        )
        self.assertEqual(card.foresight_note, "")
        self.assertIsNone(card.foresight_until)
        self.assertFalse(hasattr(card, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            card.title = "mutated"  # type: ignore[misc]
        self.assertEqual(MemoryCard.from_dict(card.to_dict()), card)

        with self.assertRaises(ValueError):
            MemoryCard(
                card_id="card-bad",
                domain="unknown_domain",
                card_type="fact",
                title="Invalid domain",
                state={},
                lifecycle_status="active",
                valid_from="2026-11-27T00:00:00Z",
                valid_to=None,
                superseded_by=None,
                tags=(),
                spans=(),
            )


class TestHookBus(unittest.TestCase):
    def test_priority_order_and_fault_isolation(self) -> None:
        bus = HookBus()
        trace: list[str] = []

        def low_priority(payload):
            trace.append(f"low:{payload}")
            return "low"

        def broken_handler(payload):
            trace.append(f"broken:{payload}")
            raise RuntimeError("handler crash")

        def high_priority(payload):
            trace.append(f"high:{payload}")
            return "high"

        bus.register("card.upsert", low_priority, plugin_name="p_low", priority=200)
        bus.register("card.upsert", broken_handler, plugin_name="p_bad", priority=100)
        bus.register("card.upsert", high_priority, plugin_name="p_high", priority=10)

        outcome = bus.emit("card.upsert", "c1")
        self.assertFalse(outcome.ok)
        self.assertEqual(trace, ["high:c1", "broken:c1", "low:c1"])
        self.assertEqual(outcome.results, ("high", "low"))
        self.assertEqual(len(outcome.errors), 1)
        self.assertEqual(outcome.errors[0].plugin_name, "p_bad")
        self.assertIn("RuntimeError: handler crash", outcome.errors[0].error)

        self.assertEqual(bus.unregister_plugin("p_bad"), 1)
        clean = bus.emit("card.upsert", "c2")
        self.assertTrue(clean.ok)
        self.assertEqual(clean.results, ("high", "low"))


class TestReijiKernel(unittest.TestCase):
    def test_env_helpers_and_home_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "REIJI_MAIGO_HOME": tmpdir,
                "REIJI_MAIGO_DISABLE_PLUGINS": "true",
                "REIJI_MAIGO_DISABLED_PLUGINS": "Alpha, beta ; Gamma",
            }
            self.assertEqual(resolve_reiji_home(env=env), Path(tmpdir).resolve())
            self.assertTrue(is_plugins_disabled(env=env))
            self.assertEqual(
                get_disabled_plugin_names(env=env),
                frozenset({"alpha", "beta", "gamma"}),
            )

    def test_three_slots_and_provider_validation(self) -> None:
        kernel = ReijiKernel(env={"REIJI_MAIGO_DISABLE_PLUGINS": "1"})
        self.assertEqual(PROVIDER_KINDS, ("reranker", "extractor"))
        self.assertEqual(PLUGIN_ENTRY_POINT_GROUP, "reiji_maigo.plugins")

        kernel.register_cli("status", lambda: {"ok": True}, help_text="Show status")
        self.assertIsNotNone(kernel.get_cli_command("status"))
        self.assertEqual(kernel.get_cli_command("status").handler(), {"ok": True})  # type: ignore[union-attr]

        kernel.register_hooks("pool.decay", lambda n: n + 1)
        self.assertEqual(kernel.hooks.emit("pool.decay", 2).results, (3,))

        reranker_a = object()
        reranker_b = object()
        extractor_a = object()
        kernel.register_providers("reranker", "a", reranker_a)
        kernel.register_providers("reranker", "b", reranker_b, default=True)
        kernel.register_providers(extractor=extractor_a, name="ext_a")

        self.assertIs(kernel.get_provider("reranker"), reranker_b)
        self.assertIs(kernel.get_provider("reranker", "a"), reranker_a)
        self.assertIs(kernel.get_provider("extractor", "ext_a"), extractor_a)

        with self.assertRaises(ValueError):
            kernel.register_providers("unsupported_kind", "x", object())

    def test_entry_points_loading_disable_and_fault_isolation(self) -> None:
        fake_eps = [
            _FakeEntryPoint("healthy_ext", lambda: _HealthyPlugin),
            _FakeEntryPoint("broken_import", lambda: (_ for _ in ()).throw(ImportError("missing dep"))),
            _FakeEntryPoint("partial_crash", lambda: _PartialCrashPlugin()),
            _FakeEntryPoint("muted_ext", lambda: _HealthyPlugin()),
        ]

        # 1) Global disable via REIJI_MAIGO_DISABLE_PLUGINS=1
        disabled_kernel = ReijiKernel(env={"REIJI_MAIGO_DISABLE_PLUGINS": "1"})
        with mock.patch("reiji_maigo.core.kernel.importlib_metadata.entry_points", return_value=fake_eps) as mocked:
            loaded = disabled_kernel.load_plugins()
            self.assertEqual(loaded, ())
            mocked.assert_not_called()

        # 2) Selective disable + fault isolation + rollback of partially registered plugin
        active_kernel = ReijiKernel(
            env={
                "REIJI_MAIGO_DISABLE_PLUGINS": "0",
                "REIJI_MAIGO_DISABLED_PLUGINS": "muted_ext",
            }
        )
        with mock.patch("reiji_maigo.core.kernel.importlib_metadata.entry_points", return_value=fake_eps) as mocked:
            records = active_kernel.load_plugins()
            mocked.assert_called_once_with(group="reiji_maigo.plugins")

        by_name = {rec.name: rec for rec in records}
        self.assertEqual(by_name["healthy_ext"].status, "loaded")
        self.assertEqual(by_name["muted_ext"].status, "skipped")
        self.assertEqual(by_name["broken_import"].status, "failed")
        self.assertEqual(by_name["partial_crash"].status, "failed")

        # Healthy plugin slots survived; crashed plugin slots were rolled back
        self.assertIn("ping", active_kernel.cli_commands)
        self.assertNotIn("ghost_cmd", active_kernel.cli_commands)
        self.assertEqual(active_kernel.hooks.emit("memory.saved", "ok").results, ("seen:ok",))
        self.assertEqual(active_kernel.get_provider("reranker"), {"kind": "rerank"})
        self.assertIsNone(active_kernel.get_provider("reranker", "ghost_reranker"))
        self.assertEqual(active_kernel.get_provider("extractor"), {"kind": "extract"})

        summary = active_kernel.status()
        self.assertEqual(summary["loaded_plugins"], ["healthy_ext"])
        self.assertEqual(summary["skipped_plugins"], ["muted_ext"])
        self.assertEqual(sorted(summary["failed_plugins"].keys()), ["broken_import", "partial_crash"])

    def test_pyproject_metadata_contract(self) -> None:
        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject_path.read_text(encoding="utf-8")
        self.assertIn('license = { text = "PolyForm-Noncommercial-1.0.0" }', text)
        self.assertIn("dependencies = []", text)
        self.assertIn('reiji = "reiji_maigo.cli:main"', text)
        self.assertIn('[project.entry-points."shinobu.plugins"]', text)
        self.assertIn('reiji_maigo = "reiji_maigo.bridges.shinobu:create_plugin"', text)


if __name__ == "__main__":
    unittest.main()
