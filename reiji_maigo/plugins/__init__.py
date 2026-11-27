"""Built-in self-bootstrapping plugin registry for Reiji-Maigo.

Aggregates the seven standard-library built-in optional plugins:
  - ``roi_ledger``: 7-decimal compute-economics ROI ledger & anti-MemoryTax guard
  - ``epoch_vault``: append-only SHA-256 verified cold epoch archive
  - ``thermal_sync``: lock-free tail reader & 250s cache warmth state machine
  - ``companion``: reversible cross-process locked companion hooks/skill installer
  - ``hybrid_reranker``: MemoRRF + Foresight + MemDream hybrid reranker provider
  - ``bootstrap_importer``: idempotent transcript/SQLite/checkpoint cold-start importer
  - ``bench_harness``: 6-dimension deterministic benchmark & scorecard harness
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import Any

from reiji_maigo.core.kernel import PluginRecord, ReijiKernel
from reiji_maigo.plugins.bench_harness import (
    BenchHarnessPlugin,
    create_plugin as _create_bench_harness_plugin,
)
from reiji_maigo.plugins.bootstrap_importer import (
    BootstrapImporterPlugin,
    create_plugin as _create_bootstrap_importer_plugin,
)
from reiji_maigo.plugins.companion import (
    CompanionPlugin,
    create_plugin as _create_companion_plugin,
)
from reiji_maigo.plugins.epoch_vault import (
    EpochVaultPlugin,
    create_plugin as _create_epoch_vault_plugin,
)
from reiji_maigo.plugins.hybrid_reranker import (
    HybridRerankerPlugin,
    create_plugin as _create_hybrid_reranker_plugin,
)
from reiji_maigo.plugins.roi_ledger import (
    RoiLedgerPlugin,
    create_plugin as _create_roi_ledger_plugin,
)
from reiji_maigo.plugins.thermal_sync import (
    ThermalSyncPlugin,
    create_plugin as _create_thermal_sync_plugin,
)

logger = logging.getLogger(__name__)

_KERNEL_SLOT_NAMES: tuple[str, ...] = (
    "register_cli",
    "register_hooks",
    "register_providers",
)


class _RegisterMethodAdapter:
    """Adapt a plugin exposing ``register(kernel)`` to ReijiKernel's slot protocol."""

    def __init__(self, wrapped: Any, *, name: str) -> None:
        self._wrapped = wrapped
        self.name = name

    def register_cli(self, kernel: ReijiKernel) -> None:
        self._wrapped.register(kernel)


def _adapt_plugin_for_kernel(plugin_obj: Any, *, name: str) -> Any:
    if any(callable(getattr(plugin_obj, slot, None)) for slot in _KERNEL_SLOT_NAMES):
        return plugin_obj
    if callable(getattr(plugin_obj, "register", None)):
        return _RegisterMethodAdapter(plugin_obj, name=name)
    return plugin_obj


BUILTIN_PLUGIN_FACTORIES: dict[str, Callable[[], Any]] = {
    "roi_ledger": _create_roi_ledger_plugin,
    "epoch_vault": _create_epoch_vault_plugin,
    "thermal_sync": _create_thermal_sync_plugin,
    "companion": _create_companion_plugin,
    "hybrid_reranker": _create_hybrid_reranker_plugin,
    "bootstrap_importer": _create_bootstrap_importer_plugin,
    "bench_harness": _create_bench_harness_plugin,
}


def register_builtin_plugins(
    kernel: ReijiKernel,
    *,
    factories: Mapping[str, Callable[[], Any]] | None = None,
) -> tuple[PluginRecord, ...]:
    """Register all built-in plugins into ``kernel`` with deduplication and fault isolation.

    Behavior:
      - Returns ``()`` immediately when ``kernel.plugins_disabled`` is True.
      - Skips any plugin whose name is already present in ``kernel.plugin_records``.
      - Delegates per-plugin disable filtering to ``kernel.disabled_plugins`` before
        invoking the factory so disabled plugins are never instantiated.
      - Isolates factory exceptions by recording a ``status="failed"`` ``PluginRecord``.
    """
    if getattr(kernel, "plugins_disabled", False):
        return ()

    registry = BUILTIN_PLUGIN_FACTORIES if factories is None else factories
    existing_lower = {name.lower() for name in kernel.plugin_records}
    disabled_lower = {
        str(name).lower() for name in getattr(kernel, "disabled_plugins", ())
    }

    records: list[PluginRecord] = []
    for raw_name, factory in registry.items():
        plugin_name = str(raw_name).strip()
        if not plugin_name or plugin_name.lower() in existing_lower:
            continue

        if plugin_name.lower() in disabled_lower:
            rec = kernel.register_plugin(None, name=plugin_name, source="builtin")
            existing_lower.add(plugin_name.lower())
            records.append(rec)
            continue

        try:
            plugin_obj = factory()
        except Exception as exc:  # noqa: BLE001 — fault isolation boundary
            logger.warning(
                "Built-in plugin factory %r failed during construction: %s",
                plugin_name,
                exc,
            )
            rec = PluginRecord(
                name=plugin_name,
                status="failed",
                source="builtin",
                error=f"{type(exc).__name__}: {exc}",
            )
            internal_records = getattr(kernel, "_plugin_records", None)
            if isinstance(internal_records, dict):
                internal_records[plugin_name] = rec
            existing_lower.add(plugin_name.lower())
            records.append(rec)
            continue

        adapted = _adapt_plugin_for_kernel(plugin_obj, name=plugin_name)
        rec = kernel.register_plugin(adapted, name=plugin_name, source="builtin")
        existing_lower.add(plugin_name.lower())
        records.append(rec)

    return tuple(records)


__all__ = [
    "BUILTIN_PLUGIN_FACTORIES",
    "BenchHarnessPlugin",
    "BootstrapImporterPlugin",
    "CompanionPlugin",
    "EpochVaultPlugin",
    "HybridRerankerPlugin",
    "RoiLedgerPlugin",
    "ThermalSyncPlugin",
    "register_builtin_plugins",
]
