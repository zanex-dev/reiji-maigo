"""Core immutable contracts and fault-isolated plugin microkernel."""

from reiji_maigo.core.contracts import (
    LOCKED_DOMAINS,
    MemoryCard,
    PoolEvent,
    SourceSpan,
    validate_domain,
)
from reiji_maigo.core.kernel import (
    PLUGIN_ENTRY_POINT_GROUP,
    PROVIDER_KINDS,
    CliCommandSpec,
    HookBus,
    HookEmitResult,
    HookError,
    PluginRecord,
    ProviderSpec,
    ReijiKernel,
    get_disabled_plugin_names,
    is_plugins_disabled,
    resolve_reiji_home,
)

__all__ = [
    "LOCKED_DOMAINS",
    "SourceSpan",
    "PoolEvent",
    "MemoryCard",
    "validate_domain",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PROVIDER_KINDS",
    "HookError",
    "HookEmitResult",
    "HookBus",
    "CliCommandSpec",
    "ProviderSpec",
    "PluginRecord",
    "ReijiKernel",
    "resolve_reiji_home",
    "is_plugins_disabled",
    "get_disabled_plugin_names",
]
