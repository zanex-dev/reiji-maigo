"""reiji-maigo: Zero-dependency temporal memory microkernel."""

from reiji_maigo.core import (
    LOCKED_DOMAINS,
    PLUGIN_ENTRY_POINT_GROUP,
    PROVIDER_KINDS,
    CliCommandSpec,
    HookBus,
    HookEmitResult,
    HookError,
    MemoryCard,
    PluginRecord,
    PoolEvent,
    ProviderSpec,
    ReijiKernel,
    SourceSpan,
    get_disabled_plugin_names,
    is_plugins_disabled,
    resolve_reiji_home,
    validate_domain,
)

__version__ = "0.2.1"

__all__ = [
    "__version__",
    "LOCKED_DOMAINS",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PROVIDER_KINDS",
    "SourceSpan",
    "PoolEvent",
    "MemoryCard",
    "validate_domain",
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
