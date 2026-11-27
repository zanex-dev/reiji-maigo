"""Fault-isolated plugin microkernel and HookBus for reiji-maigo."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
import logging
import os
from pathlib import Path
import re
from typing import Any

logger = logging.getLogger("reiji_maigo.core.kernel")

ENV_HOME = "REIJI_MAIGO_HOME"
ENV_DISABLE_PLUGINS = "REIJI_MAIGO_DISABLE_PLUGINS"
ENV_DISABLED_PLUGINS = "REIJI_MAIGO_DISABLED_PLUGINS"

PLUGIN_ENTRY_POINT_GROUP = "reiji_maigo.plugins"
PROVIDER_KINDS: tuple[str, ...] = ("reranker", "extractor")
_PROVIDER_KIND_SET: frozenset[str] = frozenset(PROVIDER_KINDS)
_TRUTHY_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def resolve_reiji_home(
    home: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the reiji-maigo home directory from argument, env, or ~/.reiji_maigo."""
    if home is not None:
        return Path(home).expanduser().resolve()
    environ = os.environ if env is None else env
    raw = (environ.get(ENV_HOME) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".reiji_maigo").resolve()


def is_plugins_disabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True when global plugin loading is disabled via environment variable."""
    environ = os.environ if env is None else env
    raw = (environ.get(ENV_DISABLE_PLUGINS) or "").strip().lower()
    return raw in _TRUTHY_VALUES


def get_disabled_plugin_names(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Parse REIJI_MAIGO_DISABLED_PLUGINS into a normalized frozenset of plugin names."""
    environ = os.environ if env is None else env
    raw = (environ.get(ENV_DISABLED_PLUGINS) or "").strip()
    if not raw:
        return frozenset()
    tokens = re.split(r"[,;\s]+", raw)
    return frozenset(tok.strip().lower() for tok in tokens if tok.strip())


@dataclass(frozen=True, slots=True)
class HookError:
    """Captured exception raised by a hook handler during HookBus.emit."""

    event: str
    plugin_name: str | None
    handler_name: str
    error: str


@dataclass(frozen=True, slots=True)
class HookEmitResult:
    """Aggregated outcome of emitting an event on HookBus."""

    event: str
    results: tuple[Any, ...] = ()
    errors: tuple[HookError, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


@dataclass(frozen=True, slots=True)
class _HookEntry:
    priority: int
    order: int
    plugin_name: str | None
    handler: Callable[..., Any]


class HookBus:
    """Fault-isolated, priority-ordered hook bus."""

    def __init__(self) -> None:
        self._entries: dict[str, list[_HookEntry]] = {}
        self._seq: int = 0

    def register(
        self,
        event: str,
        handler: Callable[..., Any],
        *,
        plugin_name: str | None = None,
        priority: int = 100,
    ) -> None:
        if not isinstance(event, str) or not event.strip():
            raise ValueError("Hook event name must be a non-empty string")
        if not callable(handler):
            raise TypeError("Hook handler must be callable")
        bucket = self._entries.setdefault(event, [])
        self._seq += 1
        bucket.append(
            _HookEntry(
                priority=int(priority),
                order=self._seq,
                plugin_name=plugin_name,
                handler=handler,
            )
        )
        bucket.sort(key=lambda item: (item.priority, item.order))

    def unregister(self, event: str, handler: Callable[..., Any]) -> bool:
        bucket = self._entries.get(event)
        if not bucket:
            return False
        before = len(bucket)
        bucket[:] = [entry for entry in bucket if entry.handler is not handler]
        if not bucket:
            self._entries.pop(event, None)
        return len(bucket) != before

    def unregister_plugin(self, plugin_name: str) -> int:
        removed = 0
        empty_events: list[str] = []
        for event, bucket in self._entries.items():
            before = len(bucket)
            bucket[:] = [entry for entry in bucket if entry.plugin_name != plugin_name]
            removed += before - len(bucket)
            if not bucket:
                empty_events.append(event)
        for event in empty_events:
            self._entries.pop(event, None)
        return removed

    def handlers(self, event: str) -> tuple[Callable[..., Any], ...]:
        return tuple(entry.handler for entry in self._entries.get(event, ()))

    def events(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries.keys()))

    def clear(self, event: str | None = None) -> None:
        if event is None:
            self._entries.clear()
        else:
            self._entries.pop(event, None)

    def emit(self, event: str, payload: Any = None, **kwargs: Any) -> HookEmitResult:
        entries = list(self._entries.get(event, ()))
        results: list[Any] = []
        errors: list[HookError] = []
        for entry in entries:
            try:
                if kwargs:
                    result = entry.handler(payload, **kwargs)
                else:
                    result = entry.handler(payload)
                results.append(result)
            except Exception as exc:  # noqa: BLE001 — fault isolation boundary
                handler_name = getattr(entry.handler, "__qualname__", repr(entry.handler))
                logger.warning(
                    "Hook handler %s (plugin=%s) failed on event %r: %s",
                    handler_name,
                    entry.plugin_name,
                    event,
                    exc,
                )
                errors.append(
                    HookError(
                        event=event,
                        plugin_name=entry.plugin_name,
                        handler_name=handler_name,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return HookEmitResult(
            event=event,
            results=tuple(results),
            errors=tuple(errors),
        )


@dataclass(frozen=True, slots=True)
class CliCommandSpec:
    """Registered CLI command entry in the microkernel."""

    name: str
    handler: Callable[..., Any]
    help_text: str = ""
    plugin_name: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Registered provider entry (reranker or extractor)."""

    kind: str
    name: str
    provider: Any
    plugin_name: str | None = None


@dataclass(frozen=True, slots=True)
class PluginRecord:
    """Status record for a discovered or directly registered plugin."""

    name: str
    status: str  # "loaded" | "skipped" | "failed"
    source: str = "direct"
    slots: tuple[str, ...] = ()
    error: str | None = None


class ReijiKernel:
    """Zero-dependency microkernel with fault-isolated plugin slots.

    Slots:
      - register_cli: CLI command registration
      - register_hooks: HookBus event handler registration
      - register_providers: reranker / extractor provider registration

    External plugins are discovered from ``entry_points("reiji_maigo.plugins")``
    unless disabled by ``REIJI_MAIGO_DISABLE_PLUGINS`` or selectively skipped
    by ``REIJI_MAIGO_DISABLED_PLUGINS``.
    """

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        disable_plugins: bool | None = None,
        disabled_plugins: Iterable[str] | None = None,
        entry_point_group: str = PLUGIN_ENTRY_POINT_GROUP,
    ) -> None:
        self._env: dict[str, str] = dict(os.environ if env is None else env)
        self.home: Path = resolve_reiji_home(home, env=self._env)
        self.plugins_disabled: bool = (
            is_plugins_disabled(self._env) if disable_plugins is None else bool(disable_plugins)
        )
        env_disabled = get_disabled_plugin_names(self._env)
        extra_disabled = frozenset(
            name.strip().lower()
            for name in (disabled_plugins or ())
            if isinstance(name, str) and name.strip()
        )
        self.disabled_plugins: frozenset[str] = env_disabled | extra_disabled
        self.entry_point_group: str = entry_point_group

        self.hooks: HookBus = HookBus()
        self._cli_commands: dict[str, CliCommandSpec] = {}
        self._providers: dict[str, dict[str, ProviderSpec]] = {
            kind: {} for kind in PROVIDER_KINDS
        }
        self._default_providers: dict[str, str] = {}
        self._plugin_records: dict[str, PluginRecord] = {}
        self._active_plugin_name: str | None = None

    # ------------------------------------------------------------------
    # Slot 1: CLI commands
    # ------------------------------------------------------------------
    def register_cli(
        self,
        name: str,
        handler: Callable[..., Any],
        *,
        help_text: str = "",
        plugin_name: str | None = None,
        replace: bool = False,
    ) -> CliCommandSpec:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("CLI command name must be a non-empty string")
        if not callable(handler):
            raise TypeError("CLI command handler must be callable")
        cmd_name = name.strip()
        if cmd_name in self._cli_commands and not replace:
            raise ValueError(f"CLI command {cmd_name!r} is already registered")
        spec = CliCommandSpec(
            name=cmd_name,
            handler=handler,
            help_text=help_text,
            plugin_name=plugin_name if plugin_name is not None else self._active_plugin_name,
        )
        self._cli_commands[cmd_name] = spec
        return spec

    def get_cli_command(self, name: str) -> CliCommandSpec | None:
        return self._cli_commands.get(name)

    @property
    def cli_commands(self) -> dict[str, CliCommandSpec]:
        return dict(self._cli_commands)

    # ------------------------------------------------------------------
    # Slot 2: Hooks
    # ------------------------------------------------------------------
    def register_hooks(
        self,
        event: str,
        handler: Callable[..., Any],
        *,
        plugin_name: str | None = None,
        priority: int = 100,
    ) -> None:
        owner = plugin_name if plugin_name is not None else self._active_plugin_name
        self.hooks.register(event, handler, plugin_name=owner, priority=priority)

    # ------------------------------------------------------------------
    # Slot 3: Providers (reranker / extractor)
    # ------------------------------------------------------------------
    def register_providers(
        self,
        kind: str | None = None,
        name: str | None = None,
        provider: Any = None,
        *,
        reranker: Any = None,
        extractor: Any = None,
        plugin_name: str | None = None,
        default: bool = False,
        replace: bool = False,
    ) -> list[ProviderSpec]:
        owner = plugin_name if plugin_name is not None else self._active_plugin_name
        registered: list[ProviderSpec] = []

        if kind is not None:
            if name is None or not isinstance(name, str) or not name.strip():
                raise ValueError("Provider name must be a non-empty string")
            if provider is None:
                raise ValueError("Provider instance must not be None")
            registered.append(
                self._store_provider(
                    kind=kind,
                    name=name.strip(),
                    provider=provider,
                    plugin_name=owner,
                    default=default,
                    replace=replace,
                )
            )

        if reranker is not None:
            prov_name = (
                name.strip()
                if (kind is None and isinstance(name, str) and name.strip())
                else owner or getattr(reranker, "name", type(reranker).__name__)
            )
            registered.append(
                self._store_provider(
                    kind="reranker",
                    name=prov_name,
                    provider=reranker,
                    plugin_name=owner,
                    default=default,
                    replace=replace,
                )
            )

        if extractor is not None:
            prov_name = (
                name.strip()
                if (kind is None and isinstance(name, str) and name.strip())
                else owner or getattr(extractor, "name", type(extractor).__name__)
            )
            registered.append(
                self._store_provider(
                    kind="extractor",
                    name=prov_name,
                    provider=extractor,
                    plugin_name=owner,
                    default=default,
                    replace=replace,
                )
            )

        if not registered:
            raise ValueError(
                "register_providers requires either (kind, name, provider) "
                "or keyword reranker/extractor"
            )
        return registered

    def _store_provider(
        self,
        *,
        kind: str,
        name: str,
        provider: Any,
        plugin_name: str | None,
        default: bool,
        replace: bool,
    ) -> ProviderSpec:
        if kind not in _PROVIDER_KIND_SET:
            allowed = ", ".join(PROVIDER_KINDS)
            raise ValueError(f"Unsupported provider kind {kind!r}; expected one of: {allowed}")
        bucket = self._providers[kind]
        if name in bucket and not replace:
            raise ValueError(f"Provider {kind}:{name!r} is already registered")
        spec = ProviderSpec(kind=kind, name=name, provider=provider, plugin_name=plugin_name)
        bucket[name] = spec
        if default or kind not in self._default_providers:
            self._default_providers[kind] = name
        return spec

    def get_provider(self, kind: str, name: str | None = None) -> Any | None:
        if kind not in _PROVIDER_KIND_SET:
            allowed = ", ".join(PROVIDER_KINDS)
            raise ValueError(f"Unsupported provider kind {kind!r}; expected one of: {allowed}")
        bucket = self._providers[kind]
        if name is None:
            default_name = self._default_providers.get(kind)
            spec = bucket.get(default_name) if default_name is not None else None
        else:
            spec = bucket.get(name)
        return None if spec is None else spec.provider

    def list_providers(self, kind: str | None = None) -> dict[str, dict[str, ProviderSpec]]:
        if kind is not None:
            if kind not in _PROVIDER_KIND_SET:
                allowed = ", ".join(PROVIDER_KINDS)
                raise ValueError(f"Unsupported provider kind {kind!r}; expected one of: {allowed}")
            return {kind: dict(self._providers[kind])}
        return {k: dict(v) for k, v in self._providers.items()}

    # ------------------------------------------------------------------
    # Plugin registration & entry_points discovery
    # ------------------------------------------------------------------
    def _rollback_plugin(self, plugin_name: str) -> None:
        self.hooks.unregister_plugin(plugin_name)
        for cmd_name in [
            key for key, spec in self._cli_commands.items() if spec.plugin_name == plugin_name
        ]:
            self._cli_commands.pop(cmd_name, None)
        for kind, bucket in self._providers.items():
            removed_names = [
                key for key, spec in bucket.items() if spec.plugin_name == plugin_name
            ]
            for key in removed_names:
                bucket.pop(key, None)
            if self._default_providers.get(kind) in removed_names:
                next_default = next(iter(bucket.keys()), None)
                if next_default is None:
                    self._default_providers.pop(kind, None)
                else:
                    self._default_providers[kind] = next_default

    def register_plugin(
        self,
        plugin: Any,
        *,
        name: str | None = None,
        source: str = "direct",
    ) -> PluginRecord:
        """Register a plugin instance/module across the three extension slots."""
        plugin_name = (
            (name or getattr(plugin, "name", None) or type(plugin).__name__).strip()
        )
        if plugin_name.lower() in self.disabled_plugins:
            record = PluginRecord(name=plugin_name, status="skipped", source=source)
            self._plugin_records[plugin_name] = record
            return record

        invoked_slots: list[str] = []
        prev_owner = self._active_plugin_name
        self._active_plugin_name = plugin_name
        try:
            for slot_name in ("register_cli", "register_hooks", "register_providers"):
                slot_fn = getattr(plugin, slot_name, None)
                if callable(slot_fn):
                    slot_fn(self)
                    invoked_slots.append(slot_name)
        except Exception as exc:  # noqa: BLE001 — fault isolation boundary
            self._rollback_plugin(plugin_name)
            logger.warning("Plugin %r failed during registration: %s", plugin_name, exc)
            record = PluginRecord(
                name=plugin_name,
                status="failed",
                source=source,
                slots=tuple(invoked_slots),
                error=f"{type(exc).__name__}: {exc}",
            )
            self._plugin_records[plugin_name] = record
            return record
        finally:
            self._active_plugin_name = prev_owner

        record = PluginRecord(
            name=plugin_name,
            status="loaded",
            source=source,
            slots=tuple(invoked_slots),
        )
        self._plugin_records[plugin_name] = record
        return record

    def load_plugins(
        self,
        *,
        entry_points_provider: Callable[[str], Iterable[Any]] | None = None,
    ) -> tuple[PluginRecord, ...]:
        """Discover and load plugins from entry_points(group='reiji_maigo.plugins')."""
        if self.plugins_disabled:
            return ()

        try:
            if entry_points_provider is not None:
                discovered = list(entry_points_provider(self.entry_point_group))
            else:
                discovered = list(
                    importlib_metadata.entry_points(group=self.entry_point_group)
                )
        except Exception as exc:  # noqa: BLE001 — fault isolation boundary
            logger.warning(
                "Failed to enumerate entry_points(%r): %s", self.entry_point_group, exc
            )
            return ()

        records: list[PluginRecord] = []
        for ep in discovered:
            ep_name = str(getattr(ep, "name", "") or "").strip() or repr(ep)
            ep_source = f"entry_point:{self.entry_point_group}:{ep_name}"

            if ep_name.lower() in self.disabled_plugins:
                rec = PluginRecord(name=ep_name, status="skipped", source=ep_source)
                self._plugin_records[ep_name] = rec
                records.append(rec)
                continue

            try:
                loaded = ep.load()
                plugin_obj = loaded() if callable(loaded) else loaded
            except Exception as exc:  # noqa: BLE001 — fault isolation boundary
                logger.warning("Entry point %r failed to load: %s", ep_name, exc)
                rec = PluginRecord(
                    name=ep_name,
                    status="failed",
                    source=ep_source,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._plugin_records[ep_name] = rec
                records.append(rec)
                continue

            rec = self.register_plugin(plugin_obj, name=ep_name, source=ep_source)
            records.append(rec)

        return tuple(records)

    @property
    def plugin_records(self) -> dict[str, PluginRecord]:
        return dict(self._plugin_records)

    @property
    def load_errors(self) -> tuple[PluginRecord, ...]:
        return tuple(
            rec for rec in self._plugin_records.values() if rec.status == "failed"
        )

    def status(self) -> dict[str, Any]:
        return {
            "home": str(self.home),
            "plugins_disabled": self.plugins_disabled,
            "disabled_plugins": sorted(self.disabled_plugins),
            "loaded_plugins": sorted(
                name for name, rec in self._plugin_records.items() if rec.status == "loaded"
            ),
            "skipped_plugins": sorted(
                name for name, rec in self._plugin_records.items() if rec.status == "skipped"
            ),
            "failed_plugins": {
                name: rec.error
                for name, rec in self._plugin_records.items()
                if rec.status == "failed"
            },
            "cli_commands": sorted(self._cli_commands.keys()),
            "hook_events": list(self.hooks.events()),
            "providers": {
                kind: sorted(bucket.keys()) for kind, bucket in self._providers.items()
            },
        }
