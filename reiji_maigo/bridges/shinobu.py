"""Zero-dependency duck-typed plugin bridge for ShinobuKernel.

Never imports ``shinobu_core`` at module top level so ``reiji_maigo`` stays
100% standalone when ``shinobu_core`` is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from reiji_maigo.core.contracts import LOCKED_DOMAINS, SourceSpan
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.security import ORIGIN_TRUST_LEVELS
from reiji_maigo.swarm.coordinator import SwarmCoordinator
from reiji_maigo.swarm.runner import SwarmModelRunner, SwarmRunnerError

_ROLE_ORIGIN_TRUST: dict[str, str] = {
    "user": "user_verified",
    "human": "user_verified",
    "assistant": "assistant",
    "system": "assistant",
    "tool": "untrusted_external",
    "function": "untrusted_external",
    "external": "untrusted_external",
}


def _resolve_origin_trust(role: str, explicit: Any = None) -> str:
    """Map host speaker role (or explicit override) to a locked origin_trust tier."""
    if isinstance(explicit, str) and explicit.strip():
        candidate = explicit.strip().lower()
        if candidate in ORIGIN_TRUST_LEVELS:
            return candidate
    return _ROLE_ORIGIN_TRUST.get(str(role or "").strip().lower(), "assistant")


@dataclass(slots=True)
class _DuckToolSpec:
    """Fallback ToolSpec-compatible carrier used when shinobu_core is absent."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def to_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


def _resolve_tool_spec_cls() -> type[Any]:
    """Lazily import shinobu_core.protocol.engine.ToolSpec if available."""
    try:
        from shinobu_core.protocol.engine import ToolSpec  # type: ignore[import-not-found]
    except ImportError:
        return _DuckToolSpec
    return ToolSpec


def _coerce_span(raw: Any) -> SourceSpan:
    if isinstance(raw, SourceSpan):
        return raw
    if isinstance(raw, dict):
        return SourceSpan.from_dict(raw)
    raise TypeError("span must be a SourceSpan or mapping with episode_id/turn_start/turn_end")


class ReijiShinobuPlugin:
    """Duck-typed ShinobuPlugin exposing Reiji-Maigo tools, CLI, and hooks."""

    name: str = "reiji_maigo"

    def __init__(
        self,
        *,
        home: Path | str | None = None,
        db_path: Path | str | None = None,
        engine: ReijiMemoryEngine | None = None,
        runner: SwarmModelRunner | None = None,
        coordinator: SwarmCoordinator | None = None,
        async_scribe: bool = False,
        debounce_sec: float = 2.0,
    ) -> None:
        self._home: Path | str | None = home
        self._db_path: Path | str | None = db_path
        self._engine: ReijiMemoryEngine | None = (
            engine if engine is not None else (coordinator.engine if coordinator is not None else None)
        )
        self._owns_engine: bool = engine is None and coordinator is None
        self._runner: SwarmModelRunner | None = runner
        self._coordinator: SwarmCoordinator | None = coordinator
        self._owns_coordinator: bool = coordinator is None
        self._async_scribe: bool = bool(async_scribe)
        self._debounce_sec: float = float(debounce_sec)

    def get_engine(self, kernel: Any = None) -> ReijiMemoryEngine:
        """Lazily initialize or return the bound ReijiMemoryEngine."""
        if self._engine is None:
            resolved_home = self._home
            if resolved_home is None and kernel is not None:
                kernel_home = getattr(kernel, "home", None)
                if kernel_home is not None:
                    resolved_home = Path(kernel_home) / "reiji_maigo"
            self._engine = ReijiMemoryEngine(
                home=resolved_home,
                db_path=self._db_path,
                load_plugins=False,
            )
            self._owns_engine = True
        return self._engine

    def get_coordinator(self, kernel: Any = None) -> SwarmCoordinator:
        """Lazily initialize or return the bound SwarmCoordinator."""
        if self._coordinator is None:
            engine = self.get_engine(kernel)
            self._coordinator = SwarmCoordinator(
                engine,
                runner=self._runner,
                async_scribe=self._async_scribe,
                debounce_sec=self._debounce_sec,
            )
            self._owns_coordinator = True
        return self._coordinator

    def close(self) -> None:
        """Close managed coordinator and engine if owned by this plugin."""
        if self._coordinator is not None and self._owns_coordinator:
            self._coordinator.close()
            self._coordinator = None
        if self._engine is not None and self._owns_engine:
            self._engine.close()
            self._engine = None

    # ------------------------------------------------------------------
    # Tool builders
    # ------------------------------------------------------------------

    def build_tools(self, kernel: Any = None) -> list[Any]:
        """Construct the four tail-end memory tools as ToolSpec-compatible objects."""
        spec_cls = _resolve_tool_spec_cls()
        engine = self.get_engine(kernel)

        def handle_pool_snapshot(
            *,
            include_undigested: bool = True,
            now_iso: str | None = None,
        ) -> str:
            snapshot = engine.pool_snapshot(
                now_iso=now_iso,
                include_undigested=bool(include_undigested),
            )
            return json.dumps({"ok": True, "snapshot": snapshot}, ensure_ascii=False)

        def handle_update_pool(
            *,
            action: str,
            event_id: str | None = None,
            title: str | None = None,
            step: str | None = None,
            span: dict[str, Any] | SourceSpan | None = None,
            scope: str = "quest",
            status: str | None = None,
            checkpoint_summary: str | None = None,
            keep_last: int = 2,
            domain: str | None = None,
            tags: list[str] | None = None,
            state: dict[str, Any] | None = None,
            card_id: str | None = None,
            now_iso: str | None = None,
        ) -> str:
            act = str(action).strip().lower()
            if act == "create":
                if not title or not step or span is None:
                    raise ValueError("action='create' requires title, step, and span")
                ev = engine.pool.create_event(
                    title,
                    step,
                    _coerce_span(span),
                    scope=scope,
                    now_iso=now_iso,
                )
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            if act == "append":
                if not event_id or not step or span is None:
                    raise ValueError("action='append' requires event_id, step, and span")
                ev = engine.pool.append_step(
                    event_id,
                    step,
                    _coerce_span(span),
                    status=status,
                    now_iso=now_iso,
                )
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            if act == "touch":
                if not event_id:
                    raise ValueError("action='touch' requires event_id")
                ev = engine.pool.touch_event(event_id, now_iso=now_iso)
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            if act == "status":
                if not event_id or not status:
                    raise ValueError("action='status' requires event_id and status")
                ev = engine.pool.update_status(event_id, status, now_iso=now_iso)
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            if act == "compact":
                if not event_id or not checkpoint_summary:
                    raise ValueError("action='compact' requires event_id and checkpoint_summary")
                ev = engine.pool.compact_chain(
                    event_id,
                    checkpoint_summary,
                    keep_last=int(keep_last),
                )
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            if act in ("archive", "hibernate"):
                if not event_id or not domain:
                    raise ValueError(f"action={act!r} requires event_id and domain")
                card = engine.archive_pool_event(
                    event_id,
                    domain=domain,
                    tags=tags or [],
                    state=dict(state or {}),
                    hibernate=(act == "hibernate"),
                    now_iso=now_iso,
                )
                return json.dumps({"ok": True, "action": act, "card": card.to_dict()}, ensure_ascii=False)

            if act == "wake":
                if not card_id or not step or span is None:
                    raise ValueError("action='wake' requires card_id, step, and span")
                ev = engine.wake_hibernated_card(
                    card_id,
                    step,
                    _coerce_span(span),
                    scope=scope if scope else None,
                    now_iso=now_iso,
                )
                return json.dumps({"ok": True, "action": act, "event": ev.to_dict()}, ensure_ascii=False)

            raise ValueError(
                f"Unsupported action {action!r}; expected create/append/touch/status/compact/archive/hibernate/wake"
            )

        def handle_recall_cards(
            *,
            query_tags: list[str] | None = None,
            domain: str | None = None,
            query: str | None = None,
            min_tag_hits: int = 1,
            top_k: int = 3,
            fallback_level3: bool = True,
            pad: int = 3,
        ) -> str:
            cards = (
                engine.recall_cards(
                    domain,
                    query_tags or [],
                    min_tag_hits=int(min_tag_hits),
                    top_k=int(top_k),
                )
                if query_tags
                else []
            )
            deep_hits: list[dict[str, Any]] = []
            if not cards and fallback_level3 and query:
                deep_hits = engine.deep_slice(query, limit=int(top_k), pad=int(pad))
            return json.dumps(
                {
                    "ok": True,
                    "cards": [c.to_dict() for c in cards],
                    "deep_hits": deep_hits,
                },
                ensure_ascii=False,
            )

        def handle_read_evidence(
            *,
            card_id: str,
            pad: int = 1,
        ) -> str:
            turns = engine.read_evidence(card_id, pad=int(pad))
            return json.dumps(
                {"ok": True, "card_id": card_id, "turns": turns},
                ensure_ascii=False,
            )

        def handle_scout_deep(
            *,
            query: str,
            domain: str | None = None,
            hint_tags: list[str] | None = None,
        ) -> str:
            coordinator = self.get_coordinator(kernel)
            try:
                briefing = coordinator.dispatch_scout(
                    query,
                    domain=domain,
                    hint_tags=hint_tags or (),
                )
            except SwarmRunnerError as exc:
                return json.dumps(
                    {
                        "ok": False,
                        "error": str(exc),
                        "found": False,
                        "abstained": True,
                        "tail_block": "",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "ok": True,
                    "query": briefing.query,
                    "found": briefing.found,
                    "abstained": briefing.abstained,
                    "briefing_text": briefing.briefing_text,
                    "tail_block": briefing.to_tail_block(),
                    "cited_card_ids": list(briefing.cited_card_ids),
                    "cited_spans": [sp.to_dict() for sp in briefing.cited_spans],
                    "rounds_used": briefing.rounds_used,
                    "raw_turns_scanned": briefing.raw_turns_scanned,
                    "distilled_tokens": briefing.distilled_tokens,
                    "compression_ratio": briefing.compression_ratio,
                },
                ensure_ascii=False,
            )

        def handle_swarm_pulse(*, format: str = "both") -> str:
            coordinator = self.get_coordinator(kernel)
            fmt = str(format).strip().lower()
            payload: dict[str, Any] = {"ok": True}
            if fmt in ("summary", "both"):
                payload["summary"] = coordinator.pulse_summary()
            if fmt in ("banner", "both"):
                payload["banner"] = coordinator.render_pulse_banner()
            if fmt not in ("summary", "banner", "both"):
                raise ValueError(f"Unsupported format {format!r}; expected summary/banner/both")
            return json.dumps(payload, ensure_ascii=False)

        def handle_security_audit(
            *,
            include_entries: bool = False,
            limit: int = 20,
        ) -> str:
            engine = self.get_engine()
            safe_limit = max(0, int(limit))
            summary = engine.security_summary()
            seals = engine.verify_card_seals()
            pending_reviews = engine.approval_gate.list_reviews(
                status="pending",
                limit=safe_limit,
            )
            payload: dict[str, Any] = {
                "ok": bool(summary["audit_chain"]["ok"] and seals["ok"]),
                "security": summary,
                "audit_chain": summary["audit_chain"],
                "card_seals": seals,
                "pending_reviews": pending_reviews,
            }
            if include_entries:
                payload["audit_entries"] = engine.audit.list_entries(limit=safe_limit)
            return json.dumps(payload, ensure_ascii=False)

        return [
            spec_cls(
                name="reiji_pool_snapshot",
                description="Render the current Reiji-Maigo ActivePool working-memory snapshot.",
                parameters={
                    "type": "object",
                    "properties": {
                        "include_undigested": {"type": "boolean", "default": True},
                        "now_iso": {"type": "string"},
                    },
                },
                handler=handle_pool_snapshot,
            ),
            spec_cls(
                name="reiji_update_pool",
                description="Create, append, compact, archive, hibernate, or wake Reiji-Maigo pool events.",
                parameters={
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [
                                "create",
                                "append",
                                "touch",
                                "status",
                                "compact",
                                "archive",
                                "hibernate",
                                "wake",
                            ],
                        },
                        "event_id": {"type": "string"},
                        "title": {"type": "string"},
                        "step": {"type": "string"},
                        "span": {"type": "object"},
                        "scope": {"type": "string", "enum": ["quest", "micro"]},
                        "status": {"type": "string"},
                        "checkpoint_summary": {"type": "string"},
                        "keep_last": {"type": "integer", "default": 2},
                        "domain": {"type": "string", "enum": list(LOCKED_DOMAINS)},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "state": {"type": "object"},
                        "card_id": {"type": "string"},
                        "now_iso": {"type": "string"},
                    },
                },
                handler=handle_update_pool,
            ),
            spec_cls(
                name="reiji_recall_cards",
                description="Recall long-term Reiji-Maigo memory cards (Level-1 PPR + optional Level-3 FTS fallback).",
                parameters={
                    "type": "object",
                    "properties": {
                        "domain": {"type": "string", "enum": list(LOCKED_DOMAINS)},
                        "query_tags": {"type": "array", "items": {"type": "string"}},
                        "query": {"type": "string"},
                        "min_tag_hits": {"type": "integer", "default": 1},
                        "top_k": {"type": "integer", "default": 3},
                        "fallback_level3": {"type": "boolean", "default": True},
                        "pad": {"type": "integer", "default": 3},
                    },
                },
                handler=handle_recall_cards,
            ),
            spec_cls(
                name="reiji_read_evidence",
                description="Read Level-2 verbatim raw conversation evidence bound to a Reiji-Maigo memory card.",
                parameters={
                    "type": "object",
                    "required": ["card_id"],
                    "properties": {
                        "card_id": {"type": "string"},
                        "pad": {"type": "integer", "default": 1},
                    },
                },
                handler=handle_read_evidence,
            ),
            spec_cls(
                name="reiji_scout_deep",
                description="Dispatch read-only ScoutAgent for 3-layer deep memory reconnaissance and compact cited briefing.",
                parameters={
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "domain": {"type": "string", "enum": list(LOCKED_DOMAINS)},
                        "hint_tags": {"type": "array", "items": {"type": "string"}},
                    },
                },
                handler=handle_scout_deep,
            ),
            spec_cls(
                name="reiji_swarm_pulse",
                description="Return Reiji-Maigo memory swarm pulse telemetry summary and/or industrial banner.",
                parameters={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["summary", "banner", "both"],
                            "default": "both",
                        },
                    },
                },
                handler=handle_swarm_pulse,
            ),
            spec_cls(
                name="reiji_security_audit",
                description="Read-only security audit returning hash-chain integrity, card HMAC seals, and pending L1 reviews.",
                parameters={
                    "type": "object",
                    "properties": {
                        "include_entries": {"type": "boolean", "default": False},
                        "limit": {"type": "integer", "default": 20},
                    },
                },
                handler=handle_security_audit,
            ),
        ]

    # ------------------------------------------------------------------
    # ShinobuPlugin duck-typed hooks
    # ------------------------------------------------------------------

    def register_tools(self, kernel: Any) -> list[Any]:
        """Register the four Reiji-Maigo tools onto a host kernel."""
        tools = self.build_tools(kernel)
        if hasattr(kernel, "register_tools") and callable(kernel.register_tools):
            kernel.register_tools(tools)
        elif hasattr(kernel, "register_tool") and callable(kernel.register_tool):
            for tool in tools:
                kernel.register_tool(tool)
        return tools

    def register_cli(self, subparsers: Any) -> Any:
        """Register a `reiji` subcommand on a host argparse subparsers group."""
        if subparsers is None or not hasattr(subparsers, "add_parser"):
            return None
        parser = subparsers.add_parser(
            "reiji",
            help="Inspect Reiji-Maigo two-tier memory engine status or snapshot.",
        )
        parser.add_argument(
            "action",
            nargs="?",
            choices=("status", "pool", "swarm"),
            default="status",
            help="Sub-action to execute (default: status).",
        )
        parser.add_argument("--json", action="store_true", help="Emit JSON output.")

        def _dispatch(args: Any) -> int:
            engine = self.get_engine()
            action = getattr(args, "action", "status")
            if action == "pool":
                snap = engine.pool_snapshot()
                if getattr(args, "json", False):
                    print(json.dumps({"snapshot": snap}, ensure_ascii=False))
                else:
                    print(snap)
            elif action == "swarm":
                coordinator = self.get_coordinator()
                if getattr(args, "json", False):
                    print(json.dumps(coordinator.pulse_summary(), ensure_ascii=False, indent=2))
                else:
                    print(coordinator.render_pulse_banner())
            else:
                info = engine.status()
                if getattr(args, "json", False):
                    print(json.dumps(info, ensure_ascii=False, indent=2))
                else:
                    print(f"Reiji-Maigo Home: {info['home']}")
            return 0

        parser.set_defaults(func=_dispatch)
        return parser

    def register_hooks(self, bus: Any) -> list[Any]:
        """Attach turn-recording, snapshot, Scribe, and Archivist hooks onto a host HookBus."""
        engine = self.get_engine()
        coordinator = self.get_coordinator()

        def _extract_data(payload: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if isinstance(payload, dict):
                merged = dict(payload)
                merged.update(kwargs)
                return merged
            return dict(kwargs)

        def _on_turn_recorded(payload: Any = None, **kwargs: Any) -> int | None:
            data = _extract_data(payload, kwargs)
            episode_id = data.get("episode_id")
            role = data.get("role")
            content = data.get("content")
            if isinstance(episode_id, str) and isinstance(role, str) and isinstance(content, str):
                trust = _resolve_origin_trust(role, data.get("origin_trust"))
                return coordinator.on_turn_completed(
                    episode_id,
                    role,
                    content,
                    record_turn=True,
                    origin_trust=trust,
                )
            return None

        def _on_session_start(*_args: Any, **_kwargs: Any) -> str:
            return engine.pool_snapshot()

        def _on_done(payload: Any = None, **kwargs: Any) -> dict[str, Any]:
            data = _extract_data(payload, kwargs)
            episode_id = str(
                data.get("episode_id")
                or data.get("session_id")
                or "default_episode"
            )
            record_turn = bool(data.get("record_turn", True))
            recorded_ids: list[int] = []

            raw_turns = data.get("turns")
            if isinstance(raw_turns, list):
                for item in raw_turns:
                    if isinstance(item, dict):
                        r = item.get("role")
                        c = item.get("content")
                        if isinstance(r, str) and isinstance(c, str) and c.strip():
                            trust = _resolve_origin_trust(
                                r,
                                item.get("origin_trust", data.get("origin_trust")),
                            )
                            recorded_ids.append(
                                coordinator.on_turn_completed(
                                    episode_id,
                                    r,
                                    c,
                                    record_turn=record_turn,
                                    origin_trust=trust,
                                )
                            )
            else:
                user_msg = data.get("user_message") or data.get("prompt")
                asst_msg = (
                    data.get("assistant_message")
                    or data.get("response")
                    or data.get("reply")
                )
                if isinstance(user_msg, str) and user_msg.strip():
                    recorded_ids.append(
                        coordinator.on_turn_completed(
                            episode_id,
                            "user",
                            user_msg,
                            record_turn=record_turn,
                            origin_trust=_resolve_origin_trust("user", data.get("user_origin_trust")),
                        )
                    )
                if isinstance(asst_msg, str) and asst_msg.strip():
                    recorded_ids.append(
                        coordinator.on_turn_completed(
                            episode_id,
                            "assistant",
                            asst_msg,
                            record_turn=record_turn,
                            origin_trust=_resolve_origin_trust("assistant", data.get("assistant_origin_trust")),
                        )
                    )
                if not recorded_ids:
                    role = data.get("role")
                    content = data.get("content") or data.get("text")
                    if isinstance(content, str) and content.strip():
                        resolved_role = role if isinstance(role, str) and role.strip() else "assistant"
                        recorded_ids.append(
                            coordinator.on_turn_completed(
                                episode_id,
                                resolved_role,
                                content,
                                record_turn=record_turn,
                                origin_trust=_resolve_origin_trust(resolved_role, data.get("origin_trust")),
                            )
                        )

            scribe_report: Any = None
            auto_digest = bool(data.get("auto_digest", True))
            if (
                auto_digest
                and coordinator.is_configured
                and coordinator.async_worker is None
            ):
                try:
                    scribe_report = coordinator.run_scribe_now(now_iso=data.get("now_iso"))
                except SwarmRunnerError:
                    scribe_report = None

            return {
                "turn_ids": recorded_ids,
                "scribe_report": scribe_report,
            }

        def _on_episode_consolidated(payload: Any = None, **kwargs: Any) -> Any:
            data = _extract_data(payload, kwargs)
            if not coordinator.is_configured:
                return None
            try:
                return coordinator.run_midnight_archivist(now_iso=data.get("now_iso"))
            except SwarmRunnerError:
                return None

        hook_bindings = (
            ("reiji.turn.recorded", _on_turn_recorded),
            ("reiji.session.start", _on_session_start),
            ("done", _on_done),
            ("episode_consolidated", _on_episode_consolidated),
        )

        registrations: list[Any] = []
        if hasattr(bus, "on") and callable(bus.on):
            for event_name, fn in hook_bindings:
                registrations.append(bus.on(event_name, fn))
        elif hasattr(bus, "register") and callable(bus.register):
            for event_name, fn in hook_bindings:
                bus.register(event_name, fn, plugin_name=self.name)
        return registrations


def create_plugin(
    home: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
    engine: ReijiMemoryEngine | None = None,
    runner: SwarmModelRunner | None = None,
    coordinator: SwarmCoordinator | None = None,
    async_scribe: bool = False,
    debounce_sec: float = 2.0,
) -> ReijiShinobuPlugin:
    """Factory creating a standalone, duck-typed Reiji-Maigo plugin for ShinobuKernel."""
    return ReijiShinobuPlugin(
        home=home,
        db_path=db_path,
        engine=engine,
        runner=runner,
        coordinator=coordinator,
        async_scribe=async_scribe,
        debounce_sec=debounce_sec,
    )


__all__ = [
    "ReijiShinobuPlugin",
    "create_plugin",
]
