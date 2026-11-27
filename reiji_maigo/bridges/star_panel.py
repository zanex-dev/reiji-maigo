"""Zero top-level UI coupling bridge mounting Reiji-Maigo onto a desktop star panel.

Uses pure duck-typing against any host exposing optional UI registration hooks:
  - ``add_tab(tab_id, title, build_tab, *, on_select=..., on_refresh=...)``
  - ``format_header_badge(fn)``
  - ``add_radar_markers(fn)``
  - ``on_radar_state_changed(fn)``

``tkinter`` is strictly imported lazily inside ``build_tab`` so headless CLI,
server, and unit-test environments never incur a GUI import overhead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.plugins.epoch_vault import (
    EpochColdVault,
    resolve_epoch_vault_dir,
)
from reiji_maigo.plugins.roi_ledger import RoiLedger

logger = logging.getLogger(__name__)

PLUGIN_NAME: str = "reiji_maigo"
PLUGIN_DESCRIPTION: str = "Reiji-Maigo temporal memory & L1 audit deck"

TAB_ID: str = "reiji"
TAB_TITLE: str = "🕰️ 零时迷子"

RADAR_COOLDOWN_SEC: float = 60.0
RADAR_TTL_SEC: float = 250.0
RADAR_COOLDOWN_FRACTION: float = RADAR_COOLDOWN_SEC / RADAR_TTL_SEC

# Black / Amber-Gold / Crimson palette for the temporal memory deck
THEME: dict[str, str] = {
    "bg_root": "#141210",
    "bg_card": "#1e1b18",
    "bg_panel": "#26211d",
    "fg_primary": "#f5f5f4",
    "fg_muted": "#a8a29e",
    "accent_gold": "#f59e0b",
    "accent_crimson": "#dc2626",
    "accent_emerald": "#22c55e",
    "border": "#3f3730",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_exists(engine: ReijiMemoryEngine, table_name: str) -> bool:
    with engine.db.read_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
    return row is not None


def _collect_pool_snapshot(engine: ReijiMemoryEngine) -> dict[str, Any]:
    """Collect active pool snapshot without assuming a single rigid helper signature."""
    max_events = int(getattr(engine.pool, "max_events", 8))
    max_tokens = int(getattr(engine.pool, "max_tokens", 900))
    events_summary: list[dict[str, Any]] = []
    active_count = 0
    total_count = 0

    list_events_fn = getattr(engine.pool, "list_events", None)
    if callable(list_events_fn):
        try:
            active_events = list(list_events_fn(include_inactive=False))
            all_events = list(list_events_fn(include_inactive=True))
            for ev in active_events:
                events_summary.append(
                    {
                        "event_id": str(getattr(ev, "event_id", "")),
                        "title": str(getattr(ev, "title", "")),
                        "scope": str(getattr(ev, "scope", "")),
                        "status": str(getattr(ev, "status", "active")),
                        "hit_count": int(getattr(ev, "hit_count", 0)),
                        "checkpoint_summary": str(
                            getattr(ev, "checkpoint_summary", "")
                        ),
                    }
                )
            return {
                "active_count": len(active_events),
                "total_count": len(all_events),
                "max_events": max_events,
                "max_tokens": max_tokens,
                "events": events_summary,
            }
        except Exception:  # noqa: BLE001 — fallback to generic probe below
            pass

    for method_name in ("list_events", "list_active_events", "get_active_events"):
        fn = getattr(engine.pool, method_name, None)
        if not callable(fn):
            continue
        try:
            raw_events = list(fn())
        except TypeError:
            try:
                raw_events = list(fn(status=None))
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue

        total_count = len(raw_events)
        for ev in raw_events:
            status = str(getattr(ev, "status", "active"))
            if status == "active":
                active_count += 1
            events_summary.append(
                {
                    "event_id": str(getattr(ev, "event_id", "")),
                    "title": str(getattr(ev, "title", "")),
                    "scope": str(getattr(ev, "scope", "")),
                    "status": status,
                    "hit_count": int(getattr(ev, "hit_count", 0)),
                    "checkpoint_summary": str(getattr(ev, "checkpoint_summary", "")),
                }
            )
        return {
            "active_count": active_count,
            "total_count": total_count,
            "max_events": max_events,
            "max_tokens": max_tokens,
            "events": events_summary,
        }

    # SQLite fallback when pool table is queried directly
    with engine.db.read_connection() as conn:
        table_row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%pool%' LIMIT 1"
        ).fetchone()
        if table_row is not None:
            tbl = str(table_row["name"])
            rows = conn.execute(
                f"SELECT * FROM {tbl} ORDER BY updated_at DESC LIMIT 32"  # noqa: S608
            ).fetchall()
            cols = set(rows[0].keys()) if rows else set()
            total_count = len(rows)
            for row in rows:
                status = str(row["status"]) if "status" in cols else "active"
                if status == "active":
                    active_count += 1
                events_summary.append(
                    {
                        "event_id": str(row["event_id"]) if "event_id" in cols else "",
                        "title": str(row["title"]) if "title" in cols else "",
                        "scope": str(row["scope"]) if "scope" in cols else "",
                        "status": status,
                        "hit_count": int(row["hit_count"]) if "hit_count" in cols else 0,
                        "checkpoint_summary": (
                            str(row["checkpoint_summary"])
                            if "checkpoint_summary" in cols
                            else ""
                        ),
                    }
                )

    return {
        "active_count": active_count,
        "total_count": total_count,
        "max_events": max_events,
        "max_tokens": max_tokens,
        "events": events_summary,
    }


def _collect_cards_snapshot(engine: ReijiMemoryEngine) -> dict[str, Any]:
    counts: dict[str, int] = {
        "active": 0,
        "superseded": 0,
        "revoked": 0,
        "total": 0,
        "cold_stubs": 0,
    }
    if not _table_exists(engine, "memory_cards"):
        return counts

    with engine.db.read_connection() as conn:
        rows = conn.execute(
            "SELECT lifecycle_status, COUNT(*) AS cnt FROM memory_cards GROUP BY lifecycle_status"
        ).fetchall()
        for row in rows:
            status = str(row["lifecycle_status"])
            cnt = int(row["cnt"])
            counts["total"] += cnt
            if status in counts:
                counts[status] = cnt

        stub_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM memory_cards
            WHERE state_json LIKE '%"_cold_stub": true%'
               OR state_json LIKE '%"_cold_stub":true%'
            """
        ).fetchone()
        if stub_row is not None:
            counts["cold_stubs"] = int(stub_row["cnt"])

    return counts


def _collect_swarm_snapshot(engine: ReijiMemoryEngine) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"calls_count": 0, "by_role": {}}
    if not _table_exists(engine, "swarm_telemetry"):
        return snapshot

    with engine.db.read_connection() as conn:
        rows = conn.execute(
            "SELECT role, COUNT(*) AS cnt FROM swarm_telemetry GROUP BY role"
        ).fetchall()
        by_role: dict[str, int] = {}
        total = 0
        for row in rows:
            role = str(row["role"])
            cnt = int(row["cnt"])
            by_role[role] = cnt
            total += cnt
        snapshot["calls_count"] = total
        snapshot["by_role"] = by_role
    return snapshot


def _collect_roi_snapshot(engine: ReijiMemoryEngine) -> dict[str, Any]:
    ledger = RoiLedger(engine.db)
    if _table_exists(engine, "swarm_telemetry"):
        try:
            ledger.sync_from_swarm_telemetry()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ROI telemetry sync skipped: %s", exc)
    return ledger.summary()


def _collect_vault_snapshot(
    engine: ReijiMemoryEngine,
    *,
    cold_stubs_count: int,
) -> dict[str, Any]:
    vault_dir = resolve_epoch_vault_dir(engine)
    index_db = vault_dir / "epoch_index.db"
    if not index_db.exists():
        return {
            "vault_dir": str(vault_dir),
            "exists": False,
            "epochs_count": 0,
            "archived_cards_count": 0,
            "archived_turns_count": 0,
            "cold_stubs_count": cold_stubs_count,
        }
    try:
        vault = EpochColdVault(vault_dir)
        summary = dict(vault.vault_summary(engine))
        summary.setdefault("vault_dir", str(vault_dir))
        summary.setdefault("exists", True)
        summary.setdefault("cold_stubs_count", cold_stubs_count)
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to inspect epoch cold vault: %s", exc)
        return {
            "vault_dir": str(vault_dir),
            "exists": True,
            "epochs_count": 0,
            "archived_cards_count": 0,
            "archived_turns_count": 0,
            "cold_stubs_count": cold_stubs_count,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _normalize_review_result(result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(result)
    rc = out.get("result_card")
    if rc is not None and callable(getattr(rc, "to_dict", None)):
        out["result_card_id"] = getattr(rc, "card_id", None)
        out["result_card"] = rc.to_dict()
    return out


class StarPanelBridgePlugin:
    """Duck-typed desktop panel bridge for Reiji-Maigo temporal memory & L1 audit."""

    name: str = PLUGIN_NAME
    description: str = PLUGIN_DESCRIPTION

    def __init__(
        self,
        *,
        home: Path | str | None = None,
        db_path: Path | str | None = None,
        engine: ReijiMemoryEngine | None = None,
        load_plugins: bool = False,
    ) -> None:
        self._home = home
        self._db_path = db_path
        self._load_plugins = bool(load_plugins)
        self._engine: ReijiMemoryEngine | None = engine
        self._owns_engine: bool = engine is None
        self._last_radar_state: dict[str, Any] = {}
        self._ui_refresh_hooks: list[Callable[[], None]] = []

    def _get_engine(self) -> ReijiMemoryEngine:
        if self._engine is None:
            self._engine = ReijiMemoryEngine(
                home=self._home,
                db_path=self._db_path,
                load_plugins=self._load_plugins,
            )
            self._owns_engine = True
        return self._engine

    @property
    def engine(self) -> ReijiMemoryEngine:
        return self._get_engine()

    def close(self) -> None:
        """Close the owned ReijiMemoryEngine instance if created by this bridge."""
        if self._engine is not None and self._owns_engine:
            self._engine.close()
            self._engine = None

    def _trigger_ui_refresh(self) -> None:
        alive: list[Callable[[], None]] = []
        for cb in list(self._ui_refresh_hooks):
            try:
                cb()
                alive.append(cb)
            except Exception:  # noqa: BLE001 — widget may have been destroyed
                continue
        self._ui_refresh_hooks = alive

    # ------------------------------------------------------------------
    # State aggregation & L1 audit controls
    # ------------------------------------------------------------------

    def collect_deck_state(self) -> dict[str, Any]:
        """Collect pool, cards, swarm, 7-decimal ROI, cold vault, L1 queue & radar state."""
        engine = self._get_engine()
        pool_info = _collect_pool_snapshot(engine)
        cards_info = _collect_cards_snapshot(engine)
        swarm_info = _collect_swarm_snapshot(engine)
        roi_info = _collect_roi_snapshot(engine)
        vault_info = _collect_vault_snapshot(
            engine,
            cold_stubs_count=int(cards_info.get("cold_stubs", 0)),
        )
        pending_reviews = engine.approval_gate.list_reviews(
            status="pending",
            limit=100,
        )
        audit_info = engine.audit.verify_chain()

        return {
            "home": str(engine.home),
            "collected_at": _utc_now_iso(),
            "pool": pool_info,
            "cards": cards_info,
            "swarm": swarm_info,
            "roi": roi_info,
            "vault": vault_info,
            "l1_reviews": {
                "pending_count": len(pending_reviews),
                "pending": pending_reviews,
            },
            "audit": {
                "ok": bool(audit_info.get("ok", False)),
                "checked": int(
                    audit_info.get("checked", audit_info.get("count", 0))
                ),
            },
            "radar": {
                "cooldown_sec": RADAR_COOLDOWN_SEC,
                "ttl_sec": RADAR_TTL_SEC,
                "cooldown_fraction": RADAR_COOLDOWN_FRACTION,
                "last_state": dict(self._last_radar_state),
            },
        }

    def approve_l1(
        self,
        review_id: str,
        approver: str = "panel_admin",
    ) -> dict[str, Any]:
        """Approve and atomically apply a pending L1 mutation review."""
        engine = self._get_engine()
        raw = engine.approval_gate.approve_review(
            review_id,
            engine.cards,
            approver=approver,
        )
        result = _normalize_review_result(raw)
        engine.kernel.hooks.emit(
            "star_panel.l1_approved",
            {
                "review_id": result.get("review_id"),
                "target_card_id": result.get("target_card_id"),
                "approved_by": result.get("approved_by"),
            },
        )
        self._trigger_ui_refresh()
        return result

    def reject_l1(
        self,
        review_id: str,
        reason: str = "",
        reviewer: str = "panel_admin",
    ) -> dict[str, Any]:
        """Reject a pending L1 mutation review without mutating the protected card."""
        engine = self._get_engine()
        raw = engine.approval_gate.reject_review(
            review_id,
            reason=reason,
            reviewer=reviewer,
        )
        result = _normalize_review_result(raw)
        engine.kernel.hooks.emit(
            "star_panel.l1_rejected",
            {
                "review_id": result.get("review_id"),
                "target_card_id": result.get("target_card_id"),
                "rejected_by": result.get("rejected_by"),
                "reason": result.get("reason", ""),
            },
        )
        self._trigger_ui_refresh()
        return result

    # ------------------------------------------------------------------
    # CLI & Kernel Hook integration
    # ------------------------------------------------------------------

    def register_cli(self, subparsers: Any) -> None:
        """Register CLI commands against either an argparse SubParsersAction or ReijiKernel."""
        if callable(getattr(subparsers, "register_cli_command", None)):
            # Loaded directly into ReijiKernel
            def _kernel_handler(args: Any, *_rest: Any) -> int:
                code = self.handle_cli(args)
                return 0 if code is None else int(code)

            subparsers.register_cli_command(
                "panel",
                _kernel_handler,
                help="Inspect Reiji-Maigo star panel deck state and manage L1 approvals",
            )
            return

        add_parser = getattr(subparsers, "add_parser", None)
        if not callable(add_parser):
            return

        parser = add_parser(
            "reiji",
            help="Reiji-Maigo temporal memory deck & L1 approval commands",
        )
        parser.set_defaults(_reiji_plugin=self, reiji_action="status")
        nested = parser.add_subparsers(dest="reiji_action")

        p_status = nested.add_parser("status", help="Show deck state snapshot")
        p_status.add_argument("--json", action="store_true", dest="as_json")
        p_status.set_defaults(_reiji_plugin=self, reiji_action="status")

        p_list = nested.add_parser("l1-list", help="List pending L1 reviews")
        p_list.add_argument("--json", action="store_true", dest="as_json")
        p_list.set_defaults(_reiji_plugin=self, reiji_action="l1-list")

        p_approve = nested.add_parser("l1-approve", help="Approve a pending L1 review")
        p_approve.add_argument("review_id")
        p_approve.add_argument("--approver", default="panel_admin")
        p_approve.add_argument("--json", action="store_true", dest="as_json")
        p_approve.set_defaults(_reiji_plugin=self, reiji_action="l1-approve")

        p_reject = nested.add_parser("l1-reject", help="Reject a pending L1 review")
        p_reject.add_argument("review_id")
        p_reject.add_argument("--reason", default="")
        p_reject.add_argument("--reviewer", default="panel_admin")
        p_reject.add_argument("--json", action="store_true", dest="as_json")
        p_reject.set_defaults(_reiji_plugin=self, reiji_action="l1-reject")

    def handle_cli(self, args: Any) -> int | None:
        """Handle parsed CLI args belonging to this plugin; return None if unclaimed."""
        owner = getattr(args, "_reiji_plugin", None)
        cmd = getattr(args, "command", None)
        action = getattr(args, "reiji_action", None)
        if owner is not self and cmd not in ("reiji", "panel") and action is None:
            return None

        resolved_action = str(action or "status")
        as_json = bool(getattr(args, "as_json", False))

        try:
            if resolved_action == "status":
                payload = self.collect_deck_state()
            elif resolved_action == "l1-list":
                payload = self.collect_deck_state()["l1_reviews"]
            elif resolved_action == "l1-approve":
                payload = self.approve_l1(
                    str(getattr(args, "review_id", "")),
                    approver=str(getattr(args, "approver", "panel_admin")),
                )
            elif resolved_action == "l1-reject":
                payload = self.reject_l1(
                    str(getattr(args, "review_id", "")),
                    reason=str(getattr(args, "reason", "")),
                    reviewer=str(getattr(args, "reviewer", "panel_admin")),
                )
            else:
                print(f"Unknown reiji action: {resolved_action}")
                return 2
        except Exception as exc:  # noqa: BLE001
            err_payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if as_json:
                print(json.dumps(err_payload, ensure_ascii=False, indent=2))
            else:
                print(f"Error: {err_payload['error']}")
            return 1

        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            if resolved_action == "status":
                badge = self._format_badge_from_state(payload)
                print(badge["text"])
            else:
                print(json.dumps(payload, ensure_ascii=False))
        return 0

    def register_hooks(self, kernel: Any) -> None:
        """Subscribe to thermal handoff and pool archival events when supported."""
        bind_fn: Callable[..., Any] | None = None
        if callable(getattr(kernel, "register_hook", None)):
            bind_fn = kernel.register_hook
        else:
            hooks_obj = getattr(kernel, "hooks", None)
            if hooks_obj is not None:
                for name in ("register", "on"):
                    candidate = getattr(hooks_obj, name, None)
                    if callable(candidate):
                        bind_fn = candidate
                        break
            if bind_fn is None:
                for name in ("register", "on"):
                    candidate = getattr(kernel, name, None)
                    if callable(candidate):
                        bind_fn = candidate
                        break
        if bind_fn is None:
            return

        def _on_thermal(payload: Any, **_kwargs: Any) -> None:
            if isinstance(payload, Mapping):
                self._last_radar_state = dict(payload)
            self._trigger_ui_refresh()

        def _on_pool_changed(_payload: Any, **_kwargs: Any) -> None:
            self._trigger_ui_refresh()

        bind_fn("thermal.handoff_ready", _on_thermal)
        bind_fn("pool.archived", _on_pool_changed)

    # ------------------------------------------------------------------
    # Duck-typed desktop UI host registration
    # ------------------------------------------------------------------

    @staticmethod
    def _format_badge_from_state(state: Mapping[str, Any]) -> dict[str, str]:
        pool_count = int(state.get("pool", {}).get("active_count", 0))
        l1_count = int(state.get("l1_reviews", {}).get("pending_count", 0))
        roi_net = str(state.get("roi", {}).get("net_roi_usd", "0.0000000"))
        return {
            "text": f"🕰️ Pool {pool_count} · L1:{l1_count} · ROI ${roi_net}",
            "fg": THEME["accent_gold"],
            "bg": THEME["bg_card"],
        }

    def _header_badge_callback(self, *_args: Any, **_kwargs: Any) -> dict[str, str]:
        state = self.collect_deck_state()
        return self._format_badge_from_state(state)

    @staticmethod
    def _radar_markers_callback(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "fraction": RADAR_COOLDOWN_FRACTION,
                "color": THEME["accent_gold"],
                "width": 1.5,
                "dash": (3, 2),
                "tag": "reiji_cooldown",
            }
        ]

    def _on_radar_state_changed(self, state: Any) -> None:
        if isinstance(state, Mapping):
            self._last_radar_state = dict(state)
        else:
            self._last_radar_state = {"raw": repr(state)}
        self._trigger_ui_refresh()

    def _build_tk_tab(self, parent: Any) -> Any:
        """Lazily import tkinter and construct the Black/Gold/Crimson memory deck."""
        import tkinter as tk  # noqa: PLC0415 — lazy UI import

        root_frame = tk.Frame(parent, bg=THEME["bg_root"], padx=10, pady=10)

        # Header row
        header = tk.Frame(root_frame, bg=THEME["bg_root"])
        header.pack(fill="x", pady=(0, 8))

        tk.Label(
            header,
            text="🕰️ 零时迷子 · Temporal Memory & L1 Audit Deck",
            fg=THEME["accent_gold"],
            bg=THEME["bg_root"],
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        status_var = tk.StringVar(value="就绪")

        # KPI cards row
        kpi_row = tk.Frame(root_frame, bg=THEME["bg_root"])
        kpi_row.pack(fill="x", pady=(0, 8))

        pool_var = tk.StringVar(value="Pool: 0/8")
        roi_var = tk.StringVar(value="ROI: $0.0000000 | Flash: $0.0000000")
        vault_var = tk.StringVar(value="Vault: 0 epochs / 0 cards")
        radar_var = tk.StringVar(value="Radar: 60s/250s cooldown | idle")

        def _make_kpi(title: str, var: Any, accent: str) -> Any:
            card = tk.Frame(
                kpi_row,
                bg=THEME["bg_card"],
                highlightbackground=THEME["border"],
                highlightthickness=1,
                padx=8,
                pady=6,
            )
            card.pack(side="left", fill="both", expand=True, padx=3)
            tk.Label(
                card,
                text=title,
                fg=accent,
                bg=THEME["bg_card"],
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).pack(fill="x")
            tk.Label(
                card,
                textvariable=var,
                fg=THEME["fg_primary"],
                bg=THEME["bg_card"],
                font=("Consolas", 9),
                anchor="w",
                justify="left",
            ).pack(fill="x", pady=(2, 0))
            return card

        _make_kpi("活跃记忆池", pool_var, THEME["accent_gold"])
        _make_kpi("蜂群 & 7位小数 ROI", roi_var, THEME["accent_emerald"])
        _make_kpi("纪元冷库", vault_var, THEME["accent_gold"])
        _make_kpi("250s 缓存余温", radar_var, THEME["accent_crimson"])

        # Active pool snapshot box
        pool_box = tk.Frame(
            root_frame,
            bg=THEME["bg_card"],
            highlightbackground=THEME["border"],
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        pool_box.pack(fill="x", pady=(0, 8))
        tk.Label(
            pool_box,
            text="活跃事件快照",
            fg=THEME["accent_gold"],
            bg=THEME["bg_card"],
            font=("Segoe UI", 9, "bold"),
            anchor="w",
        ).pack(fill="x")
        pool_listbox = tk.Listbox(
            pool_box,
            height=4,
            bg=THEME["bg_panel"],
            fg=THEME["fg_primary"],
            selectbackground=THEME["accent_gold"],
            selectforeground=THEME["bg_root"],
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9),
        )
        pool_listbox.pack(fill="x", pady=(4, 0))

        # L1 pending reviews audit panel
        l1_box = tk.Frame(
            root_frame,
            bg=THEME["bg_card"],
            highlightbackground=THEME["accent_crimson"],
            highlightthickness=1,
            padx=8,
            pady=6,
        )
        l1_box.pack(fill="both", expand=True)

        l1_header = tk.Frame(l1_box, bg=THEME["bg_card"])
        l1_header.pack(fill="x")
        l1_title_var = tk.StringVar(value="L1 受保护记忆待审队列 (0)")
        tk.Label(
            l1_header,
            textvariable=l1_title_var,
            fg=THEME["accent_crimson"],
            bg=THEME["bg_card"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")

        l1_listbox = tk.Listbox(
            l1_box,
            height=6,
            bg=THEME["bg_panel"],
            fg=THEME["fg_primary"],
            selectbackground=THEME["accent_crimson"],
            selectforeground=THEME["fg_primary"],
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9),
        )
        l1_listbox.pack(fill="both", expand=True, pady=(4, 6))

        controls = tk.Frame(l1_box, bg=THEME["bg_card"])
        controls.pack(fill="x")

        tk.Label(
            controls,
            text="驳回原因:",
            fg=THEME["fg_muted"],
            bg=THEME["bg_card"],
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(0, 4))

        reason_entry = tk.Entry(
            controls,
            bg=THEME["bg_panel"],
            fg=THEME["fg_primary"],
            insertbackground=THEME["accent_gold"],
            relief="flat",
            width=26,
        )
        reason_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        pending_ids: list[str] = []

        def _selected_review_id() -> str | None:
            sel = l1_listbox.curselection()
            if not sel:
                return None
            idx = int(sel[0])
            if 0 <= idx < len(pending_ids):
                return pending_ids[idx]
            return None

        def _refresh() -> None:
            if not root_frame.winfo_exists():
                raise RuntimeError("tab widget destroyed")
            state = self.collect_deck_state()

            pool = state["pool"]
            pool_var.set(
                f"Active: {pool['active_count']}/{pool['max_events']} (Total: {pool['total_count']})"
            )

            roi = state["roi"]
            swarm = state["swarm"]
            alert_tag = " [TAX ALERT]" if roi.get("tax_alert") else ""
            roi_var.set(
                f"Net: ${roi.get('net_roi_usd', '0.0000000')} | "
                f"Flash: ${roi.get('flash_cost_usd', '0.0000000')} | "
                f"Swarm: {swarm.get('calls_count', 0)}{alert_tag}"
            )

            vault = state["vault"]
            vault_var.set(
                f"Epochs: {vault.get('epochs_count', 0)} | "
                f"Archived: {vault.get('archived_cards_count', 0)} | "
                f"Stubs: {state['cards'].get('cold_stubs', 0)}"
            )

            radar_last = state["radar"].get("last_state", {})
            phase = str(radar_last.get("phase", "idle"))
            ttl = radar_last.get("remaining_ttl_sec", RADAR_TTL_SEC)
            radar_var.set(f"Phase: {phase} | TTL: {ttl}s (cooldown 60s)")

            pool_listbox.delete(0, tk.END)
            for ev in pool.get("events", []):
                pool_listbox.insert(
                    tk.END,
                    f"[{ev.get('status', '')}] {ev.get('event_id', '')} · {ev.get('title', '')}",
                )

            pending_items = state["l1_reviews"].get("pending", [])
            pending_ids.clear()
            l1_listbox.delete(0, tk.END)
            l1_title_var.set(f"L1 受保护记忆待审队列 ({len(pending_items)})")
            for item in pending_items:
                rid = str(item.get("review_id", ""))
                pending_ids.append(rid)
                l1_listbox.insert(
                    tk.END,
                    f"{rid} | {item.get('op_kind', '')} -> {item.get('target_card_id', '')} "
                    f"by {item.get('actor', '')} ({item.get('reason', '')})",
                )

        def _approve_selected() -> dict[str, Any] | None:
            rid = _selected_review_id()
            if not rid:
                status_var.set("请先选择一条待审 L1 记录")
                return None
            try:
                res = self.approve_l1(rid, approver="panel_admin")
                status_var.set(f"已批准 {rid} (audit #{res.get('audit_seq')})")
                return res
            except Exception as exc:  # noqa: BLE001
                status_var.set(f"批准失败: {exc}")
                return None

        def _reject_selected() -> dict[str, Any] | None:
            rid = _selected_review_id()
            if not rid:
                status_var.set("请先选择一条待审 L1 记录")
                return None
            reason_text = reason_entry.get().strip()
            try:
                res = self.reject_l1(rid, reason=reason_text, reviewer="panel_admin")
                reason_entry.delete(0, tk.END)
                status_var.set(f"已驳回 {rid} (audit #{res.get('audit_seq')})")
                return res
            except Exception as exc:  # noqa: BLE001
                status_var.set(f"驳回失败: {exc}")
                return None

        tk.Button(
            controls,
            text="✓ 批准 L1",
            command=_approve_selected,
            bg=THEME["accent_emerald"],
            fg=THEME["bg_root"],
            activebackground=THEME["accent_gold"],
            relief="flat",
            padx=8,
            pady=2,
        ).pack(side="left", padx=(0, 4))

        tk.Button(
            controls,
            text="✕ 驳回 L1",
            command=_reject_selected,
            bg=THEME["accent_crimson"],
            fg=THEME["fg_primary"],
            activebackground="#991b1b",
            relief="flat",
            padx=8,
            pady=2,
        ).pack(side="left")

        tk.Button(
            header,
            text="↻ 刷新",
            command=_refresh,
            bg=THEME["bg_panel"],
            fg=THEME["accent_gold"],
            activebackground=THEME["accent_gold"],
            activeforeground=THEME["bg_root"],
            relief="flat",
            padx=8,
            pady=2,
        ).pack(side="right")

        tk.Label(
            root_frame,
            textvariable=status_var,
            fg=THEME["fg_muted"],
            bg=THEME["bg_root"],
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", pady=(6, 0))

        self._ui_refresh_hooks.append(_refresh)
        _refresh()

        # Expose lightweight handles for deterministic testing & host refreshes
        root_frame._reiji_refresh = _refresh  # type: ignore[attr-defined]
        root_frame._reiji_approve_selected = _approve_selected  # type: ignore[attr-defined]
        root_frame._reiji_reject_selected = _reject_selected  # type: ignore[attr-defined]
        root_frame._reiji_l1_listbox = l1_listbox  # type: ignore[attr-defined]
        root_frame._reiji_reason_entry = reason_entry  # type: ignore[attr-defined]
        root_frame._reiji_status_var = status_var  # type: ignore[attr-defined]
        return root_frame

    def register_ui(self, ui_host: Any) -> dict[str, bool]:
        """Safely attach UI surfaces to any duck-typed host exposing optional slots."""
        mounted: dict[str, bool] = {
            "add_tab": False,
            "format_header_badge": False,
            "add_radar_markers": False,
            "on_radar_state_changed": False,
        }

        add_tab = getattr(ui_host, "add_tab", None)
        if callable(add_tab):
            add_tab(
                TAB_ID,
                TAB_TITLE,
                self._build_tk_tab,
                on_select=self._trigger_ui_refresh,
                on_refresh=self._trigger_ui_refresh,
            )
            mounted["add_tab"] = True

        format_header_badge = getattr(ui_host, "format_header_badge", None)
        if callable(format_header_badge):
            format_header_badge(self._header_badge_callback)
            mounted["format_header_badge"] = True

        add_radar_markers = getattr(ui_host, "add_radar_markers", None)
        if callable(add_radar_markers):
            add_radar_markers(self._radar_markers_callback)
            mounted["add_radar_markers"] = True

        on_radar_state_changed = getattr(ui_host, "on_radar_state_changed", None)
        if callable(on_radar_state_changed):
            on_radar_state_changed(self._on_radar_state_changed)
            mounted["on_radar_state_changed"] = True

        return mounted


def create_plugin(
    *,
    home: Path | str | None = None,
    db_path: Path | str | None = None,
    engine: ReijiMemoryEngine | None = None,
) -> StarPanelBridgePlugin:
    """Factory returning a configured :class:`StarPanelBridgePlugin` instance."""
    return StarPanelBridgePlugin(home=home, db_path=db_path, engine=engine)


__all__ = [
    "PLUGIN_DESCRIPTION",
    "PLUGIN_NAME",
    "RADAR_COOLDOWN_FRACTION",
    "RADAR_COOLDOWN_SEC",
    "RADAR_TTL_SEC",
    "StarPanelBridgePlugin",
    "TAB_ID",
    "TAB_TITLE",
    "THEME",
    "create_plugin",
]
