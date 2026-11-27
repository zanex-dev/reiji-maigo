"""7-decimal anti-MemoryTax compute economics ROI ledger plugin for reiji-maigo.

Tracks Flash swarm token expenditure (including L0 prefix-cache discounts)
against flagship main-window token savings and 200K pricing-cliff avoidance,
with 7-decimal USD precision (Decimal("0.0000001")), two-way reconciliation
against ``swarm_telemetry``, and anti-MemoryTax circuit-breaker alerts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from typing import Any

from reiji_maigo.storage import ReijiDatabase

PLUGIN_NAME = "roi_ledger"

USD_QUANT = Decimal("0.0000001")
NANO_PER_USD = Decimal("1000000000")
TOKENS_PER_MILLION = Decimal("1000000")

# Default unit prices in USD per 1,000,000 tokens (except cliff multiplier).
DEFAULT_PRICING: dict[str, Decimal] = {
    "flash_input": Decimal("0.15"),
    "flash_cached_l0": Decimal("0.0375"),
    "flash_output": Decimal("0.60"),
    "flagship_input": Decimal("2.50"),
    "cliff_200k_multiplier": Decimal("2.0"),
}

_ALLOWED_PRICING_KEYS: frozenset[str] = frozenset(DEFAULT_PRICING.keys())

_ROI_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS roi_ledger_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_telemetry_id INTEGER UNIQUE,
    episode_id TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_l0_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    saved_flagship_tokens INTEGER NOT NULL DEFAULT 0,
    cliff_200k_avoided INTEGER NOT NULL DEFAULT 0,
    flash_cost_nano INTEGER NOT NULL DEFAULT 0,
    gross_saved_nano INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_roi_ledger_role_created
    ON roi_ledger_entries (role, created_at DESC);

CREATE TABLE IF NOT EXISTS roi_pricing_config (
    key TEXT PRIMARY KEY,
    value_usd_per_million TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_nonneg_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _to_decimal(value: Any, *, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a numeric value, got bool")
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal: {value!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{field_name} must be finite: {value!r}")
    return dec


def _usd_to_nano(amount_usd: Decimal) -> int:
    return int((amount_usd * NANO_PER_USD).to_integral_value(rounding=ROUND_HALF_UP))


def _nano_to_usd_decimal(amount_nano: int) -> Decimal:
    return Decimal(int(amount_nano)) / NANO_PER_USD


def _format_usd(amount_usd: Decimal) -> str:
    quantized = amount_usd.quantize(USD_QUANT, rounding=ROUND_HALF_UP)
    if quantized == Decimal("0"):
        quantized = Decimal("0.0000000")
    return f"{quantized:.7f}"


def _compute_flash_cost_nano(
    *,
    prompt_tokens: int,
    cached_l0_tokens: int,
    completion_tokens: int,
    pricing: Mapping[str, Decimal],
) -> int:
    prompt = max(0, int(prompt_tokens))
    cached = max(0, int(cached_l0_tokens))
    completion = max(0, int(completion_tokens))
    if prompt > 0 and cached > prompt:
        cached = prompt
    uncached_prompt = max(0, prompt - cached)

    cost_usd = (
        Decimal(uncached_prompt) * pricing["flash_input"]
        + Decimal(cached) * pricing["flash_cached_l0"]
        + Decimal(completion) * pricing["flash_output"]
    ) / TOKENS_PER_MILLION
    return _usd_to_nano(cost_usd)


def _compute_gross_saved_nano(
    *,
    saved_flagship_tokens: int,
    cliff_200k_avoided: bool,
    pricing: Mapping[str, Decimal],
) -> int:
    saved = max(0, int(saved_flagship_tokens))
    multiplier = pricing["cliff_200k_multiplier"] if cliff_200k_avoided else Decimal("1")
    saved_usd = (Decimal(saved) * pricing["flagship_input"] * multiplier) / TOKENS_PER_MILLION
    return _usd_to_nano(saved_usd)


def _extract_telemetry_fields(role: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    prompt_tokens = _safe_nonneg_int(metrics.get("prompt_tokens"))
    cached_l0_tokens = _safe_nonneg_int(
        metrics.get("cached_l0_tokens")
        if metrics.get("cached_l0_tokens") is not None
        else (
            metrics.get("cached_prompt_tokens")
            if metrics.get("cached_prompt_tokens") is not None
            else metrics.get("cached_tokens")
        )
    )
    if prompt_tokens > 0 and cached_l0_tokens > prompt_tokens:
        cached_l0_tokens = prompt_tokens
    completion_tokens = _safe_nonneg_int(metrics.get("completion_tokens"))

    if metrics.get("saved_flagship_tokens") is not None:
        saved_flagship_tokens = _safe_nonneg_int(metrics.get("saved_flagship_tokens"))
    else:
        scanned_raw = _safe_nonneg_int(metrics.get("scanned_raw_tokens"))
        distilled = _safe_nonneg_int(metrics.get("distilled_tokens"))
        saved_flagship_tokens = max(0, scanned_raw - distilled) if scanned_raw > 0 else 0

    cliff_200k_avoided = bool(metrics.get("cliff_200k_avoided", False))
    episode_id = str(metrics.get("episode_id") or "").strip()

    return {
        "role": role,
        "episode_id": episode_id,
        "prompt_tokens": prompt_tokens,
        "cached_l0_tokens": cached_l0_tokens,
        "completion_tokens": completion_tokens,
        "saved_flagship_tokens": saved_flagship_tokens,
        "cliff_200k_avoided": cliff_200k_avoided,
    }


def _table_exists(conn: Any, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_roi_tables(db: ReijiDatabase) -> None:
    """Idempotently create ROI ledger and pricing configuration tables."""
    with db.transaction() as conn:
        conn.executescript(_ROI_SCHEMA_SQL)


class RoiLedger:
    """7-decimal USD compute economics ledger for swarm expenditure vs. savings."""

    def __init__(self, db: ReijiDatabase) -> None:
        self.db = db
        ensure_roi_tables(self.db)

    def get_pricing(self) -> dict[str, Decimal]:
        """Return effective pricing table (defaults overlaid with persisted overrides)."""
        ensure_roi_tables(self.db)
        effective = dict(DEFAULT_PRICING)
        with self.db.read_connection() as conn:
            rows = conn.execute(
                "SELECT key, value_usd_per_million FROM roi_pricing_config"
            ).fetchall()
        for row in rows:
            key = str(row["key"])
            if key in _ALLOWED_PRICING_KEYS:
                effective[key] = Decimal(str(row["value_usd_per_million"]))
        return effective

    def set_pricing(self, **rates_per_million: Any) -> dict[str, str]:
        """Persist pricing overrides (USD per 1M tokens; multiplier for 200K cliff)."""
        if not rates_per_million:
            raise ValueError("At least one pricing parameter must be provided")
        validated: dict[str, Decimal] = {}
        for key, raw_value in rates_per_million.items():
            if key not in _ALLOWED_PRICING_KEYS:
                allowed = ", ".join(sorted(_ALLOWED_PRICING_KEYS))
                raise ValueError(f"Unknown pricing key {key!r}; expected one of: {allowed}")
            dec = _to_decimal(raw_value, field_name=key)
            if key == "cliff_200k_multiplier":
                if dec < Decimal("1"):
                    raise ValueError("cliff_200k_multiplier must be >= 1")
            elif dec < Decimal("0"):
                raise ValueError(f"{key} must be non-negative")
            validated[key] = dec

        ensure_roi_tables(self.db)
        ts = _utc_now_iso()
        with self.db.transaction() as conn:
            for key, dec in validated.items():
                conn.execute(
                    """
                    INSERT INTO roi_pricing_config (key, value_usd_per_million, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_usd_per_million = excluded.value_usd_per_million,
                        updated_at = excluded.updated_at
                    """,
                    (key, str(dec), ts),
                )
        return {k: str(v) for k, v in self.get_pricing().items()}

    def sync_from_swarm_telemetry(self) -> int:
        """Idempotently ingest new rows from ``swarm_telemetry`` into ``roi_ledger_entries``."""
        ensure_roi_tables(self.db)
        pricing = self.get_pricing()

        with self.db.read_connection() as conn:
            if not _table_exists(conn, "swarm_telemetry"):
                return 0
            pending_rows = conn.execute(
                """
                SELECT t.id, t.role, t.metrics_json, t.created_at
                FROM swarm_telemetry AS t
                LEFT JOIN roi_ledger_entries AS r
                    ON r.source_telemetry_id = t.id
                WHERE r.id IS NULL
                ORDER BY t.id ASC
                """
            ).fetchall()

        if not pending_rows:
            return 0

        inserted = 0
        with self.db.transaction() as conn:
            for row in pending_rows:
                telemetry_id = int(row["id"])
                role = str(row["role"] or "unknown").strip() or "unknown"
                created_at = str(row["created_at"] or _utc_now_iso())
                try:
                    raw_metrics = json.loads(row["metrics_json"] or "{}")
                except Exception:
                    raw_metrics = {}
                metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
                fields = _extract_telemetry_fields(role, metrics)

                flash_cost_nano = _compute_flash_cost_nano(
                    prompt_tokens=fields["prompt_tokens"],
                    cached_l0_tokens=fields["cached_l0_tokens"],
                    completion_tokens=fields["completion_tokens"],
                    pricing=pricing,
                )
                gross_saved_nano = _compute_gross_saved_nano(
                    saved_flagship_tokens=fields["saved_flagship_tokens"],
                    cliff_200k_avoided=fields["cliff_200k_avoided"],
                    pricing=pricing,
                )

                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO roi_ledger_entries (
                        source_telemetry_id,
                        episode_id,
                        role,
                        prompt_tokens,
                        cached_l0_tokens,
                        completion_tokens,
                        saved_flagship_tokens,
                        cliff_200k_avoided,
                        flash_cost_nano,
                        gross_saved_nano,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        telemetry_id,
                        fields["episode_id"],
                        role,
                        fields["prompt_tokens"],
                        fields["cached_l0_tokens"],
                        fields["completion_tokens"],
                        fields["saved_flagship_tokens"],
                        1 if fields["cliff_200k_avoided"] else 0,
                        flash_cost_nano,
                        gross_saved_nano,
                        created_at,
                    ),
                )
                if cur.rowcount and cur.rowcount > 0:
                    inserted += 1
        return inserted

    def record_window_saving(
        self,
        episode_id: str,
        saved_tokens: int,
        *,
        cliff_200k_avoided: bool = False,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Record flagship main-window token savings for an episode."""
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if isinstance(saved_tokens, bool) or not isinstance(saved_tokens, int) or saved_tokens < 0:
            raise ValueError("saved_tokens must be a non-negative integer")

        ensure_roi_tables(self.db)
        pricing = self.get_pricing()
        ep = episode_id.strip()
        cliff = bool(cliff_200k_avoided)
        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        gross_saved_nano = _compute_gross_saved_nano(
            saved_flagship_tokens=saved_tokens,
            cliff_200k_avoided=cliff,
            pricing=pricing,
        )

        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO roi_ledger_entries (
                    source_telemetry_id,
                    episode_id,
                    role,
                    prompt_tokens,
                    cached_l0_tokens,
                    completion_tokens,
                    saved_flagship_tokens,
                    cliff_200k_avoided,
                    flash_cost_nano,
                    gross_saved_nano,
                    created_at
                )
                VALUES (NULL, ?, 'main_window', 0, 0, 0, ?, ?, 0, ?, ?)
                """,
                (ep, saved_tokens, 1 if cliff else 0, gross_saved_nano, ts),
            )
            entry_id = int(cur.lastrowid)

        return {
            "id": entry_id,
            "episode_id": ep,
            "role": "main_window",
            "saved_flagship_tokens": saved_tokens,
            "cliff_200k_avoided": cliff,
            "gross_saved_nano": gross_saved_nano,
            "gross_saved_usd": _format_usd(_nano_to_usd_decimal(gross_saved_nano)),
            "created_at": ts,
        }

    def summary(self) -> dict[str, Any]:
        """Compute 7-decimal USD ROI summary and anti-MemoryTax circuit-breaker alert."""
        ensure_roi_tables(self.db)
        with self.db.read_connection() as conn:
            agg = conn.execute(
                """
                SELECT
                    COUNT(*) AS entries_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(cached_l0_tokens), 0) AS cached_l0_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(saved_flagship_tokens), 0) AS saved_flagship_tokens,
                    COALESCE(SUM(cliff_200k_avoided), 0) AS cliff_200k_avoided_count,
                    COALESCE(SUM(flash_cost_nano), 0) AS flash_cost_nano,
                    COALESCE(SUM(gross_saved_nano), 0) AS gross_saved_nano
                FROM roi_ledger_entries
                """
            ).fetchone()

            role_rows = conn.execute(
                """
                SELECT
                    role,
                    COUNT(*) AS entries_count,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(cached_l0_tokens), 0) AS cached_l0_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(saved_flagship_tokens), 0) AS saved_flagship_tokens,
                    COALESCE(SUM(flash_cost_nano), 0) AS flash_cost_nano,
                    COALESCE(SUM(gross_saved_nano), 0) AS gross_saved_nano
                FROM roi_ledger_entries
                GROUP BY role
                ORDER BY role ASC
                """
            ).fetchall()

        entries_count = int(agg["entries_count"])
        flash_cost_nano = int(agg["flash_cost_nano"])
        gross_saved_nano = int(agg["gross_saved_nano"])
        net_roi_nano = gross_saved_nano - flash_cost_nano

        flash_cost_dec = _nano_to_usd_decimal(flash_cost_nano)
        gross_saved_dec = _nano_to_usd_decimal(gross_saved_nano)
        net_roi_dec = _nano_to_usd_decimal(net_roi_nano)

        if flash_cost_nano > 0:
            roi_multiple_dec = (gross_saved_dec / flash_cost_dec).quantize(
                USD_QUANT, rounding=ROUND_HALF_UP
            )
            roi_multiple = _format_usd(roi_multiple_dec)
        else:
            roi_multiple = _format_usd(Decimal("0"))

        tax_alert = flash_cost_nano > 0 and flash_cost_nano > gross_saved_nano

        by_role: dict[str, dict[str, Any]] = {}
        for r in role_rows:
            r_flash = int(r["flash_cost_nano"])
            r_saved = int(r["gross_saved_nano"])
            by_role[str(r["role"])] = {
                "entries_count": int(r["entries_count"]),
                "prompt_tokens": int(r["prompt_tokens"]),
                "cached_l0_tokens": int(r["cached_l0_tokens"]),
                "completion_tokens": int(r["completion_tokens"]),
                "saved_flagship_tokens": int(r["saved_flagship_tokens"]),
                "flash_cost_usd": _format_usd(_nano_to_usd_decimal(r_flash)),
                "gross_saved_usd": _format_usd(_nano_to_usd_decimal(r_saved)),
                "net_roi_usd": _format_usd(_nano_to_usd_decimal(r_saved - r_flash)),
            }

        return {
            "flash_cost_usd": _format_usd(flash_cost_dec),
            "gross_saved_usd": _format_usd(gross_saved_dec),
            "net_roi_usd": _format_usd(net_roi_dec),
            "roi_multiple": roi_multiple,
            "tax_alert": tax_alert,
            "tax_alert_reason": "memory_tax_exceeds_savings" if tax_alert else None,
            "entries_count": entries_count,
            "flash_cost_nano": flash_cost_nano,
            "gross_saved_nano": gross_saved_nano,
            "net_roi_nano": net_roi_nano,
            "tokens": {
                "prompt_tokens": int(agg["prompt_tokens"]),
                "cached_l0_tokens": int(agg["cached_l0_tokens"]),
                "completion_tokens": int(agg["completion_tokens"]),
                "saved_flagship_tokens": int(agg["saved_flagship_tokens"]),
                "cliff_200k_avoided_count": int(agg["cliff_200k_avoided_count"]),
            },
            "by_role": by_role,
            "pricing": {k: str(v) for k, v in self.get_pricing().items()},
        }

    def reconcile(self) -> dict[str, Any]:
        """Two-way reconciliation between ``roi_ledger_entries`` and ``swarm_telemetry``.

        Verifies ledger coverage and checks that absolute USD drift between stored
        ledger totals and recomputed raw telemetry totals is ``<= Decimal("0.0000001")``.
        """
        ensure_roi_tables(self.db)
        pricing = self.get_pricing()

        with self.db.read_connection() as conn:
            ledger_rows = conn.execute(
                """
                SELECT source_telemetry_id, flash_cost_nano, gross_saved_nano
                FROM roi_ledger_entries
                WHERE source_telemetry_id IS NOT NULL
                """
            ).fetchall()

            has_telemetry = _table_exists(conn, "swarm_telemetry")
            telemetry_rows = (
                conn.execute(
                    """
                    SELECT id, role, metrics_json
                    FROM swarm_telemetry
                    ORDER BY id ASC
                    """
                ).fetchall()
                if has_telemetry
                else []
            )

        ledger_ids: set[int] = set()
        ledger_flash_nano = 0
        ledger_saved_nano = 0
        for row in ledger_rows:
            ledger_ids.add(int(row["source_telemetry_id"]))
            ledger_flash_nano += int(row["flash_cost_nano"])
            ledger_saved_nano += int(row["gross_saved_nano"])

        telemetry_ids: set[int] = set()
        telemetry_flash_nano = 0
        telemetry_saved_nano = 0
        for row in telemetry_rows:
            tid = int(row["id"])
            telemetry_ids.add(tid)
            role = str(row["role"] or "unknown").strip() or "unknown"
            try:
                raw_metrics = json.loads(row["metrics_json"] or "{}")
            except Exception:
                raw_metrics = {}
            metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
            fields = _extract_telemetry_fields(role, metrics)
            telemetry_flash_nano += _compute_flash_cost_nano(
                prompt_tokens=fields["prompt_tokens"],
                cached_l0_tokens=fields["cached_l0_tokens"],
                completion_tokens=fields["completion_tokens"],
                pricing=pricing,
            )
            telemetry_saved_nano += _compute_gross_saved_nano(
                saved_flagship_tokens=fields["saved_flagship_tokens"],
                cliff_200k_avoided=fields["cliff_200k_avoided"],
                pricing=pricing,
            )

        missing_in_ledger = sorted(telemetry_ids - ledger_ids)
        orphan_in_ledger = sorted(ledger_ids - telemetry_ids)

        flash_drift_dec = abs(
            _nano_to_usd_decimal(ledger_flash_nano) - _nano_to_usd_decimal(telemetry_flash_nano)
        )
        saved_drift_dec = abs(
            _nano_to_usd_decimal(ledger_saved_nano) - _nano_to_usd_decimal(telemetry_saved_nano)
        )
        max_drift_dec = max(flash_drift_dec, saved_drift_dec)

        ok = (
            max_drift_dec <= USD_QUANT
            and len(missing_in_ledger) == 0
            and len(orphan_in_ledger) == 0
        )

        return {
            "ok": ok,
            "drift_usd": _format_usd(max_drift_dec),
            "flash_drift_usd": _format_usd(flash_drift_dec),
            "saved_drift_usd": _format_usd(saved_drift_dec),
            "tolerance_usd": _format_usd(USD_QUANT),
            "ledger_telemetry_count": len(ledger_ids),
            "source_telemetry_count": len(telemetry_ids),
            "missing_in_ledger": missing_in_ledger,
            "orphan_in_ledger": orphan_in_ledger,
            "ledger_flash_cost_usd": _format_usd(_nano_to_usd_decimal(ledger_flash_nano)),
            "telemetry_flash_cost_usd": _format_usd(_nano_to_usd_decimal(telemetry_flash_nano)),
            "ledger_gross_saved_usd": _format_usd(_nano_to_usd_decimal(ledger_saved_nano)),
            "telemetry_gross_saved_usd": _format_usd(_nano_to_usd_decimal(telemetry_saved_nano)),
        }


def _add_json_flag_if_missing(parser: argparse.ArgumentParser) -> None:
    if "--json" not in getattr(parser, "_option_string_actions", {}):
        parser.add_argument("--json", action="store_true", help="Emit JSON output")


def _configure_roi_subparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _add_json_flag_if_missing(parser)
    subparsers = parser.add_subparsers(dest="roi_action")

    p_status = subparsers.add_parser("status", help="Show 7-decimal ROI ledger summary")
    p_status.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing pending swarm_telemetry rows before summarizing",
    )
    _add_json_flag_if_missing(p_status)

    p_reconcile = subparsers.add_parser(
        "reconcile",
        help="Reconcile roi_ledger_entries against raw swarm_telemetry tokens",
    )
    p_reconcile.add_argument(
        "--sync",
        action="store_true",
        help="Sync pending swarm_telemetry rows before reconciling",
    )
    _add_json_flag_if_missing(p_reconcile)

    p_record = subparsers.add_parser(
        "record-saving",
        help="Record flagship main-window tokens saved for an episode",
    )
    p_record.add_argument("--episode-id", required=True, help="Episode identifier")
    p_record.add_argument(
        "--saved-tokens",
        required=True,
        type=int,
        help="Flagship main-window tokens avoided",
    )
    p_record.add_argument(
        "--cliff-200k-avoided",
        action="store_true",
        help="Apply 200K pricing-cliff multiplier to saved tokens",
    )
    p_record.add_argument("--now-iso", default=None, help="Optional UTC ISO-8601 timestamp")
    _add_json_flag_if_missing(p_record)

    p_pricing = subparsers.add_parser(
        "set-pricing",
        help="Update USD per 1M token pricing rates in roi_pricing_config",
    )
    p_pricing.add_argument("--flash-input", default=None, help="Flash uncached input USD/1M")
    p_pricing.add_argument("--flash-cached-l0", default=None, help="Flash L0 cached input USD/1M")
    p_pricing.add_argument("--flash-output", default=None, help="Flash completion output USD/1M")
    p_pricing.add_argument("--flagship-input", default=None, help="Flagship input USD/1M")
    p_pricing.add_argument(
        "--cliff-200k-multiplier",
        default=None,
        help="Multiplier when 200K pricing cliff is avoided (>= 1.0)",
    )
    _add_json_flag_if_missing(p_pricing)

    return parser


def _resolve_db_from_call(
    pos_args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[ReijiDatabase | None, argparse.Namespace, bool]:
    """Extract (db, args_namespace, owns_db) across multiple CLI dispatch conventions."""
    engine = kwargs.get("engine")
    db = kwargs.get("db")
    ns = kwargs.get("args")

    for item in pos_args:
        if isinstance(item, argparse.Namespace):
            if ns is None:
                ns = item
        elif hasattr(item, "db") and hasattr(getattr(item, "db"), "transaction"):
            if engine is None:
                engine = item
        elif hasattr(item, "transaction") and hasattr(item, "read_connection"):
            if db is None:
                db = item

    if ns is None:
        ns = argparse.Namespace()
    if db is None and engine is not None:
        db = getattr(engine, "db", None)

    owns_db = False
    if db is None:
        db_path = getattr(ns, "db", None) or getattr(ns, "db_path", None)
        if db_path:
            db = ReijiDatabase(db_path)
            owns_db = True

    return db, ns, owns_db


class _RoiCliHandler:
    """CLI handler for the ``roi`` command with ``configure_parser`` & ``extra_args`` fallback."""

    def configure_parser(self, subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return _configure_roi_subparser(subparser)

    def _parse_effective_namespace(self, ns: argparse.Namespace) -> argparse.Namespace:
        action = getattr(ns, "roi_action", None)
        extra_args = getattr(ns, "extra_args", None)
        if action is None and extra_args:
            fallback_parser = _configure_roi_subparser(argparse.ArgumentParser(prog="roi"))
            parsed = fallback_parser.parse_args(list(extra_args))
            if getattr(ns, "json", False):
                parsed.json = True
            return parsed
        return ns

    def __call__(self, *pos_args: Any, **kwargs: Any) -> int:
        db, raw_ns, owns_db = _resolve_db_from_call(pos_args, kwargs)
        if db is None:
            raise ValueError("roi command requires an active engine or ReijiDatabase instance")

        ns = self._parse_effective_namespace(raw_ns)
        as_json = bool(getattr(ns, "json", False))
        action = getattr(ns, "roi_action", None) or "status"

        try:
            ledger = RoiLedger(db)
            if action == "status":
                synced = 0 if getattr(ns, "no_sync", False) else ledger.sync_from_swarm_telemetry()
                payload = ledger.summary()
                payload["synced_rows"] = synced
                if as_json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(
                        f"ROI Summary: flash=${payload['flash_cost_usd']} "
                        f"saved=${payload['gross_saved_usd']} "
                        f"net=${payload['net_roi_usd']} "
                        f"multiple={payload['roi_multiple']}x "
                        f"tax_alert={payload['tax_alert']} "
                        f"entries={payload['entries_count']}"
                    )
                return 0

            if action == "reconcile":
                synced = ledger.sync_from_swarm_telemetry() if getattr(ns, "sync", False) else 0
                report = ledger.reconcile()
                report["synced_rows"] = synced
                if as_json:
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                else:
                    print(
                        f"ROI Reconcile: ok={report['ok']} drift=${report['drift_usd']} "
                        f"(ledger={report['ledger_telemetry_count']}, "
                        f"source={report['source_telemetry_count']})"
                    )
                return 0 if report["ok"] else 1

            if action == "record-saving":
                record = ledger.record_window_saving(
                    ns.episode_id,
                    int(ns.saved_tokens),
                    cliff_200k_avoided=bool(getattr(ns, "cliff_200k_avoided", False)),
                    now_iso=getattr(ns, "now_iso", None),
                )
                if as_json:
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                else:
                    print(
                        f"Recorded saving #{record['id']} episode={record['episode_id']} "
                        f"tokens={record['saved_flagship_tokens']} "
                        f"cliff_200k={record['cliff_200k_avoided']} "
                        f"saved=${record['gross_saved_usd']}"
                    )
                return 0

            if action == "set-pricing":
                updates: dict[str, Any] = {}
                mapping = (
                    ("flash_input", "flash_input"),
                    ("flash_cached_l0", "flash_cached_l0"),
                    ("flash_output", "flash_output"),
                    ("flagship_input", "flagship_input"),
                    ("cliff_200k_multiplier", "cliff_200k_multiplier"),
                )
                for attr, key in mapping:
                    val = getattr(ns, attr, None)
                    if val is not None:
                        updates[key] = val
                updated = ledger.set_pricing(**updates)
                payload = {"ok": True, "pricing": updated}
                if as_json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(f"Updated ROI pricing: {json.dumps(updated, ensure_ascii=False)}")
                return 0

            raise ValueError(f"Unsupported roi action: {action!r}")
        finally:
            if owns_db:
                db.close()


def _extract_db_from_payload(payload: Any) -> ReijiDatabase | None:
    if isinstance(payload, Mapping):
        db = payload.get("db")
        if db is not None and hasattr(db, "transaction"):
            return db
        engine = payload.get("engine")
        if engine is not None and hasattr(engine, "db"):
            return engine.db
    elif hasattr(payload, "db") and hasattr(getattr(payload, "db"), "transaction"):
        return payload.db
    elif hasattr(payload, "transaction") and hasattr(payload, "read_connection"):
        return payload
    return None


class RoiLedgerPlugin:
    """Self-bootstrapping optional microkernel plugin for 7-decimal ROI accounting."""

    name: str = PLUGIN_NAME

    def __init__(self) -> None:
        self._cli_handler = _RoiCliHandler()

    def register_cli(self, kernel: Any) -> None:
        kernel.register_cli(
            "roi",
            self._cli_handler,
            help_text="7-decimal anti-MemoryTax compute economics ledger",
            plugin_name=self.name,
        )

    def register_hooks(self, kernel: Any) -> None:
        kernel.register_hooks(
            "swarm.telemetry.recorded",
            self._on_swarm_telemetry,
            plugin_name=self.name,
            priority=200,
        )
        kernel.register_hooks(
            "roi.window_saved",
            self._on_window_saved,
            plugin_name=self.name,
            priority=200,
        )

    def _on_swarm_telemetry(self, payload: Any) -> dict[str, Any]:
        db = _extract_db_from_payload(payload)
        if db is None:
            return {"ok": False, "skipped": True, "reason": "no_db"}
        ledger = RoiLedger(db)
        synced = ledger.sync_from_swarm_telemetry()
        summary = ledger.summary()
        return {
            "ok": True,
            "synced": synced,
            "tax_alert": summary["tax_alert"],
            "net_roi_usd": summary["net_roi_usd"],
        }

    def _on_window_saved(self, payload: Any) -> dict[str, Any]:
        db = _extract_db_from_payload(payload)
        if db is None or not isinstance(payload, Mapping):
            return {"ok": False, "skipped": True, "reason": "invalid_payload"}
        episode_id = payload.get("episode_id")
        saved_tokens = payload.get("saved_tokens")
        if not isinstance(episode_id, str) or not episode_id.strip():
            return {"ok": False, "skipped": True, "reason": "missing_episode_id"}
        if isinstance(saved_tokens, bool) or not isinstance(saved_tokens, int):
            return {"ok": False, "skipped": True, "reason": "invalid_saved_tokens"}
        ledger = RoiLedger(db)
        return ledger.record_window_saving(
            episode_id,
            saved_tokens,
            cliff_200k_avoided=bool(payload.get("cliff_200k_avoided", False)),
            now_iso=payload.get("now_iso"),
        )


def create_plugin() -> RoiLedgerPlugin:
    """Factory entry point for microkernel plugin discovery."""
    return RoiLedgerPlugin()
