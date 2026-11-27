"""Unified memory swarm coordinator and industrial-grade telemetry pulse panel.

``SwarmCoordinator`` wires together the three physically isolated swarm workers:
  - **Scribe** (速记蜂): shadow digestion of undigested turns into ActivePool
    and Level-2 staging notes, optionally via a non-blocking ``AsyncScribeWorker``.
  - **Scout** (侦察蜂): strictly read-only 3-layer deep reconnaissance returning
    compact, source-cited ``ScoutBriefing`` payloads.
  - **Archivist** (子夜馆长蜂): midnight curation, chain compaction, card archival,
    TEPA bitemporal conflict resolution, and Level-3 backfill distillation.

When no LLM runner is configured (e.g. missing ``REIJI_FLASH_API_KEY`` /
``REIJI_FLASH_BASE_URL`` and no kernel ``extractor`` provider), ``status`` /
``pulse_summary`` / ``render_pulse_banner`` remain 100% operational with
``runner=unconfigured`` and live local SQLite telemetry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
import threading
from typing import TYPE_CHECKING, Any

from reiji_maigo.swarm.archivist import (
    ARCHIVIST_L0_PREFIX,
    ArchivistAgent,
    ArchivistReport,
)
from reiji_maigo.swarm.guards import (
    ArchivistCapabilities,
    ScoutCapabilities,
    ScribeCapabilities,
    build_role_capabilities,
    ensure_swarm_tables,
    list_swarm_telemetry,
)
from reiji_maigo.swarm.runner import (
    ENV_FLASH_API_KEY,
    ENV_FLASH_BASE_URL,
    CallableFlashRunner,
    HttpFlashRunner,
    SwarmModelRunner,
    SwarmRunnerError,
    compute_prefix_sha256,
)
from reiji_maigo.swarm.scout import (
    SCOUT_L0_PREFIX,
    ScoutAgent,
    ScoutBriefing,
)
from reiji_maigo.swarm.scribe import (
    DEFAULT_WATERMARK_KEY,
    SCRIBE_L0_PREFIX,
    AsyncScribeWorker,
    ScribeAgent,
    ScribeDigestReport,
)

if TYPE_CHECKING:
    from reiji_maigo.engine import ReijiMemoryEngine


def _resolve_runner(
    engine: ReijiMemoryEngine,
    explicit_runner: SwarmModelRunner | Any | None,
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[SwarmModelRunner | None, str]:
    """Resolve a SwarmModelRunner from explicit arg, env vars, or kernel extractor."""
    if explicit_runner is not None:
        if callable(getattr(explicit_runner, "run_shadow_json", None)):
            return explicit_runner, "explicit"
        if callable(explicit_runner):
            return CallableFlashRunner(explicit_runner), "explicit"
        raise TypeError(
            "runner must implement run_shadow_json(...) or be a callable compatible with CallableFlashRunner"
        )

    source_env: Mapping[str, str] = os.environ if env is None else env
    base_url = source_env.get(ENV_FLASH_BASE_URL, "").strip()
    api_key = source_env.get(ENV_FLASH_API_KEY, "").strip()
    if base_url and api_key:
        return HttpFlashRunner(env=source_env), "env"

    vault = getattr(engine, "vault", None)
    if vault is not None and callable(getattr(vault, "load", None)):
        try:
            vault_data = vault.load(strict=False)
        except Exception:
            vault_data = {}
        if isinstance(vault_data, Mapping):
            v_base = str(vault_data.get("flash_base_url") or "").strip()
            v_key = str(vault_data.get("flash_api_key") or "").strip()
            merged_base = base_url or v_base
            merged_key = api_key or v_key
            if merged_base and merged_key:
                merged_env = dict(source_env)
                merged_env[ENV_FLASH_BASE_URL] = merged_base
                merged_env[ENV_FLASH_API_KEY] = merged_key
                v_model = str(vault_data.get("flash_model") or "").strip()
                if v_model and not merged_env.get("REIJI_FLASH_MODEL", "").strip():
                    merged_env["REIJI_FLASH_MODEL"] = v_model
                return HttpFlashRunner(env=merged_env), "vault"

    kernel = getattr(engine, "kernel", None)
    if kernel is not None and callable(getattr(kernel, "get_provider", None)):
        try:
            extractor = kernel.get_provider("extractor")
        except Exception:
            extractor = None
        if extractor is not None:
            if callable(getattr(extractor, "run_shadow_json", None)):
                return extractor, "kernel_extractor"
            if callable(extractor):
                return CallableFlashRunner(extractor), "kernel_extractor"

    return None, "unconfigured"


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SwarmCoordinator:
    """Unified controller for Reiji-Maigo's headless three-worker memory swarm."""

    def __init__(
        self,
        engine: ReijiMemoryEngine,
        runner: SwarmModelRunner | None = None,
        *,
        async_scribe: bool = False,
        debounce_sec: float = 2.0,
    ) -> None:
        if engine is None or not hasattr(engine, "db"):
            raise TypeError("engine must be a ReijiMemoryEngine instance")
        if isinstance(debounce_sec, bool) or not isinstance(debounce_sec, (int, float)) or debounce_sec < 0:
            raise ValueError("debounce_sec must be a non-negative number")

        self._engine: ReijiMemoryEngine = engine
        self._debounce_sec: float = float(debounce_sec)
        self._async_scribe_requested: bool = bool(async_scribe)
        self._scribe_lock = threading.Lock()

        ensure_swarm_tables(engine.db)

        self._scribe_caps: ScribeCapabilities = build_role_capabilities(engine, "scribe")  # type: ignore[assignment]
        self._scout_caps: ScoutCapabilities = build_role_capabilities(engine, "scout")  # type: ignore[assignment]
        self._archivist_caps: ArchivistCapabilities = build_role_capabilities(engine, "archivist")  # type: ignore[assignment]

        resolved_runner, runner_source = _resolve_runner(engine, runner)
        self._runner: SwarmModelRunner | None = resolved_runner
        self._runner_source: str = runner_source

        self._scribe: ScribeAgent | None = None
        self._scout: ScoutAgent | None = None
        self._archivist: ArchivistAgent | None = None
        self._async_worker: AsyncScribeWorker | None = None

        if self._runner is not None:
            self._scribe = ScribeAgent(self._scribe_caps, self._runner)
            self._scout = ScoutAgent(self._scout_caps, self._runner)
            self._archivist = ArchivistAgent(self._archivist_caps, self._runner)
            if self._async_scribe_requested:
                self._async_worker = AsyncScribeWorker(
                    self._scribe,
                    debounce_sec=self._debounce_sec,
                )

    @property
    def engine(self) -> ReijiMemoryEngine:
        return self._engine

    @property
    def runner(self) -> SwarmModelRunner | None:
        return self._runner

    @property
    def is_configured(self) -> bool:
        return self._runner is not None

    @property
    def runner_source(self) -> str:
        return self._runner_source

    @property
    def scribe(self) -> ScribeAgent | None:
        return self._scribe

    @property
    def scout(self) -> ScoutAgent | None:
        return self._scout

    @property
    def archivist(self) -> ArchivistAgent | None:
        return self._archivist

    @property
    def async_worker(self) -> AsyncScribeWorker | None:
        return self._async_worker

    def _require_configured_runner(self, action: str) -> SwarmModelRunner:
        if self._runner is None:
            raise SwarmRunnerError(
                f"Cannot execute swarm {action!r}: runner is unconfigured. "
                f"Set {ENV_FLASH_BASE_URL} and {ENV_FLASH_API_KEY} or register a kernel 'extractor' provider."
            )
        return self._runner

    def _latest_turn_id(self, episode_id: str | None = None) -> int:
        with self._engine.db.read_connection() as conn:
            if episode_id:
                row = conn.execute(
                    "SELECT COALESCE(MAX(turn_id), 0) AS max_id FROM raw_turns WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COALESCE(MAX(turn_id), 0) AS max_id FROM raw_turns"
                ).fetchone()
        return int(row["max_id"]) if row is not None else 0

    def on_turn_completed(
        self,
        episode_id: str,
        role: str,
        content: str,
        *,
        record_turn: bool = True,
        origin_trust: str = "user_verified",
    ) -> int:
        """Record a completed turn (optional) and notify the async Scribe worker if active."""
        if record_turn:
            turn_id = self._engine.append_turn(
                episode_id,
                role,
                content,
                origin_trust=origin_trust,
            )
        else:
            turn_id = self._latest_turn_id(episode_id)
            if turn_id == 0:
                turn_id = self._latest_turn_id(None)

        if self._async_worker is not None and turn_id > 0:
            self._async_worker.notify(episode_id, turn_id)

        return int(turn_id)

    def flush_scribe(self, *, timeout: float | None = 5.0) -> bool:
        """Wait for any pending debounced async Scribe batch to drain."""
        if self._async_worker is None:
            return True
        return self._async_worker.flush(timeout=timeout)

    def run_scribe_now(self, *, now_iso: str | None = None) -> ScribeDigestReport:
        """Synchronously digest pending raw turns into ActivePool via ScribeAgent."""
        self._require_configured_runner("scribe")
        assert self._scribe is not None
        with self._scribe_lock:
            return self._scribe.digest_once(now_iso=now_iso)

    def dispatch_scout(
        self,
        query: str,
        *,
        domain: str | None = None,
        hint_tags: Iterable[str] = (),
    ) -> ScoutBriefing:
        """Dispatch the read-only ScoutAgent for 3-layer deep reconnaissance."""
        self._require_configured_runner("scout")
        assert self._scout is not None
        return self._scout.investigate(query, domain=domain, hint_tags=hint_tags)

    def run_midnight_archivist(self, *, now_iso: str | None = None) -> ArchivistReport:
        """Drain pending async scribe batches (if any) and execute midnight curation."""
        self._require_configured_runner("midnight")
        assert self._archivist is not None
        if self._async_worker is not None:
            self._async_worker.flush(timeout=5.0)
        return self._archivist.curate_midnight(now_iso=now_iso)

    def close(self, *, flush: bool = True, timeout: float | None = 5.0) -> None:
        """Gracefully stop the background AsyncScribeWorker if running."""
        if self._async_worker is not None:
            self._async_worker.close(flush=flush, timeout=timeout)
            self._async_worker = None

    def __enter__(self) -> SwarmCoordinator:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def pulse_summary(self) -> dict[str, Any]:
        """Collect structured telemetry across L0 prefix lock, ActivePool, and all three workers."""
        ensure_swarm_tables(self._engine.db)
        status_info = self._engine.status()
        counts = dict(status_info.get("counts", {}))

        watermark = int(self._scribe_caps.get_watermark(DEFAULT_WATERMARK_KEY))
        with self._engine.db.read_connection() as conn:
            notes_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM pool_event_notes"
            ).fetchone()
            undigested_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM raw_turns WHERE turn_id > ?",
                (watermark,),
            ).fetchone()
            telemetry_count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM swarm_telemetry"
            ).fetchone()

        event_notes_count = int(notes_row["cnt"]) if notes_row is not None else 0
        undigested_count = int(undigested_row["cnt"]) if undigested_row is not None else 0
        telemetry_total = int(telemetry_count_row["cnt"]) if telemetry_count_row is not None else 0

        rows = list_swarm_telemetry(self._engine.db, limit=500)

        scribe_stats: dict[str, Any] = {
            "runs": 0,
            "turns_digested": 0,
            "created_events": 0,
            "updated_events": 0,
            "skipped_banter": 0,
            "clamped_spans": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "last_run_at": None,
        }
        scout_stats: dict[str, Any] = {
            "runs": 0,
            "found_runs": 0,
            "abstained_runs": 0,
            "raw_turns_scanned": 0,
            "scanned_raw_tokens": 0,
            "distilled_tokens": 0,
            "avg_compression_ratio": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "last_raw_turns_scanned": 0,
            "last_distilled_tokens": 0,
            "last_compression_ratio": 0.0,
            "last_run_at": None,
        }
        archivist_stats: dict[str, Any] = {
            "runs": 0,
            "dropped_micro": 0,
            "compacted_events": 0,
            "archived_cards": 0,
            "hibernated_cards": 0,
            "superseded_cards": 0,
            "revoked_cards": 0,
            "backfilled_cards": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "last_run_at": None,
        }

        scout_ratios: list[float] = []

        for row in rows:
            role = row["role"]
            metrics = row.get("metrics") or {}
            created_at = row.get("created_at")

            if role == "scribe":
                scribe_stats["runs"] += 1
                scribe_stats["turns_digested"] += _safe_int(metrics.get("turns_digested"))
                scribe_stats["created_events"] += _safe_int(metrics.get("created_count"))
                scribe_stats["updated_events"] += _safe_int(metrics.get("updated_count"))
                scribe_stats["skipped_banter"] += _safe_int(metrics.get("skipped_banter"))
                scribe_stats["clamped_spans"] += _safe_int(metrics.get("clamped_spans"))
                scribe_stats["prompt_tokens"] += _safe_int(metrics.get("prompt_tokens"))
                scribe_stats["completion_tokens"] += _safe_int(metrics.get("completion_tokens"))
                if scribe_stats["last_run_at"] is None:
                    scribe_stats["last_run_at"] = created_at

            elif role == "scout":
                scout_stats["runs"] += 1
                if bool(metrics.get("found")):
                    scout_stats["found_runs"] += 1
                if bool(metrics.get("abstained")):
                    scout_stats["abstained_runs"] += 1
                scanned = _safe_int(metrics.get("raw_turns_scanned"))
                scanned_tok = _safe_int(metrics.get("scanned_raw_tokens"))
                distilled = _safe_int(metrics.get("distilled_tokens"))
                ratio = _safe_float(metrics.get("compression_ratio"))
                scout_stats["raw_turns_scanned"] += scanned
                scout_stats["scanned_raw_tokens"] += scanned_tok
                scout_stats["distilled_tokens"] += distilled
                scout_stats["prompt_tokens"] += _safe_int(metrics.get("prompt_tokens"))
                scout_stats["completion_tokens"] += _safe_int(metrics.get("completion_tokens"))
                if ratio > 0.0:
                    scout_ratios.append(ratio)
                if scout_stats["last_run_at"] is None:
                    scout_stats["last_run_at"] = created_at
                    scout_stats["last_raw_turns_scanned"] = scanned
                    scout_stats["last_distilled_tokens"] = distilled
                    scout_stats["last_compression_ratio"] = round(ratio, 2)

            elif role == "archivist":
                archivist_stats["runs"] += 1
                archivist_stats["dropped_micro"] += _safe_int(metrics.get("dropped_micro"))
                archivist_stats["compacted_events"] += _safe_int(metrics.get("compacted_events"))
                archivist_stats["archived_cards"] += _safe_int(metrics.get("archived_cards"))
                archivist_stats["hibernated_cards"] += _safe_int(metrics.get("hibernated_cards"))
                archivist_stats["superseded_cards"] += _safe_int(metrics.get("superseded_cards"))
                archivist_stats["revoked_cards"] += _safe_int(metrics.get("revoked_cards"))
                archivist_stats["backfilled_cards"] += _safe_int(metrics.get("backfilled_cards"))
                archivist_stats["prompt_tokens"] += _safe_int(metrics.get("prompt_tokens"))
                archivist_stats["completion_tokens"] += _safe_int(metrics.get("completion_tokens"))
                if archivist_stats["last_run_at"] is None:
                    archivist_stats["last_run_at"] = created_at

        if scout_ratios:
            scout_stats["avg_compression_ratio"] = round(sum(scout_ratios) / len(scout_ratios), 2)
        elif scout_stats["scanned_raw_tokens"] > 0 and scout_stats["distilled_tokens"] > 0:
            scout_stats["avg_compression_ratio"] = round(
                scout_stats["scanned_raw_tokens"] / scout_stats["distilled_tokens"],
                2,
            )

        runner_model = getattr(self._runner, "model", None) if self._runner is not None else None
        runner_type = type(self._runner).__name__ if self._runner is not None else None

        raw_sec = status_info.get("security")
        if not isinstance(raw_sec, Mapping):
            sec_fn = getattr(self._engine, "security_summary", None)
            raw_sec = sec_fn() if callable(sec_fn) else {}
        chain_info = raw_sec.get("audit_chain") if isinstance(raw_sec, Mapping) else None
        chain_ok = bool(chain_info.get("ok", True)) if isinstance(chain_info, Mapping) else True
        chain_checked = _safe_int(chain_info.get("checked")) if isinstance(chain_info, Mapping) else 0
        chain_broken_seq = chain_info.get("broken_seq") if isinstance(chain_info, Mapping) else None
        chain_status = "VERIFIED" if chain_ok else f"BROKEN@{chain_broken_seq}"
        seals_info = raw_sec.get("card_seals") if isinstance(raw_sec, Mapping) else None
        seals_ok = bool(seals_info.get("ok", True)) if isinstance(seals_info, Mapping) else True
        seals_checked = _safe_int(seals_info.get("checked")) if isinstance(seals_info, Mapping) else 0

        security_stats: dict[str, Any] = {
            "redactions": _safe_int(raw_sec.get("redactions")) if isinstance(raw_sec, Mapping) else 0,
            "redacted_secrets": _safe_int(raw_sec.get("redacted_secrets")) if isinstance(raw_sec, Mapping) else 0,
            "blindfolded_paths": _safe_int(raw_sec.get("blindfolded_paths")) if isinstance(raw_sec, Mapping) else 0,
            "sanitized_turns": _safe_int(raw_sec.get("sanitized_turns")) if isinstance(raw_sec, Mapping) else 0,
            "poison_blocked": _safe_int(raw_sec.get("poison_blocked")) if isinstance(raw_sec, Mapping) else 0,
            "pending_l1": _safe_int(raw_sec.get("pending_l1")) if isinstance(raw_sec, Mapping) else 0,
            "chain_ok": chain_ok,
            "chain_checked": chain_checked,
            "chain_status": chain_status,
            "seals_ok": seals_ok,
            "seals_checked": seals_checked,
        }

        economics_stats: dict[str, Any] = {
            "flash_cost_usd": "0.0000000",
            "gross_saved_usd": "0.0000000",
            "net_roi_usd": "0.0000000",
            "roi_multiple": "0.00x",
            "tax_alert": False,
            "entries_count": 0,
        }
        try:
            from reiji_maigo.plugins.roi_ledger import RoiLedger

            ledger = RoiLedger(self._engine.db)
            ledger.sync_from_swarm_telemetry()
            economics_stats = dict(ledger.summary())
        except Exception:  # noqa: BLE001 — optional plugin telemetry fallback
            pass

        cold_vault_stats: dict[str, Any] = {
            "vault_dir": str(self._engine.home / "epoch_vault"),
            "epochs_count": 0,
            "archived_cards_count": 0,
            "archived_turns_count": 0,
            "cold_stubs_count": 0,
        }
        try:
            from reiji_maigo.plugins.epoch_vault import (
                EpochColdVault,
                resolve_epoch_vault_dir,
            )

            vault_dir = resolve_epoch_vault_dir(self._engine)
            cold_vault_stats = dict(
                EpochColdVault(vault_dir).vault_summary(self._engine)
            )
        except Exception:  # noqa: BLE001 — optional plugin telemetry fallback
            pass

        return {
            "home": status_info.get("home"),
            "db_path": status_info.get("db_path"),
            "l0_cache": {
                "locked": True,
                "status": "100% LOCKED",
                "prefixes": {
                    "scribe": compute_prefix_sha256(SCRIBE_L0_PREFIX)[:12],
                    "scout": compute_prefix_sha256(SCOUT_L0_PREFIX)[:12],
                    "archivist": compute_prefix_sha256(ARCHIVIST_L0_PREFIX)[:12],
                },
            },
            "runner": {
                "configured": self._runner is not None,
                "status": "configured" if self._runner is not None else "unconfigured",
                "source": self._runner_source,
                "type": runner_type,
                "model": runner_model,
                "async_scribe": self._async_worker is not None,
            },
            "storage": counts,
            "active_pool": {
                "visible": _safe_int(counts.get("pool_visible")),
                "total": _safe_int(counts.get("pool_total")),
                "watermark": watermark,
                "undigested_turns": undigested_count,
                "event_notes": event_notes_count,
                "cards_active": _safe_int(counts.get("cards_active")),
                "cards_total": _safe_int(counts.get("cards_total")),
                "pending_backfills": _safe_int(counts.get("pending_backfills")),
            },
            "scribe": scribe_stats,
            "scout": scout_stats,
            "archivist": archivist_stats,
            "security": security_stats,
            "economics": economics_stats,
            "cold_vault": cold_vault_stats,
            "telemetry_total": telemetry_total,
        }

    def render_pulse_banner(self) -> str:
        """Render the industrial-grade swarm pulse telemetry banner."""
        summary = self.pulse_summary()
        l0 = summary["l0_cache"]
        runner_info = summary["runner"]
        pool = summary["active_pool"]
        storage = summary["storage"]
        scribe = summary["scribe"]
        scout = summary["scout"]
        archivist = summary["archivist"]
        security = summary["security"]
        economics = summary.get("economics") or {}
        cold_vault = summary.get("cold_vault") or {}

        raw_mult = economics.get("roi_multiple")
        if isinstance(raw_mult, (int, float)):
            roi_mult_str = f"{raw_mult:.2f}x"
        elif isinstance(raw_mult, str) and raw_mult.strip():
            roi_mult_str = raw_mult.strip()
            if not roi_mult_str.lower().endswith("x"):
                try:
                    roi_mult_str = f"{float(roi_mult_str):.2f}x"
                except ValueError:
                    pass
        else:
            roi_mult_str = "N/A"

        cv_epochs = _safe_int(
            cold_vault.get("epochs_count", cold_vault.get("epochs", 0))
        )
        cv_cards = _safe_int(
            cold_vault.get("archived_cards_count", cold_vault.get("cards_count", 0))
        )
        cv_turns = _safe_int(
            cold_vault.get("archived_turns_count", cold_vault.get("turns_count", 0))
        )
        cv_stubs = _safe_int(
            cold_vault.get(
                "cold_stubs_count",
                cold_vault.get("hot_cold_stubs", cold_vault.get("cold_stubs", 0)),
            )
        )

        if runner_info["configured"]:
            model_tag = runner_info["model"] or runner_info["type"] or "custom"
            runner_label = f"runner={runner_info['source']}({model_tag})"
        else:
            runner_label = "runner=unconfigured"

        mode_label = "async" if runner_info["async_scribe"] else "sync"
        prefixes = l0["prefixes"]

        lines = [
            f"[REIJI-MAIGO SWARM PULSE] L0 Cache: {l0['status']} | {runner_label} | mode={mode_label}",
            (
                f"  Prefixes    : scribe={prefixes['scribe']} "
                f"scout={prefixes['scout']} archivist={prefixes['archivist']}"
            ),
            (
                f"  Active Pool : visible={pool['visible']}/{pool['total']} | "
                f"watermark={pool['watermark']} | undigested={pool['undigested_turns']} | "
                f"notes={pool['event_notes']} | turns={_safe_int(storage.get('raw_turns'))} | "
                f"cards={pool['cards_active']}/{pool['cards_total']} | "
                f"backfills={pool['pending_backfills']}"
            ),
            (
                f"  Scribe      : runs={scribe['runs']} | digested={scribe['turns_digested']} turns | "
                f"+{scribe['created_events']}/~{scribe['updated_events']} events | "
                f"banter_filtered={scribe['skipped_banter']}"
            ),
            (
                f"  Scout       : runs={scout['runs']} | "
                f"{scout['raw_turns_scanned']} raw turns -> {scout['distilled_tokens']} distilled tok "
                f"(ratio={scout['avg_compression_ratio']:.2f}x, abstained={scout['abstained_runs']})"
            ),
            (
                f"  Archivist   : runs={archivist['runs']} | "
                f"revoked={archivist['revoked_cards']}/archived={archivist['archived_cards']}/"
                f"compacted={archivist['compacted_events']} | "
                f"hibernated={archivist['hibernated_cards']} | "
                f"superseded={archivist['superseded_cards']} | "
                f"backfilled={archivist['backfilled_cards']} | "
                f"dropped_micro={archivist['dropped_micro']}"
            ),
            (
                f"  Security    : redactions={security['redactions']} | "
                f"poison_blocked={security['poison_blocked']} | "
                f"pending_l1={security['pending_l1']} | "
                f"chain={security['chain_status']}({security['chain_checked']})"
            ),
            (
                f"  Economics   : flash=${economics.get('flash_cost_usd', '0.0000000')} | "
                f"saved=${economics.get('gross_saved_usd', '0.0000000')} | "
                f"net=${economics.get('net_roi_usd', '0.0000000')} | "
                f"roi={roi_mult_str} | "
                f"tax_alert={'ALERT' if economics.get('tax_alert') else 'ok'} | "
                f"entries={_safe_int(economics.get('entries_count'))}"
            ),
            (
                f"  Cold Vault  : epochs={cv_epochs} | "
                f"cards={cv_cards} | "
                f"turns={cv_turns} | "
                f"stubs={cv_stubs}"
            ),
        ]
        return "\n".join(lines)


__all__ = [
    "SwarmCoordinator",
]
