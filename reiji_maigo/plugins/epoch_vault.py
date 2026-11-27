"""Append-only SHA-256 verified epoch cold vault plugin for ``reiji-maigo``.

Provides:
- ``EpochColdVault``: append-only external epoch archive with ``.staging`` atomic
  commits, two-way SHA-256 verification on disk read-back, SQLite WAL index
  (``epoch_index.db``), hot-store tombstoning with HMAC seal refresh, and
  lossless verified card restoration. Exposes **no** API to delete committed
  epochs.
- ``EpochVaultPlugin`` & ``create_plugin``: microkernel plugin registering the
  ``epoch`` CLI command (``scan`` | ``seal`` | ``verify`` | ``restore``).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import threading
from typing import Any, Iterator

from reiji_maigo.core.contracts import MemoryCard
from reiji_maigo.core.kernel import ReijiKernel, resolve_reiji_home

ENV_EPOCH_VAULT_DIR: str = "REIJI_MAIGO_EPOCH_VAULT_DIR"
COLD_STUB_KEY: str = "_cold_stub"
COLD_EPOCH_ID_KEY: str = "_cold_epoch_id"
COLD_PAYLOAD_SHA256_KEY: str = "_payload_sha256"

ELIGIBLE_CARD_STATUSES: tuple[str, ...] = ("superseded", "revoked")
INACTIVE_POOL_STATUSES: frozenset[str] = frozenset({"archived", "dropped"})
VAULT_SCHEMA_VERSION: int = 1
DEFAULT_OLDER_THAN_DAYS: int = 30
DEFAULT_WATERMARK_KEY: str = "pool_digested"

_INDEX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS epochs (
    epoch_id TEXT PRIMARY KEY,
    sealed_at TEXT NOT NULL,
    older_than_days INTEGER NOT NULL,
    watermark_turn_id INTEGER NOT NULL DEFAULT 0,
    card_count INTEGER NOT NULL DEFAULT 0,
    turn_count INTEGER NOT NULL DEFAULT 0,
    cards_sha256 TEXT NOT NULL,
    turns_sha256 TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    root_sha256 TEXT NOT NULL,
    rel_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS epoch_cards (
    card_id TEXT NOT NULL,
    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
    domain TEXT NOT NULL,
    card_type TEXT NOT NULL,
    title TEXT NOT NULL,
    lifecycle_status TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    sealed_at TEXT NOT NULL,
    PRIMARY KEY (card_id, epoch_id)
);

CREATE INDEX IF NOT EXISTS idx_epoch_cards_card_id
    ON epoch_cards(card_id, sealed_at DESC);

CREATE TABLE IF NOT EXISTS epoch_turns (
    turn_id INTEGER NOT NULL,
    epoch_id TEXT NOT NULL REFERENCES epochs(epoch_id),
    episode_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    line_no INTEGER NOT NULL,
    PRIMARY KEY (turn_id, epoch_id)
);

CREATE INDEX IF NOT EXISTS idx_epoch_turns_turn_id
    ON epoch_turns(turn_id);
"""


class EpochVerificationError(ValueError):
    """Raised when disk read-back or cold vault SHA-256 verification fails."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_utc(raw: str | None) -> datetime | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compute_root_sha256(manifest_body: Mapping[str, Any]) -> str:
    core = {
        "schema_version": int(manifest_body["schema_version"]),
        "epoch_id": str(manifest_body["epoch_id"]),
        "sealed_at": str(manifest_body["sealed_at"]),
        "older_than_days": int(manifest_body["older_than_days"]),
        "watermark_turn_id": int(manifest_body["watermark_turn_id"]),
        "card_count": int(manifest_body["card_count"]),
        "turn_count": int(manifest_body["turn_count"]),
        "card_ids": list(manifest_body["card_ids"]),
        "turn_ids": [int(tid) for tid in manifest_body["turn_ids"]],
        "files": manifest_body["files"],
    }
    return _sha256_bytes(_canonical_json_bytes(core))


def _write_file_fsync(path: Path, data: bytes) -> None:
    with path.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _safe_parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_spans_from_json(raw: str | None) -> list[tuple[str, int, int]]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    spans: list[tuple[str, int, int]] = []
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        ep = item.get("episode_id")
        start = item.get("turn_start")
        end = item.get("turn_end")
        if (
            isinstance(ep, str)
            and ep.strip()
            and not isinstance(start, bool)
            and not isinstance(end, bool)
            and isinstance(start, int)
            and isinstance(end, int)
        ):
            spans.append((ep, start, end))
    return spans


def _validate_older_than_days(older_than_days: int) -> int:
    if isinstance(older_than_days, bool) or not isinstance(older_than_days, int):
        raise TypeError("older_than_days must be an int")
    if older_than_days < 0:
        raise ValueError("older_than_days must be >= 0")
    return older_than_days


def resolve_epoch_vault_dir(
    engine: Any | None = None,
    *,
    vault_dir: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the epoch cold vault directory from argument, env, or engine home."""
    if vault_dir is not None:
        return Path(vault_dir).expanduser().resolve()
    environ = os.environ if env is None else env
    raw = (environ.get(ENV_EPOCH_VAULT_DIR) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if engine is not None and getattr(engine, "home", None) is not None:
        return (Path(engine.home) / "epoch_vault").resolve()
    return (resolve_reiji_home(env=environ) / "epoch_vault").resolve()


def _resolve_effective_watermark(conn: sqlite3.Connection, watermark_key: str | None = None) -> int:
    if watermark_key is not None:
        row = conn.execute(
            "SELECT turn_id FROM meta_watermarks WHERE key = ?",
            (watermark_key,),
        ).fetchone()
        return int(row["turn_id"]) if row is not None else 0

    rows = conn.execute("SELECT key, turn_id FROM meta_watermarks").fetchall()
    if not rows:
        return 0

    positive = {str(r["key"]): int(r["turn_id"]) for r in rows if int(r["turn_id"]) > 0}
    if not positive:
        return 0
    if DEFAULT_WATERMARK_KEY in positive and len(positive) == 1:
        return positive[DEFAULT_WATERMARK_KEY]
    return min(positive.values())


def _is_turn_protected(
    episode_id: str,
    turn_id: int,
    protected_spans: dict[str, list[tuple[int, int]]],
) -> bool:
    ranges = protected_spans.get(episode_id)
    if not ranges:
        return False
    for start, end in ranges:
        if start <= turn_id <= end:
            return True
    return False


def _build_jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    if not records:
        return b""
    lines = [_canonical_json_bytes(rec) for rec in records]
    return b"\n".join(lines) + b"\n"


class EpochColdVault:
    """Append-only, SHA-256 verified cold vault for expired cards and stale turns.

    Layout under ``vault_dir``:
      - ``epochs/<epoch_id>/cards.jsonl``
      - ``epochs/<epoch_id>/turns.jsonl``
      - ``epochs/<epoch_id>/manifest.json``
      - ``epoch_index.db`` (SQLite WAL mode)

    By design, this class exposes **no** API to delete or truncate committed
    epochs.
    """

    def __init__(self, vault_dir: Path | str) -> None:
        self.vault_dir: Path = Path(vault_dir).expanduser().resolve()
        self.epochs_dir: Path = self.vault_dir / "epochs"
        self.staging_dir: Path = self.vault_dir / ".staging"
        self.index_db_path: Path = self.vault_dir / "epoch_index.db"
        self._lock = threading.RLock()
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        with self._lock:
            self.vault_dir.mkdir(parents=True, exist_ok=True)
            self.epochs_dir.mkdir(parents=True, exist_ok=True)
            with self._connect_index() as conn:
                conn.executescript(_INDEX_SCHEMA_SQL)

    @contextmanager
    def _connect_index(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.index_db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Candidate scanning
    # ------------------------------------------------------------------
    def scan_candidates(
        self,
        engine: Any,
        *,
        older_than_days: int = DEFAULT_OLDER_THAN_DAYS,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Scan hot storage for expired inactive cards and unreferenced stale turns."""
        days = _validate_older_than_days(older_than_days)
        now_dt = _utc_now() if now_iso is None else _parse_iso_utc(now_iso)
        if now_dt is None:
            raise ValueError(f"Invalid now_iso timestamp: {now_iso!r}")
        now_str = now_dt.isoformat()
        cutoff_dt = now_dt - timedelta(days=days)

        candidate_cards: list[dict[str, Any]] = []
        candidate_turns: list[dict[str, Any]] = []
        already_stubbed_cards = 0
        protected_turns_skipped = 0

        with engine.db.read_connection() as conn:
            # 1. Eligible superseded / revoked cards
            card_rows = conn.execute(
                """
                SELECT card_id, domain, card_type, title, state_json, lifecycle_status,
                       valid_from, valid_to, superseded_by, spans_json,
                       foresight_note, foresight_until, created_at, updated_at
                FROM memory_cards
                WHERE lifecycle_status IN ('superseded', 'revoked')
                ORDER BY card_id ASC
                """
            ).fetchall()

            for row in card_rows:
                state_raw = str(row["state_json"])
                state = _safe_parse_json_dict(state_raw)
                if state.get(COLD_STUB_KEY) is True:
                    already_stubbed_cards += 1
                    continue

                card_ts = (
                    _parse_iso_utc(row["updated_at"])
                    or _parse_iso_utc(row["valid_to"])
                    or _parse_iso_utc(row["created_at"])
                    or _parse_iso_utc(row["valid_from"])
                )
                if card_ts is None or card_ts > cutoff_dt:
                    continue

                candidate_cards.append(
                    {
                        "card_id": str(row["card_id"]),
                        "domain": str(row["domain"]),
                        "card_type": str(row["card_type"]),
                        "title": str(row["title"]),
                        "lifecycle_status": str(row["lifecycle_status"]),
                        "updated_at": str(row["updated_at"]),
                        "valid_to": row["valid_to"],
                        "state_bytes": len(state_raw.encode("utf-8")),
                    }
                )

            # 2. Protected turn ranges from active cards, active pool, and pending backfills
            protected_spans: dict[str, list[tuple[int, int]]] = {}

            active_card_rows = conn.execute(
                """
                SELECT spans_json
                FROM memory_cards
                WHERE lifecycle_status NOT IN ('superseded', 'revoked')
                """
            ).fetchall()
            for row in active_card_rows:
                for ep, start, end in _extract_spans_from_json(row["spans_json"]):
                    protected_spans.setdefault(ep, []).append((start, end))

            pool_rows = conn.execute(
                "SELECT status, spans_json FROM pool_events"
            ).fetchall()
            for row in pool_rows:
                if str(row["status"]).strip().lower() in INACTIVE_POOL_STATUSES:
                    continue
                for ep, start, end in _extract_spans_from_json(row["spans_json"]):
                    protected_spans.setdefault(ep, []).append((start, end))

            backfill_rows = conn.execute(
                """
                SELECT episode_id, turn_start, turn_end
                FROM pending_backfills
                WHERE status = 'pending'
                """
            ).fetchall()
            for row in backfill_rows:
                protected_spans.setdefault(str(row["episode_id"]), []).append(
                    (int(row["turn_start"]), int(row["turn_end"]))
                )

            # 3. Stale digested turns (turn_id <= watermark and not protected)
            watermark_turn_id = _resolve_effective_watermark(conn)
            if watermark_turn_id > 0:
                turn_rows = conn.execute(
                    """
                    SELECT turn_id, episode_id, role, content, created_at
                    FROM raw_turns
                    WHERE turn_id <= ?
                    ORDER BY turn_id ASC
                    """,
                    (watermark_turn_id,),
                ).fetchall()

                for row in turn_rows:
                    turn_id = int(row["turn_id"])
                    episode_id = str(row["episode_id"])
                    created_at = str(row["created_at"])
                    turn_ts = _parse_iso_utc(created_at)
                    if turn_ts is None or turn_ts > cutoff_dt:
                        continue
                    if _is_turn_protected(episode_id, turn_id, protected_spans):
                        protected_turns_skipped += 1
                        continue
                    content = str(row["content"])
                    candidate_turns.append(
                        {
                            "turn_id": turn_id,
                            "episode_id": episode_id,
                            "role": str(row["role"]),
                            "created_at": created_at,
                            "content_bytes": len(content.encode("utf-8")),
                        }
                    )

        return {
            "now_iso": now_str,
            "older_than_days": days,
            "cutoff_iso": cutoff_dt.isoformat(),
            "watermark_turn_id": watermark_turn_id,
            "candidate_cards": candidate_cards,
            "candidate_turns": candidate_turns,
            "card_ids": [item["card_id"] for item in candidate_cards],
            "turn_ids": [int(item["turn_id"]) for item in candidate_turns],
            "card_count": len(candidate_cards),
            "turn_count": len(candidate_turns),
            "already_stubbed_cards": already_stubbed_cards,
            "protected_turns_skipped": protected_turns_skipped,
        }

    # ------------------------------------------------------------------
    # Two-way disk verification helper
    # ------------------------------------------------------------------
    def _verify_epoch_dir_from_disk(
        self,
        epoch_dir: Path,
        *,
        expected_cards_sha256: str | None = None,
        expected_turns_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_root_sha256: str | None = None,
        expected_card_hashes: Mapping[str, str] | None = None,
        expected_turn_hashes: Mapping[int, str] | None = None,
    ) -> dict[str, Any]:
        manifest_path = epoch_dir / "manifest.json"
        cards_path = epoch_dir / "cards.jsonl"
        turns_path = epoch_dir / "turns.jsonl"

        for required in (manifest_path, cards_path, turns_path):
            if not required.is_file():
                raise EpochVerificationError(f"Missing cold epoch artifact: {required.name}")

        manifest_bytes = manifest_path.read_bytes()
        manifest_sha = _sha256_bytes(manifest_bytes)
        if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
            raise EpochVerificationError(
                f"manifest.json SHA-256 mismatch: expected={expected_manifest_sha256}, actual={manifest_sha}"
            )

        try:
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EpochVerificationError(f"Corrupted manifest.json: {exc}") from exc
        if not isinstance(manifest, dict):
            raise EpochVerificationError("manifest.json root must be a JSON object")

        files_meta = manifest.get("files")
        if not isinstance(files_meta, Mapping):
            raise EpochVerificationError("manifest.json missing files metadata")
        cards_meta = files_meta.get("cards.jsonl")
        turns_meta = files_meta.get("turns.jsonl")
        if not isinstance(cards_meta, Mapping) or not isinstance(turns_meta, Mapping):
            raise EpochVerificationError("manifest.json missing cards.jsonl/turns.jsonl entries")

        cards_bytes = cards_path.read_bytes()
        turns_bytes = turns_path.read_bytes()
        actual_cards_sha = _sha256_bytes(cards_bytes)
        actual_turns_sha = _sha256_bytes(turns_bytes)

        if actual_cards_sha != cards_meta.get("sha256") or len(cards_bytes) != int(cards_meta.get("bytes", -1)):
            raise EpochVerificationError("cards.jsonl disk read-back SHA-256/size mismatch")
        if expected_cards_sha256 is not None and actual_cards_sha != expected_cards_sha256:
            raise EpochVerificationError("cards.jsonl SHA-256 differs from expected commit digest")

        if actual_turns_sha != turns_meta.get("sha256") or len(turns_bytes) != int(turns_meta.get("bytes", -1)):
            raise EpochVerificationError("turns.jsonl disk read-back SHA-256/size mismatch")
        if expected_turns_sha256 is not None and actual_turns_sha != expected_turns_sha256:
            raise EpochVerificationError("turns.jsonl SHA-256 differs from expected commit digest")

        recomputed_root = _compute_root_sha256(manifest)
        if recomputed_root != manifest.get("root_sha256"):
            raise EpochVerificationError("manifest.json root_sha256 verification failed")
        if expected_root_sha256 is not None and recomputed_root != expected_root_sha256:
            raise EpochVerificationError("manifest root_sha256 differs from expected commit root")

        # Parse & validate every JSONL line
        card_records: list[dict[str, Any]] = []
        raw_card_lines = [line for line in cards_bytes.decode("utf-8").splitlines() if line]
        if len(raw_card_lines) != int(manifest.get("card_count", -1)):
            raise EpochVerificationError("cards.jsonl line count mismatch")
        disk_card_hashes: dict[str, str] = {}
        for line in raw_card_lines:
            rec = json.loads(line)
            if not isinstance(rec, dict) or not isinstance(rec.get("state"), dict):
                raise EpochVerificationError("Malformed card record in cards.jsonl")
            cid = str(rec["card_id"])
            computed_payload_sha = _sha256_bytes(_canonical_json_bytes(rec["state"]))
            if computed_payload_sha != rec.get("payload_sha256"):
                raise EpochVerificationError(f"Card payload SHA-256 mismatch for {cid!r}")
            disk_card_hashes[cid] = computed_payload_sha
            card_records.append(rec)

        if expected_card_hashes is not None and disk_card_hashes != dict(expected_card_hashes):
            raise EpochVerificationError("Disk cards.jsonl payload hashes differ from expected set")

        turn_records: list[dict[str, Any]] = []
        raw_turn_lines = [line for line in turns_bytes.decode("utf-8").splitlines() if line]
        if len(raw_turn_lines) != int(manifest.get("turn_count", -1)):
            raise EpochVerificationError("turns.jsonl line count mismatch")
        disk_turn_hashes: dict[int, str] = {}
        for line in raw_turn_lines:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise EpochVerificationError("Malformed turn record in turns.jsonl")
            tid = int(rec["turn_id"])
            turn_body = {
                "turn_id": tid,
                "episode_id": str(rec["episode_id"]),
                "role": str(rec["role"]),
                "content": str(rec["content"]),
                "created_at": str(rec["created_at"]),
            }
            computed_turn_sha = _sha256_bytes(_canonical_json_bytes(turn_body))
            if computed_turn_sha != rec.get("payload_sha256"):
                raise EpochVerificationError(f"Turn payload SHA-256 mismatch for turn_id={tid}")
            disk_turn_hashes[tid] = computed_turn_sha
            turn_records.append(rec)

        if expected_turn_hashes is not None and disk_turn_hashes != dict(expected_turn_hashes):
            raise EpochVerificationError("Disk turns.jsonl payload hashes differ from expected set")

        return {
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "cards_sha256": actual_cards_sha,
            "turns_sha256": actual_turns_sha,
            "root_sha256": recomputed_root,
            "card_records": card_records,
            "turn_records": turn_records,
        }

    # ------------------------------------------------------------------
    # Seal epoch (dry-run preview or atomic commit)
    # ------------------------------------------------------------------
    def seal_epoch(
        self,
        engine: Any,
        *,
        older_than_days: int = DEFAULT_OLDER_THAN_DAYS,
        now_iso: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Preview or atomically seal eligible cards and turns into a new epoch."""
        scan = self.scan_candidates(
            engine,
            older_than_days=older_than_days,
            now_iso=now_iso,
        )
        if not confirm:
            return {
                "ok": True,
                "dry_run": True,
                "sealed": False,
                "epoch_id": None,
                "card_count": scan["card_count"],
                "turn_count": scan["turn_count"],
                "card_ids": list(scan["card_ids"]),
                "turn_ids": list(scan["turn_ids"]),
                "scan": scan,
            }

        if scan["card_count"] == 0 and scan["turn_count"] == 0:
            return {
                "ok": True,
                "dry_run": False,
                "sealed": False,
                "reason": "no_candidates",
                "epoch_id": None,
                "card_count": 0,
                "turn_count": 0,
                "card_ids": [],
                "turn_ids": [],
                "scan": scan,
            }

        sealed_at = str(scan["now_iso"])
        sealed_dt = _parse_iso_utc(sealed_at) or _utc_now()
        epoch_id = f"ep_{sealed_dt.strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(4)}"

        card_ids: list[str] = list(scan["card_ids"])
        turn_ids: list[int] = list(scan["turn_ids"])

        # Collect full payloads from hot store
        card_records: list[dict[str, Any]] = []
        expected_card_hashes: dict[str, str] = {}
        turn_records: list[dict[str, Any]] = []
        expected_turn_hashes: dict[int, str] = {}

        with engine.db.read_connection() as conn:
            for cid in card_ids:
                card = engine.cards.get_card(cid)
                if card is None:
                    continue
                meta_row = conn.execute(
                    "SELECT created_at, updated_at FROM memory_cards WHERE card_id = ?",
                    (cid,),
                ).fetchone()
                payload_sha = _sha256_bytes(_canonical_json_bytes(card.state))
                expected_card_hashes[cid] = payload_sha
                card_records.append(
                    {
                        **card.to_dict(),
                        "created_at": str(meta_row["created_at"]) if meta_row else "",
                        "updated_at": str(meta_row["updated_at"]) if meta_row else "",
                        "payload_sha256": payload_sha,
                    }
                )

            for tid in turn_ids:
                trow = conn.execute(
                    """
                    SELECT turn_id, episode_id, role, content, created_at
                    FROM raw_turns
                    WHERE turn_id = ?
                    """,
                    (tid,),
                ).fetchone()
                if trow is None:
                    continue
                turn_body = {
                    "turn_id": int(trow["turn_id"]),
                    "episode_id": str(trow["episode_id"]),
                    "role": str(trow["role"]),
                    "content": str(trow["content"]),
                    "created_at": str(trow["created_at"]),
                }
                turn_sha = _sha256_bytes(_canonical_json_bytes(turn_body))
                expected_turn_hashes[int(trow["turn_id"])] = turn_sha
                turn_records.append({**turn_body, "payload_sha256": turn_sha})

        actual_card_ids = [rec["card_id"] for rec in card_records]
        actual_turn_ids = [int(rec["turn_id"]) for rec in turn_records]

        cards_bytes = _build_jsonl_bytes(card_records)
        turns_bytes = _build_jsonl_bytes(turn_records)
        cards_sha = _sha256_bytes(cards_bytes)
        turns_sha = _sha256_bytes(turns_bytes)

        manifest: dict[str, Any] = {
            "schema_version": VAULT_SCHEMA_VERSION,
            "epoch_id": epoch_id,
            "sealed_at": sealed_at,
            "older_than_days": int(scan["older_than_days"]),
            "watermark_turn_id": int(scan["watermark_turn_id"]),
            "card_count": len(card_records),
            "turn_count": len(turn_records),
            "card_ids": actual_card_ids,
            "turn_ids": actual_turn_ids,
            "files": {
                "cards.jsonl": {
                    "sha256": cards_sha,
                    "bytes": len(cards_bytes),
                    "lines": len(card_records),
                },
                "turns.jsonl": {
                    "sha256": turns_sha,
                    "bytes": len(turns_bytes),
                    "lines": len(turn_records),
                },
            },
        }
        root_sha = _compute_root_sha256(manifest)
        manifest["root_sha256"] = root_sha
        manifest_bytes = _canonical_json_bytes(manifest) + b"\n"
        manifest_sha = _sha256_bytes(manifest_bytes)

        with self._lock:
            self._ensure_layout()
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            stage_epoch_dir = self.staging_dir / f"{epoch_id}_{secrets.token_hex(4)}"
            final_epoch_dir = self.epochs_dir / epoch_id
            if final_epoch_dir.exists():
                raise FileExistsError(f"Committed epoch already exists: {epoch_id!r}")

            # Step 1: Write into .staging with fsync, then atomic os.replace
            stage_epoch_dir.mkdir(parents=True, exist_ok=False)
            try:
                _write_file_fsync(stage_epoch_dir / "cards.jsonl", cards_bytes)
                _write_file_fsync(stage_epoch_dir / "turns.jsonl", turns_bytes)
                _write_file_fsync(stage_epoch_dir / "manifest.json", manifest_bytes)
                os.replace(stage_epoch_dir, final_epoch_dir)
            except Exception:
                if stage_epoch_dir.exists():
                    shutil.rmtree(stage_epoch_dir, ignore_errors=True)
                raise

            # Step 2: Mandatory two-way disk read-back SHA-256 verification
            self._verify_epoch_dir_from_disk(
                final_epoch_dir,
                expected_cards_sha256=cards_sha,
                expected_turns_sha256=turns_sha,
                expected_manifest_sha256=manifest_sha,
                expected_root_sha256=root_sha,
                expected_card_hashes=expected_card_hashes,
                expected_turn_hashes=expected_turn_hashes,
            )

            # Step 3: Record immutable index entries in epoch_index.db
            rel_path = f"epochs/{epoch_id}"
            with self._connect_index() as idx_conn:
                idx_conn.execute(
                    """
                    INSERT INTO epochs (
                        epoch_id, sealed_at, older_than_days, watermark_turn_id,
                        card_count, turn_count, cards_sha256, turns_sha256,
                        manifest_sha256, root_sha256, rel_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        epoch_id,
                        sealed_at,
                        int(scan["older_than_days"]),
                        int(scan["watermark_turn_id"]),
                        len(card_records),
                        len(turn_records),
                        cards_sha,
                        turns_sha,
                        manifest_sha,
                        root_sha,
                        rel_path,
                    ),
                )
                for line_no, rec in enumerate(card_records, start=1):
                    idx_conn.execute(
                        """
                        INSERT INTO epoch_cards (
                            card_id, epoch_id, domain, card_type, title,
                            lifecycle_status, payload_sha256, line_no, sealed_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(rec["card_id"]),
                            epoch_id,
                            str(rec["domain"]),
                            str(rec["card_type"]),
                            str(rec["title"]),
                            str(rec["lifecycle_status"]),
                            str(rec["payload_sha256"]),
                            line_no,
                            sealed_at,
                        ),
                    )
                for line_no, rec in enumerate(turn_records, start=1):
                    idx_conn.execute(
                        """
                        INSERT INTO epoch_turns (
                            turn_id, epoch_id, episode_id, created_at, payload_sha256, line_no
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(rec["turn_id"]),
                            epoch_id,
                            str(rec["episode_id"]),
                            str(rec["created_at"]),
                            str(rec["payload_sha256"]),
                            line_no,
                        ),
                    )

        # Step 4: Only after verified cold commit, tombstone hot cards & prune archived turns
        with engine.db.transaction() as hot_conn:
            for rec in card_records:
                cid = str(rec["card_id"])
                stub_state = {
                    COLD_STUB_KEY: True,
                    COLD_EPOCH_ID_KEY: epoch_id,
                    COLD_PAYLOAD_SHA256_KEY: str(rec["payload_sha256"]),
                }
                hot_conn.execute(
                    """
                    UPDATE memory_cards
                    SET state_json = ?, updated_at = ?
                    WHERE card_id = ?
                    """,
                    (
                        json.dumps(stub_state, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        sealed_at,
                        cid,
                    ),
                )

            for tid in actual_turn_ids:
                hot_conn.execute("DELETE FROM raw_turns_fts WHERE rowid = ?", (tid,))
                hot_conn.execute("DELETE FROM raw_turns WHERE turn_id = ?", (tid,))

        # Step 5: Refresh HMAC seal for every tombstoned card
        for cid in actual_card_ids:
            engine.seal_card(cid, now_iso=sealed_at)

        # Step 6: Append tamper-evident audit entry
        audit_entry = engine.audit.append(
            "epoch_vault",
            "epoch_sealed",
            target_id=epoch_id,
            tier="L0",
            detail={
                "epoch_id": epoch_id,
                "card_count": len(actual_card_ids),
                "turn_count": len(actual_turn_ids),
                "card_ids": actual_card_ids,
                "root_sha256": root_sha,
                "cards_sha256": cards_sha,
                "turns_sha256": turns_sha,
                "older_than_days": int(scan["older_than_days"]),
            },
            now_iso=sealed_at,
        )

        return {
            "ok": True,
            "dry_run": False,
            "sealed": True,
            "epoch_id": epoch_id,
            "sealed_at": sealed_at,
            "epoch_dir": str(final_epoch_dir),
            "card_count": len(actual_card_ids),
            "turn_count": len(actual_turn_ids),
            "card_ids": actual_card_ids,
            "turn_ids": actual_turn_ids,
            "cards_sha256": cards_sha,
            "turns_sha256": turns_sha,
            "manifest_sha256": manifest_sha,
            "root_sha256": root_sha,
            "audit_seq": audit_entry["seq"],
        }

    # ------------------------------------------------------------------
    # Full vault verification
    # ------------------------------------------------------------------
    def verify_vault(self) -> dict[str, Any]:
        """Verify SHA-256 digests and index consistency for all committed epochs."""
        issues: list[dict[str, Any]] = []
        missing_epochs: list[str] = []
        unindexed_epochs: list[str] = []
        epochs_checked = 0
        cards_verified = 0
        turns_verified = 0

        with self._lock:
            if not self.index_db_path.is_file():
                return {
                    "ok": False,
                    "vault_dir": str(self.vault_dir),
                    "epochs_checked": 0,
                    "cards_verified": 0,
                    "turns_verified": 0,
                    "missing_epochs": [],
                    "unindexed_epochs": [],
                    "issues": [{"epoch_id": None, "reason": "index_db_missing"}],
                }

            with self._connect_index() as idx_conn:
                epoch_rows = idx_conn.execute(
                    "SELECT * FROM epochs ORDER BY sealed_at ASC, epoch_id ASC"
                ).fetchall()
                indexed_ids = {str(row["epoch_id"]) for row in epoch_rows}

                disk_ids: set[str] = set()
                if self.epochs_dir.is_dir():
                    for child in sorted(self.epochs_dir.iterdir()):
                        if child.is_dir():
                            disk_ids.add(child.name)

                for extra in sorted(disk_ids - indexed_ids):
                    unindexed_epochs.append(extra)
                    issues.append({"epoch_id": extra, "reason": "unindexed_epoch_directory"})

                for row in epoch_rows:
                    epoch_id = str(row["epoch_id"])
                    epoch_dir = self.epochs_dir / epoch_id
                    if not epoch_dir.is_dir():
                        missing_epochs.append(epoch_id)
                        issues.append({"epoch_id": epoch_id, "reason": "epoch_directory_missing"})
                        continue

                    card_index_rows = idx_conn.execute(
                        "SELECT card_id, payload_sha256 FROM epoch_cards WHERE epoch_id = ?",
                        (epoch_id,),
                    ).fetchall()
                    expected_card_hashes = {
                        str(r["card_id"]): str(r["payload_sha256"]) for r in card_index_rows
                    }

                    turn_index_rows = idx_conn.execute(
                        "SELECT turn_id, payload_sha256 FROM epoch_turns WHERE epoch_id = ?",
                        (epoch_id,),
                    ).fetchall()
                    expected_turn_hashes = {
                        int(r["turn_id"]): str(r["payload_sha256"]) for r in turn_index_rows
                    }

                    try:
                        verified = self._verify_epoch_dir_from_disk(
                            epoch_dir,
                            expected_cards_sha256=str(row["cards_sha256"]),
                            expected_turns_sha256=str(row["turns_sha256"]),
                            expected_manifest_sha256=str(row["manifest_sha256"]),
                            expected_root_sha256=str(row["root_sha256"]),
                            expected_card_hashes=expected_card_hashes,
                            expected_turn_hashes=expected_turn_hashes,
                        )
                        epochs_checked += 1
                        cards_verified += len(verified["card_records"])
                        turns_verified += len(verified["turn_records"])
                    except Exception as exc:  # noqa: BLE001 — collect all integrity issues
                        issues.append({"epoch_id": epoch_id, "reason": str(exc)})

        return {
            "ok": len(issues) == 0,
            "vault_dir": str(self.vault_dir),
            "epochs_checked": epochs_checked,
            "cards_verified": cards_verified,
            "turns_verified": turns_verified,
            "missing_epochs": missing_epochs,
            "unindexed_epochs": unindexed_epochs,
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Lossless verified card restoration
    # ------------------------------------------------------------------
    def restore_card(
        self,
        engine: Any,
        card_id: str,
        *,
        now_iso: str | None = None,
    ) -> MemoryCard:
        """Verify cold vault hashes and losslessly restore a tombstoned card into hot storage."""
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError("card_id must be a non-empty string")
        cid = card_id.strip()

        hot_card = engine.cards.get_card(cid)
        if hot_card is None:
            raise KeyError(f"Card not found in hot store: {cid!r}")

        stub_epoch_id: str | None = None
        stub_payload_sha: str | None = None
        if hot_card.state.get(COLD_STUB_KEY) is True:
            raw_ep = hot_card.state.get(COLD_EPOCH_ID_KEY)
            raw_sha = hot_card.state.get(COLD_PAYLOAD_SHA256_KEY)
            stub_epoch_id = str(raw_ep).strip() if raw_ep else None
            stub_payload_sha = str(raw_sha).strip() if raw_sha else None

        with self._lock:
            with self._connect_index() as idx_conn:
                if stub_epoch_id is not None:
                    ec_row = idx_conn.execute(
                        """
                        SELECT card_id, epoch_id, payload_sha256
                        FROM epoch_cards
                        WHERE card_id = ? AND epoch_id = ?
                        """,
                        (cid, stub_epoch_id),
                    ).fetchone()
                else:
                    ec_row = idx_conn.execute(
                        """
                        SELECT card_id, epoch_id, payload_sha256
                        FROM epoch_cards
                        WHERE card_id = ?
                        ORDER BY sealed_at DESC, epoch_id DESC
                        LIMIT 1
                        """,
                        (cid,),
                    ).fetchone()

                if ec_row is None:
                    raise KeyError(f"Card {cid!r} not found in epoch cold vault index")

                epoch_id = str(ec_row["epoch_id"])
                index_payload_sha = str(ec_row["payload_sha256"])
                if stub_payload_sha is not None and stub_payload_sha != index_payload_sha:
                    raise EpochVerificationError(
                        f"Hot tombstone SHA-256 differs from cold index for {cid!r}"
                    )

                ep_row = idx_conn.execute(
                    "SELECT * FROM epochs WHERE epoch_id = ?",
                    (epoch_id,),
                ).fetchone()
                if ep_row is None:
                    raise EpochVerificationError(f"Epoch metadata missing for {epoch_id!r}")

            epoch_dir = self.epochs_dir / epoch_id
            verified = self._verify_epoch_dir_from_disk(
                epoch_dir,
                expected_cards_sha256=str(ep_row["cards_sha256"]),
                expected_turns_sha256=str(ep_row["turns_sha256"]),
                expected_manifest_sha256=str(ep_row["manifest_sha256"]),
                expected_root_sha256=str(ep_row["root_sha256"]),
            )

            matched_record: dict[str, Any] | None = None
            for rec in verified["card_records"]:
                if str(rec.get("card_id")) == cid:
                    matched_record = rec
                    break
            if matched_record is None:
                raise EpochVerificationError(
                    f"Card {cid!r} missing from verified epoch {epoch_id!r} cards.jsonl"
                )

            restored_state = matched_record["state"]
            restored_sha = _sha256_bytes(_canonical_json_bytes(restored_state))
            if restored_sha != index_payload_sha:
                raise EpochVerificationError(
                    f"Restored payload SHA-256 mismatch for {cid!r}"
                )
            if stub_payload_sha is not None and restored_sha != stub_payload_sha:
                raise EpochVerificationError(
                    f"Restored payload SHA-256 does not match hot tombstone for {cid!r}"
                )

        ts = _utc_now_iso() if now_iso is None else str(now_iso)
        with engine.db.transaction() as hot_conn:
            hot_conn.execute(
                """
                UPDATE memory_cards
                SET state_json = ?, updated_at = ?
                WHERE card_id = ?
                """,
                (
                    json.dumps(
                        restored_state,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    ts,
                    cid,
                ),
            )

        engine.seal_card(cid, now_iso=ts)
        engine.audit.append(
            "epoch_vault",
            "epoch_card_restored",
            target_id=cid,
            tier="L0",
            detail={
                "card_id": cid,
                "epoch_id": epoch_id,
                "payload_sha256": restored_sha,
            },
            now_iso=ts,
        )

        restored_card = engine.cards.get_card(cid)
        if restored_card is None:
            raise KeyError(f"Card vanished after restore: {cid!r}")
        return restored_card

    # ------------------------------------------------------------------
    # Summary telemetry
    # ------------------------------------------------------------------
    def vault_summary(self, engine: Any | None = None) -> dict[str, Any]:
        """Return compact metrics for the epoch cold vault and optional hot tombstones."""
        with self._lock:
            epoch_count = 0
            sealed_cards = 0
            sealed_turns = 0
            latest_epoch_id: str | None = None
            latest_sealed_at: str | None = None

            if self.index_db_path.is_file():
                with self._connect_index() as idx_conn:
                    agg = idx_conn.execute(
                        """
                        SELECT
                            COUNT(*) AS epoch_count,
                            COALESCE(SUM(card_count), 0) AS sealed_cards,
                            COALESCE(SUM(turn_count), 0) AS sealed_turns
                        FROM epochs
                        """
                    ).fetchone()
                    if agg is not None:
                        epoch_count = int(agg["epoch_count"])
                        sealed_cards = int(agg["sealed_cards"])
                        sealed_turns = int(agg["sealed_turns"])
                    latest = idx_conn.execute(
                        "SELECT epoch_id, sealed_at FROM epochs ORDER BY sealed_at DESC, epoch_id DESC LIMIT 1"
                    ).fetchone()
                    if latest is not None:
                        latest_epoch_id = str(latest["epoch_id"])
                        latest_sealed_at = str(latest["sealed_at"])

            total_bytes = 0
            if self.epochs_dir.is_dir():
                for path in self.epochs_dir.rglob("*"):
                    if path.is_file():
                        total_bytes += path.stat().st_size

        hot_stub_cards: int | None = None
        if engine is not None:
            hot_stub_cards = 0
            with engine.db.read_connection() as conn:
                rows = conn.execute("SELECT state_json FROM memory_cards").fetchall()
                for row in rows:
                    state = _safe_parse_json_dict(row["state_json"])
                    if state.get(COLD_STUB_KEY) is True:
                        hot_stub_cards += 1

        return {
            "vault_dir": str(self.vault_dir),
            "exists": self.index_db_path.is_file(),
            "epoch_count": epoch_count,
            "sealed_cards": sealed_cards,
            "sealed_turns": sealed_turns,
            "total_bytes": total_bytes,
            "latest_epoch_id": latest_epoch_id,
            "latest_sealed_at": latest_sealed_at,
            "hot_stub_cards": hot_stub_cards,
        }


# ----------------------------------------------------------------------
# CLI handler & Microkernel Plugin
# ----------------------------------------------------------------------
def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _ensure_json_arg(parser: argparse.ArgumentParser) -> None:
    if "--json" not in getattr(parser, "_option_string_actions", {}):
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit machine-readable JSON output.",
        )


def _configure_epoch_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach scan|seal|verify|restore subcommands to the ``epoch`` CLI parser."""
    _ensure_json_arg(parser)
    parser.add_argument(
        "--vault-dir",
        default=None,
        help="Override epoch cold vault directory.",
    )
    parser.set_defaults(epoch_action="summary")

    sub = parser.add_subparsers(dest="epoch_action", metavar="<action>")

    p_scan = sub.add_parser("scan", help="Preview eligible expired cards and stale turns.")
    _ensure_json_arg(p_scan)
    p_scan.add_argument(
        "--older-than-days",
        type=int,
        default=DEFAULT_OLDER_THAN_DAYS,
        help=f"Minimum age in days (default: {DEFAULT_OLDER_THAN_DAYS}).",
    )
    p_scan.add_argument("--now-iso", default=None, help="Optional ISO-8601 UTC timestamp override.")
    p_scan.add_argument("--vault-dir", default=None, help="Override epoch cold vault directory.")

    p_seal = sub.add_parser("seal", help="Seal eligible cards and turns into an immutable epoch.")
    _ensure_json_arg(p_seal)
    p_seal.add_argument(
        "--older-than-days",
        type=int,
        default=DEFAULT_OLDER_THAN_DAYS,
        help=f"Minimum age in days (default: {DEFAULT_OLDER_THAN_DAYS}).",
    )
    p_seal.add_argument("--now-iso", default=None, help="Optional ISO-8601 UTC timestamp override.")
    p_seal.add_argument(
        "--confirm",
        action="store_true",
        help="Commit to cold vault and tombstone hot cards (without this flag, dry-run only).",
    )
    p_seal.add_argument("--vault-dir", default=None, help="Override epoch cold vault directory.")

    p_verify = sub.add_parser("verify", help="Verify SHA-256 integrity of all committed epochs.")
    _ensure_json_arg(p_verify)
    p_verify.add_argument("--vault-dir", default=None, help="Override epoch cold vault directory.")

    p_restore = sub.add_parser("restore", help="Verify and restore a tombstoned card into hot storage.")
    _ensure_json_arg(p_restore)
    p_restore.add_argument("card_id", help="Memory card ID to restore from the cold vault.")
    p_restore.add_argument("--now-iso", default=None, help="Optional ISO-8601 UTC timestamp override.")
    p_restore.add_argument("--vault-dir", default=None, help="Override epoch cold vault directory.")

    return parser


class _EpochCliHandler:
    """Callable CLI handler exposing ``configure_parser`` for ``reiji epoch``."""

    def __init__(self, plugin: EpochVaultPlugin) -> None:
        self._plugin = plugin

    def configure_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return _configure_epoch_parser(parser)

    def __call__(self, engine: Any, args: argparse.Namespace) -> int:
        parsed = args
        if not hasattr(args, "epoch_action"):
            sub_parser = argparse.ArgumentParser(prog="reiji epoch")
            self.configure_parser(sub_parser)
            extra = list(getattr(args, "extra_args", ()) or ())
            parsed = sub_parser.parse_args(extra)
            if getattr(args, "json", False):
                parsed.json = True

        action = getattr(parsed, "epoch_action", None) or "summary"
        vault_override = getattr(parsed, "vault_dir", None)
        vault = self._plugin.get_vault(engine, vault_dir=vault_override)
        as_json = bool(getattr(parsed, "json", False))

        if action == "summary":
            summary = vault.vault_summary(engine)
            if as_json:
                _emit_json(summary)
            else:
                print(
                    f"Epoch Cold Vault: {summary['vault_dir']} "
                    f"(epochs={summary['epoch_count']}, cards={summary['sealed_cards']}, "
                    f"turns={summary['sealed_turns']}, hot_stubs={summary['hot_stub_cards']})"
                )
            return 0

        if action == "scan":
            result = vault.scan_candidates(
                engine,
                older_than_days=int(getattr(parsed, "older_than_days", DEFAULT_OLDER_THAN_DAYS)),
                now_iso=getattr(parsed, "now_iso", None),
            )
            if as_json:
                _emit_json(result)
            else:
                print(
                    f"Epoch Scan (older_than={result['older_than_days']}d, "
                    f"watermark={result['watermark_turn_id']}): "
                    f"cards={result['card_count']}, turns={result['turn_count']}, "
                    f"already_stubbed={result['already_stubbed_cards']}"
                )
            return 0

        if action == "seal":
            result = vault.seal_epoch(
                engine,
                older_than_days=int(getattr(parsed, "older_than_days", DEFAULT_OLDER_THAN_DAYS)),
                now_iso=getattr(parsed, "now_iso", None),
                confirm=bool(getattr(parsed, "confirm", False)),
            )
            if as_json:
                _emit_json(result)
            else:
                if result.get("dry_run"):
                    print(
                        f"[Dry-Run] Eligible for epoch seal: "
                        f"cards={result['card_count']}, turns={result['turn_count']} "
                        f"(pass --confirm to commit)"
                    )
                elif result.get("sealed"):
                    print(
                        f"Sealed epoch {result['epoch_id']}: "
                        f"cards={result['card_count']}, turns={result['turn_count']}, "
                        f"root_sha256={result['root_sha256'][:16]}..."
                    )
                else:
                    print("No eligible candidates to seal.")
            return 0

        if action == "verify":
            result = vault.verify_vault()
            if as_json:
                _emit_json(result)
            else:
                status_text = "VERIFIED" if result["ok"] else "FAILED"
                print(
                    f"Epoch Cold Vault: {status_text} "
                    f"(epochs={result['epochs_checked']}, cards={result['cards_verified']}, "
                    f"turns={result['turns_verified']}, issues={len(result['issues'])})"
                )
            return 0 if result["ok"] else 1

        if action == "restore":
            card = vault.restore_card(
                engine,
                str(parsed.card_id),
                now_iso=getattr(parsed, "now_iso", None),
            )
            if as_json:
                _emit_json({"ok": True, "card": card.to_dict()})
            else:
                print(f"Restored card {card.card_id} ({card.domain}/{card.card_type}): {card.title}")
            return 0

        raise ValueError(f"Unsupported epoch action: {action!r}")


class EpochVaultPlugin:
    """Optional microkernel plugin wiring the ``epoch`` cold vault CLI command."""

    name: str = "epoch_vault"

    def __init__(self, *, vault_dir: Path | str | None = None) -> None:
        self._default_vault_dir = vault_dir
        self._cli_handler = _EpochCliHandler(self)

    def get_vault(
        self,
        engine: Any | None = None,
        *,
        vault_dir: Path | str | None = None,
    ) -> EpochColdVault:
        target = resolve_epoch_vault_dir(
            engine,
            vault_dir=vault_dir if vault_dir is not None else self._default_vault_dir,
        )
        return EpochColdVault(target)

    def register_cli(self, kernel: ReijiKernel) -> None:
        kernel.register_cli(
            "epoch",
            self._cli_handler,
            help_text="Scan, seal, verify, and restore immutable SHA-256 epoch cold vault snapshots.",
            plugin_name=self.name,
        )


def create_plugin(*, vault_dir: Path | str | None = None) -> EpochVaultPlugin:
    """Factory function for ``EpochVaultPlugin``."""
    return EpochVaultPlugin(vault_dir=vault_dir)


__all__ = [
    "COLD_EPOCH_ID_KEY",
    "COLD_PAYLOAD_SHA256_KEY",
    "COLD_STUB_KEY",
    "DEFAULT_OLDER_THAN_DAYS",
    "ELIGIBLE_CARD_STATUSES",
    "ENV_EPOCH_VAULT_DIR",
    "EpochColdVault",
    "EpochVaultPlugin",
    "EpochVerificationError",
    "VAULT_SCHEMA_VERSION",
    "create_plugin",
    "resolve_epoch_vault_dir",
]
