"""Idempotent, deduplicated cold-start historical importer plugin for ``reiji-maigo``.

Supports safe, sanitized bootstrap imports from:
  1. JSONL conversation transcripts (``import_transcript_jsonl``)
  2. Legacy read-only SQLite databases containing ``raw_turns`` / ``turns`` (``import_legacy_sqlite``)
  3. Markdown handoff/checkpoint directories with optional sealed seed ``MemoryCard`` minting
     (``import_checkpoints_dir``)

All imported turns are deduplicated via SHA-256 fingerprints stored in ``import_fingerprints``
and sanitized through ``engine.append_turn(..., sanitize=True)``. Binary media files and
filesystem reparse points / symlinks are skipped without dereferencing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any

from reiji_maigo.core.contracts import LOCKED_DOMAINS, SourceSpan, validate_domain
from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.security import can_mutate_authoritative_memory, sanitize_for_memory
from reiji_maigo.storage.migrations import get_schema_version

BINARY_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".heic",
        ".heif",
        ".avif",
        ".tif",
        ".tiff",
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".webm",
        ".flv",
        ".wmv",
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".ogg",
        ".aac",
        ".pdf",
        ".zip",
        ".gz",
        ".tar",
        ".7z",
        ".exe",
        ".dll",
        ".bin",
    }
)

_IMPORT_FINGERPRINTS_SQL = """
CREATE TABLE IF NOT EXISTS import_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_import_fingerprints_source
    ON import_fingerprints(source_kind, episode_id);
"""

_HEADING_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$")
_HEADING_H2_RE = re.compile(r"(?m)^(?=##\s+)")
_TAG_CLEAN_RE = re.compile(r"[^a-z0-9_\-\u4e00-\u9fff]+")
_FILE_ATTRIBUTE_REPARSE_POINT: int = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _is_binary_media_path(path: Path) -> bool:
    return path.suffix.lower() in BINARY_MEDIA_EXTENSIONS


def _is_reparse_or_symlink(path: Path) -> bool:
    """Detect symlinks, junctions, and Windows reparse points without following them."""
    try:
        if path.is_symlink():
            return True
    except OSError:
        return True

    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction):
        try:
            if is_junction():
                return True
        except OSError:
            return True

    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True

    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs and (attrs & _FILE_ATTRIBUTE_REPARSE_POINT):
        return True
    return False


def compute_import_fingerprint(
    source_kind: str,
    episode_id: str,
    source_seq: int | str,
    role: str,
    content: str,
) -> str:
    """Compute a deterministic SHA-256 deduplication fingerprint for an imported turn."""
    canonical = json.dumps(
        [
            str(source_kind),
            str(episode_id),
            str(source_seq),
            str(role),
            str(content),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _extract_text_content(raw_content: Any) -> str:
    """Normalize string or multimodal block lists into plain text, ignoring binary blocks."""
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").strip().lower()
                if item_type in {"image", "image_url", "input_image", "audio", "input_audio", "video", "file"}:
                    continue
                text_val = item.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    parts.append(text_val.strip())
                elif isinstance(item.get("content"), str) and item["content"].strip():
                    parts.append(item["content"].strip())
        return "\n".join(parts).strip()
    if isinstance(raw_content, dict):
        if isinstance(raw_content.get("text"), str):
            return raw_content["text"].strip()
        if isinstance(raw_content.get("content"), str):
            return raw_content["content"].strip()
    return ""


def _parse_transcript_record(
    record: Any,
    *,
    default_episode_id: str,
    explicit_episode_id: str | None,
) -> tuple[str, str, str, str | None] | None:
    """Extract ``(episode_id, role, content, created_at)`` from a JSONL transcript object."""
    if not isinstance(record, dict):
        return None

    payload = record
    if isinstance(record.get("message"), dict):
        payload = record["message"]

    raw_role = payload.get("role") or record.get("role") or payload.get("speaker") or record.get("type")
    if not isinstance(raw_role, str) or not raw_role.strip():
        return None
    role = raw_role.strip().lower()
    if role in {"human", "prompt"}:
        role = "user"
    elif role in {"model", "ai", "bot"}:
        role = "assistant"

    raw_content = payload.get("content")
    if raw_content is None:
        raw_content = payload.get("text")
    if raw_content is None:
        raw_content = record.get("content")
    if raw_content is None:
        raw_content = record.get("text")

    content = _extract_text_content(raw_content)
    if not content:
        return None

    if explicit_episode_id is not None:
        ep_id = explicit_episode_id
    else:
        raw_ep = (
            record.get("episode_id")
            or record.get("session_id")
            or record.get("conversation_id")
            or payload.get("episode_id")
        )
        ep_id = str(raw_ep).strip() if isinstance(raw_ep, str) and raw_ep.strip() else default_episode_id

    raw_ts = (
        record.get("created_at")
        or record.get("timestamp")
        or payload.get("created_at")
        or payload.get("timestamp")
    )
    created_at = str(raw_ts).strip() if isinstance(raw_ts, str) and raw_ts.strip() else None
    return ep_id, role, content, created_at


def _split_markdown_sections(markdown_text: str) -> list[str]:
    cleaned = markdown_text.strip()
    if not cleaned:
        return []
    parts = [part.strip() for part in _HEADING_H2_RE.split(cleaned) if part.strip()]
    if len(parts) >= 2:
        return parts
    return [cleaned]


def _extract_markdown_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        match = _HEADING_H1_RE.match(line)
        if match:
            title = match.group(1).strip()
            if title:
                return title[:160]
    return fallback[:160]


def _normalize_seed_tag(raw: str) -> str:
    cleaned = _TAG_CLEAN_RE.sub("-", raw.strip().lower()).strip("-")
    return cleaned[:48]


class BootstrapImporter:
    """Idempotent cold-start importer backed by ``import_fingerprints`` deduplication."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._engine.db.transaction() as conn:
            conn.executescript(_IMPORT_FINGERPRINTS_SQL)

    def _fingerprint_exists(self, fingerprint: str) -> int | None:
        with self._engine.db.read_connection() as conn:
            row = conn.execute(
                "SELECT turn_id FROM import_fingerprints WHERE fingerprint = ?",
                (fingerprint,),
            ).fetchone()
        if row is None:
            return None
        return int(row["turn_id"])

    def _import_turn_if_new(
        self,
        *,
        source_kind: str,
        episode_id: str,
        source_seq: int | str,
        role: str,
        content: str,
        created_at: str | None = None,
        origin_trust: str = "user_verified",
    ) -> tuple[bool, int]:
        """Deduplicate and append a sanitized turn. Returns ``(was_imported, turn_id)``."""
        fp = compute_import_fingerprint(source_kind, episode_id, source_seq, role, content)
        existing_turn_id = self._fingerprint_exists(fp)
        if existing_turn_id is not None:
            return False, existing_turn_id

        turn_id = self._engine.append_turn(
            episode_id,
            role,
            content,
            created_at=created_at,
            origin_trust=origin_trust,
            sanitize=True,
        )
        now = _utc_now_iso()
        with self._engine.db.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO import_fingerprints (
                    fingerprint, source_kind, episode_id, turn_id, imported_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (fp, source_kind, episode_id, turn_id, now),
            )
        return True, turn_id

    # ------------------------------------------------------------------
    # 1. Transcript JSONL importer
    # ------------------------------------------------------------------

    def import_transcript_jsonl(
        self,
        path: str | Path,
        *,
        episode_id: str | None = None,
        origin_trust: str | None = None,
    ) -> dict[str, Any]:
        """Import a JSONL transcript safely, skipping binary media and reparse points."""
        target = Path(path)
        if _is_reparse_or_symlink(target):
            return {
                "source_kind": "transcript_jsonl",
                "source_path": str(target),
                "episode_id": episode_id or f"transcript:{target.stem}",
                "imported": 0,
                "skipped_duplicates": 0,
                "skipped_invalid": 0,
                "skipped_unsafe": 1,
                "skipped_reason": "reparse_point_or_symlink",
                "turn_ids": [],
            }
        if _is_binary_media_path(target):
            return {
                "source_kind": "transcript_jsonl",
                "source_path": str(target),
                "episode_id": episode_id or f"transcript:{target.stem}",
                "imported": 0,
                "skipped_duplicates": 0,
                "skipped_invalid": 0,
                "skipped_unsafe": 1,
                "skipped_reason": "binary_media_extension",
                "turn_ids": [],
            }
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Transcript file not found: {target}")

        explicit_ep = episode_id.strip() if isinstance(episode_id, str) and episode_id.strip() else None
        default_ep = explicit_ep or f"transcript:{target.stem}"

        imported = 0
        skipped_duplicates = 0
        skipped_invalid = 0
        turn_ids: list[int] = []

        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    skipped_invalid += 1
                    continue

                parsed = _parse_transcript_record(
                    record,
                    default_episode_id=default_ep,
                    explicit_episode_id=explicit_ep,
                )
                if parsed is None:
                    skipped_invalid += 1
                    continue

                ep_id, role, content, created_at = parsed
                turn_trust = origin_trust or ("assistant" if role == "assistant" else "user_verified")
                was_new, turn_id = self._import_turn_if_new(
                    source_kind="transcript_jsonl",
                    episode_id=ep_id,
                    source_seq=line_no,
                    role=role,
                    content=content,
                    created_at=created_at,
                    origin_trust=turn_trust,
                )
                if was_new:
                    imported += 1
                    turn_ids.append(turn_id)
                else:
                    skipped_duplicates += 1

        return {
            "source_kind": "transcript_jsonl",
            "source_path": str(target),
            "episode_id": default_ep,
            "imported": imported,
            "skipped_duplicates": skipped_duplicates,
            "skipped_invalid": skipped_invalid,
            "skipped_unsafe": 0,
            "turn_ids": turn_ids,
        }

    # ------------------------------------------------------------------
    # 2. Legacy read-only SQLite importer
    # ------------------------------------------------------------------

    def import_legacy_sqlite(
        self,
        db_path: str | Path,
        *,
        episode_prefix: str = "legacy",
        origin_trust: str | None = None,
    ) -> dict[str, Any]:
        """Read ``raw_turns`` / ``turns`` from a legacy SQLite DB in strict read-only mode."""
        target = Path(db_path)
        if _is_reparse_or_symlink(target):
            return {
                "source_kind": "legacy_sqlite",
                "source_path": str(target),
                "tables_scanned": [],
                "imported": 0,
                "skipped_duplicates": 0,
                "skipped_invalid": 0,
                "skipped_unsafe": 1,
                "skipped_reason": "reparse_point_or_symlink",
                "turn_ids": [],
            }
        if _is_binary_media_path(target):
            return {
                "source_kind": "legacy_sqlite",
                "source_path": str(target),
                "tables_scanned": [],
                "imported": 0,
                "skipped_duplicates": 0,
                "skipped_invalid": 0,
                "skipped_unsafe": 1,
                "skipped_reason": "binary_media_extension",
                "turn_ids": [],
            }
        if not target.exists() or not target.is_file():
            raise FileNotFoundError(f"Legacy SQLite database not found: {target}")

        ro_uri = f"{target.resolve().as_uri()}?mode=ro"
        imported = 0
        skipped_duplicates = 0
        skipped_invalid = 0
        tables_scanned: list[str] = []
        turn_ids: list[int] = []

        src_conn = sqlite3.connect(ro_uri, uri=True)
        try:
            src_conn.row_factory = sqlite3.Row
            src_conn.execute("PRAGMA query_only = ON;")

            existing_tables = {
                str(row[0])
                for row in src_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            candidate_tables = [t for t in ("raw_turns", "turns") if t in existing_tables]

            for table_name in candidate_tables:
                cols = {
                    str(row["name"]): str(row["name"])
                    for row in src_conn.execute(f'PRAGMA table_info("{table_name}");').fetchall()
                }
                content_col = next(
                    (c for c in ("content", "text", "message", "body") if c in cols),
                    None,
                )
                if content_col is None:
                    continue

                tables_scanned.append(table_name)
                id_col = next((c for c in ("turn_id", "id", "seq") if c in cols), None)
                ep_col = next(
                    (c for c in ("episode_id", "session_id", "conversation_id") if c in cols),
                    None,
                )
                role_col = next((c for c in ("role", "speaker", "author") if c in cols), None)
                ts_col = next((c for c in ("created_at", "timestamp", "ts") if c in cols), None)

                order_expr = f'"{id_col}" ASC' if id_col is not None else "rowid ASC"
                select_id = f'"{id_col}" AS _seq' if id_col is not None else "rowid AS _seq"
                rows = src_conn.execute(
                    f'SELECT {select_id}, * FROM "{table_name}" ORDER BY {order_expr};'
                ).fetchall()

                for row in rows:
                    raw_content = row[content_col]
                    content = _extract_text_content(raw_content)
                    if not content:
                        skipped_invalid += 1
                        continue

                    raw_ep = row[ep_col] if ep_col is not None else None
                    ep_id = (
                        str(raw_ep).strip()
                        if isinstance(raw_ep, str) and raw_ep.strip()
                        else f"{episode_prefix}:{target.stem}"
                    )
                    raw_role = row[role_col] if role_col is not None else None
                    role = (
                        str(raw_role).strip().lower()
                        if isinstance(raw_role, str) and raw_role.strip()
                        else "user"
                    )
                    raw_ts = row[ts_col] if ts_col is not None else None
                    created_at = (
                        str(raw_ts).strip()
                        if isinstance(raw_ts, str) and raw_ts.strip()
                        else None
                    )
                    source_seq = f"{table_name}:{row['_seq']}"

                    turn_trust = origin_trust or ("assistant" if role == "assistant" else "user_verified")
                    was_new, turn_id = self._import_turn_if_new(
                        source_kind="legacy_sqlite",
                        episode_id=ep_id,
                        source_seq=source_seq,
                        role=role,
                        content=content,
                        created_at=created_at,
                        origin_trust=turn_trust,
                    )
                    if was_new:
                        imported += 1
                        turn_ids.append(turn_id)
                    else:
                        skipped_duplicates += 1
        finally:
            src_conn.close()

        return {
            "source_kind": "legacy_sqlite",
            "source_path": str(target),
            "tables_scanned": tables_scanned,
            "imported": imported,
            "skipped_duplicates": skipped_duplicates,
            "skipped_invalid": skipped_invalid,
            "skipped_unsafe": 0,
            "turn_ids": turn_ids,
        }

    # ------------------------------------------------------------------
    # 3. Markdown checkpoints directory importer
    # ------------------------------------------------------------------

    def _collect_safe_markdown_files(self, root_dir: Path) -> tuple[list[Path], int]:
        md_files: list[Path] = []
        skipped_unsafe = 0

        def _walk(current: Path) -> None:
            nonlocal skipped_unsafe
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
            for entry in entries:
                if _is_reparse_or_symlink(entry):
                    skipped_unsafe += 1
                    continue
                if entry.is_dir():
                    _walk(entry)
                    continue
                if _is_binary_media_path(entry):
                    skipped_unsafe += 1
                    continue
                if entry.is_file() and entry.suffix.lower() == ".md":
                    md_files.append(entry)

        _walk(root_dir)
        return md_files, skipped_unsafe

    def import_checkpoints_dir(
        self,
        dir_path: str | Path,
        *,
        domain: str = "proj",
        mint_seed_cards: bool = True,
        origin_trust: str = "user_verified",
    ) -> dict[str, Any]:
        """Import ``*.md`` checkpoint/handoff documents with ``SourceSpan`` and optional seed cards."""
        validate_domain(domain)
        root_dir = Path(dir_path)

        if _is_reparse_or_symlink(root_dir):
            return {
                "source_kind": "checkpoints_dir",
                "source_dir": str(root_dir),
                "domain": domain,
                "files_scanned": 0,
                "imported": 0,
                "skipped_duplicates": 0,
                "skipped_unsafe": 1,
                "skipped_reason": "reparse_point_or_symlink",
                "seed_cards_minted": [],
                "skipped_duplicate_cards": 0,
                "skipped_poison_cards": 0,
                "spans": [],
                "turn_ids": [],
            }
        if not root_dir.exists() or not root_dir.is_dir():
            raise FileNotFoundError(f"Checkpoints directory not found: {root_dir}")

        md_files, skipped_unsafe = self._collect_safe_markdown_files(root_dir)

        imported = 0
        skipped_duplicates = 0
        minted_cards: list[str] = []
        skipped_duplicate_cards = 0
        skipped_poison_cards = 0
        spans: list[dict[str, Any]] = []
        all_turn_ids: list[int] = []

        for md_file in md_files:
            rel_posix = md_file.relative_to(root_dir).as_posix()
            raw_text = md_file.read_text(encoding="utf-8", errors="replace")
            sections = _split_markdown_sections(raw_text)
            if not sections:
                continue

            episode_id = f"checkpoint:{rel_posix}"
            file_turn_ids: list[int] = []
            file_new_count = 0

            for idx, section in enumerate(sections, start=1):
                was_new, turn_id = self._import_turn_if_new(
                    source_kind="checkpoints_dir",
                    episode_id=episode_id,
                    source_seq=idx,
                    role="system",
                    content=section,
                    origin_trust=origin_trust,
                )
                file_turn_ids.append(turn_id)
                if was_new:
                    imported += 1
                    file_new_count += 1
                    all_turn_ids.append(turn_id)
                else:
                    skipped_duplicates += 1

            if not file_turn_ids:
                continue

            span = SourceSpan(
                episode_id=episode_id,
                turn_start=min(file_turn_ids),
                turn_end=max(file_turn_ids),
            )
            spans.append(span.to_dict())

            if mint_seed_cards and file_new_count > 0:
                seed_digest = hashlib.sha256(f"{domain}:{rel_posix}".encode("utf-8")).hexdigest()[:12]
                seed_card_id = f"seed_{seed_digest}"
                if self._engine.cards.get_card(seed_card_id) is not None:
                    skipped_duplicate_cards += 1
                    continue

                san_res = sanitize_for_memory(raw_text, origin_trust=origin_trust)
                if not can_mutate_authoritative_memory(san_res):
                    skipped_poison_cards += 1
                    continue

                title = _extract_markdown_title(san_res.clean_text, md_file.stem)
                summary_excerpt = san_res.clean_text.strip()[:400]
                stem_tag = _normalize_seed_tag(md_file.stem)
                raw_tags = ["checkpoint", "bootstrap"]
                if stem_tag and stem_tag not in raw_tags:
                    raw_tags.append(stem_tag)

                card = self._engine.cards.create_card(
                    domain=domain,
                    card_type="checkpoint_seed",
                    title=title,
                    state={
                        "source_file": rel_posix,
                        "summary": summary_excerpt,
                        "section_count": len(sections),
                    },
                    tags=tuple(raw_tags),
                    spans=(span,),
                    card_id=seed_card_id,
                )
                if hasattr(self._engine, "seal_card") and callable(self._engine.seal_card):
                    self._engine.seal_card(card)
                minted_cards.append(card.card_id)

        return {
            "source_kind": "checkpoints_dir",
            "source_dir": str(root_dir),
            "domain": domain,
            "files_scanned": len(md_files),
            "imported": imported,
            "skipped_duplicates": skipped_duplicates,
            "skipped_unsafe": skipped_unsafe,
            "seed_cards_minted": minted_cards,
            "skipped_duplicate_cards": skipped_duplicate_cards,
            "skipped_poison_cards": skipped_poison_cards,
            "spans": spans,
            "turn_ids": all_turn_ids,
        }

    # ------------------------------------------------------------------
    # 4. Status telemetry
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return import ledger statistics and current SQLite schema version."""
        with self._engine.db.read_connection() as conn:
            schema_version = get_schema_version(conn)
            total_imported = int(
                conn.execute("SELECT COUNT(*) FROM import_fingerprints").fetchone()[0]
            )
            distinct_episodes = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM import_fingerprints"
                ).fetchone()[0]
            )
            last_row = conn.execute(
                "SELECT MAX(imported_at) FROM import_fingerprints"
            ).fetchone()
            last_imported_at = None if last_row is None or last_row[0] is None else str(last_row[0])
            by_kind_rows = conn.execute(
                """
                SELECT source_kind, COUNT(*) AS cnt
                FROM import_fingerprints
                GROUP BY source_kind
                ORDER BY source_kind ASC
                """
            ).fetchall()

        by_source_kind = {str(row["source_kind"]): int(row["cnt"]) for row in by_kind_rows}
        return {
            "schema_version": schema_version,
            "total_imported": total_imported,
            "distinct_episodes": distinct_episodes,
            "by_source_kind": by_source_kind,
            "last_imported_at": last_imported_at,
        }


# ----------------------------------------------------------------------
# Microkernel plugin & CLI registration
# ----------------------------------------------------------------------


def _configure_import_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    sub_shared = argparse.ArgumentParser(add_help=False)
    sub_shared.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit machine-readable JSON output.",
    )

    parser.set_defaults(import_action="status")
    sub = parser.add_subparsers(dest="import_action", metavar="<source>")

    p_status = sub.add_parser(
        "status",
        parents=[sub_shared],
        help="Show cold-start import deduplication statistics and schema version.",
    )
    p_status.set_defaults(import_action="status")

    p_transcript = sub.add_parser(
        "transcript",
        parents=[sub_shared],
        help="Import a JSONL conversation transcript with sanitization and SHA-256 dedup.",
    )
    p_transcript.add_argument("path", help="Path to a .jsonl transcript file.")
    p_transcript.add_argument(
        "--episode-id",
        default=None,
        help="Optional episode_id override for all imported turns.",
    )
    p_transcript.set_defaults(import_action="transcript")

    p_sqlite = sub.add_parser(
        "sqlite",
        parents=[sub_shared],
        help="Import turns from a legacy SQLite database in strict read-only mode.",
    )
    p_sqlite.add_argument("db_path", help="Path to the legacy SQLite database file.")
    p_sqlite.set_defaults(import_action="sqlite")

    p_checkpoints = sub.add_parser(
        "checkpoints",
        parents=[sub_shared],
        help="Import Markdown checkpoint/handoff files and optionally mint sealed seed cards.",
    )
    p_checkpoints.add_argument("dir_path", help="Directory containing *.md handoff files.")
    p_checkpoints.add_argument(
        "--domain",
        choices=LOCKED_DOMAINS,
        default="proj",
        help="Locked L1 domain for minted seed MemoryCards (default: proj).",
    )
    p_checkpoints.add_argument(
        "--no-seed-cards",
        action="store_true",
        help="Import Markdown turns without minting seed MemoryCards.",
    )
    p_checkpoints.set_defaults(import_action="checkpoints")

    return parser


class _ImportCliHandler:
    """Callable CLI handler exposing ``configure_parser`` for ``reiji import``."""

    def configure_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return _configure_import_parser(parser)

    def __call__(self, engine: Any, args: argparse.Namespace) -> int:
        parsed = args
        if not hasattr(args, "import_action"):
            sub_parser = argparse.ArgumentParser(prog="reiji import")
            self.configure_parser(sub_parser)
            extra = list(getattr(args, "extra_args", ()) or ())
            parsed = sub_parser.parse_args(extra)
            if getattr(args, "json", False):
                parsed.json = True

        action = getattr(parsed, "import_action", None) or "status"
        as_json = bool(getattr(parsed, "json", False))
        importer = BootstrapImporter(engine)

        if action == "status":
            info = importer.status()
            if as_json:
                _emit_json(info)
            else:
                print(
                    f"Bootstrap Importer (schema=v{info['schema_version']}): "
                    f"total_imported={info['total_imported']}, "
                    f"episodes={info['distinct_episodes']}, "
                    f"by_source={info['by_source_kind']}"
                )
            return 0

        if action == "transcript":
            result = importer.import_transcript_jsonl(
                parsed.path,
                episode_id=getattr(parsed, "episode_id", None),
            )
            if as_json:
                _emit_json(result)
            else:
                print(
                    f"Imported transcript {result['source_path']}: "
                    f"imported={result['imported']}, "
                    f"duplicates={result['skipped_duplicates']}, "
                    f"invalid={result['skipped_invalid']}, "
                    f"unsafe={result['skipped_unsafe']}"
                )
            return 0

        if action == "sqlite":
            result = importer.import_legacy_sqlite(parsed.db_path)
            if as_json:
                _emit_json(result)
            else:
                print(
                    f"Imported legacy SQLite {result['source_path']}: "
                    f"tables={result['tables_scanned']}, "
                    f"imported={result['imported']}, "
                    f"duplicates={result['skipped_duplicates']}, "
                    f"unsafe={result['skipped_unsafe']}"
                )
            return 0

        if action == "checkpoints":
            result = importer.import_checkpoints_dir(
                parsed.dir_path,
                domain=str(getattr(parsed, "domain", "proj")),
                mint_seed_cards=not bool(getattr(parsed, "no_seed_cards", False)),
            )
            if as_json:
                _emit_json(result)
            else:
                print(
                    f"Imported checkpoints {result['source_dir']} ({result['domain']}): "
                    f"files={result['files_scanned']}, "
                    f"imported={result['imported']}, "
                    f"duplicates={result['skipped_duplicates']}, "
                    f"seed_cards={len(result['seed_cards_minted'])}"
                )
            return 0

        raise ValueError(f"Unsupported import action: {action!r}")


class BootstrapImporterPlugin:
    """Microkernel plugin registering the ``import`` cold-start CLI command."""

    name: str = "bootstrap_importer"

    def __init__(self) -> None:
        self._cli_handler = _ImportCliHandler()

    def register_cli(self, kernel: ReijiKernel) -> None:
        kernel.register_cli(
            "import",
            self._cli_handler,
            help_text="Idempotently import transcripts, legacy SQLite DBs, and Markdown checkpoints.",
            plugin_name=self.name,
        )


def create_plugin() -> BootstrapImporterPlugin:
    """Factory function for ``BootstrapImporterPlugin``."""
    return BootstrapImporterPlugin()


__all__ = [
    "BINARY_MEDIA_EXTENSIONS",
    "BootstrapImporter",
    "BootstrapImporterPlugin",
    "compute_import_fingerprint",
    "create_plugin",
]
