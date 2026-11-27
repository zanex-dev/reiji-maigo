"""Lock-free shared tail reader and 250s Prompt KV thermal sync plugin."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

TAIL_READ_BYTES = 16 * 1024
CACHE_SAFE_TTL_SEC = 250.0
ACTIVE_GRACE_SEC = 5.0
WARN_IDLE_SEC = 190.0
TIER_2X_WARN_TOKENS = 180_000
TIER_2X_THRESHOLD_TOKENS = 200_000

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

_FORBIDDEN_MEDIA_SUFFIXES: frozenset[str] = frozenset(
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
    }
)
_FORBIDDEN_PATH_TOKEN = ".user_uploaded"
_CURSOR_FILENAME = "thermal_cursors.json"

_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_FILE_BEGIN = 0


def _is_forbidden_media_path(path: Path) -> bool:
    if path.suffix.lower() in _FORBIDDEN_MEDIA_SUFFIXES:
        return True
    for part in path.parts:
        if _FORBIDDEN_PATH_TOKEN in part.lower():
            return True
    return False


def has_reparse_or_symlink(path: Path | str) -> bool:
    """Return True if path is a symbolic link or Windows reparse point."""
    target = Path(path)
    try:
        st = os.lstat(target)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    attrs = int(getattr(st, "st_file_attributes", 0) or 0)
    return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)


def _validate_safe_transcript_path(path: Path | str) -> Path:
    target = Path(path)
    if _is_forbidden_media_path(target):
        raise PermissionError(
            f"Refusing to access restricted media or upload path: {target.name}"
        )
    if has_reparse_or_symlink(target):
        raise PermissionError(
            f"Refusing to follow symlink or reparse point: {target.name}"
        )
    return target


def _win32_read_tail_raw(path: Path, max_bytes: int) -> tuple[bytes, int, bool]:
    """Read up to max_bytes (plus 1 sentinel byte when truncated) via Win32 shared read."""
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE

    get_file_size_ex = kernel32.GetFileSizeEx
    get_file_size_ex.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    get_file_size_ex.restype = wintypes.BOOL

    set_file_pointer_ex = kernel32.SetFilePointerEx
    set_file_pointer_ex.argtypes = [
        wintypes.HANDLE,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_longlong),
        wintypes.DWORD,
    ]
    set_file_pointer_ex.restype = wintypes.BOOL

    read_file = kernel32.ReadFile
    read_file.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    read_file.restype = wintypes.BOOL

    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    invalid_handle = ctypes.c_void_p(-1).value
    share_mode = _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE
    flags = _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT

    handle = create_file(
        os.fspath(path),
        _GENERIC_READ,
        share_mode,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if handle == invalid_handle or not handle:
        err = ctypes.get_last_error()
        raise ctypes.WinError(err)

    try:
        size_val = ctypes.c_longlong(0)
        if not get_file_size_ex(handle, ctypes.byref(size_val)):
            raise ctypes.WinError(ctypes.get_last_error())
        file_size = int(size_val.value)
        if file_size <= 0 or max_bytes <= 0:
            return b"", file_size, False

        truncated_head = file_size > max_bytes
        if truncated_head:
            read_offset = file_size - max_bytes - 1
            to_read = max_bytes + 1
        else:
            read_offset = 0
            to_read = file_size

        if read_offset > 0:
            new_pos = ctypes.c_longlong(0)
            if not set_file_pointer_ex(
                handle,
                ctypes.c_longlong(read_offset),
                ctypes.byref(new_pos),
                _FILE_BEGIN,
            ):
                raise ctypes.WinError(ctypes.get_last_error())

        chunks: list[bytes] = []
        remaining = to_read
        while remaining > 0:
            buf = ctypes.create_string_buffer(remaining)
            bytes_read = wintypes.DWORD(0)
            ok = read_file(
                handle,
                buf,
                remaining,
                ctypes.byref(bytes_read),
                None,
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            count = int(bytes_read.value)
            if count <= 0:
                break
            chunks.append(buf.raw[:count])
            remaining -= count

        return b"".join(chunks), file_size, truncated_head
    finally:
        close_handle(handle)


def _posix_read_tail_raw(path: Path, max_bytes: int) -> tuple[bytes, int, bool]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(os.fspath(path), flags)
    try:
        file_size = int(os.fstat(fd).st_size)
        if file_size <= 0 or max_bytes <= 0:
            return b"", file_size, False

        truncated_head = file_size > max_bytes
        if truncated_head:
            read_offset = file_size - max_bytes - 1
            to_read = max_bytes + 1
        else:
            read_offset = 0
            to_read = file_size

        if read_offset > 0:
            os.lseek(fd, read_offset, os.SEEK_SET)

        chunks: list[bytes] = []
        remaining = to_read
        while remaining > 0:
            piece = os.read(fd, remaining)
            if not piece:
                break
            chunks.append(piece)
            remaining -= len(piece)

        return b"".join(chunks), file_size, truncated_head
    finally:
        os.close(fd)


def _strip_partial_line_boundaries(
    raw: bytes,
    *,
    file_size: int,
    truncated_head: bool,
) -> tuple[bytes, int]:
    """Strip partial leading UTF-8/line fragments and unflushed trailing partial lines.

    Returns ``(clean_bytes, consumed_end_offset)`` where ``consumed_end_offset``
    marks the exact byte offset immediately after the last complete newline.
    """
    if not raw:
        return b"", max(0, file_size)

    if truncated_head:
        sentinel = raw[:1]
        data = raw[1:]
        if sentinel != b"\n":
            first_nl = data.find(b"\n")
            if first_nl == -1:
                # Entire tail sits inside a single partial line; hold cursor before it
                return b"", max(0, file_size - len(data))
            data = data[first_nl + 1 :]
    else:
        data = raw
        # Guard against corrupted UTF-8 continuation bytes at offset 0
        if data and 0x80 <= data[0] <= 0xBF:
            first_nl = data.find(b"\n")
            if first_nl == -1:
                return b"", 0
            data = data[first_nl + 1 :]

    if not data:
        return b"", file_size

    trailing_partial = 0
    if not data.endswith(b"\n"):
        last_nl = data.rfind(b"\n")
        if last_nl == -1:
            trailing_partial = len(data)
            data = b""
        else:
            trailing_partial = len(data) - (last_nl + 1)
            data = data[: last_nl + 1]

    consumed_end = max(0, file_size - trailing_partial)
    return data, consumed_end


def _read_shared_tail_window(
    path: Path | str,
    max_bytes: int = TAIL_READ_BYTES,
) -> tuple[bytes, int, int, bool]:
    """Read sanitized tail window and return (clean_bytes, consumed_end, file_size, truncated)."""
    target = _validate_safe_transcript_path(path)
    st = os.lstat(target)
    if stat.S_ISDIR(st.st_mode):
        raise IsADirectoryError(f"Expected file, got directory: {target.name}")
    if max_bytes <= 0:
        return b"", int(st.st_size), int(st.st_size), False

    if os.name == "nt":
        raw, file_size, truncated_head = _win32_read_tail_raw(target, max_bytes)
    else:
        raw, file_size, truncated_head = _posix_read_tail_raw(target, max_bytes)

    clean, consumed_end = _strip_partial_line_boundaries(
        raw,
        file_size=file_size,
        truncated_head=truncated_head,
    )
    return clean, consumed_end, file_size, truncated_head


def read_shared_tail_bytes(
    path: Path | str,
    max_bytes: int = TAIL_READ_BYTES,
) -> bytes:
    """Lock-free shared read of the trailing bytes of a transcript file.

    Refuses symlinks, Windows reparse points, and restricted media/upload paths.
    Safely strips partial leading UTF-8/line fragments and unflushed trailing lines.
    """
    clean, _consumed_end, _file_size, _truncated = _read_shared_tail_window(
        path,
        max_bytes=max_bytes,
    )
    return clean


@dataclass(frozen=True, slots=True)
class ThermalSnapshot:
    """Point-in-time Prompt KV cache thermal state and 200K pricing tier status."""

    phase: str  # "empty" | "active" | "hot" | "warning" | "cold"
    idle_sec: float | None
    remaining_ttl_sec: float
    pricing_tier: str  # "1x" | "near_2x" | "2x"
    context_tokens: int
    in_flight: bool
    should_digest_on_cooldown: bool
    last_activity_epoch: float | None = None
    evaluated_at_epoch: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_thermal_state(
    *,
    last_activity_epoch: float | None,
    now_epoch: float | None = None,
    context_tokens: int = 0,
    in_flight: bool = False,
) -> ThermalSnapshot:
    """Evaluate the 250s KV cache thermal phase and 200K pricing tier alert."""
    now = float(time.time() if now_epoch is None else now_epoch)
    tokens = max(0, int(context_tokens or 0))

    if tokens >= TIER_2X_THRESHOLD_TOKENS:
        pricing_tier = "2x"
    elif tokens >= TIER_2X_WARN_TOKENS:
        pricing_tier = "near_2x"
    else:
        pricing_tier = "1x"

    if last_activity_epoch is None and not in_flight:
        phase = "empty"
        idle_sec: float | None = None
        remaining_ttl_sec = 0.0
    elif last_activity_epoch is None and in_flight:
        phase = "active"
        idle_sec = 0.0
        remaining_ttl_sec = CACHE_SAFE_TTL_SEC
    else:
        last_ts = float(last_activity_epoch)  # type: ignore[arg-type]
        idle_sec = max(0.0, now - last_ts)
        remaining_ttl_sec = max(0.0, CACHE_SAFE_TTL_SEC - idle_sec)
        if in_flight or idle_sec <= ACTIVE_GRACE_SEC:
            phase = "active"
        elif idle_sec < WARN_IDLE_SEC:
            phase = "hot"
        elif idle_sec < CACHE_SAFE_TTL_SEC:
            phase = "warning"
        else:
            phase = "cold"

    should_digest = (not in_flight) and (
        phase == "cold" or pricing_tier in ("near_2x", "2x")
    )

    return ThermalSnapshot(
        phase=phase,
        idle_sec=round(idle_sec, 3) if idle_sec is not None else None,
        remaining_ttl_sec=round(remaining_ttl_sec, 3),
        pricing_tier=pricing_tier,
        context_tokens=tokens,
        in_flight=bool(in_flight),
        should_digest_on_cooldown=should_digest,
        last_activity_epoch=float(last_activity_epoch)
        if last_activity_epoch is not None
        else None,
        evaluated_at_epoch=round(now, 3),
    )


@dataclass(slots=True)
class TranscriptCursor:
    """Incremental cursor tracking (inode/dev/size/offset) for a transcript file."""

    path: str
    inode: int
    dev: int
    size: int
    offset: int
    synced_turns: int = 0
    last_activity_epoch: float | None = None
    last_context_tokens: int = 0
    last_sync_epoch: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "inode": self.inode,
            "dev": self.dev,
            "size": self.size,
            "offset": self.offset,
            "synced_turns": self.synced_turns,
            "last_activity_epoch": self.last_activity_epoch,
            "last_context_tokens": self.last_context_tokens,
            "last_sync_epoch": self.last_sync_epoch,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TranscriptCursor:
        return cls(
            path=str(data["path"]),
            inode=int(data.get("inode", 0)),
            dev=int(data.get("dev", 0)),
            size=int(data.get("size", 0)),
            offset=int(data.get("offset", 0)),
            synced_turns=int(data.get("synced_turns", 0)),
            last_activity_epoch=(
                float(data["last_activity_epoch"])
                if data.get("last_activity_epoch") is not None
                else None
            ),
            last_context_tokens=int(data.get("last_context_tokens", 0)),
            last_sync_epoch=float(data.get("last_sync_epoch", 0.0)),
        )


def _parse_epoch_value(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        num = float(value)
        if num <= 0:
            return None
        # Heuristic: convert epoch milliseconds to seconds
        if num > 1e12:
            num /= 1000.0
        return num
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return _parse_epoch_value(float(raw))
        except ValueError:
            pass
        iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            dt = datetime.fromisoformat(iso)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _epoch_to_iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _extract_text_content(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if isinstance(raw_content, Mapping):
        text = raw_content.get("text") or raw_content.get("content")
        if isinstance(text, str):
            return text
    return ""


def _extract_context_tokens(record: Mapping[str, Any]) -> int | None:
    for key in ("context_tokens", "total_tokens", "prompt_tokens", "input_tokens"):
        val = record.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
            return int(val)
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        for key in ("context_tokens", "total_tokens", "prompt_tokens"):
            val = usage.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
                return int(val)
        input_toks = usage.get("input_tokens")
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        if isinstance(input_toks, (int, float)) and not isinstance(input_toks, bool):
            total = int(input_toks)
            if isinstance(cache_read, (int, float)) and not isinstance(cache_read, bool):
                total += int(cache_read)
            if isinstance(cache_create, (int, float)) and not isinstance(cache_create, bool):
                total += int(cache_create)
            return max(0, total)
    return None


def _parse_jsonl_tail(
    data: bytes,
) -> tuple[list[dict[str, Any]], int, float | None, int | None, bool]:
    """Parse complete JSONL lines into normalized turns and thermal metadata."""
    turns: list[dict[str, Any]] = []
    parsed_lines = 0
    latest_epoch: float | None = None
    latest_tokens: int | None = None
    in_flight = False

    if not data:
        return turns, parsed_lines, latest_epoch, latest_tokens, in_flight

    text = data.decode("utf-8", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, Mapping):
            continue
        parsed_lines += 1

        ts_epoch: float | None = None
        for ts_key in ("timestamp", "created_at", "ts", "epoch", "updated_at"):
            if ts_key in obj:
                ts_epoch = _parse_epoch_value(obj.get(ts_key))
                if ts_epoch is not None:
                    break
        if ts_epoch is not None:
            if latest_epoch is None or ts_epoch > latest_epoch:
                latest_epoch = ts_epoch

        tok = _extract_context_tokens(obj)
        if tok is not None:
            latest_tokens = tok

        if isinstance(obj.get("in_flight"), bool):
            in_flight = bool(obj["in_flight"])

        msg = obj.get("message")
        role_val = obj.get("role") or obj.get("author") or obj.get("sender")
        content_raw = obj.get("content") if "content" in obj else obj.get("text")
        if isinstance(msg, Mapping):
            if not role_val:
                role_val = msg.get("role") or msg.get("author")
            if content_raw is None:
                content_raw = msg.get("content") if "content" in msg else msg.get("text")

        role = str(role_val).strip().lower() if isinstance(role_val, str) else ""
        content = _extract_text_content(content_raw)
        if role and content:
            origin_trust = obj.get("origin_trust")
            turns.append(
                {
                    "role": role,
                    "content": content,
                    "created_at": _epoch_to_iso(ts_epoch),
                    "origin_trust": (
                        str(origin_trust).strip()
                        if isinstance(origin_trust, str) and origin_trust.strip()
                        else "user_verified"
                    ),
                }
            )

    return turns, parsed_lines, latest_epoch, latest_tokens, in_flight


class ThermalSyncManager:
    """Manages lock-free JSONL tail probing, cursor-based sync, and cooldown handoff."""

    def __init__(
        self,
        engine: Any,
        *,
        cursor_store_path: Path | str | None = None,
    ) -> None:
        self.engine = engine
        self._cursors: dict[str, TranscriptCursor] = {}
        self._last_probe: dict[str, Any] | None = None

        if cursor_store_path is not None:
            self._cursor_store_path: Path | None = Path(cursor_store_path)
        else:
            home = getattr(engine, "home", None)
            self._cursor_store_path = (
                Path(home) / _CURSOR_FILENAME if home is not None else None
            )
        self._load_cursors()

    def _cursor_key(self, path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    def _load_cursors(self) -> None:
        store = self._cursor_store_path
        if store is None or not store.is_file():
            return
        try:
            payload = json.loads(store.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return
        for key, raw in payload.items():
            if isinstance(raw, Mapping) and "path" in raw:
                try:
                    self._cursors[str(key)] = TranscriptCursor.from_dict(raw)
                except (TypeError, ValueError):
                    continue

    def _save_cursors(self) -> None:
        store = self._cursor_store_path
        if store is None:
            return
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            serialized = {key: cur.to_dict() for key, cur in self._cursors.items()}
            tmp = store.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp, store)
        except OSError:
            return

    def get_cursor(self, path: Path | str) -> TranscriptCursor | None:
        return self._cursors.get(self._cursor_key(Path(path)))

    def status_summary(self) -> dict[str, Any]:
        return {
            "tracked_files": len(self._cursors),
            "cursors": [cur.to_dict() for cur in self._cursors.values()],
            "last_probe": self._last_probe,
            "constants": {
                "tail_read_bytes": TAIL_READ_BYTES,
                "cache_safe_ttl_sec": CACHE_SAFE_TTL_SEC,
                "active_grace_sec": ACTIVE_GRACE_SEC,
                "warn_idle_sec": WARN_IDLE_SEC,
                "tier_2x_warn_tokens": TIER_2X_WARN_TOKENS,
                "tier_2x_threshold_tokens": TIER_2X_THRESHOLD_TOKENS,
            },
        }

    def probe_transcript(
        self,
        path: Path | str,
        *,
        now_epoch: float | None = None,
        max_bytes: int = TAIL_READ_BYTES,
    ) -> dict[str, Any]:
        """Probe a transcript file via lock-free tail read and evaluate thermal state."""
        target = _validate_safe_transcript_path(path)
        st = os.lstat(target)
        clean, consumed_end, file_size, truncated = _read_shared_tail_window(
            target,
            max_bytes=max_bytes,
        )
        turns, parsed_lines, latest_epoch, latest_tokens, in_flight = _parse_jsonl_tail(
            clean
        )

        if file_size == 0 and latest_epoch is None:
            activity_epoch: float | None = None
        elif latest_epoch is not None:
            activity_epoch = latest_epoch
        else:
            activity_epoch = float(st.st_mtime)

        context_tokens = latest_tokens if latest_tokens is not None else 0
        snapshot = evaluate_thermal_state(
            last_activity_epoch=activity_epoch,
            now_epoch=now_epoch,
            context_tokens=context_tokens,
            in_flight=in_flight,
        )

        report = {
            "path": os.path.abspath(os.fspath(target)),
            "file_size": file_size,
            "inode": int(st.st_ino),
            "consumed_end_offset": consumed_end,
            "tail_bytes": len(clean),
            "window_truncated": truncated,
            "parsed_lines": parsed_lines,
            "turn_count": len(turns),
            "context_tokens": snapshot.context_tokens,
            "last_activity_epoch": snapshot.last_activity_epoch,
            "phase": snapshot.phase,
            "idle_sec": snapshot.idle_sec,
            "remaining_ttl_sec": snapshot.remaining_ttl_sec,
            "pricing_tier": snapshot.pricing_tier,
            "should_digest_on_cooldown": snapshot.should_digest_on_cooldown,
            "thermal": snapshot.to_dict(),
        }
        self._last_probe = report
        return report

    def sync_tail(
        self,
        path: Path | str,
        *,
        episode_id: str | None = None,
        now_epoch: float | None = None,
        trigger_scribe_on_cool: bool = False,
    ) -> dict[str, Any]:
        """Incrementally sync new JSONL turns using (inode/size/offset) cursor."""
        target = _validate_safe_transcript_path(path)
        st = os.lstat(target)
        if stat.S_ISDIR(st.st_mode):
            raise IsADirectoryError(f"Expected file, got directory: {target.name}")

        now = float(time.time() if now_epoch is None else now_epoch)
        abs_path = os.path.abspath(os.fspath(target))
        key = self._cursor_key(target)
        prev = self._cursors.get(key)

        inode = int(st.st_ino)
        dev = int(st.st_dev)
        file_size = int(st.st_size)

        cursor_reset = False
        if prev is None or prev.inode != inode or prev.dev != dev or file_size < prev.offset:
            prev_offset = 0
            prev_synced = 0
            prev_activity = None
            prev_tokens = 0
            cursor_reset = prev is not None
        else:
            prev_offset = prev.offset
            prev_synced = prev.synced_turns
            prev_activity = prev.last_activity_epoch
            prev_tokens = prev.last_context_tokens

        delta = max(0, file_size - prev_offset)
        window_truncated = False
        turns: list[dict[str, Any]] = []
        parsed_lines = 0
        latest_epoch = prev_activity
        latest_tokens = prev_tokens
        in_flight = False
        new_offset = prev_offset

        if delta > 0:
            read_limit = min(delta, TAIL_READ_BYTES)
            window_truncated = delta > TAIL_READ_BYTES
            clean, consumed_end, _cur_size, _trunc = _read_shared_tail_window(
                target,
                max_bytes=read_limit,
            )
            turns, parsed_lines, parsed_epoch, parsed_tokens, in_flight = _parse_jsonl_tail(
                clean
            )
            new_offset = max(prev_offset, consumed_end)
            if parsed_epoch is not None:
                latest_epoch = parsed_epoch
            elif file_size > 0 and latest_epoch is None:
                latest_epoch = float(st.st_mtime)
            if parsed_tokens is not None:
                latest_tokens = parsed_tokens
        elif file_size > 0 and latest_epoch is None:
            latest_epoch = float(st.st_mtime)

        ep_id = (
            episode_id.strip()
            if isinstance(episode_id, str) and episode_id.strip()
            else f"thermal:{target.stem}"
        )

        turn_ids: list[int] = []
        for turn in turns:
            tid = self.engine.append_turn(
                ep_id,
                turn["role"],
                turn["content"],
                created_at=turn["created_at"],
                origin_trust=turn["origin_trust"],
                sanitize=True,
            )
            turn_ids.append(int(tid))

        cursor = TranscriptCursor(
            path=abs_path,
            inode=inode,
            dev=dev,
            size=file_size,
            offset=new_offset,
            synced_turns=prev_synced + len(turn_ids),
            last_activity_epoch=latest_epoch,
            last_context_tokens=latest_tokens,
            last_sync_epoch=round(now, 3),
        )
        self._cursors[key] = cursor
        self._save_cursors()

        snapshot = evaluate_thermal_state(
            last_activity_epoch=latest_epoch,
            now_epoch=now,
            context_tokens=latest_tokens,
            in_flight=in_flight,
        )

        handoff_emitted = False
        scribe_triggered = False
        if snapshot.should_digest_on_cooldown:
            if snapshot.pricing_tier == "2x":
                reason = "pricing_tier_2x"
            elif snapshot.pricing_tier == "near_2x":
                reason = "pricing_tier_near_2x"
            else:
                reason = "cache_cooldown"

            handoff_payload = {
                "path": abs_path,
                "episode_id": ep_id,
                "reason": reason,
                "phase": snapshot.phase,
                "pricing_tier": snapshot.pricing_tier,
                "context_tokens": snapshot.context_tokens,
                "idle_sec": snapshot.idle_sec,
                "remaining_ttl_sec": snapshot.remaining_ttl_sec,
                "synced_turns": len(turn_ids),
                "turn_ids": list(turn_ids),
                "trigger_scribe": bool(trigger_scribe_on_cool),
                "thermal": snapshot.to_dict(),
            }
            kernel = getattr(self.engine, "kernel", None)
            hooks = getattr(kernel, "hooks", None)
            if hooks is not None:
                hooks.emit("thermal.handoff_ready", handoff_payload)
                handoff_emitted = True

            if trigger_scribe_on_cool:
                scribe_fn = (
                    getattr(self.engine, "trigger_scribe", None)
                    or getattr(self.engine, "digest_episode", None)
                    or getattr(self.engine, "run_scribe", None)
                )
                if callable(scribe_fn):
                    try:
                        scribe_fn(ep_id)
                    except TypeError:
                        scribe_fn()
                if hooks is not None:
                    hooks.emit("thermal.scribe_requested", handoff_payload)
                scribe_triggered = True

        result = {
            "ok": True,
            "path": abs_path,
            "episode_id": ep_id,
            "synced_turns": len(turn_ids),
            "turn_ids": turn_ids,
            "parsed_lines": parsed_lines,
            "cursor": cursor.to_dict(),
            "cursor_reset": cursor_reset,
            "window_truncated": window_truncated,
            "phase": snapshot.phase,
            "idle_sec": snapshot.idle_sec,
            "remaining_ttl_sec": snapshot.remaining_ttl_sec,
            "pricing_tier": snapshot.pricing_tier,
            "context_tokens": snapshot.context_tokens,
            "should_digest_on_cooldown": snapshot.should_digest_on_cooldown,
            "handoff_emitted": handoff_emitted,
            "scribe_triggered": scribe_triggered,
            "thermal": snapshot.to_dict(),
        }
        self._last_probe = {
            "path": abs_path,
            "phase": snapshot.phase,
            "pricing_tier": snapshot.pricing_tier,
            "context_tokens": snapshot.context_tokens,
            "remaining_ttl_sec": snapshot.remaining_ttl_sec,
        }
        return result


def _configure_radar_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="radar_action")

    p_status = subparsers.add_parser(
        "status",
        help="Show thermal sync cursor status and thresholds",
    )
    p_status.add_argument("--path", default=None, help="Optional transcript path to probe")
    p_status.add_argument("--now-epoch", type=float, default=None)
    p_status.add_argument("--json", action="store_true", dest="as_json")

    p_probe = subparsers.add_parser(
        "probe",
        help="Lock-free tail probe of a transcript JSONL file",
    )
    p_probe.add_argument("path", help="Transcript JSONL file path")
    p_probe.add_argument("--now-epoch", type=float, default=None)
    p_probe.add_argument("--json", action="store_true", dest="as_json")

    p_sync = subparsers.add_parser(
        "sync",
        help="Incrementally sync transcript tail turns into memory engine",
    )
    p_sync.add_argument("path", help="Transcript JSONL file path")
    p_sync.add_argument("--episode-id", default=None)
    p_sync.add_argument("--trigger-scribe", action="store_true")
    p_sync.add_argument("--now-epoch", type=float, default=None)
    p_sync.add_argument("--json", action="store_true", dest="as_json")


class ThermalSyncPlugin:
    """Optional microkernel plugin providing the ``radar`` thermal sync CLI."""

    name: str = "thermal_sync"

    def __init__(self, engine: Any | None = None) -> None:
        self._engine: Any | None = engine
        self._manager: ThermalSyncManager | None = (
            ThermalSyncManager(engine) if engine is not None else None
        )

    def get_manager(self, engine: Any | None = None) -> ThermalSyncManager:
        target_engine = engine if engine is not None else self._engine
        if self._manager is None or (
            target_engine is not None and self._manager.engine is not target_engine
        ):
            self._engine = target_engine
            self._manager = ThermalSyncManager(target_engine)
        return self._manager

    def _resolve_cli_inputs(
        self,
        *pos_args: Any,
        **kwargs: Any,
    ) -> tuple[argparse.Namespace, Any | None]:
        parsed_args: argparse.Namespace | None = kwargs.get("args")
        engine: Any | None = kwargs.get("engine")

        for item in pos_args:
            if isinstance(item, argparse.Namespace):
                parsed_args = item
            elif engine is None and item is not None:
                engine = item

        if parsed_args is None:
            parsed_args = argparse.Namespace(radar_action="status", as_json=True)
        return parsed_args, engine

    def _build_cli_handler(self) -> Any:
        def radar_handler(*pos_args: Any, **kwargs: Any) -> int:
            args, engine = self._resolve_cli_inputs(*pos_args, **kwargs)
            mgr = self.get_manager(engine)
            action = getattr(args, "radar_action", None) or "status"
            now_epoch = getattr(args, "now_epoch", None)

            if action == "probe":
                result = mgr.probe_transcript(args.path, now_epoch=now_epoch)
            elif action == "sync":
                result = mgr.sync_tail(
                    args.path,
                    episode_id=getattr(args, "episode_id", None),
                    now_epoch=now_epoch,
                    trigger_scribe_on_cool=bool(
                        getattr(args, "trigger_scribe", False)
                    ),
                )
            else:
                path_opt = getattr(args, "path", None)
                if path_opt:
                    result = mgr.probe_transcript(path_opt, now_epoch=now_epoch)
                else:
                    result = mgr.status_summary()

            sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return 0

        radar_handler.configure_parser = _configure_radar_parser  # type: ignore[attr-defined]
        return radar_handler

    def register(self, kernel: Any, engine: Any | None = None) -> None:
        if engine is not None:
            self._engine = engine
            self._manager = ThermalSyncManager(engine)
        elif self._engine is None:
            attached = getattr(kernel, "engine", None)
            if attached is not None:
                self._engine = attached
                self._manager = ThermalSyncManager(attached)

        handler = self._build_cli_handler()
        kernel.register_cli(
            "radar",
            handler,
            help_text="Inspect 250s KV cache thermal state and sync transcript tails",
            plugin_name=self.name,
        )

    def __call__(self, kernel: Any, engine: Any | None = None) -> None:
        self.register(kernel, engine=engine)


def create_plugin() -> ThermalSyncPlugin:
    """Entry-point factory returning a ThermalSyncPlugin instance."""
    return ThermalSyncPlugin()


__all__ = [
    "ACTIVE_GRACE_SEC",
    "CACHE_SAFE_TTL_SEC",
    "FILE_ATTRIBUTE_REPARSE_POINT",
    "TAIL_READ_BYTES",
    "TIER_2X_THRESHOLD_TOKENS",
    "TIER_2X_WARN_TOKENS",
    "ThermalSnapshot",
    "ThermalSyncManager",
    "ThermalSyncPlugin",
    "TranscriptCursor",
    "WARN_IDLE_SEC",
    "create_plugin",
    "evaluate_thermal_state",
    "has_reparse_or_symlink",
    "read_shared_tail_bytes",
]
