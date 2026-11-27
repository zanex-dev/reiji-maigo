"""Optional self-bootstrapping companion installer plugin for reiji-maigo.

Provides cross-process file locking, corrupt hooks.json backup protection,
POSIX-slash Python executable normalization, and non-destructive reversible
installation/uninstallation of managed hooks and the reiji-memory skill.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
from typing import Any

ENV_COMPANION_CONFIG_DIR: str = "REIJI_COMPANION_CONFIG_DIR"
MANAGED_HOOK_IDS: tuple[str, ...] = ("reiji-pool-snapshot", "reiji-turn-sync")
SKILL_DIR_NAME: str = "reiji-memory"
MANAGED_BY_TOKEN: str = "reiji-maigo"
MANAGED_SKILL_MARKER: str = "<!-- reiji-maigo:managed-skill id=reiji-memory -->"
HOOK_ID_FIELD: str = "_reiji_hook_id"
MANAGED_BY_FIELD: str = "_managed_by"

_MANAGED_HOOK_ID_SET: frozenset[str] = frozenset(MANAGED_HOOK_IDS)


def resolve_companion_config_dir(
    config_dir: Path | str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve target companion config directory without touching host files."""
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    environ = os.environ if env is None else env
    raw = (environ.get(ENV_COMPANION_CONFIG_DIR) or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".gemini" / "config").resolve()


def format_python_exe(python_exe: str | None = None) -> str:
    """Normalize a Python executable path to forward-slash POSIX notation."""
    raw = sys.executable if python_exe is None or not str(python_exe).strip() else str(python_exe).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
        raw = raw[1:-1].strip()
    normalized = raw.replace("\\", "/")
    if not normalized:
        normalized = Path(sys.executable).as_posix()
    # Preserve drive letters on Windows and clean redundant separators cross-platform.
    if os.name == "nt":
        return Path(normalized).as_posix()
    return str(PurePosixPath(normalized))


def format_python_command(
    args: str | Sequence[str],
    *,
    python_exe: str | None = None,
) -> str:
    """Build a shell command invoking Python, auto-quoting paths containing spaces."""
    exe = format_python_exe(python_exe)
    quoted_exe = f'"{exe}"' if any(ch.isspace() for ch in exe) else exe
    if isinstance(args, str):
        suffix = args.strip()
    else:
        suffix = " ".join(str(part) for part in args if str(part).strip())
    return f"{quoted_exe} {suffix}".strip()


def _resolve_lock_path(lock_target: Path | str) -> Path:
    target = Path(lock_target)
    if target.exists() and target.is_dir():
        return target / ".reiji_companion.lock"
    if target.suffix == ".lock":
        return target
    return target.parent / f"{target.name}.lock"


@contextmanager
def interprocess_file_lock(
    lock_target: Path | str,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> Iterator[Path]:
    """Acquire an exclusive inter-process advisory/byte-range file lock.

    Uses a sidecar ``.lock`` file so Windows ``os.replace`` on the target data
    file is never blocked by an open handle on the target itself.
    """
    lock_path = _resolve_lock_path(lock_target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout))
    sleep_step = max(0.005, float(poll_interval))

    fp = open(lock_path, "a+b")
    locked = False
    try:
        fp.seek(0, os.SEEK_END)
        if fp.tell() == 0:
            try:
                fp.write(b"\0")
                fp.flush()
            except OSError:
                pass

        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    fp.seek(0)
                    msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring inter-process lock on {lock_path}"
                        ) from exc
                    time.sleep(sleep_step)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Timed out acquiring inter-process lock on {lock_path}"
                        ) from exc
                    time.sleep(sleep_step)

        yield lock_path
    finally:
        try:
            if locked:
                if os.name == "nt":
                    import msvcrt

                    try:
                        fp.seek(0)
                        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
        finally:
            fp.close()


def atomic_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write UTF-8 text using a same-directory temp file + fsync + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        tmp_path = None
        return target
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def render_skill_template() -> str:
    """Render the managed reiji-memory SKILL.md template."""
    return (
        "---\n"
        f"name: {SKILL_DIR_NAME}\n"
        "description: Query reiji-maigo hot memory pool, structured memory cards, and grounded turn evidence.\n"
        "---\n\n"
        f"{MANAGED_SKILL_MARKER}\n\n"
        "# Reiji-Maigo Memory Companion\n\n"
        "Use the `reiji_maigo` CLI to inspect hot pool state and recall grounded memory:\n\n"
        "- Hot pool snapshot: `python -m reiji_maigo pool`\n"
        "- Recall cards by domain & tags: `python -m reiji_maigo recall --domain <domain> --tag <tag>`\n"
        "- Deep evidence slice: `python -m reiji_maigo recall --query \"<keywords>\" --deep`\n"
        "- Swarm pulse & scribe sync: `python -m reiji_maigo swarm status` / `python -m reiji_maigo swarm scribe --json`\n"
    )


def render_mcp_server_config(
    python_exe: str | None = None,
    *,
    home: Path | str | None = None,
) -> dict[str, Any]:
    """Render a host-compatible ``mcpServers`` JSON configuration for ``reiji mcp``."""
    import sys

    resolved_exe = str(python_exe).strip() if python_exe else sys.executable
    if not resolved_exe:
        resolved_exe = "python"

    cmd_args: list[str] = ["-m", "reiji_maigo"]
    env_map: dict[str, str] = {}
    if home is not None and str(home).strip():
        resolved_home = str(Path(home).expanduser())
        cmd_args.extend(["--home", resolved_home])
        env_map["REIJI_MAIGO_HOME"] = resolved_home
    cmd_args.append("mcp")

    server_entry: dict[str, Any] = {
        "command": resolved_exe,
        "args": cmd_args,
    }
    if env_map:
        server_entry["env"] = env_map

    return {
        "mcpServers": {
            "reiji-maigo": server_entry,
        }
    }


def _build_managed_hook_specs(python_exe: str) -> dict[str, tuple[str, dict[str, Any]]]:
    """Return mapping of hook_id -> (event_name, managed_entry_dict)."""
    snapshot_cmd = format_python_command("-m reiji_maigo pool --json", python_exe=python_exe)
    sync_cmd = format_python_command("-m reiji_maigo swarm scribe --json", python_exe=python_exe)
    return {
        "reiji-pool-snapshot": (
            "SessionStart",
            {
                "matcher": "*",
                HOOK_ID_FIELD: "reiji-pool-snapshot",
                MANAGED_BY_FIELD: MANAGED_BY_TOKEN,
                "hooks": [
                    {
                        "type": "command",
                        "command": snapshot_cmd,
                        "timeout": 10,
                        HOOK_ID_FIELD: "reiji-pool-snapshot",
                        MANAGED_BY_FIELD: MANAGED_BY_TOKEN,
                    }
                ],
            },
        ),
        "reiji-turn-sync": (
            "Stop",
            {
                "matcher": "*",
                HOOK_ID_FIELD: "reiji-turn-sync",
                MANAGED_BY_FIELD: MANAGED_BY_TOKEN,
                "hooks": [
                    {
                        "type": "command",
                        "command": sync_cmd,
                        "timeout": 15,
                        HOOK_ID_FIELD: "reiji-turn-sync",
                        MANAGED_BY_FIELD: MANAGED_BY_TOKEN,
                    }
                ],
            },
        ),
    }


def _extract_managed_hook_id(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get(HOOK_ID_FIELD)
    if isinstance(raw, str) and raw in _MANAGED_HOOK_ID_SET:
        return raw
    return None


def _scan_hooks_payload(payload: Mapping[str, Any]) -> tuple[list[str], int]:
    """Return (sorted managed hook ids present, count of third-party hook entries)."""
    hooks_section = payload.get("hooks")
    if not isinstance(hooks_section, Mapping):
        return [], 0
    found_ids: set[str] = set()
    third_party_count = 0
    for entries in hooks_section.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            managed_id = _extract_managed_hook_id(entry)
            if managed_id is not None:
                found_ids.add(managed_id)
                continue
            if isinstance(entry, dict) and isinstance(entry.get("hooks"), list):
                sub_items = entry["hooks"]
                nested_managed = False
                nested_third_party = 0
                for sub in sub_items:
                    sub_id = _extract_managed_hook_id(sub)
                    if sub_id is not None:
                        found_ids.add(sub_id)
                        nested_managed = True
                    else:
                        nested_third_party += 1
                if nested_third_party > 0 or not nested_managed:
                    third_party_count += 1
            else:
                third_party_count += 1
    ordered = [hid for hid in MANAGED_HOOK_IDS if hid in found_ids]
    return ordered, third_party_count


def _parse_hooks_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse and validate hooks.json root object structure."""
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON in hooks.json: {exc.msg} (line {exc.lineno}, col {exc.colno})"
    if not isinstance(parsed, dict):
        return None, "Invalid hooks.json structure: root JSON value must be an object"
    if "hooks" in parsed and not isinstance(parsed["hooks"], dict):
        return None, "Invalid hooks.json structure: 'hooks' field must be an object"
    return parsed, None


def _allocate_corrupt_backup_path(hooks_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = hooks_path.parent / f"{hooks_path.name}.corrupt-{stamp}.bak"
    counter = 1
    while candidate.exists():
        candidate = hooks_path.parent / f"{hooks_path.name}.corrupt-{stamp}-{counter}.bak"
        counter += 1
    return candidate


class CompanionInstaller:
    """Non-destructive, reversible installer for reiji-maigo companion hooks and skill."""

    def __init__(
        self,
        config_dir: Path | str | None = None,
        *,
        python_exe: str | None = None,
    ) -> None:
        self.config_dir: Path = resolve_companion_config_dir(config_dir)
        self.python_exe: str = format_python_exe(python_exe)

    @property
    def hooks_path(self) -> Path:
        return self.config_dir / "hooks.json"

    @property
    def skill_dir(self) -> Path:
        return self.config_dir / "skills" / SKILL_DIR_NAME

    @property
    def skill_path(self) -> Path:
        return self.skill_dir / "SKILL.md"

    def _inspect_skill(self) -> tuple[bool, bool]:
        """Return (exists, is_managed)."""
        if not self.skill_path.exists():
            return False, False
        try:
            text = self.skill_path.read_text(encoding="utf-8")
        except OSError:
            return True, False
        return True, MANAGED_SKILL_MARKER in text

    def status(self) -> dict[str, Any]:
        """Inspect current companion installation state without mutating disk."""
        hooks_exists = self.hooks_path.exists()
        hooks_valid = True
        hooks_error: str | None = None
        managed_present: list[str] = []
        third_party_count = 0

        if hooks_exists:
            raw_text = self.hooks_path.read_text(encoding="utf-8")
            parsed, hooks_error = _parse_hooks_json(raw_text)
            if parsed is None:
                hooks_valid = False
            else:
                managed_present, third_party_count = _scan_hooks_payload(parsed)

        missing = [hid for hid in MANAGED_HOOK_IDS if hid not in managed_present]
        skill_exists, skill_managed = self._inspect_skill()
        installed = hooks_valid and not missing and skill_exists and skill_managed

        return {
            "ok": True,
            "config_dir": str(self.config_dir),
            "python_exe": self.python_exe,
            "hooks_path": str(self.hooks_path),
            "hooks_file_exists": hooks_exists,
            "hooks_json_valid": hooks_valid,
            "hooks_error": hooks_error,
            "managed_hooks_present": managed_present,
            "managed_hooks_missing": missing,
            "third_party_hook_count": third_party_count,
            "skill_path": str(self.skill_path),
            "skill_installed": skill_exists,
            "skill_managed": skill_managed,
            "installed": installed,
        }

    def install(
        self,
        *,
        force: bool = False,
        backup_corrupt: bool = True,
    ) -> dict[str, Any]:
        """Non-destructively merge managed hooks and deploy reiji-memory SKILL.md."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with interprocess_file_lock(self.hooks_path):
            corrupt_backup_path: str | None = None
            hooks_file_existed = self.hooks_path.exists()
            hooks_dirty = not hooks_file_existed
            payload: dict[str, Any] = {"hooks": {}}

            if hooks_file_existed:
                raw_text = self.hooks_path.read_text(encoding="utf-8")
                parsed, parse_error = _parse_hooks_json(raw_text)
                if parsed is None:
                    if not backup_corrupt:
                        return {
                            "ok": False,
                            "action": "install",
                            "config_dir": str(self.config_dir),
                            "python_exe": self.python_exe,
                            "hooks_path": str(self.hooks_path),
                            "skill_path": str(self.skill_path),
                            "corrupt_hooks_json": True,
                            "corrupt_backup_path": None,
                            "error": (
                                f"{parse_error}. Refusing to overwrite without backup "
                                "(pass backup_corrupt=True to archive and reinitialize)."
                            ),
                        }
                    backup_target = _allocate_corrupt_backup_path(self.hooks_path)
                    atomic_write_text(backup_target, raw_text)
                    corrupt_backup_path = str(backup_target)
                    payload = {"hooks": {}}
                    hooks_dirty = True
                else:
                    payload = parsed
                    if "hooks" not in payload:
                        payload["hooks"] = {}
                        hooks_dirty = True

            hooks_map: dict[str, Any] = payload["hooks"]
            specs = _build_managed_hook_specs(self.python_exe)
            hooks_added: list[str] = []
            hooks_updated: list[str] = []
            hooks_unchanged: list[str] = []

            for hook_id in MANAGED_HOOK_IDS:
                event_name, desired_entry = specs[hook_id]
                raw_bucket = hooks_map.get(event_name)
                if raw_bucket is None:
                    bucket: list[Any] = []
                    hooks_map[event_name] = bucket
                elif isinstance(raw_bucket, list):
                    bucket = raw_bucket
                else:
                    # Preserve non-list user value in a list rather than dropping it.
                    bucket = [raw_bucket]
                    hooks_map[event_name] = bucket
                    hooks_dirty = True

                match_indices = [
                    idx
                    for idx, item in enumerate(bucket)
                    if _extract_managed_hook_id(item) == hook_id
                ]
                if not match_indices:
                    bucket.append(desired_entry)
                    hooks_added.append(hook_id)
                    hooks_dirty = True
                else:
                    primary_idx = match_indices[0]
                    # Deduplicate extra copies of the same managed hook id if present.
                    if len(match_indices) > 1:
                        for dup_idx in reversed(match_indices[1:]):
                            bucket.pop(dup_idx)
                        hooks_dirty = True
                    if force or bucket[primary_idx] != desired_entry:
                        bucket[primary_idx] = desired_entry
                        hooks_updated.append(hook_id)
                        hooks_dirty = True
                    else:
                        hooks_unchanged.append(hook_id)

            if hooks_dirty or force:
                rendered_hooks = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                atomic_write_text(self.hooks_path, rendered_hooks)

            _, third_party_preserved = _scan_hooks_payload(payload)

            # Deploy or update managed SKILL.md without clobbering custom user files.
            template_text = render_skill_template()
            skill_exists, skill_managed = self._inspect_skill()
            skill_changed = False
            if not skill_exists:
                atomic_write_text(self.skill_path, template_text)
                skill_action = "installed"
                skill_changed = True
            else:
                current_text = self.skill_path.read_text(encoding="utf-8")
                if not skill_managed and not force:
                    skill_action = "preserved_custom"
                elif current_text == template_text and not force:
                    skill_action = "unchanged"
                else:
                    atomic_write_text(self.skill_path, template_text)
                    skill_action = "updated"
                    skill_changed = True

            return {
                "ok": True,
                "action": "install",
                "config_dir": str(self.config_dir),
                "python_exe": self.python_exe,
                "hooks_path": str(self.hooks_path),
                "skill_path": str(self.skill_path),
                "hooks_added": hooks_added,
                "hooks_updated": hooks_updated,
                "hooks_unchanged": hooks_unchanged,
                "third_party_hooks_preserved": third_party_preserved,
                "corrupt_backup_path": corrupt_backup_path,
                "skill_action": skill_action,
                "changed": bool(hooks_dirty or skill_changed),
            }

    def uninstall(self) -> dict[str, Any]:
        """Reversibly remove only managed hooks and managed SKILL.md."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with interprocess_file_lock(self.hooks_path):
            hooks_removed: list[str] = []
            third_party_preserved = 0
            hooks_changed = False

            if self.hooks_path.exists():
                raw_text = self.hooks_path.read_text(encoding="utf-8")
                parsed, parse_error = _parse_hooks_json(raw_text)
                if parsed is None:
                    return {
                        "ok": False,
                        "action": "uninstall",
                        "config_dir": str(self.config_dir),
                        "hooks_path": str(self.hooks_path),
                        "skill_path": str(self.skill_path),
                        "corrupt_hooks_json": True,
                        "error": f"{parse_error}. Refusing destructive uninstall on corrupt hooks.json.",
                    }

                hooks_map = parsed.get("hooks")
                if isinstance(hooks_map, dict):
                    empty_managed_events: list[str] = []
                    for event_name, entries in list(hooks_map.items()):
                        if not isinstance(entries, list):
                            continue
                        new_entries: list[Any] = []
                        event_removed_any = False
                        for entry in entries:
                            managed_id = _extract_managed_hook_id(entry)
                            if managed_id is not None:
                                if managed_id not in hooks_removed:
                                    hooks_removed.append(managed_id)
                                event_removed_any = True
                                hooks_changed = True
                                continue
                            if isinstance(entry, dict) and isinstance(entry.get("hooks"), list):
                                kept_subs: list[Any] = []
                                sub_removed = False
                                for sub in entry["hooks"]:
                                    sub_id = _extract_managed_hook_id(sub)
                                    if sub_id is not None:
                                        if sub_id not in hooks_removed:
                                            hooks_removed.append(sub_id)
                                        sub_removed = True
                                        event_removed_any = True
                                        hooks_changed = True
                                    else:
                                        kept_subs.append(sub)
                                if sub_removed:
                                    if kept_subs:
                                        updated_entry = dict(entry)
                                        updated_entry["hooks"] = kept_subs
                                        new_entries.append(updated_entry)
                                else:
                                    new_entries.append(entry)
                            else:
                                new_entries.append(entry)

                        if event_removed_any:
                            if new_entries:
                                hooks_map[event_name] = new_entries
                            else:
                                empty_managed_events.append(event_name)

                    for event_name in empty_managed_events:
                        hooks_map.pop(event_name, None)

                _, third_party_preserved = _scan_hooks_payload(parsed)
                if hooks_changed:
                    rendered_hooks = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
                    atomic_write_text(self.hooks_path, rendered_hooks)

            skill_exists, skill_managed = self._inspect_skill()
            skill_changed = False
            if not skill_exists:
                skill_action = "absent"
            elif skill_managed:
                self.skill_path.unlink(missing_ok=True)
                skill_action = "removed"
                skill_changed = True
                # Remove skill directory only if completely empty (preserve user extras).
                try:
                    if self.skill_dir.exists() and not any(self.skill_dir.iterdir()):
                        self.skill_dir.rmdir()
                except OSError:
                    pass
            else:
                skill_action = "preserved_custom"

            return {
                "ok": True,
                "action": "uninstall",
                "config_dir": str(self.config_dir),
                "hooks_path": str(self.hooks_path),
                "skill_path": str(self.skill_path),
                "hooks_removed": hooks_removed,
                "third_party_hooks_preserved": third_party_preserved,
                "skill_action": skill_action,
                "changed": bool(hooks_changed or skill_changed),
            }


def _configure_companion_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "companion_action",
        nargs="?",
        choices=("status", "install", "uninstall", "mcp-config"),
        default="status",
        help="Companion installer action (default: status)",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help=f"Target companion config directory (overrides {ENV_COMPANION_CONFIG_DIR})",
    )
    parser.add_argument(
        "--python-exe",
        default=None,
        help="Explicit Python executable path used in generated hook commands",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rewrite managed hooks and skill template during install",
    )
    parser.add_argument(
        "--no-backup-corrupt",
        action="store_true",
        help="Refuse to overwrite corrupt hooks.json instead of creating a .corrupt-*.bak archive",
    )
    if "--json" not in getattr(parser, "_option_string_actions", {}):
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit structured JSON output",
        )


def _format_cli_summary(result: Mapping[str, Any]) -> str:
    if not result.get("ok", False):
        return f"Companion {result.get('action', 'operation')} failed: {result.get('error', 'unknown error')}"
    action = result.get("action")
    if action == "install":
        backup_note = (
            f", corrupt_backup={result['corrupt_backup_path']}"
            if result.get("corrupt_backup_path")
            else ""
        )
        return (
            f"Companion installed in {result['config_dir']} "
            f"(added={len(result['hooks_added'])}, updated={len(result['hooks_updated'])}, "
            f"unchanged={len(result['hooks_unchanged'])}, "
            f"third_party_preserved={result['third_party_hooks_preserved']}, "
            f"skill={result['skill_action']}{backup_note})"
        )
    if action == "uninstall":
        return (
            f"Companion uninstalled from {result['config_dir']} "
            f"(hooks_removed={len(result['hooks_removed'])}, "
            f"third_party_preserved={result['third_party_hooks_preserved']}, "
            f"skill={result['skill_action']})"
        )
    return (
        f"Companion status ({result['config_dir']}): installed={result['installed']}, "
        f"hooks_present={len(result['managed_hooks_present'])}/{len(MANAGED_HOOK_IDS)}, "
        f"third_party_hooks={result['third_party_hook_count']}, "
        f"skill_managed={result['skill_managed']}"
    )


class CompanionPlugin:
    """Self-bootstrapping optional plugin registering the `companion` CLI command."""

    name: str = "companion"
    version: str = "0.1.0"
    description: str = "Non-destructive reversible companion hooks & skill installer"

    def __init__(
        self,
        *,
        config_dir: Path | str | None = None,
        python_exe: str | None = None,
    ) -> None:
        self._default_config_dir = config_dir
        self._default_python_exe = python_exe

    def register(self, kernel: Any) -> None:
        def _companion_cli(*call_args: Any, **kwargs: Any) -> int:
            if len(call_args) == 1:
                args = call_args[0]
            elif len(call_args) >= 2:
                args = call_args[1]
            else:
                args = kwargs["args"]

            config_dir = getattr(args, "config_dir", None)
            if config_dir is None:
                config_dir = self._default_config_dir
            python_exe = getattr(args, "python_exe", None)
            if python_exe is None:
                python_exe = self._default_python_exe

            installer = CompanionInstaller(config_dir=config_dir, python_exe=python_exe)
            action = getattr(args, "companion_action", None) or "status"
            if action == "mcp-config":
                mcp_home = getattr(args, "home", None)
                mcp_cfg = render_mcp_server_config(python_exe=python_exe, home=mcp_home)
                if getattr(args, "json", False):
                    print(json.dumps(mcp_cfg, ensure_ascii=False, indent=2))
                else:
                    print("MCP server configuration (add to your host mcpServers config):")
                    print(json.dumps(mcp_cfg, ensure_ascii=False, indent=2))
                return 0
            if action == "install":
                result = installer.install(
                    force=bool(getattr(args, "force", False)),
                    backup_corrupt=not bool(getattr(args, "no_backup_corrupt", False)),
                )
            elif action == "uninstall":
                result = installer.uninstall()
            else:
                result = installer.status()

            if getattr(args, "json", False):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(_format_cli_summary(result))
            return 0 if result.get("ok", False) else 1

        _companion_cli.configure_parser = _configure_companion_parser  # type: ignore[attr-defined]
        kernel.register_cli(
            "companion",
            _companion_cli,
            help_text="Install, inspect, or uninstall companion hooks and reiji-memory skill",
            plugin_name=self.name,
        )


def create_plugin() -> CompanionPlugin:
    """Plugin factory entry point for reiji-maigo."""
    return CompanionPlugin()


__all__ = [
    "CompanionInstaller",
    "CompanionPlugin",
    "ENV_COMPANION_CONFIG_DIR",
    "HOOK_ID_FIELD",
    "MANAGED_BY_FIELD",
    "MANAGED_BY_TOKEN",
    "MANAGED_HOOK_IDS",
    "MANAGED_SKILL_MARKER",
    "SKILL_DIR_NAME",
    "atomic_write_text",
    "create_plugin",
    "format_python_command",
    "format_python_exe",
    "interprocess_file_lock",
    "render_mcp_server_config",
    "render_skill_template",
    "resolve_companion_config_dir",
]
