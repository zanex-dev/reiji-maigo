"""Unit tests for reiji_maigo.plugins.companion (T04)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from reiji_maigo.core.kernel import ReijiKernel
from reiji_maigo.plugins.companion import (
    CompanionInstaller,
    CompanionPlugin,
    ENV_COMPANION_CONFIG_DIR,
    HOOK_ID_FIELD,
    MANAGED_HOOK_IDS,
    MANAGED_SKILL_MARKER,
    SKILL_DIR_NAME,
    atomic_write_text,
    create_plugin,
    format_python_command,
    format_python_exe,
    interprocess_file_lock,
    resolve_companion_config_dir,
)


class TestCompanionHelpers(unittest.TestCase):
    def test_format_python_exe_posix_and_space_quoting(self) -> None:
        raw_win = r"C:\Program Files\Python312\python.exe"
        normalized = format_python_exe(raw_win)
        self.assertNotIn("\\", normalized)
        self.assertEqual(normalized, "C:/Program Files/Python312/python.exe")

        cmd = format_python_command("-m reiji_maigo pool --json", python_exe=raw_win)
        self.assertEqual(cmd, '"C:/Program Files/Python312/python.exe" -m reiji_maigo pool --json')

        no_space = format_python_command("-m reiji_maigo status", python_exe=r"C:\Python312\python.exe")
        self.assertEqual(no_space, "C:/Python312/python.exe -m reiji_maigo status")

    def test_resolve_config_dir_respects_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_companion_config_dir(
                None,
                env={ENV_COMPANION_CONFIG_DIR: tmp},
            )
            self.assertEqual(resolved, Path(tmp).resolve())

    def test_atomic_write_text_replaces_cleanly_without_leftover_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "sample.json"
            atomic_write_text(target, '{"v": 1}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"v": 1}\n')

            atomic_write_text(target, '{"v": 2}\n')
            self.assertEqual(target.read_text(encoding="utf-8"), '{"v": 2}\n')
            leftovers = [p for p in target.parent.iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])

    def test_interprocess_file_lock_blocks_concurrent_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "hooks.json"
            repo_root = Path(__file__).resolve().parents[1]

            child_code = "\n".join(
                [
                    "from pathlib import Path",
                    "import sys",
                    f"sys.path.insert(0, {str(repo_root)!r})",
                    "from reiji_maigo.plugins.companion import interprocess_file_lock",
                    f"target = Path({str(target)!r})",
                    "try:",
                    "    with interprocess_file_lock(target, timeout=0.25, poll_interval=0.05):",
                    "        print('ACQUIRED')",
                    "except TimeoutError:",
                    "    print('LOCKED_OUT')",
                ]
            )

            with interprocess_file_lock(target, timeout=2.0):
                proc = subprocess.run(
                    [sys.executable, "-c", child_code],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(proc.stdout.strip(), "LOCKED_OUT")

            # After release, a child process can acquire the lock immediately.
            proc_after = subprocess.run(
                [sys.executable, "-c", child_code],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc_after.stdout.strip(), "ACQUIRED")

    def test_interprocess_file_lock_serializes_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "hooks.json"
            events: list[str] = []

            def worker(label: str, hold: float) -> None:
                with interprocess_file_lock(target, timeout=3.0, poll_interval=0.02):
                    events.append(f"enter-{label}")
                    time.sleep(hold)
                    events.append(f"exit-{label}")

            t1 = threading.Thread(target=worker, args=("a", 0.15))
            t2 = threading.Thread(target=worker, args=("b", 0.05))
            t1.start()
            time.sleep(0.03)
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            self.assertEqual(len(events), 4)
            # First thread to enter must exit before the second enters.
            first = events[0].split("-")[1]
            second = "b" if first == "a" else "a"
            self.assertEqual(events, [f"enter-{first}", f"exit-{first}", f"enter-{second}", f"exit-{second}"])


class TestCompanionInstaller(unittest.TestCase):
    def test_install_idempotent_reinstall_and_clean_uninstall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            installer = CompanionInstaller(
                tmp,
                python_exe=r"C:\Program Files\Python312\python.exe",
            )

            before = installer.status()
            self.assertTrue(before["ok"])
            self.assertFalse(before["installed"])
            self.assertEqual(before["managed_hooks_missing"], list(MANAGED_HOOK_IDS))

            res1 = installer.install()
            self.assertTrue(res1["ok"])
            self.assertEqual(res1["hooks_added"], list(MANAGED_HOOK_IDS))
            self.assertEqual(res1["hooks_updated"], [])
            self.assertEqual(res1["skill_action"], "installed")
            self.assertTrue(res1["changed"])

            # Verify hooks.json commands use forward slashes and quote space paths.
            hooks_data = json.loads(installer.hooks_path.read_text(encoding="utf-8"))
            start_cmd = hooks_data["hooks"]["SessionStart"][0]["hooks"][0]["command"]
            stop_cmd = hooks_data["hooks"]["Stop"][0]["hooks"][0]["command"]
            self.assertIn('"C:/Program Files/Python312/python.exe"', start_cmd)
            self.assertIn('"C:/Program Files/Python312/python.exe"', stop_cmd)
            self.assertNotIn("\\", start_cmd)
            self.assertNotIn("\\", stop_cmd)

            # Verify SKILL.md deployed under <config_dir>/skills/reiji-memory/SKILL.md.
            expected_skill = Path(tmp).resolve() / "skills" / SKILL_DIR_NAME / "SKILL.md"
            self.assertEqual(installer.skill_path, expected_skill)
            self.assertTrue(expected_skill.exists())
            self.assertIn(MANAGED_SKILL_MARKER, expected_skill.read_text(encoding="utf-8"))

            after = installer.status()
            self.assertTrue(after["installed"])
            self.assertEqual(after["managed_hooks_present"], list(MANAGED_HOOK_IDS))
            self.assertEqual(after["managed_hooks_missing"], [])
            self.assertTrue(after["skill_managed"])

            # Idempotent reinstall: zero duplicates, unchanged.
            res2 = installer.install()
            self.assertTrue(res2["ok"])
            self.assertEqual(res2["hooks_added"], [])
            self.assertEqual(res2["hooks_updated"], [])
            self.assertEqual(res2["hooks_unchanged"], list(MANAGED_HOOK_IDS))
            self.assertEqual(res2["skill_action"], "unchanged")
            self.assertFalse(res2["changed"])

            hooks_after_reinstall = json.loads(installer.hooks_path.read_text(encoding="utf-8"))
            self.assertEqual(len(hooks_after_reinstall["hooks"]["SessionStart"]), 1)
            self.assertEqual(len(hooks_after_reinstall["hooks"]["Stop"]), 1)

            # Force reinstall updates cleanly without duplicating.
            res_force = installer.install(force=True)
            self.assertTrue(res_force["ok"])
            self.assertEqual(res_force["hooks_updated"], list(MANAGED_HOOK_IDS))
            self.assertEqual(res_force["skill_action"], "updated")

            # Clean uninstall removes managed hooks and managed SKILL.md.
            un = installer.uninstall()
            self.assertTrue(un["ok"])
            self.assertEqual(un["hooks_removed"], list(MANAGED_HOOK_IDS))
            self.assertEqual(un["skill_action"], "removed")
            self.assertFalse(installer.skill_path.exists())
            self.assertFalse(installer.skill_dir.exists())

            post_status = installer.status()
            self.assertFalse(post_status["installed"])
            self.assertEqual(post_status["managed_hooks_present"], [])

    def test_preserves_third_party_hooks_and_custom_user_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            original_hooks = {
                "version": 2,
                "custom_setting": {"keep": True},
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "*",
                            "hooks": [{"type": "command", "command": "echo user-session-start"}],
                        }
                    ],
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [{"type": "command", "command": "echo guard-bash"}],
                        }
                    ],
                },
            }
            (config_dir / "hooks.json").write_text(
                json.dumps(original_hooks, indent=2),
                encoding="utf-8",
            )

            installer = CompanionInstaller(config_dir, python_exe="/opt/python3/bin/python3")
            res = installer.install()
            self.assertTrue(res["ok"])
            self.assertEqual(res["third_party_hooks_preserved"], 2)

            merged = json.loads(installer.hooks_path.read_text(encoding="utf-8"))
            self.assertEqual(merged["version"], 2)
            self.assertEqual(merged["custom_setting"], {"keep": True})
            self.assertEqual(len(merged["hooks"]["SessionStart"]), 2)
            self.assertEqual(merged["hooks"]["SessionStart"][0]["hooks"][0]["command"], "echo user-session-start")
            self.assertEqual(merged["hooks"]["SessionStart"][1][HOOK_ID_FIELD], "reiji-pool-snapshot")
            self.assertEqual(merged["hooks"]["PreToolUse"], original_hooks["hooks"]["PreToolUse"])

            # Drop a user extra file inside reiji-memory dir to verify uninstall keeps it.
            user_note = installer.skill_dir / "user_notes.md"
            user_note.write_text("keep my note", encoding="utf-8")

            un = installer.uninstall()
            self.assertTrue(un["ok"])
            self.assertEqual(un["hooks_removed"], list(MANAGED_HOOK_IDS))
            self.assertEqual(un["third_party_hooks_preserved"], 2)
            self.assertFalse(installer.skill_path.exists())
            self.assertTrue(user_note.exists())

            after_uninstall = json.loads(installer.hooks_path.read_text(encoding="utf-8"))
            self.assertEqual(after_uninstall["version"], 2)
            self.assertEqual(after_uninstall["custom_setting"], {"keep": True})
            self.assertEqual(after_uninstall["hooks"]["SessionStart"], original_hooks["hooks"]["SessionStart"])
            self.assertEqual(after_uninstall["hooks"]["PreToolUse"], original_hooks["hooks"]["PreToolUse"])
            self.assertNotIn("Stop", after_uninstall["hooks"])

    def test_preserves_unmanaged_custom_skill_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            installer = CompanionInstaller(tmp)
            installer.skill_dir.mkdir(parents=True, exist_ok=True)
            installer.skill_path.write_text("# My custom skill\n", encoding="utf-8")

            res = installer.install(force=False)
            self.assertTrue(res["ok"])
            self.assertEqual(res["skill_action"], "preserved_custom")
            self.assertEqual(installer.skill_path.read_text(encoding="utf-8"), "# My custom skill\n")

            un = installer.uninstall()
            self.assertTrue(un["ok"])
            self.assertEqual(un["skill_action"], "preserved_custom")
            self.assertTrue(installer.skill_path.exists())

    def test_corrupt_hooks_json_protection_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            corrupt_content = '{"hooks": { broken json !!!'
            hooks_file = config_dir / "hooks.json"
            hooks_file.write_text(corrupt_content, encoding="utf-8")

            installer = CompanionInstaller(config_dir)

            # status reports invalid JSON safely.
            st = installer.status()
            self.assertTrue(st["ok"])
            self.assertFalse(st["hooks_json_valid"])
            self.assertIsNotNone(st["hooks_error"])

            # backup_corrupt=False refuses overwrite and keeps original file intact.
            refused = installer.install(backup_corrupt=False)
            self.assertFalse(refused["ok"])
            self.assertTrue(refused["corrupt_hooks_json"])
            self.assertIsNone(refused["corrupt_backup_path"])
            self.assertEqual(hooks_file.read_text(encoding="utf-8"), corrupt_content)
            self.assertFalse(installer.skill_path.exists())

            # uninstall also refuses destructive modification on corrupt hooks.json.
            un_refused = installer.uninstall()
            self.assertFalse(un_refused["ok"])
            self.assertEqual(hooks_file.read_text(encoding="utf-8"), corrupt_content)

            # backup_corrupt=True atomically archives .corrupt-*.bak then installs cleanly.
            recovered = installer.install(backup_corrupt=True)
            self.assertTrue(recovered["ok"])
            backup_path_str = recovered["corrupt_backup_path"]
            self.assertIsNotNone(backup_path_str)
            backup_path = Path(backup_path_str)
            self.assertTrue(backup_path.exists())
            self.assertIn("hooks.json.corrupt-", backup_path.name)
            self.assertTrue(backup_path.name.endswith(".bak"))
            self.assertEqual(backup_path.read_text(encoding="utf-8"), corrupt_content)

            rebuilt = json.loads(hooks_file.read_text(encoding="utf-8"))
            self.assertIn("SessionStart", rebuilt["hooks"])
            self.assertIn("Stop", rebuilt["hooks"])
            self.assertTrue(installer.status()["installed"])


class TestCompanionPluginRegistration(unittest.TestCase):
    def test_plugin_registers_cli_and_supports_status_install_uninstall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kernel = ReijiKernel(home=Path(tmp) / "home")
            plugin = create_plugin()
            self.assertIsInstance(plugin, CompanionPlugin)
            self.assertEqual(plugin.name, "companion")

            plugin.register(kernel)
            spec = kernel.get_cli_command("companion")
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertEqual(spec.plugin_name, "companion")
            self.assertTrue(callable(getattr(spec.handler, "configure_parser", None)))

            config_dir = Path(tmp) / "companion_cfg"
            parser = argparse.ArgumentParser(prog="reiji-maigo companion")
            spec.handler.configure_parser(parser)

            # 1. install --json
            install_args = parser.parse_args(
                [
                    "install",
                    "--config-dir",
                    str(config_dir),
                    "--python-exe",
                    r"C:\Opt\My Python\python.exe",
                    "--json",
                ]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = spec.handler(None, install_args)
            self.assertEqual(rc, 0)
            install_payload = json.loads(buf.getvalue())
            self.assertTrue(install_payload["ok"])
            self.assertEqual(install_payload["action"], "install")

            # 2. status --json
            status_args = parser.parse_args(["status", "--config-dir", str(config_dir), "--json"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = spec.handler(None, status_args)
            self.assertEqual(rc, 0)
            status_payload = json.loads(buf.getvalue())
            self.assertTrue(status_payload["installed"])

            # 3. uninstall --json
            uninstall_args = parser.parse_args(["uninstall", "--config-dir", str(config_dir), "--json"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = spec.handler(None, uninstall_args)
            self.assertEqual(rc, 0)
            uninstall_payload = json.loads(buf.getvalue())
            self.assertTrue(uninstall_payload["ok"])
            self.assertEqual(uninstall_payload["hooks_removed"], list(MANAGED_HOOK_IDS))

            # 4. --no-backup-corrupt returns exit code 1 on corrupt JSON
            (config_dir / "hooks.json").write_text("{corrupt", encoding="utf-8")
            corrupt_args = parser.parse_args(
                ["install", "--config-dir", str(config_dir), "--no-backup-corrupt", "--json"]
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = spec.handler(None, corrupt_args)
            self.assertEqual(rc, 1)
            corrupt_payload = json.loads(buf.getvalue())
            self.assertFalse(corrupt_payload["ok"])


if __name__ == "__main__":
    unittest.main()
