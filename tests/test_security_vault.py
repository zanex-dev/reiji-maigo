"""Unit tests for reiji_maigo.security.vault."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from reiji_maigo.security import vault
from reiji_maigo.security.vault import (
    AAD_AUDIT,
    AAD_BUNDLE,
    AAD_CARD,
    SecretVault,
    canonical_json_bytes,
    derive_domain_key,
    export_safe_bundle,
    import_safe_bundle,
    sign_canonical,
    verify_canonical,
)


def _create_sample_sqlite(db_path: Path, rows: list[tuple[str, str]]) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, val TEXT NOT NULL)"
        )
        conn.executemany("INSERT INTO memories (key, val) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _read_sample_sqlite(db_path: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT key, val FROM memories ORDER BY id ASC")
        return list(cur.fetchall())
    finally:
        conn.close()


class TestDomainSeparatedCrypto(unittest.TestCase):
    def test_domain_separated_hmac_roundtrip_and_isolation(self) -> None:
        key = b"\x11" * 32
        payload_a = {"b": 2, "a": 1, "nested": {"z": "中文", "y": True}}
        payload_b = {"nested": {"y": True, "z": "中文"}, "a": 1, "b": 2}

        self.assertEqual(canonical_json_bytes(payload_a), canonical_json_bytes(payload_b))

        sig_card = sign_canonical(payload_a, key, domain_tag=AAD_CARD)
        sig_audit = sign_canonical(payload_a, key, domain_tag=AAD_AUDIT)
        sig_bundle = sign_canonical(payload_a, key, domain_tag=AAD_BUNDLE)

        self.assertNotEqual(sig_card, sig_audit)
        self.assertNotEqual(sig_card, sig_bundle)
        self.assertNotEqual(sig_audit, sig_bundle)

        self.assertTrue(verify_canonical(payload_b, sig_card, key, domain_tag=AAD_CARD))
        self.assertFalse(verify_canonical(payload_b, sig_card, key, domain_tag=AAD_AUDIT))
        self.assertFalse(verify_canonical(payload_b, sig_card, key, domain_tag=AAD_BUNDLE))

        tampered = dict(payload_a, a=999)
        self.assertFalse(verify_canonical(tampered, sig_card, key, domain_tag=AAD_CARD))
        self.assertFalse(verify_canonical(payload_a, "not-hex", key, domain_tag=AAD_CARD))
        self.assertFalse(verify_canonical(payload_a, sig_card, b"\x22" * 32, domain_tag=AAD_CARD))

    def test_pbkdf2_domain_key_derivation(self) -> None:
        salt = b"unit-test-salt-16"
        k1 = derive_domain_key("master-pass", salt, domain_tag=AAD_CARD, iterations=1000)
        k2 = derive_domain_key("master-pass", salt, domain_tag=AAD_CARD, iterations=1000)
        k3 = derive_domain_key("master-pass", salt, domain_tag=AAD_AUDIT, iterations=1000)

        self.assertEqual(len(k1), 32)
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, k3)

        with self.assertRaises(ValueError):
            derive_domain_key("master-pass", salt, domain_tag=b"bad\x00tag", iterations=1000)


class TestSecretVault(unittest.TestCase):
    def test_save_and_load_roundtrip_and_at_rest_protection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            sv = SecretVault(vpath)
            self.assertEqual(sv.load(), {})

            secret_payload = {"token": "super-secret-value-999", "nested": {"k": "v"}}
            sv.save(secret_payload)
            self.assertTrue(vpath.is_file())

            raw_on_disk = vpath.read_text(encoding="utf-8")
            self.assertNotIn("super-secret-value-999", raw_on_disk)
            disk_json = json.loads(raw_on_disk)
            if sys.platform == "win32":
                self.assertEqual(disk_json["_vault"], "dpapi_v2")
            else:
                self.assertEqual(disk_json["_vault"], "posix_0600_v2")

            self.assertEqual(sv.load(), secret_payload)
            self.assertEqual(sv.load(strict=True), secret_payload)

    def test_posix_0600_branch_and_chmod_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            sv = SecretVault(vpath)
            chmod_calls: list[tuple[str, int]] = []

            def _fake_chmod(path: str, mode: int) -> None:
                chmod_calls.append((str(path), mode))

            with mock.patch.object(vault.sys, "platform", "linux"), mock.patch.object(
                vault.os, "chmod", side_effect=_fake_chmod
            ):
                sv.save({"api_key": "posix-secret"})
                disk_json = json.loads(vpath.read_text(encoding="utf-8"))
                self.assertEqual(disk_json["_vault"], "posix_0600_v2")
                self.assertEqual(sv.load(strict=True), {"api_key": "posix-secret"})

            self.assertGreaterEqual(len(chmod_calls), 2)
            self.assertTrue(all(mode == 0o600 for _, mode in chmod_calls))

    def test_corrupted_vault_strict_refusal_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            corrupted_bytes = b"{not-valid-json!!!"
            vpath.write_bytes(corrupted_bytes)

            sv = SecretVault(vpath)
            self.assertEqual(sv.load(strict=False), {})

            with self.assertRaises(ValueError):
                sv.load(strict=True)
            with self.assertRaises(ValueError):
                sv.load(for_update=True)
            with self.assertRaises(ValueError):
                sv.get_or_create_signing_key()

            # Verify original corrupted file was never overwritten.
            self.assertEqual(vpath.read_bytes(), corrupted_bytes)

    def test_tampered_encrypted_blob_refused_on_strict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            sv = SecretVault(vpath)
            sv.save({"a": 1})

            env = json.loads(vpath.read_text(encoding="utf-8"))
            env["blob"] = "00" * 16 if env["_vault"] == "dpapi_v2" else "%%%not-base64%%%"
            tampered_text = json.dumps(env)
            vpath.write_text(tampered_text, encoding="utf-8")

            with self.assertRaises(ValueError):
                sv.load(strict=True)
            with self.assertRaises(ValueError):
                sv.get_or_create_signing_key()
            self.assertEqual(vpath.read_text(encoding="utf-8"), tampered_text)

    def test_get_or_create_signing_key_persists_32_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            sv = SecretVault(vpath)
            key1 = sv.get_or_create_signing_key()
            key2 = sv.get_or_create_signing_key()

            self.assertIsInstance(key1, bytes)
            self.assertEqual(len(key1), 32)
            self.assertEqual(key1, key2)

    def test_legacy_plaintext_vault_upgrades_automatically(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vpath = Path(td) / ".reiji_vault.json"
            vpath.write_text(json.dumps({"legacy": "hello"}), encoding="utf-8")
            sv = SecretVault(vpath)
            loaded = sv.load()
            self.assertEqual(loaded, {"legacy": "hello"})

            upgraded = json.loads(vpath.read_text(encoding="utf-8"))
            self.assertIn(upgraded.get("_vault"), {"dpapi_v2", "posix_0600_v2"})


class TestSafeBundle(unittest.TestCase):
    def test_export_import_roundtrip_and_idempotent_overwrite_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.db"
            bundle = root / "export.zip"
            dst_db = root / "restored.db"
            rows = [("k1", "v1"), ("k2", "记忆归档测试")]
            _create_sample_sqlite(src_db, rows)

            key = SecretVault(root / "vault.json").get_or_create_signing_key()
            exp = export_safe_bundle(src_db, bundle, signing_key=key)
            self.assertTrue(bundle.is_file())
            self.assertEqual(exp["db_member"], "memory.sqlite3")

            with zipfile.ZipFile(bundle, "r") as zf:
                for info in zf.infolist():
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED)

            res1 = import_safe_bundle(bundle, dst_db, signing_key=key)
            self.assertEqual(res1["status"], "imported")
            self.assertFalse(res1["reused"])
            self.assertEqual(_read_sample_sqlite(dst_db), rows)

            # Idempotent re-import reuses identical target without error.
            res2 = import_safe_bundle(bundle, dst_db, signing_key=key, overwrite=False)
            self.assertEqual(res2["status"], "idempotent_reused")
            self.assertTrue(res2["reused"])

            # Modify target DB -> refuse overwrite when overwrite=False.
            conn = sqlite3.connect(str(dst_db))
            try:
                conn.execute("INSERT INTO memories (key, val) VALUES (?, ?)", ("k3", "local-edit"))
                conn.commit()
            finally:
                conn.close()

            with self.assertRaises(FileExistsError):
                import_safe_bundle(bundle, dst_db, signing_key=key, overwrite=False)
            self.assertEqual(len(_read_sample_sqlite(dst_db)), 3)

            # With overwrite=True -> replaces cleanly.
            res3 = import_safe_bundle(bundle, dst_db, signing_key=key, overwrite=True)
            self.assertEqual(res3["status"], "overwritten")
            self.assertEqual(_read_sample_sqlite(dst_db), rows)

    def test_rejects_compressed_zip_bomb_entries(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.db"
            good_bundle = root / "good.zip"
            deflated_bundle = root / "deflated.zip"
            dst_db = root / "dst.db"
            _create_sample_sqlite(src_db, [("a", "b")])
            key = b"\x33" * 32

            export_safe_bundle(src_db, good_bundle, signing_key=key)
            with zipfile.ZipFile(good_bundle, "r") as zin, zipfile.ZipFile(
                deflated_bundle, "w", compression=zipfile.ZIP_DEFLATED
            ) as zout:
                for info in zin.infolist():
                    zout.writestr(info.filename, zin.read(info.filename), compress_type=zipfile.ZIP_DEFLATED)

            with self.assertRaisesRegex(ValueError, "anti-decompression-bomb"):
                import_safe_bundle(deflated_bundle, dst_db, signing_key=key)
            self.assertFalse(dst_db.exists())

    def test_rejects_path_traversal_member(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evil_bundle = root / "evil.zip"
            dst_db = root / "dst.db"
            key = b"\x44" * 32

            with zipfile.ZipFile(evil_bundle, "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr("../escape.db", b"evil", compress_type=zipfile.ZIP_STORED)
                zf.writestr("manifest.json", b"{}", compress_type=zipfile.ZIP_STORED)

            with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                import_safe_bundle(evil_bundle, dst_db, signing_key=key)
            self.assertFalse(dst_db.exists())

    def test_enforces_max_total_bytes_circuit_breaker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.db"
            bundle = root / "bundle.zip"
            dst_db = root / "dst.db"
            _create_sample_sqlite(src_db, [("big", "X" * 10000)])
            key = b"\x55" * 32

            export_safe_bundle(src_db, bundle, signing_key=key)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                import_safe_bundle(bundle, dst_db, signing_key=key, max_total_bytes=2048)
            self.assertFalse(dst_db.exists())

    def test_rejects_bad_signature_or_tampered_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.db"
            bundle = root / "bundle.zip"
            tampered_bundle = root / "tampered.zip"
            dst_db = root / "dst.db"
            _create_sample_sqlite(src_db, [("k", "v")])
            key = b"\x66" * 32

            export_safe_bundle(src_db, bundle, signing_key=key)

            # Wrong key fails HMAC check.
            with self.assertRaisesRegex(ValueError, "HMAC verification failed"):
                import_safe_bundle(bundle, dst_db, signing_key=b"\x77" * 32)
            self.assertFalse(dst_db.exists())

            # Tampered DB byte with same length fails SHA-256 check.
            with zipfile.ZipFile(bundle, "r") as zin, zipfile.ZipFile(
                tampered_bundle, "w", compression=zipfile.ZIP_STORED
            ) as zout:
                manifest_raw = zin.read("manifest.json")
                db_raw = bytearray(zin.read("memory.sqlite3"))
                db_raw[-1] ^= 0xFF
                zout.writestr("manifest.json", manifest_raw, compress_type=zipfile.ZIP_STORED)
                zout.writestr("memory.sqlite3", bytes(db_raw), compress_type=zipfile.ZIP_STORED)

            with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
                import_safe_bundle(tampered_bundle, dst_db, signing_key=key)
            self.assertFalse(dst_db.exists())

    def test_rejects_duplicate_and_extra_members(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src_db = root / "src.db"
            bundle = root / "bundle.zip"
            extra_bundle = root / "extra.zip"
            dup_bundle = root / "dup.zip"
            dst_db = root / "dst.db"
            _create_sample_sqlite(src_db, [("k", "v")])
            key = b"\x88" * 32

            export_safe_bundle(src_db, bundle, signing_key=key)
            with zipfile.ZipFile(bundle, "r") as zin:
                m_raw = zin.read("manifest.json")
                d_raw = zin.read("memory.sqlite3")

            with zipfile.ZipFile(extra_bundle, "w", compression=zipfile.ZIP_STORED) as zout:
                zout.writestr("manifest.json", m_raw, compress_type=zipfile.ZIP_STORED)
                zout.writestr("memory.sqlite3", d_raw, compress_type=zipfile.ZIP_STORED)
                zout.writestr("sneaky.txt", b"extra", compress_type=zipfile.ZIP_STORED)

            with self.assertRaisesRegex(ValueError, "Unexpected extra archive members"):
                import_safe_bundle(extra_bundle, dst_db, signing_key=key)

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zout:
                zout.writestr("manifest.json", m_raw, compress_type=zipfile.ZIP_STORED)
                zout.writestr("manifest.json", m_raw, compress_type=zipfile.ZIP_STORED)
                zout.writestr("memory.sqlite3", d_raw, compress_type=zipfile.ZIP_STORED)
            dup_bundle.write_bytes(buf.getvalue())

            with self.assertRaisesRegex(ValueError, "Duplicate archive member"):
                import_safe_bundle(dup_bundle, dst_db, signing_key=key)


if __name__ == "__main__":
    unittest.main()
