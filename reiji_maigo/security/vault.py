"""Pure-stdlib security vault, domain-separated HMAC signing, and safe bundles.

Provides:
- ``SecretVault``: At-rest local credential store using Windows DPAPI via
  ``ctypes`` on win32 and atomic ``0600`` storage on POSIX, with strict
  corruption refusal on update paths.
- Domain-separated ``PBKDF2-HMAC-SHA256`` key derivation and canonical JSON
  ``HMAC-SHA256`` signing/verification (``AAD_CARD``, ``AAD_AUDIT``,
  ``AAD_BUNDLE``).
- ``export_safe_bundle`` / ``import_safe_bundle``: Consistent SQLite backup
  packaged into a ``ZIP_STORED`` (zero-compression) archive with HMAC-signed
  manifest, anti-decompression-bomb streaming limits, path-traversal guards,
  and SHA-256 idempotent materialization.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import sys
import tempfile
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AAD_CARD: bytes = b"reiji-card-v1"
AAD_AUDIT: bytes = b"reiji-audit-v1"
AAD_BUNDLE: bytes = b"reiji-bundle-v1"

DEFAULT_VAULT_FILENAME: str = ".reiji_vault.json"
DEFAULT_MAX_BUNDLE_BYTES: int = 64 * 1024 * 1024
_MAX_MANIFEST_BYTES: int = 1024 * 1024
_STREAM_CHUNK_BYTES: int = 1024 * 1024
_DPAPI_ENTROPY: bytes = b"reiji-maigo-vault-dpapi-v1"
_BUNDLE_FORMAT: str = "reiji-safe-bundle-v1"
_DB_MEMBER_NAME: str = "memory.sqlite3"
_MANIFEST_MEMBER_NAME: str = "manifest.json"
_SQLITE_MAGIC: bytes = b"SQLite format 3\x00"
_SAFE_INNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

__all__ = [
    "AAD_CARD",
    "AAD_AUDIT",
    "AAD_BUNDLE",
    "DEFAULT_MAX_BUNDLE_BYTES",
    "DEFAULT_VAULT_FILENAME",
    "SecretVault",
    "canonical_json_bytes",
    "derive_domain_key",
    "sign_canonical",
    "verify_canonical",
    "export_safe_bundle",
    "import_safe_bundle",
]


# ==================== Windows DPAPI Helpers (stdlib ctypes) ====================
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CRYPTPROTECT_UI_FORBIDDEN = 0x01

    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    _crypt32.CryptProtectData.restype = wintypes.BOOL

    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL

    _kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    _kernel32.LocalFree.restype = wintypes.HLOCAL

    def _make_blob(raw: bytes) -> tuple[_DATA_BLOB, Any]:
        buf = ctypes.create_string_buffer(raw, len(raw))
        blob = _DATA_BLOB(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
        return blob, buf

    def _dpapi_seal(raw: bytes) -> bytes:
        in_blob, _in_ref = _make_blob(raw)
        ent_blob, _ent_ref = _make_blob(_DPAPI_ENTROPY)
        out_blob = _DATA_BLOB()
        if not _crypt32.CryptProtectData(
            ctypes.byref(in_blob),
            "reiji-maigo-vault",
            ctypes.byref(ent_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        ):
            raise OSError(f"DPAPI CryptProtectData failed: {ctypes.get_last_error()}")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            if out_blob.pbData:
                _kernel32.LocalFree(out_blob.pbData)

    def _dpapi_open(enc: bytes) -> bytes:
        in_blob, _in_ref = _make_blob(enc)
        ent_blob, _ent_ref = _make_blob(_DPAPI_ENTROPY)
        out_blob = _DATA_BLOB()
        if not _crypt32.CryptUnprotectData(
            ctypes.byref(in_blob),
            None,
            ctypes.byref(ent_blob),
            None,
            None,
            _CRYPTPROTECT_UI_FORBIDDEN,
            ctypes.byref(out_blob),
        ):
            raise OSError(f"DPAPI CryptUnprotectData failed: {ctypes.get_last_error()}")
        try:
            return ctypes.string_at(out_blob.pbData, out_blob.cbData)
        finally:
            if out_blob.pbData:
                _kernel32.LocalFree(out_blob.pbData)

else:

    def _dpapi_seal(raw: bytes) -> bytes:  # pragma: no cover - non-Windows fallback
        raise OSError("Windows DPAPI is unavailable on this platform")

    def _dpapi_open(enc: bytes) -> bytes:  # pragma: no cover - non-Windows fallback
        raise OSError("Windows DPAPI is unavailable on this platform")


# ==================== Domain-Separated PBKDF2 / HMAC-SHA256 ====================
def _validate_domain_tag(domain_tag: bytes) -> bytes:
    if not isinstance(domain_tag, (bytes, bytearray)) or not domain_tag:
        raise ValueError("domain_tag must be non-empty bytes")
    tag = bytes(domain_tag)
    if b"\x00" in tag:
        raise ValueError("domain_tag must not contain NUL bytes")
    return tag


def _validate_key(key: bytes) -> bytes:
    if not isinstance(key, (bytes, bytearray)) or not key:
        raise ValueError("signing key must be non-empty bytes")
    return bytes(key)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a mapping to deterministic UTF-8 canonical JSON bytes."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a Mapping")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def derive_domain_key(
    secret: bytes | str,
    salt: bytes,
    *,
    domain_tag: bytes = AAD_CARD,
    iterations: int = 200_000,
    dklen: int = 32,
) -> bytes:
    """Derive a domain-separated key using stdlib PBKDF2-HMAC-SHA256."""
    tag = _validate_domain_tag(domain_tag)
    if isinstance(secret, str):
        secret_bytes = secret.encode("utf-8")
    elif isinstance(secret, (bytes, bytearray)):
        secret_bytes = bytes(secret)
    else:
        raise TypeError("secret must be bytes or str")
    if not secret_bytes:
        raise ValueError("secret must not be empty")
    if not isinstance(salt, (bytes, bytearray)) or not salt:
        raise ValueError("salt must be non-empty bytes")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if dklen < 16:
        raise ValueError("dklen must be >= 16")
    return hashlib.pbkdf2_hmac(
        "sha256",
        secret_bytes,
        tag + b"\x00" + bytes(salt),
        iterations,
        dklen=dklen,
    )


def sign_canonical(
    payload: Mapping[str, Any],
    key: bytes,
    *,
    domain_tag: bytes = AAD_CARD,
) -> str:
    """Compute domain-separated HMAC-SHA256 over canonical JSON payload."""
    key_bytes = _validate_key(key)
    tag = _validate_domain_tag(domain_tag)
    body = canonical_json_bytes(payload)
    return hmac.new(key_bytes, tag + b"\x00" + body, hashlib.sha256).hexdigest()


def verify_canonical(
    payload: Mapping[str, Any],
    signature_hex: str,
    key: bytes,
    *,
    domain_tag: bytes = AAD_CARD,
) -> bool:
    """Constant-time verification of a domain-separated canonical JSON HMAC."""
    if not isinstance(signature_hex, str):
        return False
    candidate = signature_hex.strip().lower()
    if not _HEX_64_RE.fullmatch(candidate):
        return False
    try:
        expected = sign_canonical(payload, key, domain_tag=domain_tag)
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(expected, candidate)


# ==================== At-Rest Credential Vault ====================
class SecretVault:
    """Local credential vault with Win32 DPAPI / POSIX 0600 atomic writes.

    Refuses to overwrite corrupted vault files when ``strict=True`` or
    ``for_update=True`` is specified.
    """

    def __init__(self, vault_path: Path | str | None = None) -> None:
        if vault_path is None:
            self.vault_path = Path.home() / DEFAULT_VAULT_FILENAME
        else:
            self.vault_path = Path(vault_path)

    def _read_strict(self) -> tuple[dict[str, Any], bool]:
        """Strictly read and decode the vault; raises ValueError on corruption.

        Returns ``(data, was_legacy_plaintext)``.
        """
        try:
            raw_text = self.vault_path.read_text(encoding="utf-8")
            obj = json.loads(raw_text)
        except Exception as exc:
            raise ValueError(f"Corrupted vault JSON in {self.vault_path}") from exc

        if not isinstance(obj, dict):
            raise ValueError(f"Corrupted vault root structure in {self.vault_path}")

        vault_kind = obj.get("_vault")
        if vault_kind == "dpapi_v2":
            if sys.platform != "win32":
                raise ValueError(
                    "Cannot decrypt Windows DPAPI vault on a non-Windows platform"
                )
            blob = obj.get("blob")
            if not isinstance(blob, str) or not blob:
                raise ValueError(f"Missing DPAPI blob in {self.vault_path}")
            try:
                decrypted = _dpapi_open(bytes.fromhex(blob)).decode("utf-8")
                dec_obj = json.loads(decrypted)
            except Exception as exc:
                raise ValueError(
                    f"Failed to decrypt DPAPI vault payload in {self.vault_path}"
                ) from exc
            if not isinstance(dec_obj, dict):
                raise ValueError(
                    f"Corrupted decrypted DPAPI vault payload in {self.vault_path}"
                )
            return dec_obj, False

        if vault_kind == "posix_0600_v2":
            blob = obj.get("blob")
            if not isinstance(blob, str) or not blob:
                raise ValueError(f"Missing POSIX vault blob in {self.vault_path}")
            try:
                decoded = base64.b64decode(blob.encode("ascii"), validate=True).decode(
                    "utf-8"
                )
                dec_obj = json.loads(decoded)
            except Exception as exc:
                raise ValueError(
                    f"Failed to decode POSIX vault payload in {self.vault_path}"
                ) from exc
            if not isinstance(dec_obj, dict):
                raise ValueError(
                    f"Corrupted decoded POSIX vault payload in {self.vault_path}"
                )
            return dec_obj, False

        if vault_kind is not None:
            raise ValueError(
                f"Unsupported vault format {vault_kind!r} in {self.vault_path}"
            )

        return dict(obj), True

    def load(
        self, *, strict: bool = False, for_update: bool = False
    ) -> dict[str, Any]:
        """Load secrets from disk.

        When ``strict=True`` or ``for_update=True``, any corruption or
        decryption error raises ``ValueError`` so callers never silently
        clobber an existing vault.
        """
        require_strict = bool(strict or for_update)
        if not self.vault_path.exists():
            return {}
        try:
            data, was_legacy = self._read_strict()
        except ValueError:
            if require_strict:
                raise
            return {}
        if was_legacy and not require_strict:
            try:
                self.save(data)
            except OSError:
                pass
        return data

    def save(self, data: Mapping[str, Any]) -> None:
        """Atomically persist secrets using Win32 DPAPI or POSIX 0600."""
        if not isinstance(data, Mapping):
            raise TypeError("vault data must be a Mapping")

        parent = self.vault_path.resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(
            dict(data),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

        if sys.platform == "win32":
            envelope = {"_vault": "dpapi_v2", "blob": _dpapi_seal(raw).hex()}
        else:
            envelope = {
                "_vault": "posix_0600_v2",
                "blob": base64.b64encode(raw).decode("ascii"),
            }

        fd, tmp_path = tempfile.mkstemp(
            dir=str(parent), prefix=".reiji_vault_", suffix=".tmp"
        )
        try:
            if sys.platform != "win32":
                os.chmod(tmp_path, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(envelope, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, str(self.vault_path))
            if sys.platform != "win32":
                os.chmod(str(self.vault_path), 0o600)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def get_or_create_signing_key(self, *, key_name: str = "signing_key_hex") -> bytes:
        """Return the persistent 32-byte master signing key, creating if absent.

        Uses strict loading so a corrupted vault raises ``ValueError`` instead
        of being silently overwritten with a fresh key.
        """
        data = self.load(for_update=True)
        existing = data.get(key_name)
        if existing is not None:
            if not isinstance(existing, str) or not _HEX_64_RE.fullmatch(existing):
                raise ValueError(
                    f"Corrupted signing key entry {key_name!r} in {self.vault_path}"
                )
            return bytes.fromhex(existing)

        key_bytes = secrets.token_bytes(32)
        data[key_name] = key_bytes.hex()
        self.save(data)
        return key_bytes


# ==================== Anti-Bomb Portable Safe Bundle ====================
def _file_sha256_hex(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_STREAM_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
            total += len(chunk)
    return hasher.hexdigest(), total


def _validate_member_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Empty archive member name rejected")
    if (
        "/" in name
        or "\\" in name
        or ":" in name
        or ".." in name
        or "\x00" in name
        or name.startswith(".")
        or not _SAFE_INNER_RE.fullmatch(name)
    ):
        raise ValueError(f"Unsafe archive member path rejected: {name!r}")


def _backup_sqlite_consistent(src_db: Path, dst_db: Path) -> None:
    src_conn = sqlite3.connect(str(src_db))
    dst_conn = sqlite3.connect(str(dst_db))
    try:
        src_conn.backup(dst_conn)
        dst_conn.commit()
    finally:
        dst_conn.close()
        src_conn.close()


def export_safe_bundle(
    db_path: Path | str,
    bundle_path: Path | str,
    *,
    signing_key: bytes,
) -> dict[str, Any]:
    """Export a consistent SQLite snapshot into a ZIP_STORED signed bundle."""
    key_bytes = _validate_key(signing_key)
    src_path = Path(db_path)
    out_path = Path(bundle_path)

    if not src_path.is_file():
        raise FileNotFoundError(f"Source SQLite database not found: {src_path}")

    out_parent = out_path.resolve().parent
    out_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".reiji_snap_") as snap_dir:
        snapshot_path = Path(snap_dir) / _DB_MEMBER_NAME
        _backup_sqlite_consistent(src_path, snapshot_path)
        db_sha256, db_size = _file_sha256_hex(snapshot_path)

        manifest_payload: dict[str, Any] = {
            "format": _BUNDLE_FORMAT,
            "compression": "ZIP_STORED",
            "created_at": int(time.time()),
            "db_member": _DB_MEMBER_NAME,
            "db_sha256": db_sha256,
            "db_size": db_size,
            "files": [
                {
                    "inner": _DB_MEMBER_NAME,
                    "sha256": db_sha256,
                    "size": db_size,
                }
            ],
        }
        signature = sign_canonical(
            manifest_payload, key_bytes, domain_tag=AAD_BUNDLE
        )
        manifest_doc = {
            **manifest_payload,
            "signature": signature,
            "signature_domain": AAD_BUNDLE.decode("ascii"),
        }
        manifest_bytes = json.dumps(
            manifest_doc,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")

        fd, tmp_bundle = tempfile.mkstemp(
            dir=str(out_parent), prefix=".reiji_bundle_", suffix=".tmp"
        )
        os.close(fd)
        try:
            with zipfile.ZipFile(
                tmp_bundle, mode="w", compression=zipfile.ZIP_STORED
            ) as zf:
                zf.writestr(
                    _MANIFEST_MEMBER_NAME,
                    manifest_bytes,
                    compress_type=zipfile.ZIP_STORED,
                )
                zf.write(
                    snapshot_path,
                    arcname=_DB_MEMBER_NAME,
                    compress_type=zipfile.ZIP_STORED,
                )
            os.replace(tmp_bundle, str(out_path))
        except Exception:
            if os.path.exists(tmp_bundle):
                try:
                    os.remove(tmp_bundle)
                except OSError:
                    pass
            raise

    return {
        "bundle_path": str(out_path),
        "db_member": _DB_MEMBER_NAME,
        "db_sha256": db_sha256,
        "db_size": db_size,
        "signature": signature,
        "manifest": manifest_doc,
    }


def import_safe_bundle(
    bundle_path: Path | str,
    target_db_path: Path | str,
    *,
    signing_key: bytes,
    max_total_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify and safely materialize a ZIP_STORED signed memory bundle.

    Security invariants:
    - Rejects any non-ZIP_STORED entry (anti-decompression-bomb policy).
    - Rejects duplicate member names and any path-traversal member names.
    - Enforces streaming byte cap ``max_total_bytes`` before and during read.
    - Verifies ``AAD_BUNDLE`` HMAC signature over canonical manifest JSON and
      exact payload SHA-256 / byte length.
    - Performs SHA-256 idempotent materialization: identical existing target is
      reused without touching it; modified existing target raises
      ``FileExistsError`` unless ``overwrite=True``.
    """
    key_bytes = _validate_key(signing_key)
    if not isinstance(max_total_bytes, int) or max_total_bytes <= 0:
        raise ValueError("max_total_bytes must be a positive integer")

    src_bundle = Path(bundle_path)
    if not src_bundle.is_file():
        raise FileNotFoundError(f"Bundle archive not found: {src_bundle}")

    dst_path = Path(target_db_path)
    dst_parent = dst_path.resolve().parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    real_dst_dir = os.path.realpath(str(dst_parent))
    real_dst_file = os.path.realpath(str(dst_path))
    if (
        os.path.commonpath([real_dst_dir, real_dst_file]) != real_dst_dir
        or real_dst_file == real_dst_dir
    ):
        raise ValueError("Target database path escapes target directory")

    total_streamed = 0

    with zipfile.ZipFile(src_bundle, mode="r") as zf:
        infos = zf.infolist()
        if not infos:
            raise ValueError("Empty archive rejected")

        seen_names: set[str] = set()
        declared_total = 0
        for info in infos:
            _validate_member_name(info.filename)
            if info.filename in seen_names:
                raise ValueError(
                    f"Duplicate archive member rejected: {info.filename!r}"
                )
            seen_names.add(info.filename)

            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    "Compressed ZIP entries are forbidden (anti-decompression-bomb policy)"
                )
            if info.file_size < 0 or info.compress_size < 0:
                raise ValueError("Invalid negative archive entry size")
            if info.compress_size != info.file_size:
                raise ValueError(
                    "ZIP_STORED entry size mismatch (possible header tampering)"
                )
            if info.file_size > max_total_bytes:
                raise ValueError("Archive entry exceeds maximum allowed size")
            declared_total += info.file_size
            if declared_total > max_total_bytes:
                raise ValueError("Total archive size exceeds safety limit")

        if _MANIFEST_MEMBER_NAME not in seen_names:
            raise ValueError("Bundle is missing manifest.json")

        m_info = zf.getinfo(_MANIFEST_MEMBER_NAME)
        if m_info.file_size > _MAX_MANIFEST_BYTES:
            raise ValueError("Manifest exceeds 1 MiB limit")

        manifest_chunks: list[bytes] = []
        with zf.open(m_info, "r") as mf:
            while True:
                chunk = mf.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                total_streamed += len(chunk)
                if total_streamed > max_total_bytes or total_streamed > _MAX_MANIFEST_BYTES:
                    raise ValueError("Manifest streamed bytes exceeded safety limit")
                manifest_chunks.append(chunk)

        try:
            manifest_doc = json.loads(b"".join(manifest_chunks).decode("utf-8"))
        except Exception as exc:
            raise ValueError("Invalid manifest.json encoding or JSON syntax") from exc

        if not isinstance(manifest_doc, dict):
            raise ValueError("Invalid manifest root structure")

        signature = manifest_doc.get("signature")
        sig_domain = manifest_doc.get("signature_domain")
        if sig_domain is not None and sig_domain != AAD_BUNDLE.decode("ascii"):
            raise ValueError(f"Unexpected manifest signature domain: {sig_domain!r}")

        manifest_payload = {
            k: v
            for k, v in manifest_doc.items()
            if k not in {"signature", "signature_domain"}
        }
        if not isinstance(signature, str) or not verify_canonical(
            manifest_payload, signature, key_bytes, domain_tag=AAD_BUNDLE
        ):
            raise ValueError("Bundle manifest HMAC verification failed")

        if manifest_payload.get("format") != _BUNDLE_FORMAT:
            raise ValueError(
                f"Unsupported bundle format: {manifest_payload.get('format')!r}"
            )
        if manifest_payload.get("compression") != "ZIP_STORED":
            raise ValueError("Manifest declares non-ZIP_STORED compression")

        db_member = manifest_payload.get("db_member")
        if not isinstance(db_member, str):
            raise ValueError("Manifest missing valid db_member")
        _validate_member_name(db_member)
        if db_member == _MANIFEST_MEMBER_NAME or db_member not in seen_names:
            raise ValueError(f"Declared db_member not found in bundle: {db_member!r}")

        allowed_members = {_MANIFEST_MEMBER_NAME, db_member}
        unexpected = seen_names - allowed_members
        if unexpected:
            raise ValueError(
                f"Unexpected extra archive members rejected: {sorted(unexpected)!r}"
            )

        expected_sha256 = manifest_payload.get("db_sha256")
        expected_size = manifest_payload.get("db_size")
        if not isinstance(expected_sha256, str) or not _HEX_64_RE.fullmatch(
            expected_sha256
        ):
            raise ValueError("Manifest contains invalid db_sha256")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("Manifest contains invalid db_size")

        db_info = zf.getinfo(db_member)
        if db_info.file_size != expected_size:
            raise ValueError("Archive member size does not match signed manifest")

        fd, tmp_path = tempfile.mkstemp(
            dir=real_dst_dir, prefix=".reiji_import_", suffix=".tmp"
        )
        hasher = hashlib.sha256()
        entry_written = 0
        header_sample = bytearray()
        try:
            with zf.open(db_info, "r") as src_f, os.fdopen(fd, "wb") as dst_f:
                while True:
                    chunk = src_f.read(_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    total_streamed += len(chunk)
                    entry_written += len(chunk)
                    if total_streamed > max_total_bytes:
                        raise ValueError("Streamed bytes exceeded safety limit")
                    if len(header_sample) < len(_SQLITE_MAGIC):
                        need = len(_SQLITE_MAGIC) - len(header_sample)
                        header_sample.extend(chunk[:need])
                    hasher.update(chunk)
                    dst_f.write(chunk)
                dst_f.flush()
                os.fsync(dst_f.fileno())

            actual_sha256 = hasher.hexdigest()
            if entry_written != expected_size or not hmac.compare_digest(
                actual_sha256, expected_sha256.lower()
            ):
                raise ValueError("Bundle database payload SHA-256 verification failed")

            if entry_written > 0 and bytes(header_sample) != _SQLITE_MAGIC:
                raise ValueError("Bundle database payload is not a valid SQLite file")

            if os.path.exists(real_dst_file):
                existing_sha256, _ = _file_sha256_hex(Path(real_dst_file))
                if hmac.compare_digest(existing_sha256, actual_sha256):
                    os.remove(tmp_path)
                    return {
                        "status": "idempotent_reused",
                        "reused": True,
                        "target_db_path": str(dst_path),
                        "db_sha256": actual_sha256,
                        "db_size": entry_written,
                        "manifest": manifest_doc,
                    }
                if not overwrite:
                    os.remove(tmp_path)
                    raise FileExistsError(
                        f"Refusing to overwrite modified local database: {dst_path}"
                    )
                os.replace(tmp_path, real_dst_file)
                return {
                    "status": "overwritten",
                    "reused": False,
                    "target_db_path": str(dst_path),
                    "db_sha256": actual_sha256,
                    "db_size": entry_written,
                    "manifest": manifest_doc,
                }

            os.replace(tmp_path, real_dst_file)
            return {
                "status": "imported",
                "reused": False,
                "target_db_path": str(dst_path),
                "db_sha256": actual_sha256,
                "db_size": entry_written,
                "manifest": manifest_doc,
            }
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise
