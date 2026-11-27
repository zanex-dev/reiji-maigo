"""Security armor package for ``reiji_maigo``.

Aggregates the four pure-stdlib defense layers:
- ``sanitizer``: zero-latency secret redaction, privacy blindfold, and anti-poisoning.
- ``vault``: Win32 DPAPI / POSIX 0600 credential vault, domain-separated HMAC, and safe bundles.
- ``audit_gate``: hash-chained tamper-evident audit ledger and L0/L1/L2 mutation approval gate.
- ``loopback_guard``: exclusive 127.0.0.1 HTTP server, anti-DNS-rebinding, and replay guard.
"""

from __future__ import annotations

from reiji_maigo.security.audit_gate import (
    GENESIS_PREV_HASH,
    PROTECTED_CARD_TYPES,
    AuditChainLedger,
    MutationApprovalGate,
    MutationTier,
    ensure_security_tables,
)
from reiji_maigo.security.loopback_guard import (
    ADMIN_TOKEN_HEADER,
    DEFAULT_REPLAY_CACHE_SIZE,
    REQUEST_ID_HEADER,
    ExclusiveLoopbackHTTPServer,
    LoopbackGuardedRequestHandler,
    ReplayGuard,
    is_allowed_host,
    is_allowed_origin,
    is_allowed_sec_fetch_site,
    validate_loopback_request,
    verify_admin_token,
)
from reiji_maigo.security.sanitizer import (
    ORIGIN_TRUST_LEVELS,
    SanitizeResult,
    can_mutate_authoritative_memory,
    sanitize_for_memory,
)
from reiji_maigo.security.vault import (
    AAD_AUDIT,
    AAD_BUNDLE,
    AAD_CARD,
    DEFAULT_MAX_BUNDLE_BYTES,
    DEFAULT_VAULT_FILENAME,
    SecretVault,
    canonical_json_bytes,
    derive_domain_key,
    export_safe_bundle,
    import_safe_bundle,
    sign_canonical,
    verify_canonical,
)

__all__ = [
    # sanitizer
    "ORIGIN_TRUST_LEVELS",
    "SanitizeResult",
    "sanitize_for_memory",
    "can_mutate_authoritative_memory",
    # vault
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
    # audit_gate
    "PROTECTED_CARD_TYPES",
    "GENESIS_PREV_HASH",
    "MutationTier",
    "ensure_security_tables",
    "AuditChainLedger",
    "MutationApprovalGate",
    # loopback_guard
    "ADMIN_TOKEN_HEADER",
    "REQUEST_ID_HEADER",
    "DEFAULT_REPLAY_CACHE_SIZE",
    "ReplayGuard",
    "ExclusiveLoopbackHTTPServer",
    "LoopbackGuardedRequestHandler",
    "is_allowed_host",
    "is_allowed_origin",
    "is_allowed_sec_fetch_site",
    "verify_admin_token",
    "validate_loopback_request",
]
