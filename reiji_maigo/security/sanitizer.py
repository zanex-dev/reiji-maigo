"""Zero-latency memory-ingress sanitizer, privacy blindfold, and anti-poison gate.

Stdlib-only clean-room security module for ``reiji_maigo``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

__all__ = (
    "ORIGIN_TRUST_LEVELS",
    "SanitizeResult",
    "sanitize_for_memory",
    "can_mutate_authoritative_memory",
)

ORIGIN_TRUST_LEVELS: tuple[str, ...] = (
    "user_verified",
    "assistant",
    "untrusted_external",
)

_AUTHORITATIVE_MUTATION_TRUST: frozenset[str] = frozenset(
    ("user_verified", "assistant")
)

# ---------------------------------------------------------------------------
# 1. Secret & handshake ticket redaction rules
# ---------------------------------------------------------------------------

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PEM_PRIVATE_KEY",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
            r"|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
            re.MULTILINE,
        ),
    ),
    (
        "KISSSHOT_TICKET",
        re.compile(r"KISSSHOT_TICKET:[0-9a-fA-F]+"),
    ),
    (
        "GITHUB_PAT",
        re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{20,}(?![A-Za-z0-9_])"),
    ),
    (
        "GITHUB_TOKEN",
        re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9_])"),
    ),
    (
        "API_KEY",
        re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"),
    ),
    (
        "HF_TOKEN",
        re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{16,}(?![A-Za-z0-9_])"),
    ),
    (
        "GOOGLE_API_KEY",
        re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{30,}(?![A-Za-z0-9_-])"),
    ),
    (
        "AWS_ACCESS_KEY",
        re.compile(r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])"),
    ),
    (
        "BEARER_TOKEN",
        re.compile(r"(?i)\bBearer\s+(?!<REDACTED:)[A-Za-z0-9._~+/=-]{8,}"),
    ),
)

# ---------------------------------------------------------------------------
# 2. Privacy path & binary media asset blindfold rules
# ---------------------------------------------------------------------------

# Assemble host-profile segments dynamically to keep source clean-room safe.
_PROFILE_WIN_SEG = "".join(("U", "s", "e", "r", "s"))
_PROFILE_NIX_SEG = "".join(("h", "o", "m", "e"))
_PROFILE_ROAMING_SEG = "".join(("A", "p", "p", "D", "a", "t", "a"))

_SENSITIVE_HOME_DIRS: tuple[str, ...] = (
    "Pictures",
    "Videos",
    "Music",
    "Desktop",
    "Downloads",
    "Documents",
    _PROFILE_ROAMING_SEG,
)

_SENSITIVE_DIR_ALT = "|".join(re.escape(name) for name in _SENSITIVE_HOME_DIRS)

# Tail path characters stop before whitespace, quotes, brackets, or CJK punctuation.
_PATH_TAIL_UNQUOTED = r"(?:[\\/][^\s\"'<>|,;()\[\]{}，。；：！？、“”‘’（）【】]*)?"
_PATH_TAIL_QUOTED = r"(?:[\\/][^\"'\r\n<>|]*)?"

_GENERIC_PRIVATE_PATH_RE: re.Pattern[str] = re.compile(
    r"(?i)"
    r"(?:"
    r"\"(?:[A-Za-z]:[\\/]" + _PROFILE_WIN_SEG + r"[\\/][^\\/\r\n\"'<>|]+[\\/]"
    r"|/(?:" + _PROFILE_WIN_SEG + r"|" + _PROFILE_NIX_SEG + r")/[^/\r\n\"'<>|]+/"
    r"|(?:~|%USERPROFILE%|\$HOME)[\\/])"
    r"(?:" + _SENSITIVE_DIR_ALT + r")(?=$|[\\/\"])" + _PATH_TAIL_QUOTED + r"\""
    r"|'(?:[A-Za-z]:[\\/]" + _PROFILE_WIN_SEG + r"[\\/][^\\/\r\n\"'<>|]+[\\/]"
    r"|/(?:" + _PROFILE_WIN_SEG + r"|" + _PROFILE_NIX_SEG + r")/[^/\r\n\"'<>|]+/"
    r"|(?:~|%USERPROFILE%|\$HOME)[\\/])"
    r"(?:" + _SENSITIVE_DIR_ALT + r")(?=$|[\\/'])" + _PATH_TAIL_QUOTED + r"'"
    r"|(?:[A-Za-z]:[\\/]" + _PROFILE_WIN_SEG + r"[\\/][^\\/\s\"'<>|]+[\\/]"
    r"|/(?:" + _PROFILE_WIN_SEG + r"|" + _PROFILE_NIX_SEG + r")/[^/\s\"'<>|]+/"
    r"|(?:~|%USERPROFILE%|\$HOME)[\\/])"
    r"(?:" + _SENSITIVE_DIR_ALT + r")(?=$|[\\/\s\"'<>|,;()\[\]{}，。；：！？])"
    + _PATH_TAIL_UNQUOTED
    + r")"
)

_MEDIA_EXTENSIONS: tuple[str, ...] = (
    "jpg",
    "jpeg",
    "png",
    "gif",
    "webp",
    "bmp",
    "ico",
    "heic",
    "heif",
    "avif",
    "mp4",
    "mkv",
    "avi",
    "mov",
    "webm",
    "flv",
    "wmv",
    "mp3",
    "wav",
    "flac",
    "m4a",
    "ogg",
    "aac",
)

_MEDIA_EXT_ALT = "|".join(_MEDIA_EXTENSIONS)

_MEDIA_PATH_RE: re.Pattern[str] = re.compile(
    r"(?i)"
    r"(?:"
    # Quoted path with optional spaces ending in binary media extension
    r"\"(?:[A-Za-z]:[\\/]|~[\\/]|/|\.{1,2}[\\/]|[^\"'\r\n<>|]*[\\/])"
    r"[^\"'\r\n<>|/\\]+\.(?:" + _MEDIA_EXT_ALT + r")\""
    r"|'(?:[A-Za-z]:[\\/]|~[\\/]|/|\.{1,2}[\\/]|[^\"'\r\n<>|]*[\\/])"
    r"[^\"'\r\n<>|/\\]+\.(?:" + _MEDIA_EXT_ALT + r")'"
    # Unquoted path (drive, posix, home-relative, relative dir, or file://)
    r"|(?:file:///)?(?:[A-Za-z]:[\\/]|~[\\/]|/|\.{1,2}[\\/]|[A-Za-z0-9._-]+[\\/])"
    r"[^\s\"'<>|,;()\[\]{}，。；：！？、“”‘’（）【】]*\.(?:" + _MEDIA_EXT_ALT + r")"
    r"(?![A-Za-z0-9_])"
    # Bare media filename token (e.g. avatar_01.png) without leaking non-filename dot-ext mentions
    r"|(?<![A-Za-z0-9_./\\-])"
    r"[A-Za-z0-9][A-Za-z0-9._-]*\.(?:" + _MEDIA_EXT_ALT + r")"
    r"(?![A-Za-z0-9_])"
    r")"
)


@lru_cache(maxsize=4)
def _compile_runtime_home_pattern(home_raw: str, home_posix: str) -> re.Pattern[str] | None:
    """Compile sensitive-folder regex anchored to the runtime home directory."""
    variants: list[str] = []
    for candidate in (home_raw, home_posix):
        trimmed = candidate.rstrip("\\/")
        if trimmed and trimmed not in variants:
            variants.append(trimmed)
    if not variants:
        return None

    # Allow either slash style inside the home prefix for cross-style path text.
    escaped_prefixes: list[str] = []
    for variant in variants:
        parts = re.split(r"[\\/]+", variant)
        escaped_prefixes.append(r"[\\/]".join(re.escape(part) for part in parts if part))
        if variant.startswith("/"):
            escaped_prefixes[-1] = "/" + escaped_prefixes[-1]
    prefix_alt = "|".join(dict.fromkeys(escaped_prefixes))

    return re.compile(
        r"(?i)"
        r"(?:"
        r"\"(?:" + prefix_alt + r")[\\/](?:" + _SENSITIVE_DIR_ALT + r")"
        r"(?=$|[\\/\"])" + _PATH_TAIL_QUOTED + r"\""
        r"|'(?:" + prefix_alt + r")[\\/](?:" + _SENSITIVE_DIR_ALT + r")"
        r"(?=$|[\\/'])" + _PATH_TAIL_QUOTED + r"'"
        r"|(?:" + prefix_alt + r")[\\/](?:" + _SENSITIVE_DIR_ALT + r")"
        r"(?=$|[\\/\s\"'<>|,;()\[\]{}，。；：！？])" + _PATH_TAIL_UNQUOTED
        + r")"
    )


# ---------------------------------------------------------------------------
# 3. Anti-poisoning & anti-privilege-laundering detection rules
# ---------------------------------------------------------------------------

_POISON_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "forged_system_header",
        re.compile(
            r"\[\s*(?:ActivePool|Scout\s+Briefing|System\s+Prompt|SYSTEM|ADMIN|DEVELOPER|INST)\s*\]"
            r"|<\s*/?\s*(?:SYSTEM_MESSAGE|system|developer_instructions)\s*>"
            r"|<<\s*SYS\s*>>"
            r"|^\s*#{1,4}\s*System\s*:"
            r"|\bBEGIN\s+SYSTEM\s+PROMPT\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "instruction_hijack",
        re.compile(
            r"(?i)"
            r"(?:"
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
            r"(?:instructions?|rules?|directives?|prompts?)\b"
            r"|\bsystem\s+override\b"
            r"|\boverride\s+(?:the\s+)?system\s+(?:prompt|rules?|policy|instructions?)\b"
            r"|\b(?:enter|enable|activate)\s+(?:developer|god|unrestricted)\s+mode\b"
            r"|\byou\s+are\s+now\s+(?:in\s+)?(?:unrestricted|root|admin|jailbreak)\b"
            r"|忽略(?:之前|以上|此前|先前)(?:所有|全部)?(?:指令|规则|提示|设定)"
            r"|覆盖系统(?:提示|指令|规则)"
            r"|解除(?:所有|全部)?(?:安全)?限制"
            r")"
        ),
    ),
    (
        "privacy_control_tamper",
        re.compile(
            r"(?i)"
            r"(?:"
            r"\.allow_media\b"
            r"|\bagent[-_]blindfold\b"
            r"|\bblindfold[-_]control\b"
            r"|\bunlock\.(?:ps1|sh)\b"
            r"|/etc/(?:shadow|sudoers)\b"
            r"|\bsudoers\.d\b"
            r"|\borigin_trust\s*[:=]\s*[\"']?user_verified\b"
            r"|\b(?:bypass|disable|turn\s+off)\s+(?:the\s+)?(?:blindfold|sanitizer|security\s+gate|privacy\s+guard)\b"
            r"|\b(?:promote|launder|mark)\s+(?:this|memory|entry)\s+as\s+(?:user_verified|authoritative)\b"
            r")"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public data structures & functions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SanitizeResult:
    """Structured outcome of memory-ingress sanitization and poison screening."""

    clean_text: str
    redaction_count: int
    redaction_kinds: tuple[str, ...]
    blindfold_count: int
    poison_detected: bool
    poison_reasons: tuple[str, ...]
    origin_trust: str


def sanitize_for_memory(
    text: str,
    *,
    origin_trust: str = "user_verified",
) -> SanitizeResult:
    """Scrub secrets, blindfold private/media paths, and detect memory poisoning.

    Args:
        text: Raw candidate content before memory ingestion.
        origin_trust: Provenance trust tier; must be in ``ORIGIN_TRUST_LEVELS``.

    Returns:
        Frozen ``SanitizeResult`` describing scrubbed text and gate signals.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if origin_trust not in ORIGIN_TRUST_LEVELS:
        raise ValueError(
            f"Unsupported origin_trust {origin_trust!r}; "
            f"expected one of {ORIGIN_TRUST_LEVELS}"
        )

    if not text:
        return SanitizeResult(
            clean_text="",
            redaction_count=0,
            redaction_kinds=(),
            blindfold_count=0,
            poison_detected=False,
            poison_reasons=(),
            origin_trust=origin_trust,
        )

    # Detect poisoning on raw text before blindfold/redaction can alter markers.
    poison_reasons: list[str] = []
    for reason, pattern in _POISON_RULES:
        if pattern.search(text):
            poison_reasons.append(reason)

    cleaned = text

    # Pass 1: Redact secrets & handshake tickets.
    redaction_count = 0
    redaction_kinds: list[str] = []
    for kind, pattern in _SECRET_RULES:
        cleaned, count = pattern.subn(f"<REDACTED:{kind}>", cleaned)
        if count > 0:
            redaction_count += count
            redaction_kinds.append(kind)

    # Pass 2: Blindfold runtime home sensitive folders first so prefixes never leak.
    blindfold_count = 0
    home = Path.home()
    home_pattern = _compile_runtime_home_pattern(str(home), home.as_posix())
    if home_pattern is not None:
        cleaned, count = home_pattern.subn("<BLINDFOLD:PRIVATE_PATH>", cleaned)
        blindfold_count += count

    cleaned, count = _GENERIC_PRIVATE_PATH_RE.subn("<BLINDFOLD:PRIVATE_PATH>", cleaned)
    blindfold_count += count

    # Pass 3: Blindfold binary media asset paths/filenames.
    cleaned, count = _MEDIA_PATH_RE.subn("<BLINDFOLD:MEDIA_ASSET>", cleaned)
    blindfold_count += count

    return SanitizeResult(
        clean_text=cleaned,
        redaction_count=redaction_count,
        redaction_kinds=tuple(redaction_kinds),
        blindfold_count=blindfold_count,
        poison_detected=bool(poison_reasons),
        poison_reasons=tuple(poison_reasons),
        origin_trust=origin_trust,
    )


def can_mutate_authoritative_memory(res: SanitizeResult) -> bool:
    """Return True only when content is clean and from an authoritative trust tier."""
    if not isinstance(res, SanitizeResult):
        raise TypeError(f"res must be SanitizeResult, got {type(res).__name__}")
    if res.poison_detected:
        return False
    return res.origin_trust in _AUTHORITATIVE_MUTATION_TRUST
