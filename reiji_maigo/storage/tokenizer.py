"""Mixed CJK Bigram + ASCII identifier tokenizer and safe FTS5 query builder."""

from __future__ import annotations

import re
import unicodedata

# Match contiguous CJK ideographs/kana runs OR ASCII alphanumeric/underscore identifiers.
_SEGMENT_RE = re.compile(
    r"(?P<cjk>[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)"
    r"|(?P<word>[A-Za-z0-9_]+)"
)


def _cjk_bigrams(run: str) -> list[str]:
    """Slice a contiguous CJK run into overlapping 2-char bigrams, keeping single chars."""
    if len(run) <= 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def tokenize_for_fts(text: str) -> str:
    """Tokenize mixed CJK/English text into space-separated tokens for unicode61 FTS5.

    - Contiguous CJK runs of length >= 2 are sliced into overlapping 2-character bigrams.
    - Single CJK characters (length == 1) are preserved as-is so single-char queries still work.
    - ASCII alphanumeric/underscore identifiers are preserved in lowercase.
    """
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKC", text)
    tokens: list[str] = []
    for match in _SEGMENT_RE.finditer(normalized):
        cjk = match.group("cjk")
        if cjk is not None:
            tokens.extend(_cjk_bigrams(cjk))
        else:
            word = match.group("word")
            if word:
                tokens.append(word.lower())
    return " ".join(tokens)


def build_fts_query(query: str) -> str:
    """Safely escape user input into an FTS5 MATCH boolean phrase expression.

    Every extracted segment is double-quoted so FTS5 operators (AND/OR/NOT/NEAR),
    wildcards (*), column prefixes (:), and unbalanced quotes cannot inject syntax errors.
    Contiguous CJK runs become exact bigram phrases (e.g. "配额问题" -> '"配额 额问 问题"'),
    while 2-char CJK words directly hit the Bigram FTS5 index (e.g. "配额" -> '"配额"').
    """
    if not query or not query.strip():
        return ""

    normalized = unicodedata.normalize("NFKC", query)
    clauses: list[str] = []
    seen: set[str] = set()

    for match in _SEGMENT_RE.finditer(normalized):
        cjk = match.group("cjk")
        if cjk is not None:
            bigrams = _cjk_bigrams(cjk)
            phrase = " ".join(bigrams)
            clause = f'"{phrase}"'
        else:
            word = match.group("word")
            if not word:
                continue
            escaped = word.lower().replace('"', '""')
            clause = f'"{escaped}"'

        if clause not in seen:
            seen.add(clause)
            clauses.append(clause)

    return " AND ".join(clauses)
