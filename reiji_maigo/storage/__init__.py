"""Storage and Bigram FTS5 search subsystem for reiji-maigo."""

from reiji_maigo.storage.sqlite_engine import MAX_SPAN_TURNS, ReijiDatabase
from reiji_maigo.storage.tokenizer import build_fts_query, tokenize_for_fts

__all__ = [
    "MAX_SPAN_TURNS",
    "ReijiDatabase",
    "build_fts_query",
    "tokenize_for_fts",
]
