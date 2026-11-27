"""MeMo vocabulary projection, RRF hybrid reranking, Foresight sweep, and MemDream plugin.

Fills the first official ``reranker`` provider slot (``memo_rrf``) in the
reiji-maigo microkernel using strictly stdlib-only components:

1. **MeMo Controlled Vocabulary Projection**: aligns free-form queries against
   registered controlled L2 tags and their aliases via exact/substring matching,
   CJK-Bigram overlap, and an optional L0-prefix-locked shadow runner.
2. **Reciprocal Rank Fusion (RRF)**: fuses Level-1 PPR recall rank, active-card
   lexical Bigram rank, and raw-turn FTS5 provenance rank via
   ``sum(1 / (rrf_k + rank))`` while strictly preserving active-only visibility.
3. **Foresight Expiration Inspection & Penalty**: parses ISO-8601
   ``foresight_until`` timestamps to sweep expired active cards and penalize
   them during hybrid recall with ``expired_foresight=True`` and
   ``foresight_warning="[EXPIRED_FORESIGHT]"``.
4. **MemDream Self-Probing Alias Completion**: generates colloquial retrieval
   probes for active cards and automatically registers high-signal missed-probe
   phrases/bigrams into the primary tag's controlled ``aliases``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import logging
import re
import unicodedata
from typing import Any

from reiji_maigo.core.contracts import LOCKED_DOMAINS, MemoryCard, validate_domain
from reiji_maigo.storage.tokenizer import _cjk_bigrams, tokenize_for_fts
from reiji_maigo.swarm.runner import (
    CallableFlashRunner,
    SwarmJSONError,
    SwarmRunnerError,
)

logger = logging.getLogger("reiji_maigo.plugins.hybrid_reranker")

PLUGIN_NAME: str = "hybrid_reranker"
PROVIDER_NAME: str = "memo_rrf"
EXPIRED_FORESIGHT_WARNING: str = "[EXPIRED_FORESIGHT]"

# Immutable L0 system prefixes for optional stateless Flash shadow invocations.
MEMO_PROJECTION_L0_PREFIX: str = (
    "You are the Reiji-Maigo MeMo controlled-vocabulary tag projector.\n"
    "Given a user query and a closed vocabulary of canonical tags with aliases,\n"
    "select ONLY canonical tag names from the provided vocabulary that semantically\n"
    "match the query. Never invent new tags.\n"
    'Respond strictly with a single JSON object: {"tags": ["<canonical_tag>", ...]}.'
)

MEM_DREAM_L0_PREFIX: str = (
    "You are the Reiji-Maigo MemDream self-probe synthesizer.\n"
    "Given a structured long-term memory card, generate concise colloquial user\n"
    "retrieval questions (probes) and short synonym phrases (aliases) a user might\n"
    "say when recalling this memory without repeating its formal title.\n"
    'Respond strictly with a single JSON object: {"probes": ["..."], "aliases": ["..."]}.'
)

_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_SPLIT_RE = re.compile(
    r"(?:为什么|怎么办|是不是|有没有|请问|帮我|一下|怎么|什么|如何|关于|处理|解决|情况|问题|现在|目前|已经|最近|之前|这个|那个|我的|你的|我们|他们|是否|可以|需要|[的了呢吗吧啊着过把被给在是有和与跟就也都还很太])+"
)

_STOPWORDS: frozenset[str] = frozenset(
    {
        # Common CJK conversational stopwords & bigrams
        "什么",
        "怎么",
        "如何",
        "为什",
        "什么",
        "请问",
        "帮我",
        "一下",
        "现在",
        "目前",
        "关于",
        "这个",
        "那个",
        "我的",
        "你的",
        "我们",
        "他们",
        "是否",
        "可以",
        "需要",
        "处理",
        "解决",
        "情况",
        "问题",
        "之前",
        "最近",
        "已经",
        "还有",
        "不是",
        "没有",
        "怎么",
        "么办",
        "办呢",
        "状态",
        "进展",
        # Common English stopwords
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "would",
        "your",
    }
)

_METADATA_STATE_KEYS: frozenset[str] = frozenset(
    {
        "pool_state",
        "hibernated",
        "source_event_id",
        "source_scope",
        "source_status",
        "hit_count",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp or date string into a timezone-aware UTC datetime."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Empty ISO timestamp")
    if cleaned.endswith(("Z", "z")):
        cleaned = f"{cleaned[:-1]}+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text).strip()).lower()


def _tokenize_set(text: str) -> set[str]:
    tok_str = tokenize_for_fts(text)
    if not tok_str:
        return set()
    return set(tok_str.split())


def _tokenize_list(text: str) -> list[str]:
    tok_str = tokenize_for_fts(text)
    if not tok_str:
        return []
    return tok_str.split()


def _extract_state_texts(state: Any, *, skip_metadata: bool = False) -> list[str]:
    """Recursively flatten textual values from a MemoryCard.state payload."""
    parts: list[str] = []
    if isinstance(state, Mapping):
        for key, val in state.items():
            key_str = str(key)
            if skip_metadata and key_str in _METADATA_STATE_KEYS:
                continue
            parts.extend(_extract_state_texts(val, skip_metadata=skip_metadata))
    elif isinstance(state, (list, tuple, set)):
        for item in state:
            parts.extend(_extract_state_texts(item, skip_metadata=skip_metadata))
    elif isinstance(state, str):
        cleaned = state.strip()
        if cleaned:
            parts.append(cleaned)
    elif isinstance(state, (int, float)) and not isinstance(state, bool):
        parts.append(str(state))
    return parts


def _parse_tag_args(raw_tags: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for item in raw_tags or ():
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            cleaned = part.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _load_active_cards(engine: Any, *, domain: str | None = None) -> list[MemoryCard]:
    """Load all active MemoryCards from ``engine`` (strictly excluding revoked/superseded)."""
    if domain is not None:
        validate_domain(domain)

    with engine.db.read_connection() as conn:
        if domain is None:
            rows = conn.execute(
                """
                SELECT card_id
                FROM memory_cards
                WHERE lifecycle_status = 'active'
                ORDER BY updated_at DESC, card_id ASC
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT card_id
                FROM memory_cards
                WHERE lifecycle_status = 'active' AND domain = ?
                ORDER BY updated_at DESC, card_id ASC
                """,
                (domain,),
            ).fetchall()

    cards: list[MemoryCard] = []
    for row in rows:
        card = engine.cards.get_card(str(row["card_id"]))
        if card is not None and card.lifecycle_status == "active":
            cards.append(card)
    return cards


def _invoke_locked_runner(
    runner: Any,
    *,
    role: str,
    locked_system_prefix: str,
    user_payload: str,
) -> dict[str, Any] | None:
    """Run an optional L0-prefix-locked shadow JSON call, returning parsed dict or None."""
    if runner is None:
        return None
    try:
        if hasattr(runner, "run_shadow_json") and callable(runner.run_shadow_json):
            res = runner.run_shadow_json(
                role=role,
                locked_system_prefix=locked_system_prefix,
                user_payload=user_payload,
            )
            data = getattr(res, "data", res)
            return dict(data) if isinstance(data, Mapping) else None
        adapter = CallableFlashRunner(runner)
        res = adapter.run_shadow_json(
            role=role,
            locked_system_prefix=locked_system_prefix,
            user_payload=user_payload,
        )
        return dict(res.data)
    except (SwarmRunnerError, SwarmJSONError, TypeError, ValueError) as exc:
        logger.warning("Shadow runner failed for role=%r: %s", role, exc)
        return None


def inspect_foresight(
    card: MemoryCard,
    *,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Inspect a card's ``foresight_until`` deadline and determine whether it has expired."""
    if not isinstance(card, MemoryCard):
        raise TypeError("card must be a MemoryCard instance")

    now_dt = _utc_now() if now_iso is None else _parse_iso_datetime(str(now_iso))
    checked_at = now_dt.isoformat()
    raw_until = (card.foresight_until or "").strip() if card.foresight_until is not None else ""

    if not raw_until:
        return {
            "card_id": card.card_id,
            "domain": card.domain,
            "title": card.title,
            "has_foresight": False,
            "expired": False,
            "foresight_until": card.foresight_until,
            "foresight_note": card.foresight_note,
            "checked_at": checked_at,
            "remaining_seconds": None,
            "warning": "",
        }

    try:
        until_dt = _parse_iso_datetime(raw_until)
    except ValueError as exc:
        return {
            "card_id": card.card_id,
            "domain": card.domain,
            "title": card.title,
            "has_foresight": True,
            "expired": False,
            "foresight_until": raw_until,
            "foresight_note": card.foresight_note,
            "checked_at": checked_at,
            "remaining_seconds": None,
            "warning": "",
            "parse_error": str(exc),
        }

    remaining = (until_dt - now_dt).total_seconds()
    expired = now_dt > until_dt
    return {
        "card_id": card.card_id,
        "domain": card.domain,
        "title": card.title,
        "has_foresight": True,
        "expired": expired,
        "foresight_until": raw_until,
        "foresight_note": card.foresight_note,
        "checked_at": checked_at,
        "remaining_seconds": round(remaining, 3),
        "warning": EXPIRED_FORESIGHT_WARNING if expired else "",
    }


def sweep_expired_foresight(
    engine: Any,
    *,
    domain: str | None = None,
    now_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Scan all active cards and return inspection records for expired ``foresight_until`` entries."""
    active_cards = _load_active_cards(engine, domain=domain)
    effective_now = _utc_now_iso() if now_iso is None else str(now_iso)
    expired_items: list[dict[str, Any]] = []
    for card in active_cards:
        report = inspect_foresight(card, now_iso=effective_now)
        if report["expired"]:
            expired_items.append(report)

    expired_items.sort(
        key=lambda item: (str(item.get("foresight_until") or ""), str(item["card_id"]))
    )
    return expired_items


def _score_form_against_query(
    norm_query: str,
    query_tokens: set[str],
    raw_form: str,
) -> float:
    """Compute lexical + CJK-Bigram similarity score between a tag surface form and query."""
    norm_form = _normalize_text(raw_form)
    if not norm_form or not norm_query:
        return 0.0

    # 1. Exact equality
    if norm_form == norm_query:
        return 1.0

    # 2. Direct substring containment (query contains tag/alias or vice versa)
    if len(norm_form) >= 2 and norm_form in norm_query:
        coverage_bonus = 0.05 * min(len(norm_form) / max(len(norm_query), 1), 1.0)
        return 0.90 + coverage_bonus

    if len(norm_query) >= 2 and norm_query in norm_form:
        coverage_bonus = 0.05 * min(len(norm_query) / max(len(norm_form), 1), 1.0)
        return 0.80 + coverage_bonus

    # 3. CJK-Bigram / ASCII identifier token overlap
    form_tokens = _tokenize_set(norm_form)
    if not form_tokens or not query_tokens:
        return 0.0

    overlap = query_tokens & form_tokens
    if not overlap:
        return 0.0

    # Filter trivial 1-char overlaps unless exact single-token form
    meaningful_overlap = {tok for tok in overlap if len(tok) >= 2 or len(form_tokens) == 1}
    if not meaningful_overlap:
        return 0.0

    dice = (2.0 * len(meaningful_overlap)) / float(len(query_tokens) + len(form_tokens))
    form_cov = len(meaningful_overlap) / float(len(form_tokens))
    query_cov = len(meaningful_overlap) / float(len(query_tokens))
    score = max(dice, 0.72 * form_cov, 0.65 * query_cov)

    if form_cov >= 0.25 or dice >= 0.20 or len(meaningful_overlap) >= 2:
        return min(score, 0.88)
    return 0.0


def _score_card_lexical(
    card: MemoryCard,
    norm_query: str,
    query_tokens: set[str],
) -> float:
    """Compute CJK-Bigram + substring lexical relevance score for an active card."""
    if not norm_query and not query_tokens:
        return 0.0

    norm_title = _normalize_text(card.title)
    title_tokens = _tokenize_set(card.title)

    state_text = " ".join(_extract_state_texts(card.state, skip_metadata=True))
    body_parts = [state_text, card.foresight_note, " ".join(card.tags)]
    norm_body = _normalize_text(" ".join(p for p in body_parts if p))
    body_tokens = _tokenize_set(norm_body)

    score = 0.0
    q_denom = float(max(len(query_tokens), 1))

    if norm_query and len(norm_query) >= 2:
        if norm_query in norm_title:
            score += 2.0
        elif norm_title and len(norm_title) >= 2 and norm_title in norm_query:
            score += 1.4
        if norm_body and norm_query in norm_body:
            score += 1.0

    if query_tokens:
        title_overlap = query_tokens & title_tokens
        if title_overlap:
            score += 1.8 * (len(title_overlap) / q_denom)
            score += 0.4 * (len(title_overlap) / float(max(len(title_tokens), 1)))

        body_overlap = query_tokens & body_tokens
        if body_overlap:
            score += 1.0 * (len(body_overlap) / q_denom)
            score += 0.2 * (len(body_overlap) / float(max(len(body_tokens), 1)))

    return score


def _extract_probe_aliases(
    probes: Iterable[str],
    *,
    suggested_aliases: Iterable[str] = (),
    exclude_terms: Iterable[str] = (),
    max_aliases: int = 6,
) -> list[str]:
    """Extract high-signal 2-char CJK bigrams, phrases, and ASCII terms from missed probes."""
    excluded: set[str] = {_normalize_text(t) for t in exclude_terms if isinstance(t, str) and t.strip()}
    selected: list[str] = []
    seen: set[str] = set()

    def _try_add(term: str) -> None:
        norm = _normalize_text(term)
        if (
            not norm
            or len(norm) < 2
            or len(norm) > 24
            or norm in excluded
            or norm in _STOPWORDS
            or norm in seen
        ):
            return
        seen.add(norm)
        selected.append(norm)

    # First honor any explicit aliases provided by a shadow runner.
    for alias in suggested_aliases:
        if isinstance(alias, str):
            _try_add(alias)
        if len(selected) >= max_aliases:
            return selected

    phrase_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()
    ascii_counts: Counter[str] = Counter()

    for probe in probes:
        if not isinstance(probe, str) or not probe.strip():
            continue
        normalized = unicodedata.normalize("NFKC", probe)

        for match in _CJK_RUN_RE.finditer(normalized):
            run = match.group(0)
            for chunk in _CJK_SPLIT_RE.split(run):
                chunk = chunk.strip()
                if len(chunk) < 2:
                    continue
                if 2 <= len(chunk) <= 8 and chunk not in _STOPWORDS:
                    phrase_counts[chunk] += 1
                for bg in _cjk_bigrams(chunk):
                    if len(bg) == 2 and bg not in _STOPWORDS:
                        bigram_counts[bg] += 1

        for match in _ASCII_WORD_RE.finditer(normalized):
            word = match.group(0).lower()
            if len(word) >= 2 and word not in _STOPWORDS:
                ascii_counts[word] += 1

    # Prefer concise meaningful phrases, then high-frequency CJK bigrams and ASCII terms.
    ranked_candidates: list[tuple[int, int, str]] = []
    for phrase, cnt in phrase_counts.items():
        # Give clean 2-6 char phrases priority while breaking ties by frequency and length.
        weight = cnt * 3 + (2 if 2 <= len(phrase) <= 4 else 1)
        ranked_candidates.append((weight, cnt, phrase))
    for bg, cnt in bigram_counts.items():
        if bg not in phrase_counts:
            ranked_candidates.append((cnt * 2, cnt, bg))
    for word, cnt in ascii_counts.items():
        ranked_candidates.append((cnt * 2 + 1, cnt, word))

    ranked_candidates.sort(key=lambda item: (-item[0], -item[1], len(item[2]), item[2]))
    for _, _, term in ranked_candidates:
        _try_add(term)
        if len(selected) >= max_aliases:
            break

    return selected


def _generate_offline_probes(engine: Any, card: MemoryCard) -> list[str]:
    """Synthesize deterministic colloquial probes for a card when no LLM runner is supplied."""
    probes: list[str] = []
    seen: set[str] = set()

    def _add_probe(text: str) -> None:
        cleaned = _WHITESPACE_RE.sub(" ", text).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            probes.append(cleaned)

    # 1. Explicit colloquial / hint fields in card.state if present.
    for key in ("colloquial", "probe", "probes", "question", "symptom", "alias_hint", "hint"):
        val = card.state.get(key)
        if isinstance(val, str) and val.strip():
            _add_probe(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    _add_probe(item)

    # 2. Foresight note or descriptive state text.
    if card.foresight_note.strip():
        _add_probe(f"之前提到的{card.foresight_note.strip()}现在怎么处理？")

    state_texts = _extract_state_texts(card.state, skip_metadata=True)
    for snippet in state_texts[:2]:
        if snippet and snippet != card.title:
            _add_probe(f"关于{snippet[:60]}现在是什么情况？")

    # 3. User-side bound evidence turns (captures original colloquial wording).
    try:
        turns = engine.cards.read_evidence_level2(card.card_id, pad=0)
    except Exception:
        turns = []
    for turn in turns:
        if str(turn.get("role") or "").lower() == "user":
            content = str(turn.get("content") or "").strip()
            if content:
                _add_probe(content[:120])
                break

    # 4. Fallback to title-based colloquial question.
    if not probes:
        _add_probe(f"{card.title}目前是什么进展？")

    return probes[:3]


class MemoRrfReranker:
    """Official MeMo + RRF + Foresight + MemDream reranker provider."""

    name: str = PROVIDER_NAME

    def project_query_tags(
        self,
        engine: Any,
        query: str,
        *,
        domain: str | None = None,
        extra_tags: Iterable[str] = (),
        runner: Any = None,
    ) -> list[str]:
        """Project a free-form query onto canonical controlled L2 tags.

        Combines exact/substring alias alignment, CJK-Bigram token similarity,
        and an optional L0-prefix-locked shadow runner over ``engine.cards.list_tags``.
        """
        if domain is not None:
            validate_domain(domain)

        vocab = engine.cards.list_tags(domain=domain)

        # Build fast lookup maps from normalized surface forms to canonical tag names.
        form_to_canonical: dict[str, str] = {}
        canonical_usage: dict[str, int] = {}
        canonical_set: set[str] = set()
        for entry in vocab:
            canonical = str(entry["canonical_name"])
            canonical_norm = _normalize_text(canonical)
            canonical_set.add(canonical)
            canonical_usage[canonical] = max(
                canonical_usage.get(canonical, 0), int(entry.get("usage_count", 0))
            )
            form_to_canonical.setdefault(canonical_norm, canonical)
            for alias in entry.get("aliases", ()):
                alias_norm = _normalize_text(str(alias))
                if alias_norm:
                    form_to_canonical.setdefault(alias_norm, canonical)

        result: list[str] = []
        seen: set[str] = set()

        def _append_tag(raw: str) -> None:
            norm = _normalize_text(raw)
            if not norm:
                return
            resolved = form_to_canonical.get(norm, norm)
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)

        # 1. Explicit caller-supplied extra_tags (normalized against controlled vocabulary).
        for raw_tag in _parse_tag_args(extra_tags):
            _append_tag(raw_tag)

        norm_query = _normalize_text(query) if isinstance(query, str) else ""
        if not norm_query or not vocab:
            return result

        query_tokens = _tokenize_set(norm_query)
        scored: dict[str, float] = {}

        # 2. Deterministic substring + CJK-Bigram projection over controlled vocabulary.
        for entry in vocab:
            canonical = str(entry["canonical_name"])
            forms = [canonical, *(str(a) for a in entry.get("aliases", ()))]
            best = 0.0
            for form in forms:
                s = _score_form_against_query(norm_query, query_tokens, form)
                if s > best:
                    best = s
            if best > 0.0 and best > scored.get(canonical, 0.0):
                scored[canonical] = best

        # 3. Optional L0-prefix-locked shadow runner semantic mapping (closed-vocab validated).
        if runner is not None:
            vocab_payload = [
                {
                    "canonical": str(entry["canonical_name"]),
                    "domain": str(entry["domain"]),
                    "aliases": list(entry.get("aliases", ())),
                }
                for entry in vocab
            ]
            payload_str = json.dumps(
                {
                    "query": query.strip(),
                    "domain": domain,
                    "vocabulary": vocab_payload,
                },
                ensure_ascii=False,
            )
            runner_data = _invoke_locked_runner(
                runner,
                role="memo_projector",
                locked_system_prefix=MEMO_PROJECTION_L0_PREFIX,
                user_payload=payload_str,
            )
            if isinstance(runner_data, Mapping):
                raw_runner_tags = (
                    runner_data.get("tags")
                    or runner_data.get("canonical_tags")
                    or runner_data.get("projected_tags")
                    or ()
                )
                if isinstance(raw_runner_tags, list):
                    for item in raw_runner_tags:
                        if not isinstance(item, str) or not item.strip():
                            continue
                        norm_item = _normalize_text(item)
                        resolved = form_to_canonical.get(norm_item)
                        if resolved is not None and resolved in canonical_set:
                            scored[resolved] = max(scored.get(resolved, 0.0), 0.96)

        ranked_vocab = sorted(
            scored.items(),
            key=lambda kv: (-kv[1], -canonical_usage.get(kv[0], 0), kv[0]),
        )
        for canonical, _ in ranked_vocab:
            if canonical not in seen:
                seen.add(canonical)
                result.append(canonical)

        return result

    def hybrid_recall(
        self,
        engine: Any,
        query: str,
        *,
        domain: str | None = None,
        query_tags: Iterable[str] = (),
        top_k: int = 3,
        now_iso: str | None = None,
        rrf_k: int = 60,
        expired_penalty: float = 0.5,
        runner: Any = None,
    ) -> list[dict[str, Any]]:
        """Fuse PPR, active-card lexical Bigram, and FTS5-turn ranks with RRF + Foresight decay."""
        if domain is not None:
            validate_domain(domain)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative int")
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
            raise ValueError("rrf_k must be a positive int")
        if (
            isinstance(expired_penalty, bool)
            or not isinstance(expired_penalty, (int, float))
            or float(expired_penalty) < 0.0
        ):
            raise ValueError("expired_penalty must be a non-negative number")
        if top_k == 0:
            return []

        active_list = _load_active_cards(engine, domain=domain)
        if not active_list:
            return []
        active_by_id: dict[str, MemoryCard] = {card.card_id: card for card in active_list}

        projected_tags = self.project_query_tags(
            engine,
            query,
            domain=domain,
            extra_tags=query_tags,
            runner=runner,
        )

        # Channel 1: Level-1 PPR rank over projected controlled tags.
        ppr_ranks: dict[str, int] = {}
        if projected_tags:
            ppr_limit = max(len(active_by_id), top_k * 4, 20)
            ppr_cards = engine.cards.recall_level1(
                domain,
                projected_tags,
                min_tag_hits=1,
                top_k=ppr_limit,
            )
            for rank, card in enumerate(ppr_cards, start=1):
                if card.card_id in active_by_id and card.card_id not in ppr_ranks:
                    ppr_ranks[card.card_id] = rank

        # Channel 2: Active card title / state / foresight / tag CJK-Bigram lexical rank.
        lexical_ranks: dict[str, int] = {}
        norm_query = _normalize_text(query) if isinstance(query, str) else ""
        query_tokens = _tokenize_set(norm_query)
        if norm_query:
            lex_scored: list[tuple[float, str]] = []
            for cid, card in active_by_id.items():
                score = _score_card_lexical(card, norm_query, query_tokens)
                if score > 0.0:
                    lex_scored.append((score, cid))
            lex_scored.sort(key=lambda item: (-item[0], item[1]))
            for rank, (_, cid) in enumerate(lex_scored, start=1):
                lexical_ranks[cid] = rank

        # Channel 3: Raw-turn FTS5 search hits mapped back to active cards via SourceSpans.
        fts_ranks: dict[str, int] = {}
        if isinstance(query, str) and query.strip():
            fts_limit = max(top_k * 8, 25)
            try:
                fts_hits = engine.db.search_turns_fts(query, limit=fts_limit)
            except Exception as exc:
                logger.warning("FTS5 search failed during hybrid_recall: %s", exc)
                fts_hits = []

            if fts_hits:
                card_fts_score: dict[str, float] = {}
                card_best_hit: dict[str, int] = {}
                for hit_idx, hit in enumerate(fts_hits):
                    hit_ep = str(hit.get("episode_id") or "")
                    try:
                        hit_tid = int(hit.get("turn_id"))
                    except (TypeError, ValueError):
                        continue
                    hit_weight = 1.0 / float(1 + hit_idx)
                    for cid, card in active_by_id.items():
                        for span in card.spans:
                            if (
                                span.episode_id == hit_ep
                                and span.turn_start <= hit_tid <= span.turn_end
                            ):
                                card_fts_score[cid] = card_fts_score.get(cid, 0.0) + hit_weight
                                if cid not in card_best_hit:
                                    card_best_hit[cid] = hit_idx
                                break

                ranked_fts = sorted(
                    card_fts_score.items(),
                    key=lambda kv: (-kv[1], card_best_hit.get(kv[0], 0), kv[0]),
                )
                for rank, (cid, _) in enumerate(ranked_fts, start=1):
                    fts_ranks[cid] = rank

        candidate_ids = set(ppr_ranks) | set(lexical_ranks) | set(fts_ranks)
        if not candidate_ids:
            return []

        effective_now = _utc_now_iso() if now_iso is None else str(now_iso)
        penalty = float(expired_penalty)
        fused: list[dict[str, Any]] = []

        for cid in candidate_ids:
            card = active_by_id[cid]
            channel_ranks: dict[str, int] = {}
            base_rrf = 0.0
            for channel_name, rank_map in (
                ("ppr", ppr_ranks),
                ("lexical", lexical_ranks),
                ("fts", fts_ranks),
            ):
                rank = rank_map.get(cid)
                if rank is not None:
                    channel_ranks[channel_name] = rank
                    base_rrf += 1.0 / float(rrf_k + rank)

            foresight_info = inspect_foresight(card, now_iso=effective_now)
            is_expired = bool(foresight_info["expired"])
            final_score = base_rrf * penalty if is_expired else base_rrf

            entry: dict[str, Any] = {
                "card_id": card.card_id,
                "card": card.to_dict(),
                "score": round(final_score, 10),
                "base_rrf_score": round(base_rrf, 10),
                "rrf_score": round(final_score, 10),
                "ranks": channel_ranks,
                "projected_tags": list(projected_tags),
                "expired_foresight": is_expired,
                "foresight_until": card.foresight_until,
                "foresight_note": card.foresight_note,
            }
            if is_expired:
                entry["foresight_warning"] = EXPIRED_FORESIGHT_WARNING
            fused.append(entry)

        fused.sort(
            key=lambda item: (
                -float(item["score"]),
                -float(item["base_rrf_score"]),
                str(item["card_id"]),
            )
        )
        return fused[:top_k]

    def run_mem_dream(
        self,
        engine: Any,
        *,
        card_ids: Iterable[str] | None = None,
        runner: Any = None,
        top_k: int = 3,
        now_iso: str | None = None,
    ) -> dict[str, Any]:
        """Self-probe active cards and auto-complete primary tag aliases on recall misses."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be an int >= 1")

        if card_ids is None:
            target_cards = _load_active_cards(engine)
        else:
            target_cards = []
            seen_ids: set[str] = set()
            for cid in card_ids:
                if not isinstance(cid, str) or not cid.strip():
                    continue
                cleaned = cid.strip()
                if cleaned in seen_ids:
                    continue
                seen_ids.add(cleaned)
                card = engine.cards.get_card(cleaned)
                if card is not None and card.lifecycle_status == "active":
                    target_cards.append(card)

        effective_now = _utc_now_iso() if now_iso is None else str(now_iso)
        probes_tested = 0
        missed_cards: list[str] = []
        aliases_added_total = 0
        updates: list[dict[str, Any]] = []

        for card in target_cards:
            if not card.tags:
                continue
            primary_tag = card.tags[0]

            # Generate colloquial probe queries (runner-assisted or deterministic offline).
            probes: list[str] = []
            suggested_aliases: list[str] = []

            if runner is not None:
                card_payload = json.dumps(
                    {
                        "card_id": card.card_id,
                        "domain": card.domain,
                        "card_type": card.card_type,
                        "title": card.title,
                        "tags": list(card.tags),
                        "state": card.state,
                        "foresight_note": card.foresight_note,
                    },
                    ensure_ascii=False,
                )
                runner_data = _invoke_locked_runner(
                    runner,
                    role="mem_dream",
                    locked_system_prefix=MEM_DREAM_L0_PREFIX,
                    user_payload=card_payload,
                )
                if isinstance(runner_data, Mapping):
                    raw_probes = runner_data.get("probes") or runner_data.get("queries") or ()
                    if isinstance(raw_probes, list):
                        probes = [str(p).strip() for p in raw_probes if isinstance(p, str) and p.strip()]
                    raw_aliases = runner_data.get("aliases") or ()
                    if isinstance(raw_aliases, list):
                        suggested_aliases = [
                            str(a).strip() for a in raw_aliases if isinstance(a, str) and a.strip()
                        ]

            if not probes:
                probes = _generate_offline_probes(engine, card)

            missed_probes: list[str] = []
            for probe in probes:
                probes_tested += 1
                hits = self.hybrid_recall(
                    engine,
                    probe,
                    domain=card.domain,
                    top_k=top_k,
                    now_iso=effective_now,
                )
                hit_ids = {item["card_id"] for item in hits}
                if card.card_id not in hit_ids:
                    missed_probes.append(probe)

            if not missed_probes:
                continue

            missed_cards.append(card.card_id)

            # Inspect existing aliases for primary_tag so we only report newly added ones.
            existing_tags = engine.cards.list_tags(domain=card.domain)
            existing_aliases: set[str] = set()
            for entry in existing_tags:
                if str(entry["canonical_name"]) == primary_tag:
                    existing_aliases = {
                        _normalize_text(str(a)) for a in entry.get("aliases", ())
                    }
                    break

            candidate_aliases = _extract_probe_aliases(
                missed_probes,
                suggested_aliases=suggested_aliases,
                exclude_terms={primary_tag, *card.tags, *existing_aliases},
            )
            if not candidate_aliases:
                continue

            canonical = engine.cards.register_or_resolve_tag(
                card.domain,
                primary_tag,
                aliases=candidate_aliases,
            )
            aliases_added_total += len(candidate_aliases)
            updates.append(
                {
                    "card_id": card.card_id,
                    "domain": card.domain,
                    "primary_tag": canonical,
                    "probes": probes,
                    "missed_probes": missed_probes,
                    "added_aliases": candidate_aliases,
                }
            )

        return {
            "cards_scanned": len(target_cards),
            "probes_tested": probes_tested,
            "missed_cards": missed_cards,
            "aliases_added": aliases_added_total,
            "updates": updates,
        }


def _add_json_flag_if_missing(parser: argparse.ArgumentParser) -> None:
    if "--json" not in getattr(parser, "_option_string_actions", {}):
        parser.add_argument("--json", action="store_true", help="Emit JSON output")


def _configure_rerank_subparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    _add_json_flag_if_missing(parser)
    subparsers = parser.add_subparsers(dest="rerank_action", metavar="<action>")

    p_recall = subparsers.add_parser(
        "recall",
        help="Run MeMo tag projection + RRF hybrid recall with Foresight decay",
    )
    p_recall.add_argument("query_pos", nargs="?", default=None, help="Free-form search query")
    p_recall.add_argument("--query", default=None, help="Free-form search query")
    p_recall.add_argument(
        "--domain",
        choices=LOCKED_DOMAINS,
        default=None,
        help="Optional locked L1 domain filter",
    )
    p_recall.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Explicit query tag(s); repeatable or comma-separated",
    )
    p_recall.add_argument("--top-k", type=int, default=3, help="Maximum results to return")
    p_recall.add_argument("--now-iso", default=None, help="Optional ISO-8601 timestamp override")
    p_recall.add_argument("--rrf-k", type=int, default=60, help="RRF rank smoothing constant")
    p_recall.add_argument(
        "--expired-penalty",
        type=float,
        default=0.5,
        help="Score multiplier for cards past foresight_until (default: 0.5)",
    )
    _add_json_flag_if_missing(p_recall)

    p_foresight = subparsers.add_parser(
        "foresight",
        help="Sweep active cards for expired foresight_until timestamps",
    )
    p_foresight.add_argument(
        "--domain",
        choices=LOCKED_DOMAINS,
        default=None,
        help="Optional locked L1 domain filter",
    )
    p_foresight.add_argument("--now-iso", default=None, help="Optional ISO-8601 timestamp override")
    _add_json_flag_if_missing(p_foresight)

    p_dream = subparsers.add_parser(
        "dream",
        help="Run MemDream self-probe cycle to complete missing tag aliases",
    )
    p_dream.add_argument(
        "--card-id",
        action="append",
        default=[],
        help="Target active card ID(s) to probe (repeatable; defaults to all active cards)",
    )
    p_dream.add_argument("--top-k", type=int, default=3, help="Recall cutoff for self-probe check")
    p_dream.add_argument("--now-iso", default=None, help="Optional ISO-8601 timestamp override")
    _add_json_flag_if_missing(p_dream)

    return parser


def _resolve_engine_and_args(
    pos_args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any | None, argparse.Namespace]:
    engine = kwargs.get("engine")
    ns = kwargs.get("args")
    for item in pos_args:
        if isinstance(item, argparse.Namespace):
            if ns is None:
                ns = item
        elif hasattr(item, "cards") and hasattr(item, "db"):
            if engine is None:
                engine = item
    if ns is None:
        ns = argparse.Namespace()
    return engine, ns


class _RerankCliHandler:
    """CLI handler for ``reiji rerank <recall|foresight|dream>``."""

    def __init__(self, reranker: MemoRrfReranker) -> None:
        self._reranker = reranker

    def configure_parser(self, subparser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return _configure_rerank_subparser(subparser)

    def _parse_effective_namespace(self, ns: argparse.Namespace) -> argparse.Namespace:
        action = getattr(ns, "rerank_action", None)
        extra_args = getattr(ns, "extra_args", None)
        if action is None and extra_args:
            fallback = _configure_rerank_subparser(argparse.ArgumentParser(prog="rerank"))
            parsed = fallback.parse_args(list(extra_args))
            if getattr(ns, "json", False):
                parsed.json = True
            return parsed
        return ns

    def __call__(self, *pos_args: Any, **kwargs: Any) -> int:
        engine, raw_ns = _resolve_engine_and_args(pos_args, kwargs)
        if engine is None:
            raise ValueError("rerank command requires an active ReijiMemoryEngine instance")

        ns = self._parse_effective_namespace(raw_ns)
        as_json = bool(getattr(ns, "json", False))
        action = getattr(ns, "rerank_action", None) or "foresight"

        if action == "recall":
            query = getattr(ns, "query", None) or getattr(ns, "query_pos", None) or ""
            tags = _parse_tag_args(getattr(ns, "tag", ()))
            if not str(query).strip() and not tags:
                raise ValueError("rerank recall requires a query or at least one --tag")
            results = self._reranker.hybrid_recall(
                engine,
                str(query),
                domain=getattr(ns, "domain", None),
                query_tags=tags,
                top_k=int(getattr(ns, "top_k", 3)),
                now_iso=getattr(ns, "now_iso", None),
                rrf_k=int(getattr(ns, "rrf_k", 60)),
                expired_penalty=float(getattr(ns, "expired_penalty", 0.5)),
            )
            payload = {
                "query": str(query),
                "domain": getattr(ns, "domain", None),
                "count": len(results),
                "results": results,
            }
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                if not results:
                    print("(no matching active cards)")
                for idx, item in enumerate(results, start=1):
                    card_info = item["card"]
                    warn = f" {item['foresight_warning']}" if item.get("foresight_warning") else ""
                    print(
                        f"{idx}. [{card_info['domain']}] {card_info['title']} "
                        f"(id={item['card_id']}, score={item['score']:.6f}, ranks={item['ranks']}){warn}"
                    )
            return 0

        if action == "foresight":
            expired = sweep_expired_foresight(
                engine,
                domain=getattr(ns, "domain", None),
                now_iso=getattr(ns, "now_iso", None),
            )
            payload = {
                "domain": getattr(ns, "domain", None),
                "expired_count": len(expired),
                "expired": expired,
            }
            if as_json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                if not expired:
                    print("(no expired foresight cards)")
                for item in expired:
                    print(
                        f"- {EXPIRED_FORESIGHT_WARNING} [{item['domain']}] {item['title']} "
                        f"(id={item['card_id']}, until={item['foresight_until']})"
                    )
            return 0

        if action == "dream":
            raw_card_ids = getattr(ns, "card_id", None) or []
            card_ids = [cid for cid in raw_card_ids if isinstance(cid, str) and cid.strip()] or None
            report = self._reranker.run_mem_dream(
                engine,
                card_ids=card_ids,
                top_k=int(getattr(ns, "top_k", 3)),
                now_iso=getattr(ns, "now_iso", None),
            )
            if as_json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(
                    f"MemDream complete: scanned={report['cards_scanned']}, "
                    f"probes={report['probes_tested']}, "
                    f"missed={len(report['missed_cards'])}, "
                    f"aliases_added={report['aliases_added']}"
                )
            return 0

        raise ValueError(f"Unsupported rerank action: {action!r}")


class HybridRerankerPlugin:
    """Official microkernel plugin providing the default ``memo_rrf`` reranker and CLI."""

    name: str = PLUGIN_NAME

    def __init__(self) -> None:
        self.reranker: MemoRrfReranker = MemoRrfReranker()
        self._cli_handler: _RerankCliHandler = _RerankCliHandler(self.reranker)

    def register_providers(self, kernel: Any) -> None:
        kernel.register_providers(
            "reranker",
            PROVIDER_NAME,
            self.reranker,
            plugin_name=self.name,
            default=True,
        )

    def register_cli(self, kernel: Any) -> None:
        kernel.register_cli(
            "rerank",
            self._cli_handler,
            help_text="MeMo tag projection, RRF hybrid recall, foresight sweep, and MemDream",
            plugin_name=self.name,
        )


def create_plugin() -> HybridRerankerPlugin:
    """Factory entry point for microkernel plugin discovery."""
    return HybridRerankerPlugin()


__all__ = [
    "EXPIRED_FORESIGHT_WARNING",
    "HybridRerankerPlugin",
    "MEMO_PROJECTION_L0_PREFIX",
    "MEM_DREAM_L0_PREFIX",
    "MemoRrfReranker",
    "PLUGIN_NAME",
    "PROVIDER_NAME",
    "create_plugin",
    "inspect_foresight",
    "sweep_expired_foresight",
]
