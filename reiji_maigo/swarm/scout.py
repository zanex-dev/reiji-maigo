"""Read-only deep-reconnaissance swarm worker (ScoutAgent / 侦察蜂).

Runs strictly inside isolated single-turn shadow calls via ``SwarmModelRunner``
with a byte-locked L0 system prefix (``SCOUT_L0_PREFIX``) and a physically
read-only ``ScoutCapabilities`` sandbox.

Progressive 3-layer drawer protocol:
  - Layer 1: ActivePool snapshot, controlled tag lexicon, and Level-1 card shells.
  - Layer 2: Card Level-2 ``detail`` drawers & staged event notes.
  - Layer 3: Raw conversation turn slicing (``slice_card_evidence``,
    ``deep_fts_search``, ``expand_span``) to digest lengthy histories in the
    shadow window and return only a compact, cited ``ScoutBriefing``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from reiji_maigo import SourceSpan
from reiji_maigo.swarm.guards import ScoutCapabilities
from reiji_maigo.swarm.runner import SwarmModelRunner

SCOUT_L0_PREFIX: str = (
    "You are Reiji-Maigo Scout (侦察蜂), a strictly read-only deep-reconnaissance memory worker.\n"
    "Your job is to inspect memory drawers inside an isolated shadow window and return ONLY a compact,\n"
    "fact-dense, source-cited briefing for the main conversation window.\n"
    "\n"
    "MANDATORY RULES:\n"
    "1. Progressive 3-Layer Drawer Protocol:\n"
    "   - Layer 1 (Shells): Inspect ActivePool snapshot, controlled tags, and Level-1 card shells first.\n"
    "   - Layer 2 (Details): Inspect Level-2 card details or event notes. If Layer-2 details already answer\n"
    "     the query accurately and completely, STOP immediately and emit final_briefing in Round 1.\n"
    "   - Layer 3 (Raw Turns): Only when details are insufficient (e.g. exact parameters, verbatim quotes,\n"
    "     line numbers, timelines) or no card covers the query, drill down into raw turns using\n"
    "     slice_card_evidence, deep_fts_search, or expand_span.\n"
    "2. LongMemEval Honest-Abstention Iron Law:\n"
    "   - NEVER guess, extrapolate, or fabricate facts not grounded in observed records.\n"
    "   - If the memory store contains no verifiable evidence answering the query, set\n"
    "     \"abstained\": true, \"found\": false, and \"briefing_text\": \"\".\n"
    "3. Compression & Citations:\n"
    "   - Digest verbose histories into a concise briefing strictly within max_briefing_tokens.\n"
    "   - Always cite supporting card_ids and exact SourceSpans (episode_id, turn_start, turn_end).\n"
    "\n"
    "OUTPUT FORMAT (return a SINGLE valid JSON object only, no prose outside JSON):\n"
    "- To drill deeper (only when must_finalize is false):\n"
    "  {\n"
    "    \"status\": \"need_more\",\n"
    "    \"reasoning\": \"short reason\",\n"
    "    \"next_actions\": [\n"
    "      {\"action\": \"inspect_card_detail\", \"card_id\": \"...\"},\n"
    "      {\"action\": \"slice_card_evidence\", \"card_id\": \"...\", \"pad\": 2},\n"
    "      {\"action\": \"deep_fts_search\", \"query\": \"...\", \"limit\": 3, \"pad\": 3},\n"
    "      {\"action\": \"expand_span\", \"episode_id\": \"...\", \"turn_start\": 1, \"turn_end\": 10, \"pad\": 3},\n"
    "      {\"action\": \"inspect_event_note\", \"event_id\": \"...\"},\n"
    "      {\"action\": \"recall_by_tags\", \"tags\": [\"...\"], \"domain\": null, \"top_k\": 5}\n"
    "    ]\n"
    "  }\n"
    "- To finish (or abstain):\n"
    "  {\n"
    "    \"status\": \"final\",\n"
    "    \"final_briefing\": {\n"
    "      \"found\": true,\n"
    "      \"abstained\": false,\n"
    "      \"briefing_text\": \"compact cited answer within max_briefing_tokens\",\n"
    "      \"cited_card_ids\": [\"card_id_1\"],\n"
    "      \"cited_spans\": [{\"episode_id\": \"ep_1\", \"turn_start\": 1, \"turn_end\": 5}]\n"
    "    }\n"
    "  }\n"
)

_MAX_ACTIONS_PER_ROUND = 6
_INITIAL_DETAIL_LIMIT = 3
_MAX_TAG_CATALOG = 80
_MAX_POOL_SNAPSHOT_CHARS = 2400
_SPAN_STR_RE = re.compile(r"^\s*([^:\s]+)\s*:\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )


def estimate_tokens(text: str) -> int:
    """Zero-dependency bilingual token estimator (CJK ~1 char/tok, ASCII ~4 chars/tok)."""
    if not text:
        return 0
    cjk_count = 0
    other_non_space = 0
    for ch in text:
        if _is_cjk_char(ch):
            cjk_count += 1
        elif not ch.isspace():
            other_non_space += 1
    ascii_tokens = (other_non_space + 3) // 4 if other_non_space else 0
    total = cjk_count + ascii_tokens
    return max(1, total) if (cjk_count or other_non_space) else 0


def _truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Trim briefing text cleanly if it exceeds the configured token budget."""
    cleaned = text.strip()
    if not cleaned or estimate_tokens(cleaned) <= max_tokens:
        return cleaned
    if max_tokens <= 1:
        return cleaned[:1]

    budget = max(1, max_tokens - 1)
    cjk_count = 0
    other_non_space = 0
    cutoff = len(cleaned)
    for idx, ch in enumerate(cleaned):
        if _is_cjk_char(ch):
            cjk_count += 1
        elif not ch.isspace():
            other_non_space += 1
        est = cjk_count + ((other_non_space + 3) // 4 if other_non_space else 0)
        if est > budget:
            cutoff = idx
            break
    trimmed = cleaned[:cutoff].rstrip(" ,;:\n\t")
    return f"{trimmed}..." if trimmed else cleaned[:1]


@dataclass(frozen=True, slots=True)
class ScoutBriefing:
    """Compact, source-cited reconnaissance result returned to the main window."""

    query: str
    found: bool
    abstained: bool
    briefing_text: str
    cited_card_ids: tuple[str, ...]
    cited_spans: tuple[SourceSpan, ...]
    rounds_used: int
    raw_turns_scanned: int
    distilled_tokens: int
    compression_ratio: float

    def to_tail_block(self) -> str:
        """Render a compact tail block for main-window injection, or '' if abstained/empty."""
        text = self.briefing_text.strip()
        if self.abstained or not self.found or not text:
            return ""
        meta_parts: list[str] = []
        if self.cited_card_ids:
            meta_parts.append("cards=" + ",".join(self.cited_card_ids))
        if self.cited_spans:
            span_strs = [
                f"{sp.episode_id}:{sp.turn_start}-{sp.turn_end}"
                for sp in self.cited_spans
            ]
            meta_parts.append("spans=" + ",".join(span_strs))
        meta_parts.append(f"rounds={self.rounds_used}")
        if self.raw_turns_scanned > 0:
            meta_parts.append(f"scanned_turns={self.raw_turns_scanned}")
        if self.compression_ratio > 0.0:
            meta_parts.append(f"ratio={self.compression_ratio:.2f}x")
        header = f"[Scout Briefing | {' | '.join(meta_parts)}]"
        return f"{header}\n{text}"


def _extract_span_dicts(raw_spans: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not raw_spans:
        return result
    for item in raw_spans:
        sp = _parse_source_span(item)
        if sp is not None:
            result.append(sp.to_dict())
    return result


def _parse_source_span(item: Any) -> SourceSpan | None:
    if isinstance(item, SourceSpan):
        return item
    if isinstance(item, Mapping):
        ep = item.get("episode_id")
        ts = item.get("turn_start")
        te = item.get("turn_end", ts)
        if (
            isinstance(ep, str)
            and ep.strip()
            and not isinstance(ts, bool)
            and not isinstance(te, bool)
        ):
            try:
                start_i = int(ts)
                end_i = int(te)
            except (TypeError, ValueError):
                return None
            if start_i >= 1 and end_i >= start_i:
                return SourceSpan(
                    episode_id=ep.strip(),
                    turn_start=start_i,
                    turn_end=end_i,
                )
        return None
    if isinstance(item, str):
        match = _SPAN_STR_RE.match(item)
        if match:
            ep = match.group(1)
            start_i = int(match.group(2))
            end_i = int(match.group(3)) if match.group(3) else start_i
            if start_i >= 1 and end_i >= start_i:
                return SourceSpan(episode_id=ep, turn_start=start_i, turn_end=end_i)
    return None


def _extract_card_fields(card: Any) -> dict[str, Any]:
    """Safely normalize a MemoryCard object or mapping into shell + Layer-2 detail fields."""
    base: dict[str, Any] = {}
    if isinstance(card, Mapping):
        base = dict(card)
    else:
        to_dict = getattr(card, "to_dict", None)
        if callable(to_dict):
            maybe = to_dict()
            if isinstance(maybe, Mapping):
                base = dict(maybe)
        for attr in (
            "card_id",
            "domain",
            "card_type",
            "title",
            "summary",
            "state",
            "detail",
            "tags",
            "spans",
            "lifecycle_status",
            "valid_from",
            "valid_to",
            "foresight_note",
            "updated_at",
        ):
            if attr not in base and hasattr(card, attr):
                base[attr] = getattr(card, attr)

    card_id = str(base.get("card_id", "")).strip()
    domain = str(base.get("domain", "")).strip()
    card_type = str(base.get("card_type", "")).strip()
    title = str(base.get("title", "")).strip()
    raw_state = base.get("state")
    state_dict: dict[str, Any] = dict(raw_state) if isinstance(raw_state, Mapping) else {}

    summary = str(
        base.get("summary")
        or state_dict.get("summary")
        or title
        or ""
    ).strip()
    raw_tags = base.get("tags") or ()
    tags = [str(t).strip() for t in raw_tags if str(t).strip()]
    spans = _extract_span_dicts(base.get("spans"))

    raw_detail = base.get("detail")
    if raw_detail is None and "detail" in state_dict:
        raw_detail = state_dict["detail"]

    if raw_detail is not None and raw_detail != "":
        if isinstance(raw_detail, (dict, list)):
            detail_text = json.dumps(raw_detail, ensure_ascii=False)
        else:
            detail_text = str(raw_detail).strip()
    elif state_dict:
        detail_text = json.dumps(state_dict, ensure_ascii=False)
    else:
        detail_text = ""

    return {
        "card_id": card_id,
        "domain": domain,
        "card_type": card_type,
        "title": title,
        "summary": summary,
        "state": state_dict,
        "tags": tags,
        "spans": spans,
        "lifecycle_status": str(base.get("lifecycle_status", "active")),
        "valid_from": str(base.get("valid_from", "")),
        "foresight_note": str(base.get("foresight_note", "")),
        "updated_at": str(base.get("updated_at", "")),
        "detail": detail_text,
    }


def _card_shell(fields: Mapping[str, Any]) -> dict[str, Any]:
    shell: dict[str, Any] = {
        "card_id": fields["card_id"],
        "domain": fields["domain"],
        "card_type": fields["card_type"],
        "title": fields["title"],
        "summary": fields["summary"],
        "tags": fields["tags"],
        "spans": fields["spans"],
        "has_detail": bool(fields["detail"]),
    }
    if fields.get("foresight_note"):
        shell["foresight_note"] = fields["foresight_note"]
    return shell


def _extract_tag_names(raw_tags: Any) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    if not raw_tags:
        return names
    for item in raw_tags:
        candidate = ""
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, Mapping):
            for key in ("canonical_name", "tag", "name", "label"):
                val = item.get(key)
                if isinstance(val, str) and val.strip():
                    candidate = val.strip()
                    break
        else:
            for attr in ("canonical_name", "tag", "name", "label"):
                val = getattr(item, attr, None)
                if isinstance(val, str) and val.strip():
                    candidate = val.strip()
                    break
        if candidate and candidate not in seen:
            seen.add(candidate)
            names.append(candidate)
    return names


def _match_query_tags(
    query: str,
    hint_tags: Iterable[str],
    catalog_tags: Iterable[str],
) -> list[str]:
    selected: list[str] = []
    seen_lower: set[str] = set()

    for raw in hint_tags:
        cleaned = str(raw).strip()
        if cleaned and cleaned.lower() not in seen_lower:
            seen_lower.add(cleaned.lower())
            selected.append(cleaned)

    q_lower = query.lower()
    q_tokens = {tok for tok in re.findall(r"[a-z0-9_.\-]+", q_lower) if len(tok) >= 2}
    for tag in catalog_tags:
        t_lower = tag.lower()
        if t_lower in seen_lower:
            continue
        if len(t_lower) >= 2 and (t_lower in q_lower or t_lower in q_tokens):
            seen_lower.add(t_lower)
            selected.append(tag)

    return selected


def _normalize_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": str(turn.get("episode_id", "")),
        "turn_id": int(turn.get("turn_id", 0)),
        "role": str(turn.get("role", "")),
        "content": str(turn.get("content", "")),
        "created_at": str(turn.get("created_at", "")),
    }


def _bounded_int(val: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(val, bool):
        return default
    try:
        parsed = int(val)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


class ScoutAgent:
    """Deep-reconnaissance shadow worker that digests old records into compact briefings."""

    def __init__(
        self,
        caps: ScoutCapabilities,
        runner: SwarmModelRunner,
        *,
        max_rounds: int = 3,
        max_briefing_tokens: int = 240,
    ) -> None:
        if caps is None or not isinstance(caps, ScoutCapabilities):
            raise TypeError("caps must be a ScoutCapabilities instance")
        if runner is None or not isinstance(runner, SwarmModelRunner):
            raise TypeError("runner must be a SwarmModelRunner instance")
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be an int >= 1")
        if (
            isinstance(max_briefing_tokens, bool)
            or not isinstance(max_briefing_tokens, int)
            or max_briefing_tokens < 1
        ):
            raise ValueError("max_briefing_tokens must be an int >= 1")

        self._caps = caps
        self._runner = runner
        self._max_rounds = max_rounds
        self._max_briefing_tokens = max_briefing_tokens

    def investigate(
        self,
        query: str,
        *,
        domain: str | None = None,
        hint_tags: Iterable[str] = (),
    ) -> ScoutBriefing:
        """Run multi-round read-only reconnaissance in an isolated shadow window."""
        cleaned_query = query.strip() if isinstance(query, str) else ""
        cleaned_hints = tuple(
            str(tag).strip() for tag in hint_tags if str(tag).strip()
        )
        if not cleaned_query:
            briefing = self._abstained_briefing(
                query="",
                rounds_used=0,
                raw_turns_scanned=0,
            )
            self._emit_telemetry(
                query="",
                domain=domain,
                briefing=briefing,
                scanned_raw_tokens=0,
                actions_executed=[],
                prompt_tokens=0,
                completion_tokens=0,
                elapsed_ms=0,
                prefix_sha256="",
            )
            return briefing

        # Track unique scanned raw turns & scanned material token volume for real compression ratio.
        scanned_turns: dict[tuple[str, int], dict[str, Any]] = {}
        scanned_detail_tokens: dict[str, int] = {}
        scanned_note_tokens: dict[str, int] = {}
        scanned_shell_tokens: dict[str, int] = {}
        executed_action_keys: set[str] = set()
        actions_log: list[str] = []

        # --- Layer 1 + Top-K Layer 2 initial read-only prefetch ---
        observations: dict[str, Any] = self._build_initial_observations(
            query=cleaned_query,
            domain=domain,
            hint_tags=cleaned_hints,
            scanned_shell_tokens=scanned_shell_tokens,
            scanned_detail_tokens=scanned_detail_tokens,
        )
        action_results: list[dict[str, Any]] = []
        observations["action_results"] = action_results

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_elapsed_ms = 0
        last_prefix_sha256 = ""
        rounds_used = 0
        final_payload: Mapping[str, Any] | None = None

        for round_idx in range(1, self._max_rounds + 1):
            rounds_used = round_idx
            is_final_round = round_idx >= self._max_rounds
            user_payload = json.dumps(
                {
                    "query": cleaned_query,
                    "domain": domain,
                    "hint_tags": list(cleaned_hints),
                    "round": round_idx,
                    "max_rounds": self._max_rounds,
                    "must_finalize": is_final_round,
                    "max_briefing_tokens": self._max_briefing_tokens,
                    "observations": observations,
                },
                ensure_ascii=False,
            )

            call_result = self._runner.run_shadow_json(
                role="scout",
                locked_system_prefix=SCOUT_L0_PREFIX,
                user_payload=user_payload,
            )
            total_prompt_tokens += call_result.prompt_tokens
            total_completion_tokens += call_result.completion_tokens
            total_elapsed_ms += call_result.elapsed_ms
            last_prefix_sha256 = call_result.prefix_sha256

            data = call_result.data
            extracted_final = self._extract_final_payload(data)
            next_actions = self._extract_next_actions(data)

            if extracted_final is not None:
                final_payload = extracted_final
                break

            if is_final_round or not next_actions:
                # Model neither finalized nor requested valid drill-down actions -> honest abstain.
                final_payload = None
                break

            new_observations_added = False
            for raw_action in next_actions[:_MAX_ACTIONS_PER_ROUND]:
                outcome, dedup_key = self._execute_readonly_action(
                    raw_action,
                    default_domain=domain,
                    executed_action_keys=executed_action_keys,
                    scanned_turns=scanned_turns,
                    scanned_detail_tokens=scanned_detail_tokens,
                    scanned_note_tokens=scanned_note_tokens,
                    scanned_shell_tokens=scanned_shell_tokens,
                )
                if outcome is not None:
                    action_results.append(outcome)
                    actions_log.append(str(outcome.get("action", "unknown")))
                    if dedup_key is not None:
                        executed_action_keys.add(dedup_key)
                    new_observations_added = True

            if not new_observations_added:
                # Avoid fruitless spin when every requested action was already executed.
                final_payload = None
                break

        scanned_raw_tokens = self._compute_scanned_raw_tokens(
            scanned_turns=scanned_turns,
            scanned_detail_tokens=scanned_detail_tokens,
            scanned_note_tokens=scanned_note_tokens,
            scanned_shell_tokens=scanned_shell_tokens,
        )
        raw_turns_scanned = len(scanned_turns)

        briefing = self._build_briefing(
            query=cleaned_query,
            final_payload=final_payload,
            rounds_used=rounds_used,
            raw_turns_scanned=raw_turns_scanned,
            scanned_raw_tokens=scanned_raw_tokens,
        )

        self._emit_telemetry(
            query=cleaned_query,
            domain=domain,
            briefing=briefing,
            scanned_raw_tokens=scanned_raw_tokens,
            actions_executed=actions_log,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            elapsed_ms=total_elapsed_ms,
            prefix_sha256=last_prefix_sha256,
        )
        return briefing

    def _build_initial_observations(
        self,
        *,
        query: str,
        domain: str | None,
        hint_tags: tuple[str, ...],
        scanned_shell_tokens: dict[str, int],
        scanned_detail_tokens: dict[str, int],
    ) -> dict[str, Any]:
        pool_text = ""
        try:
            raw_pool = self._caps.pool_snapshot(include_undigested=False)
            if isinstance(raw_pool, str):
                pool_text = raw_pool[:_MAX_POOL_SNAPSHOT_CHARS]
        except Exception:
            pool_text = ""

        catalog_tags = self._safe_list_tags(domain=domain)
        seed_tags = _match_query_tags(query, hint_tags, catalog_tags)

        card_shells: list[dict[str, Any]] = []
        prefetched_details: dict[str, str] = {}
        if seed_tags:
            try:
                recalled = self._caps.recall_level1(
                    domain,
                    seed_tags,
                    min_tag_hits=1,
                    top_k=5,
                )
            except Exception:
                recalled = []
            for idx, card in enumerate(recalled or ()):
                fields = _extract_card_fields(card)
                cid = fields["card_id"]
                if not cid:
                    continue
                shell = _card_shell(fields)
                card_shells.append(shell)
                scanned_shell_tokens[cid] = estimate_tokens(
                    f"{fields['summary']} {' '.join(fields['tags'])}"
                )
                if idx < _INITIAL_DETAIL_LIMIT and fields["detail"]:
                    prefetched_details[cid] = fields["detail"]
                    scanned_detail_tokens[cid] = estimate_tokens(fields["detail"])

        fts_preview: list[dict[str, Any]] = []
        if not card_shells:
            try:
                hits = self._caps.search_turns_fts(query, limit=3)
            except Exception:
                hits = []
            for hit in hits or ():
                if isinstance(hit, Mapping):
                    fts_preview.append(
                        {
                            "episode_id": str(hit.get("episode_id", "")),
                            "turn_id": int(hit.get("turn_id", 0)),
                            "snippet": str(
                                hit.get("snippet") or hit.get("content") or ""
                            )[:240],
                        }
                    )

        return {
            "pool_snapshot": pool_text,
            "available_tags": catalog_tags[:_MAX_TAG_CATALOG],
            "matched_seed_tags": seed_tags,
            "level1_cards": card_shells,
            "prefetched_card_details": prefetched_details,
            "fts_preview": fts_preview,
        }

    def _safe_list_tags(self, *, domain: str | None) -> list[str]:
        try:
            raw = (
                self._caps.list_tags(domain=domain)
                if domain is not None
                else self._caps.list_tags()
            )
        except TypeError:
            try:
                raw = self._caps.list_tags()
            except Exception:
                return []
        except Exception:
            return []
        return _extract_tag_names(raw)

    @staticmethod
    def _extract_final_payload(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
        if not isinstance(data, Mapping):
            return None
        nested = data.get("final_briefing")
        if isinstance(nested, Mapping):
            return nested
        status = str(data.get("status", "")).strip().lower()
        if status in ("final", "done", "complete", "completed", "abstain", "abstained"):
            return data
        has_actions = bool(data.get("next_actions") or data.get("actions"))
        if not has_actions and any(
            k in data for k in ("briefing_text", "abstained", "found")
        ):
            return data
        return None

    @staticmethod
    def _extract_next_actions(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if not isinstance(data, Mapping):
            return []
        raw = data.get("next_actions")
        if raw is None:
            raw = data.get("actions")
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, Mapping)]

    def _execute_readonly_action(
        self,
        raw_action: Mapping[str, Any],
        *,
        default_domain: str | None,
        executed_action_keys: set[str],
        scanned_turns: dict[tuple[str, int], dict[str, Any]],
        scanned_detail_tokens: dict[str, int],
        scanned_note_tokens: dict[str, int],
        scanned_shell_tokens: dict[str, int],
    ) -> tuple[dict[str, Any] | None, str | None]:
        action_name = str(raw_action.get("action") or raw_action.get("type") or "").strip()
        if not action_name:
            return None, None

        if action_name == "inspect_card_detail":
            card_id = str(raw_action.get("card_id", "")).strip()
            if not card_id:
                return {"action": action_name, "error": "missing card_id"}, None
            dedup_key = f"inspect_card_detail:{card_id}"
            if dedup_key in executed_action_keys:
                return None, None
            card = self._caps.get_card(card_id)
            if card is None:
                return {
                    "action": action_name,
                    "card_id": card_id,
                    "found": False,
                }, dedup_key
            fields = _extract_card_fields(card)
            if fields["detail"]:
                scanned_detail_tokens[card_id] = estimate_tokens(fields["detail"])
            scanned_shell_tokens.setdefault(
                card_id,
                estimate_tokens(f"{fields['summary']} {' '.join(fields['tags'])}"),
            )
            return {
                "action": action_name,
                "card_id": card_id,
                "found": True,
                "card": fields,
            }, dedup_key

        if action_name == "slice_card_evidence":
            card_id = str(raw_action.get("card_id", "")).strip()
            pad = _bounded_int(raw_action.get("pad"), default=2, minimum=0, maximum=20)
            if not card_id:
                return {"action": action_name, "error": "missing card_id"}, None
            dedup_key = f"slice_card_evidence:{card_id}:{pad}"
            if dedup_key in executed_action_keys:
                return None, None
            try:
                raw_turns = self._caps.read_evidence_level2(card_id, pad=pad)
            except KeyError:
                return {
                    "action": action_name,
                    "card_id": card_id,
                    "found": False,
                    "turns": [],
                }, dedup_key
            normalized = self._record_turns(raw_turns, scanned_turns)
            return {
                "action": action_name,
                "card_id": card_id,
                "pad": pad,
                "turn_count": len(normalized),
                "turns": normalized,
            }, dedup_key

        if action_name == "deep_fts_search":
            sub_query = str(raw_action.get("query", "")).strip()
            limit = _bounded_int(raw_action.get("limit"), default=3, minimum=1, maximum=8)
            pad = _bounded_int(raw_action.get("pad"), default=3, minimum=0, maximum=15)
            if not sub_query:
                return {"action": action_name, "error": "missing query"}, None
            dedup_key = f"deep_fts_search:{sub_query.lower()}:{limit}:{pad}"
            if dedup_key in executed_action_keys:
                return None, None
            enqueue_backfill = bool(raw_action.get("enqueue_backfill", True))
            entries: list[dict[str, Any]] = []
            if enqueue_backfill:
                slices = self._caps.deep_slice_level3(sub_query, limit=limit, pad=pad)
                for item in slices or ():
                    if not isinstance(item, Mapping):
                        continue
                    turns_norm = self._record_turns(item.get("turns") or (), scanned_turns)
                    entries.append(
                        {
                            "span": item.get("span"),
                            "turn_count": len(turns_norm),
                            "turns": turns_norm,
                        }
                    )
            else:
                hits = self._caps.search_turns_fts(sub_query, limit=limit)
                for hit in hits or ():
                    if not isinstance(hit, Mapping):
                        continue
                    ep = str(hit.get("episode_id", "")).strip()
                    tid = _bounded_int(hit.get("turn_id"), default=0, minimum=0, maximum=10**9)
                    if not ep or tid < 1:
                        continue
                    span = SourceSpan(episode_id=ep, turn_start=tid, turn_end=tid)
                    turns_norm = self._record_turns(
                        self._caps.read_span(span, pad=pad),
                        scanned_turns,
                    )
                    entries.append(
                        {
                            "span": span.to_dict(),
                            "turn_count": len(turns_norm),
                            "turns": turns_norm,
                        }
                    )
            return {
                "action": action_name,
                "query": sub_query,
                "matches": entries,
            }, dedup_key

        if action_name == "expand_span":
            ep = str(raw_action.get("episode_id", "")).strip()
            turn_start = _bounded_int(
                raw_action.get("turn_start"), default=0, minimum=0, maximum=10**9
            )
            turn_end = _bounded_int(
                raw_action.get("turn_end", turn_start),
                default=turn_start,
                minimum=0,
                maximum=10**9,
            )
            pad = _bounded_int(raw_action.get("pad"), default=3, minimum=0, maximum=20)
            max_turns = _bounded_int(
                raw_action.get("max_turns"), default=40, minimum=1, maximum=80
            )
            if not ep or turn_start < 1 or turn_end < turn_start:
                return {"action": action_name, "error": "invalid span bounds"}, None
            dedup_key = f"expand_span:{ep}:{turn_start}:{turn_end}:{pad}:{max_turns}"
            if dedup_key in executed_action_keys:
                return None, None
            span = SourceSpan(episode_id=ep, turn_start=turn_start, turn_end=turn_end)
            normalized = self._record_turns(
                self._caps.read_span(span, pad=pad, max_turns=max_turns),
                scanned_turns,
            )
            return {
                "action": action_name,
                "span": span.to_dict(),
                "pad": pad,
                "turn_count": len(normalized),
                "turns": normalized,
            }, dedup_key

        if action_name == "inspect_event_note":
            event_id = str(raw_action.get("event_id", "")).strip()
            if not event_id:
                return {"action": action_name, "error": "missing event_id"}, None
            dedup_key = f"inspect_event_note:{event_id}"
            if dedup_key in executed_action_keys:
                return None, None
            event_obj = self._caps.get_event(event_id)
            note = self._caps.get_event_note(event_id)
            if note is not None:
                scanned_note_tokens[event_id] = estimate_tokens(
                    json.dumps(note, ensure_ascii=False)
                )
            event_repr: Any = None
            if event_obj is not None:
                to_dict = getattr(event_obj, "to_dict", None)
                event_repr = to_dict() if callable(to_dict) else str(event_obj)
            return {
                "action": action_name,
                "event_id": event_id,
                "event": event_repr,
                "note": note,
            }, dedup_key

        if action_name == "recall_by_tags":
            raw_tags = raw_action.get("tags") or ()
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
            target_domain = raw_action.get("domain", default_domain)
            if target_domain is not None:
                target_domain = str(target_domain).strip() or None
            top_k = _bounded_int(raw_action.get("top_k"), default=5, minimum=1, maximum=10)
            if not tags:
                return {"action": action_name, "error": "missing tags"}, None
            dedup_key = f"recall_by_tags:{target_domain}:{','.join(sorted(t.lower() for t in tags))}:{top_k}"
            if dedup_key in executed_action_keys:
                return None, None
            cards = self._caps.recall_level1(
                target_domain,
                tags,
                min_tag_hits=1,
                top_k=top_k,
            )
            shells: list[dict[str, Any]] = []
            details: dict[str, str] = {}
            for card in cards or ():
                fields = _extract_card_fields(card)
                cid = fields["card_id"]
                if not cid:
                    continue
                shells.append(_card_shell(fields))
                scanned_shell_tokens[cid] = estimate_tokens(
                    f"{fields['summary']} {' '.join(fields['tags'])}"
                )
                if fields["detail"]:
                    details[cid] = fields["detail"]
                    scanned_detail_tokens[cid] = estimate_tokens(fields["detail"])
            return {
                "action": action_name,
                "tags": tags,
                "cards": shells,
                "card_details": details,
            }, dedup_key

        return {"action": action_name, "error": f"unsupported action: {action_name}"}, None

    @staticmethod
    def _record_turns(
        raw_turns: Iterable[Mapping[str, Any]],
        scanned_turns: dict[tuple[str, int], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        normalized_list: list[dict[str, Any]] = []
        for turn in raw_turns:
            if not isinstance(turn, Mapping):
                continue
            norm = _normalize_turn(turn)
            key = (norm["episode_id"], norm["turn_id"])
            if key not in scanned_turns:
                scanned_turns[key] = norm
            normalized_list.append(norm)
        return normalized_list

    @staticmethod
    def _compute_scanned_raw_tokens(
        *,
        scanned_turns: Mapping[tuple[str, int], Mapping[str, Any]],
        scanned_detail_tokens: Mapping[str, int],
        scanned_note_tokens: Mapping[str, int],
        scanned_shell_tokens: Mapping[str, int],
    ) -> int:
        turn_tokens = sum(
            estimate_tokens(f"{t.get('role', '')}: {t.get('content', '')}")
            for t in scanned_turns.values()
        )
        return (
            turn_tokens
            + sum(scanned_detail_tokens.values())
            + sum(scanned_note_tokens.values())
            + sum(scanned_shell_tokens.values())
        )

    def _build_briefing(
        self,
        *,
        query: str,
        final_payload: Mapping[str, Any] | None,
        rounds_used: int,
        raw_turns_scanned: int,
        scanned_raw_tokens: int,
    ) -> ScoutBriefing:
        if final_payload is None:
            return self._abstained_briefing(
                query=query,
                rounds_used=rounds_used,
                raw_turns_scanned=raw_turns_scanned,
            )

        raw_abstained = final_payload.get("abstained")
        raw_found = final_payload.get("found")
        raw_text = str(final_payload.get("briefing_text") or "").strip()

        if raw_abstained is True:
            abstained = True
        elif raw_found is False:
            abstained = True
        elif not raw_text:
            abstained = True
        else:
            abstained = False

        if abstained:
            return self._abstained_briefing(
                query=query,
                rounds_used=rounds_used,
                raw_turns_scanned=raw_turns_scanned,
            )

        distilled_text = _truncate_to_token_budget(raw_text, self._max_briefing_tokens)
        distilled_tokens = estimate_tokens(distilled_text)

        cited_card_ids_list: list[str] = []
        seen_cids: set[str] = set()
        raw_cids = final_payload.get("cited_card_ids") or ()
        if isinstance(raw_cids, (list, tuple)):
            for cid in raw_cids:
                s = str(cid).strip()
                if s and s not in seen_cids:
                    seen_cids.add(s)
                    cited_card_ids_list.append(s)

        cited_spans_list: list[SourceSpan] = []
        seen_spans: set[tuple[str, int, int]] = set()
        raw_spans = final_payload.get("cited_spans") or ()
        if isinstance(raw_spans, (list, tuple)):
            for item in raw_spans:
                sp = _parse_source_span(item)
                if sp is not None:
                    key = (sp.episode_id, sp.turn_start, sp.turn_end)
                    if key not in seen_spans:
                        seen_spans.add(key)
                        cited_spans_list.append(sp)

        if distilled_tokens > 0 and scanned_raw_tokens > 0:
            compression_ratio = round(scanned_raw_tokens / distilled_tokens, 2)
        else:
            compression_ratio = 0.0

        return ScoutBriefing(
            query=query,
            found=True,
            abstained=False,
            briefing_text=distilled_text,
            cited_card_ids=tuple(cited_card_ids_list),
            cited_spans=tuple(cited_spans_list),
            rounds_used=rounds_used,
            raw_turns_scanned=raw_turns_scanned,
            distilled_tokens=distilled_tokens,
            compression_ratio=compression_ratio,
        )

    @staticmethod
    def _abstained_briefing(
        *,
        query: str,
        rounds_used: int,
        raw_turns_scanned: int,
    ) -> ScoutBriefing:
        return ScoutBriefing(
            query=query,
            found=False,
            abstained=True,
            briefing_text="",
            cited_card_ids=(),
            cited_spans=(),
            rounds_used=rounds_used,
            raw_turns_scanned=raw_turns_scanned,
            distilled_tokens=0,
            compression_ratio=0.0,
        )

    def _emit_telemetry(
        self,
        *,
        query: str,
        domain: str | None,
        briefing: ScoutBriefing,
        scanned_raw_tokens: int,
        actions_executed: list[str],
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: int,
        prefix_sha256: str,
    ) -> None:
        self._caps.record_telemetry(
            "investigate",
            {
                "query": query[:120],
                "domain": domain,
                "found": briefing.found,
                "abstained": briefing.abstained,
                "rounds_used": briefing.rounds_used,
                "raw_turns_scanned": briefing.raw_turns_scanned,
                "scanned_raw_tokens": scanned_raw_tokens,
                "distilled_tokens": briefing.distilled_tokens,
                "compression_ratio": briefing.compression_ratio,
                "actions_executed": actions_executed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": elapsed_ms,
                "prefix_sha256": prefix_sha256,
            },
        )


__all__ = [
    "SCOUT_L0_PREFIX",
    "ScoutAgent",
    "ScoutBriefing",
    "estimate_tokens",
]
