"""Zero-dependency CLI entry point for reiji-maigo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
import threading
from typing import Any
from urllib.parse import urlparse

from reiji_maigo.core.contracts import LOCKED_DOMAINS
from reiji_maigo.engine import ReijiMemoryEngine
from reiji_maigo.security import (
    ADMIN_TOKEN_HEADER,
    ExclusiveLoopbackHTTPServer,
    LoopbackGuardedRequestHandler,
    export_safe_bundle,
    import_safe_bundle,
)
from reiji_maigo.swarm.coordinator import SwarmCoordinator
from reiji_maigo.swarm.runner import SwarmRunnerError

_MAX_SERVE_BODY_BYTES: int = 64 * 1024


def _parse_tags(raw_tags: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for item in raw_tags or ():
        for part in item.split(","):
            cleaned = part.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cmd_status(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    info = engine.status()
    if args.json:
        _emit_json(info)
        return 0
    counts = info["counts"]
    print(f"Reiji-Maigo Home: {info['home']}")
    print(f"Database: {info['db_path']}")
    print(f"Plugins Disabled: {info['plugins_disabled']}")
    print(
        f"Counts: turns={counts['raw_turns']}, "
        f"pool={counts['pool_visible']}/{counts['pool_total']}, "
        f"cards={counts['cards_active']}/{counts['cards_total']}, "
        f"tags={counts['tags']}, "
        f"pending_backfills={counts['pending_backfills']}"
    )
    return 0


def _cmd_pool(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    if args.triage:
        buckets = engine.pool.plan_midnight_triage(
            max_steps=args.max_steps,
            max_dormant_nights=args.max_dormant_nights,
        )
        payload = {
            name: [ev.to_dict() for ev in events]
            for name, events in buckets.items()
        }
        if args.json:
            _emit_json(payload)
        else:
            for name, events in buckets.items():
                print(f"[{name}] ({len(events)})")
                for ev in events:
                    print(f"  - {ev.event_id}: {ev.title} [{ev.status}|{ev.scope}]")
        return 0

    if args.list_events:
        events = engine.pool.list_events(include_inactive=args.include_inactive)
        if args.json:
            _emit_json([ev.to_dict() for ev in events])
        else:
            if not events:
                print("(no pool events)")
            for ev in events:
                print(f"- {ev.event_id} [{ev.status}|{ev.scope}] {ev.title} (hits={ev.hit_count})")
        return 0

    snapshot = engine.pool_snapshot(include_undigested=not args.no_undigested)
    if args.json:
        _emit_json({"snapshot": snapshot})
    else:
        print(snapshot)
    return 0


def _cmd_recall(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    if args.evidence:
        turns = engine.read_evidence(args.evidence, pad=args.pad)
        if args.json:
            _emit_json({"card_id": args.evidence, "turns": turns})
        else:
            for turn in turns:
                print(f"#{turn['turn_id']} ({turn['episode_id']}/{turn['role']}): {turn['content']}")
        return 0

    tags = _parse_tags(args.tag)
    cards = (
        engine.recall_cards(
            args.domain,
            tags,
            min_tag_hits=args.min_tag_hits,
            top_k=args.top_k,
        )
        if tags
        else []
    )

    deep_hits: list[dict[str, Any]] = []
    if args.query and (not cards or args.deep):
        deep_hits = engine.deep_slice(args.query, limit=args.top_k, pad=args.pad)

    if args.json:
        _emit_json(
            {
                "cards": [card.to_dict() for card in cards],
                "deep_hits": deep_hits,
            }
        )
        return 0

    if cards:
        for card in cards:
            tag_str = ", ".join(card.tags)
            print(f"- [{card.domain}/{card.card_type}] {card.title} (id={card.card_id}, tags={tag_str})")
    elif not deep_hits:
        print("(no matching memory cards)")

    if deep_hits:
        print("[Level-3 Deep Slice]")
        for item in deep_hits:
            span = item["span"]
            print(
                f"- backfill#{item['backfill_id']} "
                f"{span['episode_id']}:{span['turn_start']}-{span['turn_end']}"
            )
    return 0


def _cmd_turn(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    turn_id = engine.append_turn(args.episode, args.role, args.content)
    if args.json:
        _emit_json({"turn_id": turn_id, "episode_id": args.episode, "role": args.role})
    else:
        print(f"Recorded turn #{turn_id} ({args.episode}/{args.role})")
    return 0


def _cmd_swarm(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    action = getattr(args, "swarm_action", None) or "status"
    with SwarmCoordinator(engine) as coordinator:
        if action == "status":
            if args.json:
                _emit_json(coordinator.pulse_summary())
            else:
                print(coordinator.render_pulse_banner())
            return 0

        if action == "scribe":
            report = coordinator.run_scribe_now(now_iso=getattr(args, "now_iso", None))
            payload = {
                "turns_digested": report.turns_digested,
                "watermark_after": report.watermark_after,
                "created_ids": list(report.created_ids),
                "updated_ids": list(report.updated_ids),
                "skipped_banter": report.skipped_banter,
                "elapsed_ms": report.elapsed_ms,
            }
            if args.json:
                _emit_json(payload)
            else:
                print(
                    f"Scribe digested {report.turns_digested} turn(s) "
                    f"(watermark={report.watermark_after}, created={len(report.created_ids)}, "
                    f"updated={len(report.updated_ids)}, banter_filtered={report.skipped_banter}, "
                    f"elapsed={report.elapsed_ms}ms)"
                )
            return 0

        if action == "scout":
            raw_query = getattr(args, "query", None) or getattr(args, "query_pos", None)
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ValueError("swarm scout requires a non-empty query (--query or positional query)")
            briefing = coordinator.dispatch_scout(
                raw_query.strip(),
                domain=getattr(args, "domain", None),
                hint_tags=_parse_tags(getattr(args, "tag", [])),
            )
            payload = {
                "query": briefing.query,
                "found": briefing.found,
                "abstained": briefing.abstained,
                "briefing_text": briefing.briefing_text,
                "tail_block": briefing.to_tail_block(),
                "cited_card_ids": list(briefing.cited_card_ids),
                "cited_spans": [sp.to_dict() for sp in briefing.cited_spans],
                "rounds_used": briefing.rounds_used,
                "raw_turns_scanned": briefing.raw_turns_scanned,
                "distilled_tokens": briefing.distilled_tokens,
                "compression_ratio": briefing.compression_ratio,
            }
            if args.json:
                _emit_json(payload)
            else:
                tail_block = briefing.to_tail_block()
                print(tail_block if tail_block else "[Scout Abstained] No grounded evidence found.")
            return 0

        if action == "midnight":
            report = coordinator.run_midnight_archivist(now_iso=getattr(args, "now_iso", None))
            payload = {
                "dropped_micro": list(report.dropped_micro),
                "compacted_events": list(report.compacted_events),
                "archived_cards": list(report.archived_cards),
                "hibernated_cards": list(report.hibernated_cards),
                "superseded_cards": list(report.superseded_cards),
                "revoked_cards": list(report.revoked_cards),
                "backfilled_cards": list(report.backfilled_cards),
                "elapsed_ms": report.elapsed_ms,
            }
            if args.json:
                _emit_json(payload)
            else:
                print(
                    f"Archivist midnight curation complete: "
                    f"dropped_micro={len(report.dropped_micro)}, "
                    f"compacted={len(report.compacted_events)}, "
                    f"archived={len(report.archived_cards)}, "
                    f"hibernated={len(report.hibernated_cards)}, "
                    f"superseded={len(report.superseded_cards)}, "
                    f"revoked={len(report.revoked_cards)}, "
                    f"backfilled={len(report.backfilled_cards)}, "
                    f"elapsed={report.elapsed_ms}ms"
                )
            return 0

        raise ValueError(f"Unsupported swarm action: {action!r}")


def _cmd_audit(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    action = getattr(args, "audit_action", None) or "status"

    if action == "status":
        summary = engine.security_summary()
        if args.json:
            _emit_json(summary)
        else:
            chain = summary["audit_chain"]
            seals = summary["card_seals"]
            print(
                f"Audit Chain: {'VERIFIED' if chain['ok'] else 'BROKEN'} "
                f"(checked={chain['checked']}, reason={chain['reason']})"
            )
            print(
                f"Card Seals: {'VERIFIED' if seals['ok'] else 'TAMPERED'} "
                f"(checked={seals['checked']}, sealed={seals['sealed']}, "
                f"tampered={seals['tampered_count']}, unsealed={seals['unsealed_count']})"
            )
            print(
                f"Security Counters: redactions={summary['redactions']} "
                f"(secrets={summary['redacted_secrets']}, blindfolded={summary['blindfolded_paths']}), "
                f"poison_blocked={summary['poison_blocked']}, pending_l1={summary['pending_l1']}"
            )
        return 0

    if action == "verify":
        chain = engine.audit.verify_chain()
        seals = engine.verify_card_seals()
        ok = bool(chain["ok"] and seals["ok"])
        payload = {
            "ok": ok,
            "audit_chain": chain,
            "card_seals": seals,
        }
        if args.json:
            _emit_json(payload)
        else:
            print(
                f"Audit Chain: {'VERIFIED' if chain['ok'] else 'BROKEN'} "
                f"(checked={chain['checked']}, broken_seq={chain['broken_seq']}, reason={chain['reason']})"
            )
            print(
                f"Card Seals: {'VERIFIED' if seals['ok'] else 'TAMPERED'} "
                f"(checked={seals['checked']}, sealed={seals['sealed']}, "
                f"tampered={len(seals['tampered'])}, unsealed={len(seals['unsealed'])}, "
                f"orphans={len(seals['orphan_seals'])})"
            )
        return 0 if ok else 1

    if action == "pending":
        limit = max(0, int(getattr(args, "limit", 100)))
        reviews = engine.approval_gate.list_reviews(status="pending", limit=limit)
        if args.json:
            _emit_json({"count": len(reviews), "pending": reviews})
        else:
            if not reviews:
                print("(no pending L1 reviews)")
            for rev in reviews:
                print(
                    f"- {rev['review_id']} [{rev['tier']}/{rev['op_kind']}] "
                    f"target={rev['target_card_id']} actor={rev['actor']} reason={rev['reason']}"
                )
        return 0

    if action == "approve":
        approver = getattr(args, "approver", "human_admin") or "human_admin"
        res = engine.approval_gate.approve_review(
            args.review_id,
            engine.cards,
            approver=approver,
        )
        result_card = res.get("result_card")
        payload = dict(res)
        payload["result_card"] = (
            result_card.to_dict() if result_card is not None else None
        )
        if args.json:
            _emit_json(payload)
        else:
            print(
                f"Approved review {payload['review_id']} "
                f"({payload['op_kind']} target={payload['target_card_id']}, audit_seq={payload['audit_seq']})"
            )
        return 0

    if action == "reject":
        reviewer = getattr(args, "reviewer", "human_admin") or "human_admin"
        reason = getattr(args, "reason", "") or ""
        res = engine.approval_gate.reject_review(
            args.review_id,
            reason=reason,
            reviewer=reviewer,
        )
        if args.json:
            _emit_json(res)
        else:
            print(
                f"Rejected review {res['review_id']} "
                f"({res['op_kind']} target={res['target_card_id']}, audit_seq={res['audit_seq']})"
            )
        return 0

    raise ValueError(f"Unsupported audit action: {action!r}")


def _cmd_vault(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    action = getattr(args, "vault_action", None) or "status"

    if action == "status":
        data = engine.vault.load(strict=False)
        vault_path_str = str(engine.vault.vault_path)
        ephemeral = vault_path_str == ":memory:"
        exists = False if ephemeral else engine.vault.vault_path.exists()
        base_url = str(data.get("flash_base_url") or "").strip()
        api_key = str(data.get("flash_api_key") or "").strip()
        model = str(data.get("flash_model") or "").strip()
        payload = {
            "vault_path": vault_path_str,
            "exists": exists,
            "ephemeral": ephemeral,
            "has_signing_key": bool(data.get("signing_key_hex")),
            "flash_configured": bool(base_url and api_key),
            "flash_base_url": base_url,
            "flash_model": model,
            "keys": sorted(str(k) for k in data.keys()),
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Vault Path: {payload['vault_path']} (exists={payload['exists']}, ephemeral={payload['ephemeral']})")
            print(f"Signing Key Present: {payload['has_signing_key']}")
            print(
                f"Flash Credentials: {'configured' if payload['flash_configured'] else 'unconfigured'}"
                + (f" (base_url={base_url}, model={model or 'default'})" if payload["flash_configured"] else "")
            )
        return 0

    if action in ("set-flash", "set-key"):
        base_url = str(getattr(args, "base_url", "") or "").strip()
        api_key = str(getattr(args, "api_key", "") or "").strip()
        model = str(getattr(args, "model", "") or "").strip()
        if not base_url:
            raise ValueError("--base-url must be a non-empty string")
        if not api_key:
            raise ValueError("--api-key must be a non-empty string")
        data = engine.vault.load(for_update=True)
        data["flash_base_url"] = base_url
        data["flash_api_key"] = api_key
        if model:
            data["flash_model"] = model
        engine.vault.save(data)
        engine.audit.append(
            "human_admin",
            "vault_set_flash",
            tier="L0",
            detail={
                "flash_base_url": base_url,
                "has_model": bool(model),
            },
        )
        payload = {
            "ok": True,
            "vault_path": str(engine.vault.vault_path),
            "flash_configured": True,
            "flash_base_url": base_url,
            "flash_model": model or str(data.get("flash_model") or ""),
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Saved encrypted Flash credentials to {payload['vault_path']}")
        return 0

    if action == "export-key":
        signing_key = engine.vault.get_or_create_signing_key()
        key_hex = signing_key.hex()
        engine.audit.append(
            "human_admin",
            "vault_export_key",
            tier="L0",
            detail={"vault_path": str(engine.vault.vault_path)},
        )
        payload = {
            "ok": True,
            "vault_path": str(engine.vault.vault_path),
            "signing_key_hex": key_hex,
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Signing Key (hex): {key_hex}")
        return 0

    if action == "export":
        if str(engine.db.db_path) == ":memory:":
            raise ValueError("Cannot export safe bundle from an in-memory database")
        raw_key_hex = str(getattr(args, "signing_key", "") or "").strip()
        if raw_key_hex:
            if len(raw_key_hex) != 64:
                raise ValueError("--signing-key must be a 64-character hex string (32 bytes)")
            signing_key = bytes.fromhex(raw_key_hex)
        else:
            signing_key = engine.vault.get_or_create_signing_key()
        res = export_safe_bundle(
            engine.db.db_path,
            args.zip_path,
            signing_key=signing_key,
        )
        if args.json:
            _emit_json(res)
        else:
            print(
                f"Exported safe bundle to {res['bundle_path']} "
                f"(size={res['db_size']} bytes, sha256={res['db_sha256'][:16]}...)"
            )
        return 0

    if action == "import":
        if str(engine.db.db_path) == ":memory:":
            raise ValueError("Cannot import safe bundle into an in-memory database")
        target_db = Path(engine.db.db_path)
        raw_key_hex = str(getattr(args, "signing_key", "") or "").strip()
        explicit_key = bool(raw_key_hex)
        if explicit_key:
            if len(raw_key_hex) != 64:
                raise ValueError("--signing-key must be a 64-character hex string (32 bytes)")
            signing_key = bytes.fromhex(raw_key_hex)
        else:
            signing_key = engine.vault.get_or_create_signing_key()

        # Detect whether CLI startup just initialized an empty schema DB in a fresh --home.
        with engine.db.read_connection() as conn:
            pristine_empty = (
                int(conn.execute("SELECT COUNT(*) FROM raw_turns").fetchone()[0]) == 0
                and int(conn.execute("SELECT COUNT(*) FROM pool_events").fetchone()[0]) == 0
                and int(conn.execute("SELECT COUNT(*) FROM memory_cards").fetchone()[0]) == 0
                and int(conn.execute("SELECT COUNT(*) FROM audit_chain").fetchone()[0]) == 0
            )
        effective_overwrite = bool(getattr(args, "overwrite", False)) or pristine_empty

        # Close active SQLite handles before replacing the DB file (required on Windows).
        engine.close()
        res = import_safe_bundle(
            args.zip_path,
            target_db,
            signing_key=signing_key,
            overwrite=effective_overwrite,
        )
        if explicit_key and str(engine.vault.vault_path) != ":memory:":
            vault_data = engine.vault.load(for_update=True)
            vault_data["signing_key_hex"] = signing_key.hex()
            engine.vault.save(vault_data)
        if args.json:
            _emit_json(res)
        else:
            print(
                f"Imported safe bundle ({res['status']}) -> {res['target_db_path']} "
                f"(size={res['db_size']} bytes, sha256={res['db_sha256'][:16]}...)"
            )
        return 0

    raise ValueError(f"Unsupported vault action: {action!r}")


def create_loopback_server(
    engine: ReijiMemoryEngine,
    *,
    host: str = "127.0.0.1",
    port: int = 18721,
    admin_token: str | None = None,
    require_request_id: bool = False,
    quiet: bool = True,
) -> ExclusiveLoopbackHTTPServer:
    """Create a zero-trust 127.0.0.1 HTTP server exposing /v1/status, /v1/pulse, /v1/recall."""
    engine_lock = threading.Lock()

    class _ReijiServeHandler(LoopbackGuardedRequestHandler):
        # Enforce Host/Origin/Sec-Fetch-Site/Admin-Token/ReplayGuard once in parse_request().
        auto_guard = True

        def do_GET(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/v1/status":
                with engine_lock:
                    info = engine.status()
                self.send_json(200, {"ok": True, "status": info})
                return
            if path == "/v1/pulse":
                with engine_lock:
                    with SwarmCoordinator(engine) as coordinator:
                        summary = coordinator.pulse_summary()
                self.send_json(200, {"ok": True, "pulse": summary})
                return
            self.send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path != "/v1/recall":
                self.send_json(404, {"ok": False, "error": "not found"})
                return

            raw_len = self.headers.get("Content-Length", "0")
            try:
                content_len = int(raw_len)
            except (TypeError, ValueError):
                self.send_json(400, {"ok": False, "error": "invalid Content-Length"})
                return
            if content_len < 0 or content_len > _MAX_SERVE_BODY_BYTES:
                self.send_json(413, {"ok": False, "error": "request body too large"})
                return

            raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body = json.loads(raw_body.decode("utf-8") or "{}")
            except Exception:
                self.send_json(400, {"ok": False, "error": "invalid JSON body"})
                return
            if not isinstance(body, dict):
                self.send_json(400, {"ok": False, "error": "JSON body must be an object"})
                return

            domain = body.get("domain")
            if domain is not None and domain not in LOCKED_DOMAINS:
                self.send_json(400, {"ok": False, "error": f"invalid domain: {domain!r}"})
                return

            raw_tags = body.get("tags")
            if isinstance(raw_tags, str):
                tags = _parse_tags([raw_tags])
            elif isinstance(raw_tags, list):
                tags = _parse_tags([str(t) for t in raw_tags])
            elif raw_tags is None:
                tags = []
            else:
                self.send_json(400, {"ok": False, "error": "tags must be a list or string"})
                return

            query = body.get("query")
            if query is not None and not isinstance(query, str):
                self.send_json(400, {"ok": False, "error": "query must be a string"})
                return

            evidence = body.get("evidence")
            if evidence is not None and not isinstance(evidence, str):
                self.send_json(400, {"ok": False, "error": "evidence must be a string"})
                return

            try:
                min_tag_hits = int(body.get("min_tag_hits", 1))
                top_k = int(body.get("top_k", 3))
                pad = int(body.get("pad", 1))
                deep = bool(body.get("deep", False))
            except (TypeError, ValueError):
                self.send_json(400, {"ok": False, "error": "invalid numeric recall parameter"})
                return

            try:
                with engine_lock:
                    if evidence:
                        turns = engine.read_evidence(evidence, pad=pad)
                        self.send_json(
                            200,
                            {"ok": True, "card_id": evidence, "turns": turns},
                        )
                        return
                    cards = (
                        engine.recall_cards(
                            domain,
                            tags,
                            min_tag_hits=min_tag_hits,
                            top_k=top_k,
                        )
                        if tags
                        else []
                    )
                    deep_hits: list[dict[str, Any]] = []
                    if query and (not cards or deep):
                        deep_hits = engine.deep_slice(query, limit=top_k, pad=pad)
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "cards": [card.to_dict() for card in cards],
                        "deep_hits": deep_hits,
                    },
                )
            except KeyError as exc:
                self.send_json(404, {"ok": False, "error": str(exc)})
            except ValueError as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})

    return ExclusiveLoopbackHTTPServer(
        (host, int(port)),
        _ReijiServeHandler,
        admin_token=admin_token,
        require_admin_token=True,
        require_request_id=require_request_id,
        quiet=quiet,
    )


def _cmd_serve(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    server = create_loopback_server(
        engine,
        host=args.host,
        port=args.port,
        admin_token=args.token,
        require_request_id=bool(getattr(args, "require_request_id", False)),
        quiet=not bool(getattr(args, "verbose", False)),
    )
    bound_host, bound_port = server.server_address[:2]
    startup = {
        "host": bound_host,
        "port": bound_port,
        "admin_token_header": ADMIN_TOKEN_HEADER,
        "admin_token": server.admin_token,
        "endpoints": [
            "GET /v1/status",
            "GET /v1/pulse",
            "POST /v1/recall",
        ],
    }
    if args.json:
        _emit_json(startup)
    else:
        print(f"Reiji-Maigo loopback server listening on http://{bound_host}:{bound_port}")
        print(f"Auth Header: {ADMIN_TOKEN_HEADER}: {server.admin_token}")
        print("Endpoints: GET /v1/status, GET /v1/pulse, POST /v1/recall")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def _cmd_mcp(engine: ReijiMemoryEngine, args: argparse.Namespace) -> int:
    from reiji_maigo.bridges.mcp_stdio import ReijiMcpServer

    server = ReijiMcpServer(
        engine=engine,
        read_only=bool(getattr(args, "read_only", False)),
    )
    server.serve_stdio(framing=str(getattr(args, "framing", "auto") or "auto"))
    return 0


def build_parser(
    *,
    plugin_commands: dict[str, Any] | None = None,
) -> argparse.ArgumentParser:
    """Build the top-level argparse parser for `reiji` CLI."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--home",
        default=None,
        help="Override REIJI_MAIGO_HOME directory.",
    )
    shared.add_argument(
        "--db",
        default=None,
        help="Override SQLite database path (supports :memory:).",
    )
    shared.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit machine-readable JSON output.",
    )

    parser = argparse.ArgumentParser(
        prog="reiji",
        description="Reiji-Maigo: zero-dependency two-tier memory engine CLI.",
        parents=[shared],
    )
    parser.set_defaults(json=False)
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    p_status = subparsers.add_parser(
        "status",
        parents=[shared],
        help="Show kernel, plugin, and memory storage status.",
    )
    p_status.set_defaults(_handler=_cmd_status)

    p_pool = subparsers.add_parser(
        "pool",
        parents=[shared],
        help="Inspect ActivePool working-memory snapshot, events, or midnight triage plan.",
    )
    p_pool.add_argument(
        "--list",
        dest="list_events",
        action="store_true",
        help="List structured pool events instead of rendering snapshot.",
    )
    p_pool.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include archived/hibernated/dropped events when used with --list.",
    )
    p_pool.add_argument(
        "--no-undigested",
        action="store_true",
        help="Omit undigested raw turn tail from snapshot output.",
    )
    p_pool.add_argument(
        "--triage",
        action="store_true",
        help="Display dry-run midnight triage buckets.",
    )
    p_pool.add_argument("--max-steps", type=int, default=5, help="Step threshold for triage.")
    p_pool.add_argument(
        "--max-dormant-nights",
        type=int,
        default=7,
        help="Dormant nights threshold for triage.",
    )
    p_pool.set_defaults(_handler=_cmd_pool)

    p_recall = subparsers.add_parser(
        "recall",
        parents=[shared],
        help="Recall long-term memory cards (Level 1/2/3).",
    )
    p_recall.add_argument(
        "--domain",
        choices=LOCKED_DOMAINS,
        default=None,
        help="Filter by locked L1 domain.",
    )
    p_recall.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Query tag(s); repeatable or comma-separated.",
    )
    p_recall.add_argument(
        "--query",
        default=None,
        help="Full-text fallback query for Level-3 deep slice.",
    )
    p_recall.add_argument(
        "--deep",
        action="store_true",
        help="Run Level-3 FTS deep slice even when Level-1 cards match.",
    )
    p_recall.add_argument(
        "--evidence",
        default=None,
        help="Card ID for Level-2 raw evidence slicing.",
    )
    p_recall.add_argument("--min-tag-hits", type=int, default=1, help="Minimum direct tag hits.")
    p_recall.add_argument("--top-k", type=int, default=3, help="Maximum number of results.")
    p_recall.add_argument("--pad", type=int, default=1, help="Surrounding turn padding.")
    p_recall.set_defaults(_handler=_cmd_recall)

    p_turn = subparsers.add_parser(
        "turn",
        parents=[shared],
        help="Append a raw conversation turn into the FTS5-indexed ledger.",
    )
    p_turn.add_argument("--episode", required=True, help="Episode identifier.")
    p_turn.add_argument("--role", default="user", help="Speaker role (default: user).")
    p_turn.add_argument("--content", required=True, help="Turn text content.")
    p_turn.set_defaults(_handler=_cmd_turn)

    p_swarm = subparsers.add_parser(
        "swarm",
        parents=[shared],
        help="Inspect or trigger headless memory swarm workers (status | scribe | scout | midnight).",
    )
    p_swarm.set_defaults(_handler=_cmd_swarm, swarm_action="status")
    swarm_sub = p_swarm.add_subparsers(dest="swarm_action", metavar="<action>")

    p_swarm_status = swarm_sub.add_parser(
        "status",
        parents=[shared],
        help="Show swarm pulse telemetry banner or JSON summary.",
    )
    p_swarm_status.set_defaults(_handler=_cmd_swarm, swarm_action="status")

    p_swarm_scribe = swarm_sub.add_parser(
        "scribe",
        parents=[shared],
        help="Synchronously digest pending raw turns into ActivePool via ScribeAgent.",
    )
    p_swarm_scribe.add_argument("--now-iso", default=None, help="Optional ISO timestamp override.")
    p_swarm_scribe.set_defaults(_handler=_cmd_swarm, swarm_action="scribe")

    p_swarm_scout = swarm_sub.add_parser(
        "scout",
        parents=[shared],
        help="Run read-only deep reconnaissance via ScoutAgent.",
    )
    p_swarm_scout.add_argument("query_pos", nargs="?", default=None, help="Reconnaissance query text.")
    p_swarm_scout.add_argument("--query", default=None, help="Reconnaissance query text.")
    p_swarm_scout.add_argument(
        "--domain",
        choices=LOCKED_DOMAINS,
        default=None,
        help="Optional locked L1 domain filter.",
    )
    p_swarm_scout.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Optional hint tag(s); repeatable or comma-separated.",
    )
    p_swarm_scout.set_defaults(_handler=_cmd_swarm, swarm_action="scout")

    p_swarm_midnight = swarm_sub.add_parser(
        "midnight",
        parents=[shared],
        help="Execute midnight archival, compaction, and backfill curation via ArchivistAgent.",
    )
    p_swarm_midnight.add_argument("--now-iso", default=None, help="Optional ISO timestamp override.")
    p_swarm_midnight.set_defaults(_handler=_cmd_swarm, swarm_action="midnight")

    p_audit = subparsers.add_parser(
        "audit",
        parents=[shared],
        help="Inspect tamper-evident audit chain, card seals, and L1 mutation approvals.",
    )
    p_audit.set_defaults(_handler=_cmd_audit, audit_action="status")
    audit_sub = p_audit.add_subparsers(dest="audit_action", metavar="<action>")

    p_audit_status = audit_sub.add_parser(
        "status",
        parents=[shared],
        help="Show aggregated security summary (chain, card seals, L1 queue, sanitizer).",
    )
    p_audit_status.set_defaults(_handler=_cmd_audit, audit_action="status")

    p_audit_verify = audit_sub.add_parser(
        "verify",
        parents=[shared],
        help="Verify audit hash chain continuity and all card HMAC-SHA256 seals.",
    )
    p_audit_verify.set_defaults(_handler=_cmd_audit, audit_action="verify")

    p_audit_pending = audit_sub.add_parser(
        "pending",
        parents=[shared],
        help="List pending L1 human-approval mutation reviews.",
    )
    p_audit_pending.add_argument("--limit", type=int, default=100, help="Maximum pending reviews to list.")
    p_audit_pending.set_defaults(_handler=_cmd_audit, audit_action="pending")

    p_audit_approve = audit_sub.add_parser(
        "approve",
        parents=[shared],
        help="Approve and execute a pending L1 mutation review.",
    )
    p_audit_approve.add_argument("review_id", help="Pending review identifier.")
    p_audit_approve.add_argument("--approver", default="human_admin", help="Approver actor name.")
    p_audit_approve.set_defaults(_handler=_cmd_audit, audit_action="approve")

    p_audit_reject = audit_sub.add_parser(
        "reject",
        parents=[shared],
        help="Reject a pending L1 mutation review without mutating the card.",
    )
    p_audit_reject.add_argument("review_id", help="Pending review identifier.")
    p_audit_reject.add_argument("--reason", default="", help="Optional rejection reason.")
    p_audit_reject.add_argument("--reviewer", default="human_admin", help="Reviewer actor name.")
    p_audit_reject.set_defaults(_handler=_cmd_audit, audit_action="reject")

    p_vault = subparsers.add_parser(
        "vault",
        parents=[shared],
        help="Manage encrypted DPAPI/0600 credential vault and anti-bomb portable bundles.",
    )
    p_vault.set_defaults(_handler=_cmd_vault, vault_action="status")
    vault_sub = p_vault.add_subparsers(dest="vault_action", metavar="<action>")

    p_vault_status = vault_sub.add_parser(
        "status",
        parents=[shared],
        help="Show redacted vault and Flash credential configuration status.",
    )
    p_vault_status.set_defaults(_handler=_cmd_vault, vault_action="status")

    p_vault_set_flash = vault_sub.add_parser(
        "set-flash",
        aliases=["set-key"],
        parents=[shared],
        help="Encrypt and store Flash API credentials in the local DPAPI/0600 vault.",
    )
    p_vault_set_flash.add_argument("--base-url", required=True, help="OpenAI-compatible Flash base URL.")
    p_vault_set_flash.add_argument("--api-key", required=True, help="Flash API key (stored encrypted).")
    p_vault_set_flash.add_argument("--model", default=None, help="Optional Flash model name.")
    p_vault_set_flash.set_defaults(_handler=_cmd_vault, vault_action="set-flash")

    p_vault_export_key = vault_sub.add_parser(
        "export-key",
        parents=[shared],
        help="Export the 64-hex signing key for cross-machine portable bundle verification.",
    )
    p_vault_export_key.set_defaults(_handler=_cmd_vault, vault_action="export-key")

    p_vault_export = vault_sub.add_parser(
        "export",
        parents=[shared],
        help="Export a consistent, HMAC-signed ZIP_STORED safe memory bundle.",
    )
    p_vault_export.add_argument("zip_path", help="Output bundle archive path.")
    p_vault_export.add_argument(
        "--signing-key",
        default=None,
        help="Optional explicit 64-hex signing key (defaults to local vault key).",
    )
    p_vault_export.set_defaults(_handler=_cmd_vault, vault_action="export")

    p_vault_import = vault_sub.add_parser(
        "import",
        parents=[shared],
        help="Verify and import a signed ZIP_STORED safe memory bundle.",
    )
    p_vault_import.add_argument("zip_path", help="Input bundle archive path.")
    p_vault_import.add_argument(
        "--signing-key",
        default=None,
        help="Origin 64-hex signing key for cross-machine bundle import (adopted into local vault).",
    )
    p_vault_import.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing modified local SQLite database.",
    )
    p_vault_import.set_defaults(_handler=_cmd_vault, vault_action="import")

    p_serve = subparsers.add_parser(
        "serve",
        parents=[shared],
        help="Run zero-trust 127.0.0.1 loopback HTTP API (/v1/status, /v1/pulse, /v1/recall).",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Loopback bind host (must be 127.0.0.1).")
    p_serve.add_argument("--port", type=int, default=18721, help="TCP port to bind (default: 18721).")
    p_serve.add_argument("--token", default=None, help="Optional explicit X-Reiji-Admin-Token.")
    p_serve.add_argument(
        "--require-request-id",
        action="store_true",
        help="Require X-Reiji-Request-Id header on every request.",
    )
    p_serve.add_argument("--verbose", action="store_true", help="Enable HTTP access logging.")
    p_serve.set_defaults(_handler=_cmd_serve)

    p_mcp = subparsers.add_parser(
        "mcp",
        parents=[shared],
        help="Run zero-dependency stdio JSON-RPC 2.0 MCP server (tools & read-only resources).",
    )
    p_mcp.add_argument(
        "--framing",
        choices=("auto", "ndjson", "lsp"),
        default="auto",
        help="Stdio framing mode: auto (detect), ndjson, or lsp Content-Length (default: auto).",
    )
    p_mcp.add_argument(
        "--read-only",
        action="store_true",
        help="Hide and block mutating MCP tools (e.g., reiji_update_pool).",
    )
    p_mcp.set_defaults(_handler=_cmd_mcp)

    for cmd_name, spec in sorted((plugin_commands or {}).items()):
        if cmd_name in {"status", "pool", "recall", "turn", "swarm", "audit", "vault", "serve", "mcp"}:
            continue
        p_ext = subparsers.add_parser(
            cmd_name,
            parents=[shared],
            help=getattr(spec, "help_text", "") or f"Plugin command: {cmd_name}",
        )
        configure_parser = getattr(spec.handler, "configure_parser", None)
        if callable(configure_parser):
            configure_parser(p_ext)
        else:
            p_ext.add_argument("extra_args", nargs=argparse.REMAINDER)
        p_ext.set_defaults(_plugin_spec=spec)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the `reiji` CLI and return a POSIX exit code."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # First-pass probe for --home / --db so plugin CLI commands can be discovered safely.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--home", default=None)
    probe.add_argument("--db", default=None)
    known, _ = probe.parse_known_args(raw_argv)

    with ReijiMemoryEngine(home=known.home, db_path=known.db) as engine:
        parser = build_parser(plugin_commands=engine.kernel.cli_commands)
        args = parser.parse_args(raw_argv)

        if not args.command:
            parser.print_help()
            return 0

        plugin_spec = getattr(args, "_plugin_spec", None)
        if plugin_spec is not None:
            try:
                result = plugin_spec.handler(engine, args)
                return int(result) if isinstance(result, int) else 0
            except (KeyError, ValueError, FileNotFoundError, FileExistsError, SwarmRunnerError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2

        handler = getattr(args, "_handler", None)
        if handler is None:
            parser.print_help()
            return 0
        try:
            return int(handler(engine, args))
        except (KeyError, ValueError, FileNotFoundError, FileExistsError, SwarmRunnerError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
