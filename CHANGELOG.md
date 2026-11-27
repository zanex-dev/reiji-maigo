# Changelog

**English** | [简体中文](./CHANGELOG_ZH.md)

All notable changes to **`reiji-maigo`** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-27 — Midnight Flagship (Stdio MCP, MemoRRF Self-Probing, 7-Decimal Anti-MemoryTax Suite & Six-Dimensional Adversarial Harness)

> *"At the stroke of midnight, the vessel restores itself whole — zero broken caches, zero forgotten truths, and zero third-party bloat."*

### Added
- **Zero-Dependency Dual-Frame Stdio MCP Server (`reiji_maigo/bridges/mcp_stdio.py`, `reiji mcp`)**:
  - Pure Python stdlib JSON-RPC 2.0 Model Context Protocol server auto-negotiating both newline-delimited JSON (`NDJSON`) and `Content-Length` LSP framing over `stdin/stdout` with strict `stderr`-only diagnostics.
  - Exposes 8 tail-mounted memory tools (`reiji_pool_snapshot`, `reiji_update_pool`, `reiji_recall_cards`, `reiji_hybrid_recall`, `reiji_read_evidence`, `reiji_scout_deep`, `reiji_swarm_pulse`, `reiji_security_audit`) and 3 live resources (`reiji://pool/snapshot`, `reiji://swarm/pulse`, `reiji://security/summary`), plus `--read-only` mutation gating.
- **Official `MemoRRF` Hybrid Reranker, Foresight Expiry & `MemDream` Sleep Self-Probe (`reiji_maigo/plugins/hybrid_reranker.py`, `reiji rerank`)**:
  - Registers the default `reranker:memo_rrf` provider on `ReijiKernel`, projecting natural-language queries onto controlled L2 tag vocabularies and fusing Level-1 `PPR` graph scores, CJK-Bigram card lexical ranks, and `FTS5` evidence hits via Reciprocal Rank Fusion (`RRF`).
  - Adds `Foresight` temporal expiry sweeping (`[EXPIRED_FORESIGHT]` demotion & warnings) and `MemDream` sleep-time self-probing that automatically backfills missed colloquial synonyms into `memory_tags.aliases_json`.
- **Idempotent Schema Migration Gate & Cold-Start Bootstrap Importer (`reiji_maigo/storage/migrations.py`, `reiji_maigo/plugins/bootstrap_importer.py`, `reiji import`)**:
  - Tracks SQLite `PRAGMA user_version` (`CURRENT_SCHEMA_VERSION = 2`) with `schema_migrations` ledgering and automatic pre-upgrade `.bak-v<old>` SQLite online backups for file-backed databases.
  - Adds `SHA-256`-fingerprinted idempotent cold-start ingestion across JSONL conversation transcripts (`import transcript`), read-only legacy SQLite stores (`import sqlite`), and Markdown handoff checkpoints with sealed seed-card minting (`import checkpoints`).
- **Six-Dimensional Adversarial Benchmark & Chaos Harness (`reiji_maigo/plugins/bench_harness.py`, `reiji bench`)**:
  - Evaluates `single_hop`, `multi_hop`, `temporal_foresight`, `tepa_reversal` (3-step `A -> B -> A'` cascading override with zero ghost leakage), `abstention`, and `anti_poison_banter` in an isolated sandbox (`--mode deterministic` or `--mode live` against real Flash APIs), rendering JSON and Markdown scorecards with P50/P95 latency and 7-decimal Net ROI.
- **Four Operational Self-Bootstrapping Plugins & Desktop Star-Panel UI Bridge (`reiji_maigo/plugins/*`, `reiji_maigo/bridges/star_panel.py`)**:
  - `roi_ledger` (`reiji roi`): 7-decimal USD compute-economics ledger tracking Flash spend (with L0 cache discount) vs avoided flagship & 200K 2x-cliff tokens, complete with `1e-7` USD reconciliation and negative-ROI `MemoryTax` alerts.
  - `epoch_vault` (`reiji epoch`): Append-only cold epoch vault with `.staging` atomic commits, bidirectional `SHA-256` verification, compact hot-DB tombstone stubs with refreshed `card_seals` HMACs, and lossless `restore`.
  - `thermal_sync` (`reiji radar`): Win32 `FILE_SHARE_READ | WRITE | DELETE` lock-free 16KB tail reader, symlink/reparse-point blocker, and calibrated `250s` KV-cache warmth + `180K/200K` cliff state machine.
  - `companion` (`reiji companion`): Cross-process file-locked, `.corrupt-*.bak`-protected reversible installer for `hooks.json`, `reiji-memory/SKILL.md`, and `mcp-config` snippets.
  - `star_panel`: Zero-top-level-`tkinter` duck-typed desktop UI bridge mounting the dark-gold-crimson `🕰️ 零时迷子` deck tab, header badge, radar cooldown marker, and one-click `L1` mutation Approve/Reject controls.

### Security & Verification
- Expanded the dual-mode (`REIJI_MAIGO_DISABLE_PLUGINS=1` bare kernel and `=0` full 7-plugin suite) verification matrix and clean-room static leak scanner to **`224/224` passing tests**, alongside GitHub Actions multi-OS CI (`Ubuntu` / `Windows` / `macOS` × Python `3.10`–`3.12`).

## [0.1.0] - 2026-11-27 — Foundation (Two-Tier Microkernel, Three-Bee Swarm & Four-Layer Zero-Trust Armor)

### Added
- **Zero-Dependency Microkernel & Immutable Contracts (`reiji_maigo/core/`)**:
  - Fault-isolated `ReijiKernel` (`register_cli`, `register_hooks`, `register_providers`), priority `HookBus`, `entry_points("reiji_maigo.plugins")` loader, locked 5-domain taxonomy (`proj`, `infra`, `research`, `companion`, `life`), and `@dataclass(frozen=True, slots=True)` `SourceSpan`, `PoolEvent`, and `MemoryCard` contracts.
- **Two-Tier Memory Engine (`reiji_maigo/storage/`, `pool/`, `cards/`, `engine.py`)**:
  - Single-file SQLite WAL storage with CJK-Bigram + Latin `unicode61` `FTS5` virtual indexing, `ActivePool` 4D Zeigarnik retention decay & lossless-span chain compaction, `CardStore` controlled L2 tags, 12-iteration stdlib `Personalized PageRank`, `TEPA` bitemporal supersede/revoke, and Level 1–3 progressive recall.
- **Headless Three-Bee Memory Swarm (`reiji_maigo/swarm/`)**:
  - `HttpFlashRunner` / `CallableFlashRunner` with `SHA-256` L0 prefix locking, physical capability proxies (`ScribeCapabilities`, `ScoutCapabilities`, `ArchivistCapabilities`), debounced `ScribeAgent`, read-only 3-drawer `ScoutAgent` with honest abstention, midnight `ArchivistAgent`, and `SwarmCoordinator` pulse banner.
- **Four-Layer Zero-Trust Security Armor & Companion Bridge (`reiji_maigo/security/`, `bridges/shinobu.py`)**:
  - Zero-latency secret/ticket redaction, media/private path blindfold, `ORIGIN_TRUST_LEVELS` anti-poisoning gate, Win32 `DPAPI` / POSIX `0600` `SecretVault`, domain-separated `HMAC-SHA256` card seals, anti-bomb `ZIP_STORED` bundle export/import, `AuditChainLedger`, `L0/L1/L2` `MutationApprovalGate`, `127.0.0.1` `ExclusiveLoopbackHTTPServer` with `ReplayGuard`, and zero-coupling `ReijiShinobuPlugin` (`129/129` tests passing).
