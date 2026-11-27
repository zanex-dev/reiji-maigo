# AGENTS.md — `reiji-maigo` 零依赖记忆微内核开发约定

## 1. 架构定位与硬边界
- **微内核底座（白帽开源）**：`reiji_maigo` 核心仅依赖 Python 3.10+ 标准库（`dependencies = []`），遵循 `PolyForm-Noncommercial-1.0.0` 非商用许可。
- **物理洁净红线**：源码、测试与文档严禁写入本机绝对路径或任何私有扩展包名称；外部能力一律通过 `entry_points("reiji_maigo.plugins")` 单向注入。
- **插件容错隔离**：所有插件通过 `ReijiKernel` 加载并受环境变量控制；任一插件加载或钩子执行异常不得影响微内核与其他插件运行。

## 2. 模块地图（v0.2.0）
- `reiji_maigo/core/contracts.py`：锁定记忆域 `LOCKED_DOMAINS` 及不可变核心数据契约（`SourceSpan`、`PoolEvent`、`MemoryCard`）。
- `reiji_maigo/core/kernel.py`：容错事件总线 `HookBus`、插件微内核 `ReijiKernel`、三插槽注册（`register_cli` / `register_hooks` / `register_providers`）与 `entry_points("reiji_maigo.plugins")` 加载器。
- `reiji_maigo/storage/`（含 `sqlite_engine.py`、`migrations.py`）、`reiji_maigo/pool/`、`reiji_maigo/cards/`、`reiji_maigo/engine.py`：SQLite WAL + CJK FTS5 底层账本与幂等 Schema 迁移引擎（`CURRENT_SCHEMA_VERSION = 2`）、活跃记忆池与衰减机制、双时态记忆卡与 PPR 检索及统一门面 `ReijiMemoryEngine`（含 `hybrid_recall` 重排委派与裸核回退）。
- `reiji_maigo/swarm/`：无头三蜂记忆蜂群子系统：
  - `runner.py`：零依赖 `SwarmModelRunner` 协议、`HttpFlashRunner`、`CallableFlashRunner` 及 L0 前缀 SHA-256 锁定校验。
  - `guards.py`：物理隔离角色能力沙箱（`ScribeCapabilities` / `ScoutCapabilities` / `ArchivistCapabilities`）、`pool_event_notes` 二层暂存表与 `swarm_telemetry` 遥测账本。
  - `scribe.py`：速记蜂 `ScribeAgent` 与非阻塞去抖队列 `AsyncScribeWorker`（闲聊门禁、代词消解、真实 `SourceSpan` 裁剪）。
  - `scout.py`：只读深潜侦察蜂 `ScoutAgent` 与 `ScoutBriefing`（三层抽屉渐进式钻取、诚实拒答与压缩简报）。
  - `archivist.py`：子夜馆长蜂 `ArchivistAgent` 与 `ArchivistReport`（`micro` 丢弃、长链无损压实、归档/休眠制卡、TEPA 冲突消解与缺页回填）。
  - `coordinator.py`：蜂群总控 `SwarmCoordinator` 与工业级「蜂群脉搏遥测面板」（`pulse_summary` / `render_pulse_banner`，汇入 7 位小数 `economics` 与 `cold_vault` 遥测）。
- `reiji_maigo/security/`：零依赖全维安全装甲四层防线：
  - `sanitizer.py`：零延迟凭据/票据擦除、隐私与媒体路径盲罩、来源信任分级（`ORIGIN_TRUST_LEVELS`）与反记忆投毒/防权限洗白净水闸。
  - `vault.py`：Win32 `DPAPI` / POSIX `0600` 加密凭据保险箱（`SecretVault`）、域分离 `PBKDF2`/`HMAC-SHA256` 封签与防解压炸弹便携记忆归档（`export_safe_bundle` / `import_safe_bundle`）。
  - `audit_gate.py`：防篡改哈希链审计账本（`AuditChainLedger`）与高危记忆卡 `L0/L1/L2` 变更审批闸门（`MutationApprovalGate`）。
  - `loopback_guard.py`：独占 `127.0.0.1` 回环服务（`ExclusiveLoopbackHTTPServer`）、反 DNS 重绑定/跨站拦截与有界反重放守卫（`ReplayGuard`）。
- `reiji_maigo/plugins/`：7 大内置可选自举插件包（受 `REIJI_MAIGO_DISABLE_PLUGINS` / `REIJI_MAIGO_DISABLED_PLUGINS` 管控）：
  - `__init__.py`：`BUILTIN_PLUGIN_FACTORIES` 与去重故障隔离自举器 `register_builtin_plugins(kernel)`。
  - `roi_ledger.py`：7 位小数反 `MemoryTax` 算力经济学账本 `RoiLedger` 与 `roi` CLI（`status|reconcile|record-saving|set-pricing`）。
  - `epoch_vault.py`：双向 `SHA-256` 校验、只增不删纪元冷库 `EpochColdVault` 与 `epoch` CLI（`scan|seal|verify|restore`）。
  - `thermal_sync.py`：Win32 共享尾读与 250s 缓存余温 / 200K 悬崖同步雷达 `ThermalSyncManager` 与 `radar` CLI（`status|probe|sync`）。
  - `companion.py`：跨进程文件锁、坏 JSON 保护备份、非破坏性可逆装配器 `CompanionInstaller`、`render_mcp_server_config` 与 `companion` CLI（`status|install|uninstall|mcp-config`）。
  - `hybrid_reranker.py`：MemoRRF 三通道混合重排、`Foresight` 时效过期降权、`MemDream` 自探针别名补全与 `rerank` CLI（`recall|foresight|dream`），注册默认 `reranker:memo_rrf` provider。
  - `bootstrap_importer.py`：幂等冷启动迁移器 `BootstrapImporter`（转录 JSONL / 旧版 SQLite / 检查点目录导入）与 `import` CLI。
  - `bench_harness.py`：6 维确定性基准靶场 `ReijiBenchHarness`（含 Markdown 成绩单渲染）与 `bench` CLI（`run|list-cases`）。
- `reiji_maigo/cli.py`：`reiji` 命令行入口（核心命令 `status` / `pool` / `recall` / `turn` / `swarm` / `audit` / `vault` / `serve` / `mcp`，及支持 `configure_parser` 的 7 大内置插件命令 `roi` / `epoch` / `radar` / `companion` / `rerank` / `import` / `bench`）。
- `reiji_maigo/bridges/`：零顶层耦合外部宿主桥接模块：
  - `mcp_stdio.py`：零依赖原生 stdio JSON-RPC 2.0 MCP 服务 `ReijiMcpServer`（支持 `auto`/`ndjson`/`lsp` 双帧、8 工具 + 3 只读资源、`--read-only` 门禁）。
  - `shinobu.py`：伴侣宿主桥接 `ReijiShinobuPlugin`（尾部工具含 `reiji_scout_deep` / `reiji_swarm_pulse` / `reiji_security_audit`，钩子联动 `done` 按角色映射 `origin_trust` 喂速记蜂与 `episode_consolidated` 唤醒馆长蜂）。
  - `star_panel.py`：零顶层 `tkinter` 耦合桌面星盘 UI 桥接 `StarPanelBridgePlugin`（鸭子类型挂载「🕰️ 零时迷子」页签、顶栏徽章、250s 雷达冷却刻度线及 `L1` 待审记忆批准/驳回闭环）。
- `tests/`：微内核、存储与迁移、记忆池、记忆卡、三蜂单测、7 大内置插件、原生 MCP 桥接、桌面桥接、全链路联调与旗舰总门禁测试。

## 3. 核心契约与环境变量
- **锁定域（`LOCKED_DOMAINS`）**：`("proj", "infra", "research", "companion", "life")`，`MemoryCard.domain` 必须在锁定域集合内。
- **数据模型**：`SourceSpan`、`PoolEvent`、`MemoryCard` 均使用 `@dataclass(frozen=True, slots=True)`。
- **环境变量**：
  - `REIJI_MAIGO_HOME`：自定义本地数据目录（默认 `~/.reiji_maigo`）。
  - `REIJI_MAIGO_DISABLE_PLUGINS`：设为 `1`/`true`/`yes`/`on` 时全局禁用外部与内置插件加载。
  - `REIJI_MAIGO_DISABLED_PLUGINS`：逗号或分号分隔的禁用插件名列表（大小写不敏感）。
  - `REIJI_MAIGO_EPOCH_VAULT_DIR`：自定义纪元冷库路径（默认 `<home>/epoch_vault`）。
  - `REIJI_COMPANION_CONFIG_DIR`：自定义伴侣配置根目录（默认 `~/.gemini/config`）。
- **三插槽约定**：插件对象可选实现 `register_cli(kernel)`、`register_hooks(kernel)`、`register_providers(kernel)`（provider 类型仅限 `reranker` 与 `extractor`）。

## 4. 本地验证命令
```powershell
$env:REIJI_MAIGO_DISABLE_PLUGINS="1"; python -m unittest discover -s tests -v
$env:REIJI_MAIGO_DISABLE_PLUGINS="0"; python -m unittest discover -s tests -v
Remove-Item Env:REIJI_MAIGO_DISABLE_PLUGINS -ErrorAction SilentlyContinue
```
