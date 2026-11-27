# 更新日志 (Changelog)

[English](./CHANGELOG.md) | **简体中文**

**`reiji-maigo`（零时迷子）** 的所有重要版本演进均记录于此文件中。

日志格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本 2.0.0 (SemVer)](https://semver.org/lang/zh-CN/)。

## [0.2.0] - 2026-11-27 — 子夜旗舰版（原生 Stdio MCP · MemoRRF 自探针重排 · 7 位小数反 MemoryTax 套件 · 六维对抗靶场）

> **「当零点的钟声敲响，容器内损耗的一切如初复原——零碎裂缓存、零遗忘真相、零第三方臃肿依赖。」**

### 新增功能
- **零依赖双帧自适应 Stdio MCP 服务端（`reiji_maigo/bridges/mcp_stdio.py`，`reiji mcp`）**：
  - 纯 Python 标准库实现 JSON-RPC 2.0 MCP 协议，自动识别并回写 `NDJSON` 行分隔与 `Content-Length` 帧头，诊断日志严格隔离至 `stderr`；
  - 默认暴露 8 项尾部记忆工具（含 `reiji_hybrid_recall`、`reiji_scout_deep`、`reiji_security_audit` 等）与 3 项实时只读资源，支持 `--read-only` 物理隐藏写入工具，即插即用接入 Antigravity、Claude Code、Cursor 等宿主且 100% 保护 L0 前缀缓存。
- **官方首个 `MemoRRF` 混合重排、`Foresight` 时效巡检与 `MemDream` 睡眠自探针插件（`reiji_maigo/plugins/hybrid_reranker.py`，`reiji rerank`）**：
  - 填补微内核默认 `reranker:memo_rrf` 插槽：将自然语言查询语义投影至受控 L2 标签词典，通过倒数秩融合（`RRF`）合流 Level-1 `PPR` 图扩散、卡片 CJK 双字词法分与 `FTS5` 证据轮次命中；
  - 新增 `foresight_until` 时效过期巡检与自动降权提醒（`[EXPIRED_FORESIGHT]`），以及 `MemDream` 子夜口语自探针——对未命中卡片自动抽提口语双字/短语回填至 `memory_tags.aliases_json`。
- **SQLite `user_version` 留底迁移闸门与三源防重冷启动导入器（`reiji_maigo/storage/migrations.py`、`plugins/bootstrap_importer.py`，`reiji import`）**：
  - 维护 `PRAGMA user_version`（`CURRENT_SCHEMA_VERSION = 2`）与 `schema_migrations` 账本，文件型数据库跨版升级前自动执行 SQLite Online Backup 生成 `.bak-v<旧版本>` 安全留底；
  - 支持基于 `SHA-256` 指纹幂等去重的三类存量数据净水导入：会话转录 `JSONL`（`import transcript`）、只读旧版 SQLite 记忆库（`import sqlite`）及 Markdown 阶段交接单目录并铸造带封签种子卡（`import checkpoints`）。
- **对齐 LongMemEval / TEPA 的六维记忆对抗靶场（`reiji_maigo/plugins/bench_harness.py`，`reiji bench`）**：
  - 在隔离沙箱内对 **单跳精确提取、跨窗多跳拼图、时效区间推理、TEPA 三连环冲突覆写（`A -> B -> A'` 零幽灵泄漏）、无证据诚实拒答、玩梗反讽与外部投毒免疫** 六大维度执行端到端检验（支持零成本 `deterministic` 与真实模型 `live` 双模），一键输出 JSON 与 Markdown 跑分成绩单。
- **猎户座四大运维自举插件与桌面星盘 UI 桥接页签（`reiji_maigo/plugins/*`、`bridges/star_panel.py`）**：
  - `roi_ledger`（`reiji roi`）：7 位小数美元精度核算三蜂 Flash 花销（含 L0 缓存命中折扣）对比主窗省下的旗舰 Token 与 `200K` 翻倍档规避收益，支持 `1e-7` 误差对账与负收益熔断告警；
  - `epoch_vault`（`reiji epoch`）：只增不删外部纪元冷库，经 `.staging` 原子落盘与双向 `SHA-256` 核验后在热库替换为轻量墓碑桩并自动重签 `card_seals` HMAC，支持一键无损回灌；
  - `thermal_sync`（`reiji radar`）：Win32 `FILE_SHARE_READ | WRITE | DELETE` 无锁共享读取末尾 `16KB`，拦截软链接/重解析点，内置 `250s` 缓存余温与 `180K/200K` 悬崖同步状态机；
  - `companion`（`reiji companion`）：带跨进程文件锁与 `.corrupt-*.bak` 坏文件保护备份的非破坏性可逆钩子/技能装配器及 `mcp-config` 生成器；
  - `star_panel`：零顶层 `tkinter` 耦合桌面页签桥，鸭子类型挂载黑金绯红「🕰️ 零时迷子」面板、顶栏徽章、250s 雷达冷却刻度线及 `L1` 待审记忆一键批准/驳回按钮。

### 安全自审与测试加固
- 裸核模式（`REIJI_MAIGO_DISABLE_PLUGINS=1`）与 7 插件全开模式（`=0`）双态单元测试扩容至 **`224/224` 全绿**，静态洁净室扫描覆盖全部 `.py`、`README` 与 `AGENTS.md`，并接入 GitHub Actions 三大操作系统跨版本矩阵 CI。

## [0.1.0] - 2026-11-27 — 奠基版（零依赖双层微内核 · 无头三蜂 · 四层零信任安全装甲）

### 新增功能
- **零依赖微内核与不可变契约（`reiji_maigo/core/`）**：实现故障隔离 `ReijiKernel` 三插槽（`register_cli` / `register_hooks` / `register_providers`）、优先级 `HookBus`、五大锁定记忆域（`proj` / `infra` / `research` / `companion` / `life`）及不可变 `SourceSpan`、`PoolEvent`、`MemoryCard` 契约。
- **双层记忆底座（`storage/`、`pool/`、`cards/`、`engine.py`）**：单文件 SQLite WAL + CJK 双字 `FTS5` 全文账本、`ActivePool` 四维蔡加尼克衰减与长链无损压实、`CardStore` 受控二级标签、12 轮纯标准库 `PPR` 多跳扩散、`TEPA` 双时态覆写/撤销与三级渐进召回。
- **无头三蜂记忆蜂群（`reiji_maigo/swarm/`）**：`SHA-256` 锁定 L0 前缀运行器、物理隔离角色能力代理、非阻塞去抖速记蜂 `ScribeAgent`、只读三层抽屉侦察蜂 `ScoutAgent`（含无证据诚实拒答）、子夜馆长蜂 `ArchivistAgent` 与工业级蜂群脉搏遥测面板。
- **四层零信任安全装甲与伴侣桥（`reiji_maigo/security/`、`bridges/shinobu.py`）**：入库凭据擦除与媒体路径盲罩、来源信任反投毒闸门、Win32 `DPAPI` / POSIX `0600` 加密保险箱、域分离 `HMAC-SHA256` 卡片封签、防炸弹 `ZIP_STORED` 便携包、哈希链审计账本、`L0/L1/L2` 变更审批闸门、`127.0.0.1` 独占回环服务与 `ReijiShinobuPlugin` 桥接（`129/129` 单测全绿）。
