# Reconcile Report — 2026-07-09

## Summary
- Feature Creep: 2 items
- Missed Scope: 3 items
- Contract Violations: 4 items

---

## Feature Creep（代码有，但规范文档未提及）

### FC-1. `norm_std` 双阈值弱信号判定
- **代码**: `src/prompt_linter/norm_scanner.py:163-170` — `_mark_weak_chunks` 使用双重判定：`mean_norm < 15%` **或** `norm_std < 15%`
- **规范**: `ARCHITECTURE.md:101-103` 只写了"标记低于分位数"，`ADR-002` 只写了严格 `<` 算子
- **原因**: 开发过程中发现单纯均值判定对"过于均匀"的退化文本无效；方差坍塌才是更强信号
- **建议**: 更新 `ARCHITECTURE.md:101-103` 和 `ADR-002`，记录双重判定规则

### FC-2. `z_score` 字段 + z-score 异常检测
- **代码**: `src/prompt_linter/entropy_analyzer.py:88-162` — 逐位置熵差 + z-score 异常检测，每个结果含 `z_score` 字段
- **规范**: `ARCHITECTURE.md:80-84` 写的是"取最后一个 Token" + 固定阈值 2.0/1.5
- **原因**: 原始算法只看了最后一个 Token 的注意力分布，对中间位置的"逻辑死结"完全无感
- **建议**: 更新 `ARCHITECTURE.md:80-84` 和 Flow 1 数据流描述

---

## Missed Scope（规范承诺了，但代码未实现）

### MS-1. 截断/分片处理
- **规范**: `ARCHITECTURE.md:128` "输入过长：截断或分片处理"
- **代码**: `src/prompt_linter/prompt_linter.py:71-75` 只检查了 `max_length` 并抛出 `InputTooLongError`，无截断逻辑
- **影响**: 超过 32K tokens 的输入直接报错退出，而非截断后继续分析
- **建议**: 在 `PromptLinter.analyze()` 中加入截断逻辑（保留前 N tokens），或更新规范降低承诺

### MS-2. `DB_SCHEMA.sql` 和 `API_CONTRACT.yaml` 未完成的 DB/类型定义
- **规范**: `/arch-workflow` 产出物清单包含 `DB_SCHEMA.sql`
- **代码**: 不存在此文件
- **原因**: 该项目无数据库依赖，所有分析在内存中完成
- **建议**: 从产出物清单中删除 `DB_SCHEMA.sql`，或添加空文件注明"本版本无持久化需求"

### MS-3. `TEST_INTENT.md` 未生成
- **规范**: 阶段二模板要求 `TEST_INTENT.md` 记录每个测试的防御意图
- **代码**: `tests/` 目录没有此文件
- **原因**: 开发过程中跳过了这一步
- **建议**: 生成或明确记录为不需要

---

## Contract Violations（代码实现与规范文档不一致）

### CV-1. 熵变率算法：逐位置 + z-score vs 仅最后一个 Token + 固定阈值
- **规范**: `ARCHITECTURE.md:80-82` "取最后一个 Token [:, -1, :] → 单个标量 → delta > 2.0 / 1.5"
- **代码**: `entropy_analyzer.py:112-140` 逐位置计算 `[:, i, :]` → 多个 delta → z-score 异常检测
- **偏差程度**: 大。算法完全不同
- **建议**: 更新 `ARCHITECTURE.md` 中的 Flow 1 描述

### CV-2. 范数扫描：均值+方差双阈值 vs 仅均值
- **规范**: `ARCHITECTURE.md:101-103` "按 chunk_size=128 聚合求平均 → 标记低于 15% 分位数"
- **代码**: `norm_scanner.py:158-170` 同时比较 `mean_norm` 和 `norm_std`，任一低于分位数即标记
- **偏差程度**: 中。增加了 std 维度，未改变 API 结构
- **建议**: 更新 `ARCHITECTURE.md:101-103` 和 `ADR-002`

### CV-3. chunk_size 默认值不同
- **规范**: `ARCHITECTURE.md:101` "chunk_size=128"
- **代码**: `norm_scanner.py:21` `DEFAULT_CHUNK_SIZE = 32`
- **偏差程度**: 小。仅为默认值变更，用户可在 UI 调整
- **建议**: 更新 `ARCHITECTURE.md:101`

### CV-4. 运行延迟超出承诺
- **规范**: `ARCHITECTURE.md:135` "单次分析 < 10 秒"
- **实际**: 首次需下载 1GB 模型（分钟级）；后续分析 1K tokens 约 2-5 秒，10K tokens 约 10-30 秒
- **影响**: 短文本满足承诺，长文本和首次加载超出
- **建议**: 更新约束说明，区分"首次加载"和"后续分析"，长文本时间标注为 10-30 秒

---

## 实际效果与 PRD 愿景的差距（非规范覆盖，但值得记录）

### GAP-1. 信噪比不足
- PRD 愿景：像 ESLint 一样一眼看出问题
- 实际：坏 Prompt 与正常文本的指标差异不够显著（delta 0.96 vs 0.15），需要调参才能稳定区分
- 结论：作为"量化测量仪"可用，作为"自动诊断工具"不可靠

### GAP-2. 小模型 ≠ 大模型
- PRD 核心假设：0.5B 的浅层 attention 模式与 70B 同构
- 实际：未经实验验证，Demo 阶段无法证明相关性
- 结论：白皮书中的"同源假设"仍是理论推测

### GAP-3. 文本分类精度不足
- "请用Python写一个快速排序" 被标记为高风险（Chinese→English code transition），但文本本身并无问题
- 说明算法对语言混写、代码等自然场景的假阳性过高

---

## 建议操作

| # | 类型 | 操作 | 工作量 |
|---|------|------|--------|
| CV-1 | 更新 `ARCHITECTURE.md` | 重写 Flow 1 数据流描述，匹配逐位置 + z-score 实现 | 小 |
| CV-2/3 | 更新 `ARCHITECTURE.md` 和 `ADR-002` | 记录 norm_std 双阈值 + chunk_size=32 默认值 | 小 |
| FC-1/2 | 补充 ADR（可选） | 记录 z-score 和 std 双阈值的引入原因 | 小 |
| MS-1 | 截断实现或规范降级 | 在 PromptLinter 中加入前 N tokens 截断，或从 ARCHITECTURE.md 移除承诺 | 中 |
| GAP-1~3 | 记录在 `docs/lessons.md` | 作为项目复盘记录，供后续参考 | 小 |

> **未自动修改任何文件。** 以上漂移项需人工确认操作路径。
