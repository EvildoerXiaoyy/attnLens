# PRD: Prompt 注意力诊断器 (Linter) — v0.1 Demo

## Problem Statement

用户将 Prompt 发送给昂贵大模型（如 GPT-4、Qwen-72B）后，结果往往不如预期，但**无法定位问题根源**：

- 是 Prompt 写得逻辑混乱，还是模型本身的随机波动？
- RAG 拼接了 10K 长文本后模型遗漏关键信息——是检索没召回，还是模型内部"忽略"了这部分内容？
- 让 AI 改写 Prompt 后，没有量化指标验证改写是否真的有效。

用户需要一款工具，能在**发送给大模型之前**，对 Prompt 文本进行量化诊断，用物理指标（而非经验猜测）定位文本结构脆弱性。

## Solution

构建一款基于**同源小模型代理**的静态代码检查器（类比 ESLint），通过对 Prompt 文本进行物理级信号扫描，输出 Token 级热力图和 Chunk 级信号强度报告：

- **注意力熵变率扫描**：检测句法层面的"逻辑死结"，定位到具体 Token
- **隐藏态范数扫描**：检测语义层面的"表征洼地"，定位到文本块

工具**不修改用户文本**，只输出量化指标；**不依赖 GPU**，可在 Mac M 系列或纯 CPU 上运行；运行时**完全离线**。

## User Stories

1. 作为 Prompt 工程师，我想输入一段 Prompt 并立即看到 Token 级风险热力图，以便快速定位可能导致模型"逻辑混乱"的词汇。
2. 作为 Prompt 工程师，我想看到分析结果中高风险 Token 在原文中被高亮标记，以便直观地感知问题位置。
3. 作为 RAG 系统开发者，我想对拼接后的长文本执行块级信号强度扫描，以便发现被模型内部"压缩坍塌"的语义洼地。
4. 作为 RAG 系统开发者，我想看到弱信号块的具体文本预览和信号分数，以便决定是删除、截断还是提前到文本开头。
5. 作为 LLM 应用开发者，我想在两个标签页之间切换（短 Prompt 分析 / 长文本扫描），以便覆盖不同的诊断场景。
6. 作为技术决策者，我想在 UI 中看到模型信息和分析耗时，以便评估工具的性能成本。
7. 作为新用户，我想在首次使用时看到清晰的工具边界声明，以便正确理解诊断结果的局限性（0.5B ≠ 70B、仅限静态分析等）。
8. 作为评估者，我想用一组标准测试用例验证工具的检测能力，以便确认其可用性。
9. 作为开发者，我想通过统一的 `analyze()` 接口调用所有分析能力，以便在 CI/CD 管道中集成该工具。
10. 作为付费用户（未来），我想在 70B 级别大模型上运行段落消融测试以获得精准因果推断，以便验证小模型诊断结果与真实表现的关联性。

## Implementation Decisions

### 技术栈
- **代理模型**：`Qwen/Qwen2.5-0.5B` — 与主流生产级大模型同源（Qwen 系列），Tokenizer 100% 对齐，约 1GB 内存即可运行。
- **推理框架**：HuggingFace `transformers` + PyTorch，配置 `output_attentions=True` 和 `output_hidden_states=True`。
- **UI 框架**：Streamlit + Plotly（柱状图/热力图渲染）。
- **运行环境**：Mac M 系列芯片或纯 CPU，无需 GPU，运行时可离线。

### 模块架构（四模块，契约驱动）

```
model-loader → entropy-analyzer + norm-scanner（并行）→ ui-layer
```

各模块职责：
- **model-loader**：加载 Qwen2.5-0.5B，管理 Tokenizer 生命周期，提供 `model` / `tokenizer` 实例。
- **entropy-analyzer**：提取模型最后一层和倒数第三层的注意力矩阵，计算最后一个 Token 处的熵变率（delta），delta > 2.0 标红，delta > 1.5 标黄。
- **norm-scanner**：提取模型倒数第三层的 Hidden State L2 范数，按 128 Token 为块聚合，低于全局 15% 分位数标记为弱信号。
- **ui-layer**：Streamlit 双标签页界面，提供输入框、分析按钮、Plotly 图表、Token 高亮展示。

### 接口契约
详见 `API_CONTRACT.yaml`，核心接口 `PromptLinter.analyze(text, chunk_size?)` 返回统一结构：
- `token_risks[]`：每个 Token 的熵差值 + 风险等级
- `chunk_risks[]`：每个文本块的平均范数 + 弱信号标记
- `metadata`：模型名称、总 Tokens 数、耗时

### 预留接口
`run_ablation_on_70B()` 方法桩已定义在契约中，标注为 `NotImplementedError`，占位未来付费版的段落消融测试能力。

### 架构决策
见 `docs/adr/ADR-001-agent-model-selection.md` 和 `docs/adr/ADR-002-core-detection-algorithm.md`。

## Testing Decisions

### 测试哲学
- 只测外部行为，不测实现细节。
- 以 **模块级 API**（`PromptLinter.analyze()`）为主要测试接缝。
- 所有测试使用 Mock 模型输出（无需加载真实模型），确保测试快速且确定。

### 测试覆盖范围

| 测试目标 | 方法 | 关键断言 |
|---------|------|---------|
| 输入校验 | `analyze("")`, `analyze("   ")` | 抛出 `InputEmptyError` |
| 输入过长 | `analyze(>32K tokens)` | 抛出 `InputTooLongError` |
| 返回结构 | `analyze("test")` with mock | 返回含 `token_risks`, `chunk_risks`, `metadata` 的 dict |
| 风险判定 | entropy delta > 2.0 | `risk_level == "high"` |
| 弱信号检测 | norm < 15% percentile | `is_weak == True` |
| 短文本保护 | 输入 < 2 tokens | 安全回退，全 low |
| 预留接口 | `run_ablation_on_70B()` | 抛出 `NotImplementedError` |
| 模型生命周期 | `is_ready`, `unload_model()` | 状态转换正确 |

### 测试工具
- pytest + `unittest.mock`（或等效 mock 方式）
- Mock 模型输出：构造固定的 attention 矩阵和 hidden states 张量，使熵差和范数在已知范围内

## Out of Scope

- **不实现** Prompt 改写或生成功能（本工具是"测量仪"非"生成器"）。
- **不实现** 真实 70B 模型的消融测试（`run_ablation_on_70B()` 仅为占位桩）。
- **不实现** 向量数据库检索质量检测（仅检测拼接后上下文的表征洼地）。
- **不实现** Prompt 事实正确性判断（仅做结构静态分析）。
- **不实现** 多用户并发、认证授权、持久化存储等生产级功能。
- **不验证** 0.5B 诊断结果与 70B 真实表现的相关性（v0.1 阶段不纳入 scope）。

## Further Notes

### 技术边界（须在 UI 和文档中显著标注）

1. **0.5B ≠ 70B**：本工具测量的"高熵"不代表 70B 一定会犯错，仅代表该处文本结构不符合 Transformer 浅层认知规律。
2. **RAG 归因限制**：范数扫描检测的是"拼接后上下文中的表征洼地"，无法检测向量数据库中的检索漏召。真正的 RAG 遗漏需使用段落消融测试在主模型上离线验证。
3. **仅限静态分析**：本工具不做推理生成，不涉及模型知识库，无法判断 Prompt 内容的事实正确性。

### 开发顺序

按模块依赖顺序分三个阶段实施（并行窗口已在依赖图中标注）：
1. **model-loader**：模型懒加载、生命周期管理
2. **entropy-analyzer + norm-scanner**（并行）：两种核心算法实现
3. **ui-layer**：Streamlit 双标签页界面 + Plotly 可视化
