# Handoff — Prompt 注意力诊断器 (attnLens)

## 项目状态

**结论：方向性终止。** 实验验证了"基于浅层物理量（熵 + 范数）的静态 Linter"在工业级输入下不可行。详见 `docs/experiment-report.md`。

### 核心教训（必须阅读）

1. **信噪比不足**：0.5B 模型无法区分"逻辑谬误"与"复杂指令"。正常代码题（写快排）的熵 delta 达 0.78，仅略低于坏 Prompt 的 0.96。
2. **平均池化无效**：128 chunk_size 下范数差异完全淹没，需缩到 32 才能观测——但这就失去了泛化能力。
3. **学术有价值，产品无出路**：作为模型可解释性实验有参考价值，作为开箱即用的 Linter 不能满足需求。

### 本次尝试的经验总结

1. **浅层物理量的信噪比天花板**：0.5B 模型的 attention entropy 和 hidden norm 对于区分"逻辑混乱"和"复杂但正常"的文本，信噪比约在 2:1 到 3:1 之间（坏文本 delta=0.96 vs 正常复杂文本 0.78）。这个比例不足以做自动化诊断，只能做辅助探测。
2. **范数的"尺度敏感"问题**：L2 Norm 对 chunk_size 极度敏感。128 下无法观测，32 下才可见。这意味着该指标无法直接应用到真实 RAG 系统的常规分块策略（256-512 tokens）。
3. **特征工程陷阱**：开发过程中逐渐加入了 z-score 异常检测、norm_std 双阈值、逐位置熵差等"补丁"——每项单独看都有道理，合在一起说明原始假设（一个简单物理量即可工作）不成立。
4. **测试驱动开发的价值**：86 个测试、5 轮外部审查保证了代码质量，也让方向性问题更早暴露。如果没有严格的测试，可能会在错误的方向上走得更远。

### 项目资产（可复用）

| 资产 | 路径 | 说明 |
|------|------|------|
| 架构文档 | `ARCHITECTURE.md` | 已更新匹配最终实现 |
| 契约 | `API_CONTRACT.yaml` | ModelLoader 和 PromptLinter 接口定义 |
| ADRs | `docs/adr/ADR-001.md`, `ADR-002.md` | 已定稿，标记 Final |
| 测试套件 | `tests/` | 86 个测试全部通过 |
| Mock 基础设施 | `tests/mocks/model_loader_mock.py` | 逐层注意力、确定性 Generator、zlib hash |
| 实验报告 | `docs/experiment-report.md` | 漂移分析与终止结论 |
| 漂移报告 | `REconcile-20260709.md` | 规范 vs 实现差距分析 |
| 审查记录 | `review-request/` | 5 轮 GLM 5.2 外部审查 |

### 未解决问题

1. **截断逻辑未实现**（MS-1）：ARCHITECTURE.md 承诺"输入过长时截断"，但 `PromptLinter.analyze()` 仅抛出 `InputTooLongError`。如需复活项目，应在编排层加入前 N tokens 截断。
2. **`TEST_INTENT.md` 未生成**（MS-2）：阶段二模板要求但跳过了。
3. **`DB_SCHEMA.sql` 未生成**（MS-3）：项目无持久化需求，产出物清单应删去此项。

### Context Budget — 关键文件

| 优先级 | 文件 | 为什么重要 |
|--------|------|-----------|
| 🔴 P0 | `docs/experiment-report.md` | 实验结论，任何人接触项目前必须先读 |
| 🔴 P0 | `src/prompt_linter/entropy_analyzer.py` | 核心算法（逐位置熵差 + z-score），也是问题所在 |
| 🟡 P1 | `src/prompt_linter/norm_scanner.py` | 核心算法（mean+std 双阈值），同样的问题 |
| 🟡 P1 | `tests/test_entropy_analyzer.py` | 34 个测试，含精确 delta→risk 映射 |
| 🟢 P2 | `ARCHITECTURE.md` | 架构蓝图（已更新） |
| 🟢 P2 | `docs/adr/ADR-002-core-detection-algorithm.md` | 算法决策与偏差记录 |

### 建议的切入方式

若后续想基于本次经验做新的尝试，建议：

1. **换个切入方向**：从"内部物理量"换到"logits 不确定性"或"embedding 余弦相似度"——这些指标在现有文献中已有更成熟的证据。
2. **换个大模型**：如果非要走 attention 分析路线，至少用 7B 模型。0.5B 的浅层信号太弱。
3. **利用测试套件**：86 个测试用例可以作为任何新方法的基线 benchmark。
4. **运行方式**：`.venv/bin/streamlit run ui/app.py`（需先装依赖 `.venv/bin/pip install -r requirements.txt`）

### Suggested Skills

Next agent should invoke these skills (in order):
1. `explore` — explore `src/prompt_linter/entropy_analyzer.py` and `norm_scanner.py` for current implementation
2. `research` — research alternative approaches (logits uncertainty, embedding cosine similarity) before writing new code
