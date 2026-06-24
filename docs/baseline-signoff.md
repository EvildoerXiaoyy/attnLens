# 基线签字记录

**项目**: Prompt 注意力诊断器 (Linter) — v0.1 Demo
**签字人**: 用户（确认人）
**日期**: 2026-07-09
**阶段**: 阶段一（整体规划层）— 正式关闭

## 已确认的基线文档

| 文档 | 版本 |
|------|------|
| `ARCHITECTURE.md` | v1 — 四模块组件图、数据流、外部依赖、容错策略、关键约束 |
| `API_CONTRACT.yaml` | v1 — 形式化接口定义（analyze / analyze_entropy / scan_norm / run_ablation_on_70B） |
| `MODULE_DEPENDENCY.md` | v1 — model-loader → entropy-analyzer + norm-scanner（并行）→ ui-layer |
| `PRD.md` | v1 — 问题陈述、10 条 user stories、实现决策、测试决策、out of scope |
| `docs/adr/ADR-001-agent-model-selection.md` | Proposed — 选择 Qwen2.5-0.5B |
| `docs/adr/ADR-002-core-detection-algorithm.md` | Proposed — 熵变率 + 范数扫描 |
| `TODOS.md` | v1 — 耗时预估、报告导出（标记为 deferred） |

## 已确认的审查决议

### /grill-me（8 项设计决策）
1. 同源假设 → v0.1 不做验证，技术边界声明强度够
2. 阈值来源 → 启发式默认值，代码可配置，READ ME 标注"实验性"
3. 单次 forward → 合并为一次（同时开 attention + hidden states）
4. 超长输入 → 截断模式
5. 模型下载失败 → 进度条 + 故障指引 + 预留镜像可配置
6. 空/过短输入 → UI 提示"建议至少 20 字符"
7. 重复推理 → 缓存最近分析结果
8. Demo 冷启动 → --preload 预加载模式

### /plan-eng-review（EM 审查）
- 范围接受（12 文件 / 4 类，无缩减）
- Architecture: 同步调用 + 单例模式
- Code Quality: run_ablation_on_70B 留在 PromptLinter 中
- Test: mock model.forward + 真实 tokenizer
- Performance: 同步 + Spinner
- 0 critical gaps

## 基线锁定声明

> 上述文档和决策构成 v0.1 Demo 的实现基线。阶段二的模块实现必须遵循此基线。
> 如在实现过程中发现需要调整基线，须通过 `/amend-contract` 发起受控修订，
> 并标记受影响模块为"需重新验证"。
> 任何静默偏离将在阶段四的 `/reconcile` 漂移检测中被捕获。
