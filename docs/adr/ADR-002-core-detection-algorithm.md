# ADR-002: 使用注意力熵变率 + 隐藏态范数扫描作为核心检测算法

## Status

Proposed

## Context

本工具的核心问题是：如何量化 Prompt 文本的"结构健康度"？候选方案：

1. **注意力熵变率 + 隐藏态范数扫描**（最终选择）— 读取模型内部物理量
2. **Perplexity 评分** — 计算模型对文本的困惑度，越低表示越"自然"
3. **Saliency Map（梯度归因）** — 计算每个 Token 对输出的梯度贡献
4. **纯规则启发式** — 基于长度、重复率、标点密度等表面特征

选择方案 A+C 的理由：

- **Attention Entropy Delta**：浅层到深层的熵值突变标志着"句法依赖断裂"，有强物理意义。当深层比浅层熵突然飙升，说明模型在该处的注意力分散，需要在多个可能的句法结构中"硬猜"。
- **Hidden State Norm**：范数极低的 Token 在模型内部被"压缩坍塌"，无论 Attention 是否关注它，它都已失去语义激活强度。这对 RAG 长文本特别重要——拼接的段落可能在某些层被静默忽略。

Perplexity 作为整体指标无法定位到 Token 级问题；Saliency Map 需要梯度计算，计算量大且梯度噪声高；纯规则完全无法捕捉 Transformer 内部表征。

## Decision

同时实现两种互补的检测算法：

- **熵变率**（方案 A）：检测句法层面的"逻辑死结"，定位到具体 Token
- **范数扫描**（方案 C）：检测语义层面的"表征洼地"，定位到文本块

两个算法共用同一个模型推理结果（一次 forward pass 同时输出 attention 和 hidden states），计算开销几乎不变。

## Consequences

- **更好**：从两个正交维度（句法 + 语义）覆盖文本问题，检测更全面
- **更好**：共享一次模型推理，性能开销小
- **更好**：输出形式互补——Token 级热力图 + Chunk 级块扫描，UI 展示丰富
- **更差**：算法物理意义需要向非技术用户解释，增加教育成本
- **更差**：两个算法的阈值（delta > 2.0、15% 分位数、chunk_size=32）需要实验调优，v0.1 使用启发式默认值

> **范数扫描算子约定（F1 修复）：** 弱信号判定采用**严格小于**（`norm_score < percentile_threshold`）。
> - `percentile=0` 时没有任何块满足 `norm < 最小值`，故 `weak_count == 0`
> - `percentile=100` 时阈值等于最大值，最大值块不满足严格 `<`，故 `weak_count == len - 1`
> 此约定已在 `test_zero_percentile` 和 `test_hundred_percentile` 中固化。
