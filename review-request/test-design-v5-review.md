# Review Request: test-design-v5 — 终审

## Context

对 Prompt 注意力诊断器 v0.1 Demo 的测试设计进行**第五次（终审）独立审查**。前四轮 GLM 5.2 共计发现 34 项问题，全部已解决。上一轮（v4）裁定 **approve**，仅留 3 项 Low 建议——本轮已全部应用。

**本轮的变更仅仅是上一轮 3 项 Low 建议的具体落地。** 如果通过，测试设计正式定稿，可以进入 TDD 实现阶段。

---

## Diff

仅 3 处微小变更：

| 文件 | 变更 |
|------|------|
| `tests/test_entropy_analyzer.py:194` | `assert len(results) > 0` → `assert len(results) == 100` |
| `tests/mocks/model_loader_mock.py:53` | `# 兼容旧测试` → `# 全局默认` |
| `tests/test_entropy_analyzer.py:430` + `tests/test_norm_scanner.py:369` | TODO 区前添加注释说明"有意推迟" |

---

## 四轮审查完整追溯

### 一轮（14 项）— 全部通过
H1/H2/M1-M7/L1-L5

### 二轮（11 项 F1-F11）— 全部通过
F1 percentile 算子 → ADR-002 记录严格 `<`
F2/F3 hidden pattern → `set_hidden_pattern("zero"|"huge")`
F4/F5 真 mismatch → 覆盖 config 字段
F6/F7 medium 管线 → 完整 `calc_entropy_delta` → `risk_level` 断言
F8 三元组自洽 → whitespace/negative/zero 修复
F9/F10 注释 → 修正
F11 触发隔离 → 层数给足

### 三轮（6 项）— 全部通过
M-1 截断移出 Mock → 截断是编排层职责
L-1 mismatch 单向 → 自适应实际维度
L-2 norm handler → 与 entropy 对齐
L-3/L-4/L-5 → 文档/注释/可接受留痕

### 四轮（3 项 Low）— ✅ 本轮已落地
L-1 `test_input_at_max_length` → `len(results) == 100`
L-2 mock docstring → "全局默认"
L-3 TODO 注释 → 添加"有意推迟"说明

---

## Trade-offs to Respect

> **Decision:** Qwen2.5-0.5B > Llama-3.2-1B
> **Decision:** 熵变率 + 范数扫描 > 替代方案
> **Decision:** 阈值 delta > 2.0/1.5 为启发式默认值（可配置）
> **Decision:** 超长输入截断模式 > 滑窗分片
> **Decision:** 单次 forward 同时输出 attention + hidden_states
> **Decision:** Mock model.forward + MockTokenizer（crc32 确定性）
> **Decision:** 弱信号严格小于 `norm_score < percentile_threshold`（ADR-002）
> **Decision:** 截断是编排层职责，非 Mock/分析器职责

---

## What Reasonix Already Checked

四轮审查累计覆盖：三元组自洽、配对矛盾、try/except 失败路径、注释一致性、架构一致性、Mock 完备性。

全部 34 项一审结论及其行号参考如下：

**Round 1 High:**
- `mock:86` 注意力逐层不变 → 已修: `set_layer_attention`
- `mock:101` hidden 种子化 → 已修: `torch.Generator`

**Round 1 Medium:**
- M1 `norm:25` 默认值 → 已修: `TestDefaultConfig`
- M2 `entropy:84` 顺序假绿 → 已修: `set_fixed_ids` + `token_id` 比对
- M3 `entropy:128` 截断 → 已修: M-1 裁决定稿
- M4 `mock:137` hash → 已修: `zlib.crc32`
- M5 `mock:76` output_* → 已修: forward 参数校验
- M6 `entropy:138` delta→risk → 已修: 表驱动 + 管线断言
- M7 `entropy:84` 规约 → 已修: docstring + ARCHITECTURE.md

**Round 2 P0:**
- F1 `norm:197/205` 互斥算子 → 已修: 严格 `<` + ADR-002

**Round 2 P1:**
- F2 `norm:297` 全零假 → 已修: `set_hidden_pattern("zero")`
- F3 `norm:344` 不 extreme → 已修: `set_hidden_pattern("huge")`
- F4 `entropy:369/norm:319` 无 mismatch → 已修: config 覆盖
- F5 `entropy:355` 合法形状 → 已修: `num_attention_heads` 冲突

**Round 3 M-1:**
- `entropy:185/mock:186` 截断在 Mock → 已修: 移出 Mock

---

## What We Want From You

这是**终审**。只需确认：

1. 上一轮 3 项 Low 建议是否已正确落地？
2. 有无新的回归？
3. 最终裁定：**approve** → 测试定稿，可进入 TDD

---

## Response Format

```json
{
  "review": {
    "overall": "approve|conditional|reject",
    "conflict_with_reasonix": false,
    "summary": { "high": 0, "medium": 0, "low": 0 },
    "findings": []
  }
}
```

如果 approve，findings 应为空数组。
