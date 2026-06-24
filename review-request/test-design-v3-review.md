# Review Request: test-design-v3 — 语义一致性修复验证

## Context

对 Prompt 注意力诊断器 v0.1 Demo 的测试设计进行**第三次独立审查**。前两轮 GLM 5.2 分别发现了 14 个问题（2H+7M+5L）和 11 个语义一致性问题（F1-F11），**本轮所有问题已修复**。本轮审查目标是确认修复彻底性、无回归、无新问题。

---

## Diff（变更概览，共 1251 行，5 个文件）

> Diff 超过 500 行，已压缩为变更摘要 + 关键代码片段。全部文件为核心业务/测试逻辑，无自动生成代码。

### 文件清单

| 文件 | 行数 | 本轮变更类型 |
|------|------|-------------|
| `tests/mocks/model_loader_mock.py` | 297 | 新增 `set_hidden_pattern()`、`num_attention_heads` 到 config |
| `tests/test_entropy_analyzer.py` | 484 | 修复 F4/F5/F8/F9/F11，删除无意义测试 |
| `tests/test_norm_scanner.py` | 415 | 修复 F1/F2/F3/F4/F10 |
| `tests/conftest.py` | 12 | 无变更 |
| `docs/adr/ADR-002-core-detection-algorithm.md` | 43 | 新增严格 `<` 算子约定 |

### 核心变更摘要

#### 1. `tests/mocks/model_loader_mock.py` — Mock 新增 hidden pattern + config 字段

```python
# 新增（F2/F3 修复）：hidden states 生成模式
def set_hidden_pattern(self, pattern: str):
    """"normal": randn*0.1 | "zero": 全零 | "huge": 1e6"""
    self._hidden_pattern = pattern

def _make_hidden_state(self, layer_idx, batch, seq):
    if pattern == "zero":
        return torch.zeros(batch, seq, hidden_dim)
    if pattern == "huge":
        return torch.ones(batch, seq, hidden_dim) * 1e6
    # "normal": 种子化 randn*0.1

# 新增（F5 修复）：config 增加 num_attention_heads 字段
class MockConfig:
    def __init__(self, ..., num_attention_heads: int = 4):
        self.num_attention_heads = num_attention_heads
```

#### 2. `tests/test_entropy_analyzer.py` — 8 项修复

**F4** — 删除无意义的 `test_hidden_size_mismatch`（熵分析器不消费 hidden states）
**F5** — `test_wrong_attention_shape` 改为真 mismatch：
```python
model.config.num_attention_heads = 2  # config 说 2，实际张量 4
```
**F8** — `test_negative_threshold` 改为构造真负 delta：
```python
model = _setup_entropy_delta_model(deep_pattern="focused", shallow_pattern="uniform")
```
**F8** — `test_zero_threshold` 收紧断言：
```python
assert all(r["risk_level"] == "high" for r in results if r["entropy_delta"] > 0)
```
**F9** — 注释 `"< 4"` 改为 `"层数 < 3"`
**F11** — `test_single_char_input` 层数给足以隔离触发条件

#### 3. `tests/test_norm_scanner.py` — 5 项修复

**F1** — `test_hundred_percentile` 改为严格 `<` 语义：
```python
assert weak_count == len(results) - 1  # 最大值块不算弱
```
**F2** — `test_all_zero_hidden_states` 注入全零：
```python
model.set_hidden_pattern("zero")
assert c["norm_score"] == 0.0
```
**F3** — `test_extreme_norm_values` 注入 1e6：
```python
model.set_hidden_pattern("huge")
assert c["norm_score"] > 1e5
```
**F4** — `test_hidden_state_dim_mismatch` 构造真 mismatch：
```python
model.config.hidden_size = 16  # 实际张量 hidden_dim=8
```
**F10** — 注释修正：tuple 负索引越界 → IndexError

#### 4. `docs/adr/ADR-002-core-detection-algorithm.md` — 算子约定

新增严格小于约定（F1）：
> 范数扫描算子约定：弱信号判定采用**严格小于**（`norm_score < percentile_threshold`）。
> - `percentile=0` → `weak_count == 0`
> - `percentile=100` → `weak_count == len - 1`

---

## Architecture Context

测试覆盖两个核心算法模块 + Mock 基础设施 + ADR 约定：

```
model-loader (Mock) ← set_hidden_pattern("zero"|"huge"|"normal")
  ├─→ entropy-analyzer — calc_entropy_delta(model, tokenizer, text)
  └─→ norm-scanner     — scan_signal_strength(model, tokenizer, text, chunk_size)
```

详见 `ARCHITECTURE.md`、`API_CONTRACT.yaml`、`ADR-002`。

---

## Trade-offs to Respect

> **Decision:** Qwen2.5-0.5B > Llama-3.2-1B — 同 Tokenizer 对齐 + 低资源
> **Decision:** 熵变率 + 范数扫描 > 替代方案 — 句法 + 语义双维度
> **Decision:** 阈值 delta > 2.0/1.5 为启发式默认值 — 代码可配置
> **Decision:** 超长输入截断模式 > 滑窗分片 — 避免计算膨胀
> **Decision:** 单次 forward 同时输出 attention + hidden states
> **Decision:** Mock model.forward + MockTokenizer (crc32 确定性)
> **Decision:** 弱信号严格小于 `norm_score < percentile_threshold`（ADR-002 已记录）

---

## What Reasonix Already Checked

### 一轮审查（14 项）
| # | 条目 | 状态 |
|---|------|------|
| H1 | 注意力矩阵逐层不变 | ✅ 已修 |
| H2 | hidden states 未种子 randn | ✅ 已修 |
| M1 | 默认 chunk_size 未锁定 | ✅ 已修 |
| M2 | 顺序假绿 | ✅ 已修 |
| M3 | 截断零测试 | ✅ 已修 |
| M4 | tokenizer hash 不确定 | ✅ 已修 |
| M5 | output_* 开关被丢弃 | ✅ 已修 |
| M6 | 阈值→risk 未钉住 | ✅ 已修 |
| M7 | 规约歧义 | ✅ 已修 |
| L1-L5 | 5 项 Low | ✅ 4 修 + 1 注释声明 |

### 二轮审查（11 项 F1-F11）
| ID | 条目 | 状态 |
|----|------|------|
| F1 | percentile 互斥算子 | ✅ ADR-002 记录 + 测试收紧 |
| F2 | zero test 实际非零 | ✅ set_hidden_pattern("zero") |
| F3 | extreme 不 extreme | ✅ set_hidden_pattern("huge") |
| F4 | mismatch 没 mismatch | ✅ 构造函数内覆盖 config |
| F5 | wrong shape 用合法形状 | ✅ num_attention_heads 冲突 |
| F6 | 边界测试承诺 medium | ✅ 先前修复留存 |
| F7 | medium 未走管线 | ✅ 先前修复留存 |
| F8 | premise/assertion 脱节 | ✅ 3 个测试全部修复 |
| F9 | 注释 "<4" 矛盾 | ✅ 改为 "<3" |
| F10 | tuple vs tensor 负索引 | ✅ 注释修正 |
| F11 | 单字符混合触发条件 | ✅ 层数隔离 |

---

## What We Want From You

Apply the **Systematic Scan Checklist** to every test function in the diff:

1. **Triple-Check**: docstring → constructed input → assert verification (self-consistent?)
2. **Pairwise Check**: related tests have contradictory assumptions?
3. **try/except Failure Path**: any bare `Exception` swallowing bugs?
4. **Threshold & Comment Consistency**: numeric thresholds match LTL claims?
5. **Cross-Validation**: do the "fixed" items truly fix the original problem without introducing new false-green risks?

Focus areas:
- Are all 25 previous findings properly resolved?
- Any regressions from the changes?
- Any new false-green risks from the fixes?
- Semantic consistency: do tests named `test_X` actually construct and verify X?
- CONFLICT judgment: compare High findings against Reasonix's 25-item conclusion list

---

## Response Format

**JSON as source of truth, Markdown as summary only.**

```json
{
  "review": {
    "overall": "approve|conditional|reject",
    "conflict_with_reasonix": false,
    "summary": {"high": 0, "medium": 0, "low": 0},
    "findings": [
      {
        "severity": "high",
        "file": "path/to/file.py",
        "line": 42,
        "issue": "description",
        "category": "false_green|semantic_mismatch|assertion_too_weak|edge_case|other",
        "conflict": false
      }
    ]
  }
}
```
