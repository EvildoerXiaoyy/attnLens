# Review Request: test-design-v2 — 二审修复验证

## Context

对 Prompt 注意力诊断器 v0.1 Demo 的测试设计进行**第二次独立审查**。本次审查的对象是**修复后的版本**——两周前 GLM 5.2 一审发现了 2 High + 7 Medium + 5 Low 问题，所有问题已在当前版本修复。本次二审的目的是确认修复是否彻底、是否有新问题引入。

## Diff（变更概览，共 1153 行，4 个文件）

> Diff 超过 500 行，已压缩为变更摘要 + 关键代码片段。全部文件均为核心业务/测试逻辑，无自动生成代码。

### 文件清单

| 文件 | 行数 | 变更类型 |
|------|------|---------|
| `tests/mocks/model_loader_mock.py` | 263 | 重写（H1, H2, M4, M5 修复） |
| `tests/test_entropy_analyzer.py` | 479 | 重写 + 新增 20 个测试 |
| `tests/test_norm_scanner.py` | 399 | 重写 + 新增 18 个测试 |
| `tests/conftest.py` | 12 | **新增** |

### 核心变更摘要

#### 1. `tests/mocks/model_loader_mock.py` — Mock 基础设施修复

**H1 修复 — 注意力逐层可变：**
```python
# 新增 API
def set_layer_attention(self, layer_idx: int, pattern: str):
    """设置特定层的注意力模式，覆盖全局默认"""
    self._layer_patterns[layer_idx] = pattern

# forward() 改为按层查询模式
def _get_pattern(self, layer_idx: int) -> str:
    return self._layer_patterns.get(layer_idx, self._global_pattern)
```

**H2 修复 — 确定性 hidden states：**
```python
# 使用种子化 Generator，不用全局 RNG
def _make_hidden_state(self, layer_idx, batch, seq):
    seed = 42 + layer_idx * 7
    self._rng.manual_seed(seed)
    h = torch.randn(batch, seq, hidden_dim, generator=self._rng) * 0.1
```

**M4 修复 — 确定性 Tokenizer hash：**
```python
# zlib.crc32 替代 Python hash()，跨进程稳定
ids = [zlib.crc32(c.encode("utf-8")) & 0x7FFFFFFF % self.vocab_size for c in text]

# convert_ids_to_tokens 反映真实 token id
def convert_ids_to_tokens(self, ids):
    return [f"id_{tid}" for tid in ids]  # 之前返回 tok_0..tok_n
```

**M5 修复 — output_* 开关校验：**
```python
def forward(self, input_ids=None, output_attentions=False, output_hidden_states=False, **kwargs):
    # 未收到 True 时返回 None
    if not output_attentions:
        attentions = None
    if not output_hidden_states:
        hidden_states = None
```

#### 2. `tests/test_entropy_analyzer.py` — 从 15 到 32 个活跃测试

**新增测试类别：**
- `TestDefaultConfig`（M1）— 验证默认阈值 2.0/1.5
- `TestDeltaToRiskLevel`（M6）— 表驱动测试验证 delta→risk_level 映射
- 剩余异常/边界：空 tokenizer、attention 形状异常、hidden_size 不匹配、负阈值、零阈值、纯空白输入、emoji 输入、全零注意力、超大矩阵、负注意力权重、幂等性

**关键假绿修复（M2）：**
```python
# 之前：assert len(results) > 0（假绿）
def test_result_order_matches_input(self, analyzer, mock_tokenizer):
    encoded = tokenizer(text, return_tensors="pt")
    expected_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    for i, r in enumerate(results):
        assert r["token"] == expected_tokens[i]
```

#### 3. `tests/test_norm_scanner.py` — 从 14 到 32 个活跃测试

**新增测试类别：**
- `TestDefaultConfig`（M1）— 验证默认 chunk_size=128/percentile=15
- chunk_size 边界（=1/0/负数）防御测试
- percentile 边界（0/100/负数）防御测试
- 输入边界：空白、Unicode、精确块边界
- 错误处理：空 tokenizer、hidden state 维度不匹配
- 幂等性：连续 10 次调用

**连续性断言补充（L4）：**
```python
def test_chunk_size_respected(self, scanner, mock_tokenizer):
    # 之前只查 end-start<=4
    # 新增：
    for i in range(1, len(results)):
        assert results[i]["start_token"] == results[i-1]["end_token"]
```

#### 4. `tests/conftest.py` — 新增（L1）

```python
# 集中管理 sys.path，测试文件不再各自插入
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
MOCKS_DIR = Path(__file__).resolve().parent / "mocks"
for p in [SRC_DIR, MOCKS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

---

## Architecture Context

测试设计覆盖两个核心算法模块（熵变率 + 范数扫描），共享 model-loader Mock。

```
model-loader (Mock)
  ├─→ entropy-analyzer ─→ calc_entropy_delta(model, tokenizer, text) → list[token_risks]
  └─→ norm-scanner     ─→ scan_signal_strength(model, tokenizer, text, chunk_size) → list[chunk_risks]
```

详见 `ARCHITECTURE.md`、`API_CONTRACT.yaml`。

---

## Trade-offs to Respect

这些是已确认的架构决策，**不要建议推翻它们**。

> **Decision:** Qwen2.5-0.5B > Llama-3.2-1B — 同 Tokenizer 对齐 + 低资源
> **Decision:** 熵变率 + 范数扫描 > 替代方案 — 句法+语义双维度
> **Decision:** 阈值 delta > 2.0/1.5 为启发式默认值 — 代码可配置
> **Decision:** 超长输入截断模式 > 滑窗分片 — 避免计算膨胀
> **Decision:** 单次 forward 同时输出 attention + hidden states
> **Decision:** Mock model.forward + 真实 tokenizer

---

## What Reasonix Already Checked

一审（上次 /review-request）已覆盖：

1. **代码质量**：Mock 设计、测试结构、断言强度
2. **测试完备性**：Happy Path + TODO 占位
3. **Mock 充分性**：MockModel 能否覆盖真实模型边界
4. **架构一致性**：测试与 API_CONTRACT.yaml / ARCHITECTURE.md 对齐

## Reasonix 一审结论（High/Medium 问题列表）

> **High**
> - `tests/mocks/model_loader_mock.py:86` — 注意力矩阵逐层不变，delta 恒为 0
> - `tests/mocks/model_loader_mock.py:101` — hidden states 未种子 randn，破坏确定性
>
> **Medium**
> - `tests/test_norm_scanner.py:25` — chunk_size 默认值 128 未被锁定
> - `tests/test_entropy_analyzer.py:84` — test_result_order_matches_input 假绿
> - `tests/test_entropy_analyzer.py:128` — 截断无 active 测试
> - `tests/mocks/model_loader_mock.py:137` — MockTokenizer hash 不确定
> - `tests/mocks/model_loader_mock.py:76` — forward(**kwargs) 丢弃 output_* 开关
> - `tests/test_entropy_analyzer.py:138` — 阈值→风险等级未被 delta 精确钉住
> - `tests/test_entropy_analyzer.py:84` — 规约歧义（per-token vs 全局 delta）
>
> **Low**
> - `tests/test_entropy_analyzer.py:8` — sys.path 重复
> - `tests/test_entropy_analyzer.py:188` — 测试私有方法 _entropy
> - `tests/test_entropy_analyzer.py:70` — MockModel 内联构造绕过 fixture
> - `tests/test_entropy_analyzer.py:46` — 断言偏弱（聚合性假绿）
> - `API_CONTRACT.yaml:28` — metadata 零断言

---

## What We Want From You

Focus on things the primary review might miss this round:

- **修复彻底性**：一审指出的 14 个问题是否已完全修复？有无遗漏？
- **回归风险**：修复过程中是否引入了新问题？
- **断言强度**：新增的 20+ 测试是否存在"假绿"或"过松"的断言？
- **Mock 完备性**：改进后的 Mock 是否足够覆盖测试需求？有无引入新漏洞？
- **一致性**：当前测试是否与 `ARCHITECTURE.md`、`API_CONTRACT.yaml`、`ADR-002` 一致？
- **冲突判断**：对比你的发现与一审结论——如果在一审标记为"已修复"的行上发现新问题，标记 `CONFLICT: Yes`

---

## Response Format

**输出原则：** JSON 是唯一完整数据源（Source of Truth），Markdown 是摘要。

### Markdown（摘要）

```
## Summary
- High: X | Medium: Y | Low: Z
- Overall: approve / conditional / reject
- CONFLICT: Yes/No
```

### JSON（完整数据源）

```json
{
  "review": {
    "overall": "approve|conditional|reject",
    "conflict_with_reasonix": false,
    "summary": {"high": 0, "medium": 0, "low": 0},
    "findings": [
      {
        "severity": "high",
        "file": "tests/mocks/model_loader_mock.py",
        "line": 86,
        "issue": "description",
        "conflict": false
      }
    ]
  }
}
```
