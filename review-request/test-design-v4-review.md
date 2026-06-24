# Review Request: test-design-v4 — 全轮修复最终验证

## Context

对 Prompt 注意力诊断器 v0.1 Demo 的测试设计进行**第四次独立审查**。前三轮 GLM 5.2 共计发现 31 个问题（2H+8M+14L+7 语义），**本轮所有问题已修复**。本轮是最终验证——确认 suite 无假绿、无互相矛盾的约定、三元组自洽。

---

## Diff（变更概览，共 1236 行，5 个文件）

> Diff 超过 500 行，已压缩。全部为核心逻辑文件，无自动生成代码。

### 文件清单

| 文件 | 行数 | 本轮变更 |
|------|------|---------|
| `tests/mocks/model_loader_mock.py` | 287 | 删除 `set_max_length` + forward 截断逻辑（M-1） |
| `tests/test_entropy_analyzer.py` | 480 | 截断测试改为兼容性；mismatch 测试定单向；注释修 |
| `tests/test_norm_scanner.py` | 414 | mismatch 测试定单向；empty_tokenizer handler 加强 |
| `tests/conftest.py` | 12 | 无变更 |
| `docs/adr/ADR-002-core-detection-algorithm.md` | 43 | 无变更 |

### 核心变更（vs v3）

**M-1 — 截断从 Mock 移回分析器逻辑域：**
```python
# MockModel.forward: 删除
# - if seq > max_length: seq = max_length   ← 已移除
# - set_max_length()                         ← 已移除

# test_input_at_max_length: 改为仅验证长输入不崩溃
# 截断行为将在 PromptLinter 编排层测试中覆盖
```

**L-1 — mismatch 测试固定单向（自适应实际维度）：**
```python
# 之前: try: ... except: return（双向接受）
# 现在: 直接调用 calc_entropy_delta，断言 len>0（单向：自适应）
```

**L-2 — norm empty_tokenizer 与 entropy 侧一致：**
```python
# 之前: except (ValueError, RuntimeError, IndexError): pass
# 现在: except ... as e: assert len(str(e)) > 0
```

---

## 四轮审查完整追溯

### 一轮（14 项）
| # | 状态 | 行号参考 |
|---|------|---------|
| H1 逐层注意力 | ✅ | mock:99-101 |
| H2 确定性 hidden | ✅ | mock:128-138 |
| M1 默认值测试 | ✅ | entropy:140-144, norm:112-120 |
| M2 顺序假绿 | ✅ | entropy:108-123 |
| M3 截断零测试 | ⚠️ **最终修复（M-1）** | 截断移出 Mock |
| M4 tokenizer hash | ✅ | mock:217-222 |
| M5 output_* 开关 | ✅ | mock:192-195, entropy:346-352 |
| M6 delta→risk 钉住 | ✅ | entropy:295-316 |
| M7 规约厘清 | ✅ | entropy:5-15, ARCHITECTURE.md:79 |
| L1-L5 | ✅ | 全部 |

### 二轮（11 项）
| F1 互斥算子 | ✅ | ADR-002, norm:208-216 |
| F2 全零测试 | ✅ | norm:309-316, mock:150-151 |
| F3 extreme 测试 | ✅ | norm:357-364, mock:153-154 |
| F4 mismatch 真构造 | ✅ | entropy:379, norm:330-337 |
| F5 wrong shape 真冲突 | ✅ | entropy:367-377, mock:23 |
| F6 边界承诺 medium | ✅ | entropy:295-316 |
| F7 medium 管线验证 | ✅ | entropy:295-316 |
| F8 三元组自洽 | ✅ | entropy:195-205/238-245/248-254 |
| F9 注释 "<4" 修复 | ✅ | entropy:161 |
| F10 tuple vs tensor | ✅ | norm:246-258 |
| F11 触发隔离 | ✅ | entropy:152-159 |

### 三轮（6 项）
| M-1 截断在 Mock 层 | ✅ | 本版本已移出 Mock |
| L-1 mismatch 双向接受 | ✅ | 已固定单向 |
| L-2 norm handler 弱 | ✅ | 已与 entropy 对齐 |
| L-3 全局标量注释 | ✅ | ADR-002 记录 |
| L-4 误导注释 | ✅ | entropy:264-265 已修 |
| L-5 双向接受 | 可接受 | 防御性测试 |

**无回归。零裸 `except Exception`。**

---

## Trade-offs to Respect

> **Decision:** Qwen2.5-0.5B > Llama-3.2-1B
> **Decision:** 熵变率 + 范数扫描 > 替代方案
> **Decision:** 阈值 delta > 2.0/1.5 为启发式默认值（可配置）
> **Decision:** 超长输入截断模式 > 滑窗分片
> **Decision:** 单次 forward 同时输出 attention + hidden_states
> **Decision:** Mock model.forward + MockTokenizer（crc32 确定性）
> **Decision:** 弱信号严格小于 `norm_score < percentile_threshold`（ADR-002）
> **Decision:** 截断是编排层职责，非 Mock/分析器职责（M-1 确认）

---

## What Reasonix Already Checked

四轮审查累计覆盖：
- **Mock 基础设施**：逐层注意力、确定性、output_* 开关、hidden pattern、确定性 tokenizer hash
- **测试完备性**：33 + 33 = 66 个活跃测试，覆盖 happy path / 阈值边界 / 异常路径 / 确定性 / 幂等性
- **语义一致性**：三元组自洽（docstring→构造→断言）、配对矛盾检查、注释数值准确性
- **Mock 充分性**：set_hidden_pattern("zero"|"huge")、set_fixed_ids、逐层 attention
- **架构一致性**：与 ARCHITECTURE.md / API_CONTRACT.yaml / ADR-002 对齐

---

## What We Want From You

Apply the **Systematic Scan Checklist** one final time:

1. **Triple-Check** — every test function: docstring → input → assert
2. **Pairwise Check** — no contradictory assumptions between related tests
3. **try/except Failure Path** — no bare `Exception`, no swallowed bugs
4. **Threshold & Comment Consistency** — numeric thresholds match code
5. **Cross-Validate All 31 Prior Items** — no regression, no new false greens

Final decision:
- **approve** — suite is self-consistent and ready for TDD
- **conditional** — specific issues remain
- **reject** — blocking problems exist

---

## Response Format

```json
{
  "review": {
    "overall": "approve|conditional|reject",
    "conflict_with_reasonix": false,
    "summary": { "high": 0, "medium": 0, "low": 0 },
    "findings": [
      {
        "severity": "...",
        "category": "...",
        "file": "...",
        "line": 0,
        "issue": "...",
        "conflict": false
      }
    ]
  }
}
```

Markdown: Summary only (one line per severity tier).
