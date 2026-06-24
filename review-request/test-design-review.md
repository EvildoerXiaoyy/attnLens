# Review Request: entropy-analyzer + norm-scanner 测试设计

## Context

对 Prompt 注意力诊断器 v0.1 Demo 的两个核心算法模块（注意力熵变率分析 + 隐藏态范数扫描）的测试骨架进行二审。当前处于**测试驱动开发的测试生成阶段**——生产代码尚未编写，测试骨架已由 AI 生成，人类尚未补充边界用例。

## Diff（测试设计文件）

两个测试文件共 549 行：

### `tests/test_entropy_analyzer.py`
- 文件归属：`entropy-analyzer` 模块
- 测试框架：pytest
- 依赖 Mock：`tests/mocks/model_loader_mock.py`（MockModel / MockTokenizer）
- 已实现：15 个 Happy Path 用例
- TODO 占位：12 个（Unicode 输入、超大矩阵、并发安全、幂等性等）

### `tests/test_norm_scanner.py`
- 文件归属：`norm-scanner` 模块
- 测试框架：pytest
- 依赖 Mock：`tests/mocks/model_loader_mock.py`（MockModel / MockTokenizer）
- 已实现：14 个 Happy Path 用例
- TODO 占位：15 个（边界 chunk_size、全零 hidden states、超长文本、并发安全等）

详细结构见下文。

---

## Architecture Context

两个算法模块共享同一个 `model-loader` 依赖（Mock 已生成），可并行开发：

```
model-loader (Mock)
  ├─→ entropy-analyzer ─→ calc_entropy_delta(model, tokenizer, text) → list[token_risks]
  └─→ norm-scanner     ─→ scan_signal_strength(model, tokenizer, text, chunk_size) → list[chunk_risks]
```

核心算法：
- **熵变率**：取最后一层 vs 倒数第三层的注意力熵差，delta > 2.0 高风险
- **范数扫描**：取倒数第三层 Hidden State 的 L2 范数，按块聚合，低于 15% 分位数标记弱信号

详见 `ARCHITECTURE.md`、`API_CONTRACT.yaml`、`docs/adr/ADR-002-core-detection-algorithm.md`。

---

## Trade-offs to Respect

这些是已确认的架构决策，**不要建议推翻它们**。

> **Decision:** Qwen2.5-0.5B > Llama-3.2-1B — 更低资源需求，同 Tokenizer 对齐
> **Decision:** 熵变率 + 范数扫描 > 其他方案（Perplexity / Saliency / 规则） — 句法 + 语义双维度覆盖
> **Decision:** 阈值 delta > 2.0 / 1.5 为启发式默认值 — 代码可配置
> **Decision:** 超长输入采用截断模式而非滑窗分片 — 避免计算膨胀
> **Decision:** 一次 forward pass 同时输出 attention + hidden states — 避免重复推理
> **Decision:** Mock model.forward + 真实 tokenizer — 测试速度优先

---

## What Reasonix Already Checked

Reasonix 一审在此阶段**仅审查了测试设计本身**（生产代码尚未编写）：

- 测试覆盖完整性（Happy Path + TODO 占位）
- Mock 策略合理性（mock model.forward + 真实 tokenizer）
- 测试接缝选择（模块级 API 为主要接缝）
- 测试分类（Happy Path / Edge / Error / Concurrency / Performance / Consistency）

---

## Reasonix 一审结论

现阶段无生产代码，一审结论聚焦测试设计的结构性检查：

**High（0）**
无 High 级别问题。测试设计结构完整，Mock 策略合理。

**Medium（0）**
无 Middle 级别问题。

**Low（2）**
- 所有 TODO 占位标记了待人工补充的边界用例，但未区分优先级（哪些是必须的、哪些是锦上添花）
- 两个测试文件的结构高度对称（同一模板生成），可通过共享 fixture 减少重复代码——但这是生产代码编写后的优化时机

---

## What We Want From You

测试设计二审，重点关注：

- **测试完备性**：是否有缺失的关键场景？
- **Mock 合理性**：MockModel 的行为能否覆盖真实模型的边界情况？
- **断言强度**：现有断言是否足够严格？是否存在"假绿"（测试通过但未真正验证行为）的风险？
- **TODO 优先级**：27 个 TODO 占位中，哪些必须在此版本完成、哪些可延迟？
- **架构一致性**：测试设计是否与 `ARCHITECTURE.md` 和 `API_CONTRACT.yaml` 一致？
- **测试可维护性**：是否有过度耦合或脆弱的测试模式？

---

## Response Format

**输出原则：** JSON 是唯一完整数据源（Source of Truth），Markdown 是摘要。

### Markdown（摘要）

```
## Summary
- High: X | Medium: Y | Low: Z
- Overall: approve / conditional / reject
```

### JSON（完整数据源）

```json
{
  "review": {
    "overall": "approve|conditional|reject",
    "summary": {
      "high": 0,
      "medium": 0,
      "low": 0
    },
    "findings": [
      {
        "severity": "high",
        "file": "tests/test_entropy_analyzer.py",
        "line": 42,
        "issue": "description"
      }
    ]
  }
}
```

---

## 附录：测试文件完整结构

### `tests/test_entropy_analyzer.py` 结构树

```
TestHappyPath
  ├─ test_basic_entropy_analysis       — 正常文本返回 Token 级结果
  ├─ test_entropy_delta_is_float        — 熵差值为浮点数
  ├─ test_risk_level_is_valid_string    — 风险等级为 high/medium/low
  ├─ test_uniform_attention_all_high_risk  — 均匀注意力→高风险
  ├─ test_focused_attention_low_risk    — 集中注意力→低风险
  └─ test_result_order_matches_input    — 结果顺序与输入一致

TestShortText
  ├─ test_single_char_input             — 单字符输入→安全回退
  ├─ test_two_layer_model               — 层数不足→安全回退
  ├─ test_two_token_input               — 2 Token 极短输入
  └─ [3 TODO] whitespace / unicode / max_length

TestThresholdConfig
  ├─ test_custom_thresholds             — 低阈值→更多高风险
  ├─ test_high_threshold_no_high_risk   — 高阈值→无高风险
  └─ [2 TODO] negative / zero threshold

TestErrorHandling: [4 TODO]
TestNumericalAccuracy: 3 active + [3 TODO]
TestConcurrency: [2 TODO]
TestPerformance: [2 TODO]
TestConsistency: 1 active + [1 TODO]
```

### `tests/test_norm_scanner.py` 结构树

```
TestHappyPath
  ├─ test_basic_scan                    — 正常文本返回块级结果
  ├─ test_chunk_size_respected          — 分块大小与配置一致
  ├─ test_norm_score_is_float           — 范数分数为浮点数
  ├─ test_is_weak_is_bool               — is_weak 为布尔值
  ├─ test_text_snippet_non_empty        — 文本片段不为空
  └─ test_weak_chunk_detection          — 弱信号块正确标记

TestChunkSize
  ├─ test_custom_chunk_size             — 自定义 chunk_size
  ├─ test_small_chunk_size              — 小 chunk_size→更多块
  ├─ test_chunk_size_larger_than_input  — chunk_size > 输入→自动调整
  └─ [5 TODO] size=1 / equals_input / zero / negative

TestWeakPercentile
  ├─ test_high_percentile_more_weak     — 高百分位→更多弱信号
  ├─ test_low_percentile_fewer_weak     — 低百分位→更少弱信号
  └─ [3 TODO] 0 / 100 / negative percentile

TestEdgeCases: 1 active + [5 TODO]
TestErrorHandling: [4 TODO]
TestNumericalAccuracy: 2 active + [2 TODO]
TestConcurrency: [3 TODO]
TestConsistency: 1 active + [1 TODO]
```
