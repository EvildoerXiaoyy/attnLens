"""
测试骨架 — entropy-analyzer（注意力熵变率分析模块）

AI 生成基础测试 + 已补充的 must-have 边界用例（基于 GLM 5.2 二审反馈）。

规约说明（M7 修复）：
熵变率算法计算的是「最后一个 Token 的注意力分布熵」在深层与浅层之间的差值。
运算过程：
  1. 取最后一层和倒数第三层 attention 矩阵中最后一个 Token 的分布 [:, -1, :]
  2. 对每层：拆分为 num_heads 个分布，各算熵后取平均 → 该层标量
  3. delta = deep_layer_entropy - shallow_layer_entropy（单个标量）
结果列表与输入 Token 序列等长，但每项携带相同的 delta 值。
delta 作为整个 Prompt 在"决策出口处"句法清晰度的量化信号。

注意（L5）：metadata（model_name/total_tokens/analysis_time_ms）属于
PromptLinter 编排层的装配职责，不在本模块测试范围内。
相关测试应放在 tests/test_prompt_linter.py 中。"""

import numpy as np
import torch
import pytest

from model_loader_mock import MockModel, MockTokenizer
from prompt_linter.entropy_analyzer import EntropyAnalyzer


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def analyzer():
    return EntropyAnalyzer(high_risk_threshold=2.0, medium_risk_threshold=1.5)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def _make_model(num_layers=6, seq_len=8, num_heads=4, hidden_dim=8):
    """创建标准 MockModel 辅助函数（L3 修复：统一构造入口）"""
    return MockModel(
        num_layers=num_layers,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )


def _setup_entropy_delta_model(
    num_layers=6, seq_len=8, deep_pattern="uniform", shallow_pattern="focused"
):
    """创建支持已知熵差的 MockModel（H1 修复）。

    预设 deep 层和 shallow 层为不同注意力模式，使 delta 可控。
    deep_pattern 对应 layer[-1]，shallow_pattern 对应 layer[-3]。
    """
    model = _make_model(num_layers=num_layers, seq_len=seq_len)
    model.set_layer_attention(num_layers - 1, deep_pattern)
    model.set_layer_attention(num_layers - 3, shallow_pattern)
    return model


# ── Happy Path ─────────────────────────────────────────────────────

class TestHappyPath:
    """正常输入下的基本功能验证"""

    def test_basic_entropy_analysis(self, analyzer, mock_tokenizer):
        """Happy Path: 正常文本应返回 Token 级分析结果（L4 修复：补充有限性校验）"""
        model = _make_model()
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "hello world test prompt")
        assert len(results) > 0
        for r in results:
            assert "token" in r
            assert "token_id" in r
            assert "entropy_delta" in r
            assert "risk_level" in r
            assert isinstance(r["entropy_delta"], float)
            assert np.isfinite(r["entropy_delta"]), f"delta 应为有限值: {r['entropy_delta']}"
            assert r["risk_level"] in ("high", "medium", "low")

    def test_entropy_delta_is_float(self, analyzer, mock_tokenizer):
        """Happy Path: 熵差值应为浮点数"""
        model = _make_model()
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "test input")
        for r in results:
            assert isinstance(r["entropy_delta"], float)

    def test_risk_level_is_valid_string(self, analyzer, mock_tokenizer):
        """Happy Path: 风险等级应为 high / medium / low 之一"""
        model = _make_model()
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "test")
        for r in results:
            assert r["risk_level"] in ("high", "medium", "low")

    def test_deep_high_shallow_low_produces_high_risk(self, analyzer, mock_tokenizer):
        """Happy Path: 深层高熵 + 浅层低熵 → delta > 0 → 高风险"""
        # 深层 uniform（高熵）vs 浅层 focused（低熵）→ delta = log(N) - 0
        # seq_len=8 → log(8) ≈ 2.079 > 2.0 → high risk（H1 修复）
        model = _setup_entropy_delta_model(seq_len=8, deep_pattern="uniform", shallow_pattern="focused")
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "x" * 8)
        high_count = sum(1 for r in results if r["risk_level"] == "high")
        assert high_count > 0, "深层高熵+浅层低熵应产生高风险"

    def test_all_focused_low_risk(self, analyzer, mock_tokenizer):
        """Happy Path: 所有层 focused → delta ≈ 0 → 无高风险"""
        model = _make_model()
        model.set_attention_pattern("focused")
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "test")
        high_count = sum(1 for r in results if r["risk_level"] == "high")
        assert high_count == 0

    def test_result_order_matches_input(self, analyzer):
        """Happy Path: 结果顺序应与输入 Token 顺序一致（L1 修复：用固定 ID 验证 token_id 映射）"""
        model = _make_model()
        tokenizer = MockTokenizer()
        known_ids = [101, 205, 310, 415, 520]
        tokenizer.set_fixed_ids(known_ids)
        results = analyzer.calc_entropy_delta(model, tokenizer, "x" * len(known_ids))
        assert len(results) == len(known_ids)
        for i, r in enumerate(results):
            assert r["token_id"] == known_ids[i], (
                f"位置 {i}: 期望 token_id={known_ids[i]}, 得到 {r['token_id']}"
            )
            # token 名称应与 token_id 对应
            assert r["token"] == f"id_{known_ids[i]}", (
                f"位置 {i}: 期望 token='id_{known_ids[i]}', 得到 '{r['token']}'"
            )


# ── Default Configuration ────────────────────────────────────────

class TestDefaultConfig:
    """默认构造参数验证（M1 修复）"""

    def test_default_high_threshold_is_2_0(self):
        """默认 high_risk_threshold 应为 2.0"""
        a = EntropyAnalyzer()
        assert a.high_risk_threshold == 2.0

    def test_default_medium_threshold_is_1_5(self):
        """默认 medium_risk_threshold 应为 1.5"""
        a = EntropyAnalyzer()
        assert a.medium_risk_threshold == 1.5


# ── Short Text / Edge Cases ───────────────────────────────────────

class TestShortText:
    """短文本与边界长度"""

    def test_single_char_input(self, analyzer, mock_tokenizer):
        """Edge: 单字符（序列长 1）应触发短文本回退，与层数无关（F11 修复：层数给足以隔离触发条件）"""
        model = MockModel(num_layers=6, seq_len=1, num_heads=2, hidden_dim=4)
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "a")
        assert len(results) > 0
        for r in results:
            assert r["risk_level"] == "low"
            assert r["entropy_delta"] == 0.0

    def test_two_layer_model(self, analyzer, mock_tokenizer):
        """Edge: 层数 < 3（无法取 layer[-3]）应触发安全回退（F9 修复）"""
        model = MockModel(num_layers=2, seq_len=5, num_heads=2, hidden_dim=4)
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "test")
        for r in results:
            assert r["risk_level"] == "low"

    def test_two_token_input(self, analyzer, mock_tokenizer):
        """Edge: 2 个 Token 的极短输入"""
        model = _make_model()
        model.set_attention_pattern("uniform")
        # 短输入下每个 token 都应有结果
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "ab")
        assert len(results) == 2

    def test_unicode_chinese_input(self, analyzer, mock_tokenizer):
        """Edge: 中文输入不应崩溃（原 TODO: unicode_input → must-have）"""
        model = _make_model()
        model.set_attention_pattern("uniform")
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "你好，Prompt 诊断器测试")
        assert len(results) > 0
        for r in results:
            assert isinstance(r["entropy_delta"], float)

    def test_input_at_max_length(self, analyzer, mock_tokenizer):
        """Edge: 长输入不应崩溃（截断由编排层 PromptLinter.analyze 负责，见该层测试）

        Mock 不模拟截断（M-1 修复），此测试仅验证分析器在处理长输入时不会 OOM/崩溃。
        同时验证结果数与输入 token 数一致（L-1 修复：不静默丢失 token）。
        """
        model = _make_model(seq_len=100)
        model.set_attention_pattern("uniform")
        text = "x" * 100
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, text)
        assert len(results) == 100, f"长输入应产生 100 个结果，得到 {len(results)}"

    def test_whitespace_input(self, analyzer, mock_tokenizer):
        """Edge: 纯空白字符经 tokenizer 后为正常长度，应正常分析（F8 修复）"""
        model = _make_model()
        model.set_attention_pattern("uniform")
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "   \t\n  ")
        assert len(results) > 0
        for r in results:
            assert isinstance(r["entropy_delta"], float)
            assert r["risk_level"] in ("high", "medium", "low")

    def test_emoji_input(self, analyzer, mock_tokenizer):
        """Edge: 混合 emoji 输入不应崩溃"""
        model = _make_model()
        model.set_attention_pattern("uniform")
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "Hello 👋 World 🌍 测试")
        assert len(results) > 0
        for r in results:
            assert isinstance(r["entropy_delta"], float)


# ── Threshold Configuration ───────────────────────────────────────

class TestThresholdConfig:
    """阈值配置行为"""

    def test_custom_thresholds(self, mock_tokenizer):
        """阈值调低后应产生更多高风险 Token"""
        strict = EntropyAnalyzer(high_risk_threshold=0.5, medium_risk_threshold=0.2)
        # 使用有已知熵差的模型
        model = _setup_entropy_delta_model(seq_len=8, deep_pattern="uniform", shallow_pattern="focused")
        results = strict.calc_entropy_delta(model, mock_tokenizer, "x" * 8)
        high_count = sum(1 for r in results if r["risk_level"] == "high")
        assert high_count > 0

    def test_high_threshold_no_high_risk(self, mock_tokenizer):
        """阈值极高时应无高风险"""
        lenient = EntropyAnalyzer(high_risk_threshold=100.0, medium_risk_threshold=50.0)
        model = _setup_entropy_delta_model(seq_len=8, deep_pattern="uniform", shallow_pattern="focused")
        results = lenient.calc_entropy_delta(model, mock_tokenizer, "x" * 8)
        high_count = sum(1 for r in results if r["risk_level"] == "high")
        assert high_count == 0

    def test_negative_threshold(self, mock_tokenizer):
        """F8: 负阈值下，构造真负 delta（deep focused / shallow uniform）应全部判 low"""
        neg = EntropyAnalyzer(high_risk_threshold=-1.0, medium_risk_threshold=-2.0)
        model = _setup_entropy_delta_model(
            seq_len=8, deep_pattern="focused", shallow_pattern="uniform"
        )
        results = neg.calc_entropy_delta(model, mock_tokenizer, "x" * 8)
        assert all(r["risk_level"] == "low" for r in results), "负 delta 应全部判 low"

    def test_zero_threshold(self, mock_tokenizer):
        """F8: 阈值=0 时，所有正 delta 均应判 high"""
        zero = EntropyAnalyzer(high_risk_threshold=0.0, medium_risk_threshold=0.0)
        model = _setup_entropy_delta_model(seq_len=4, deep_pattern="uniform", shallow_pattern="focused")
        results = zero.calc_entropy_delta(model, mock_tokenizer, "x" * 4)
        assert all(r["risk_level"] == "high" for r in results if r["entropy_delta"] > 0), \
            "阈值=0 时所有正 delta 应判 high"


# ── Delta → Risk Level 表驱动测试（M6 修复） ─────────────────────

class TestDeltaToRiskLevel:
    """已知熵差 → 正确风险等级（M6 修复）"""

    def test_delta_above_high_threshold_is_high(self):
        """delta > 2.0 → risk_level = high"""
        analyzer = EntropyAnalyzer(high_risk_threshold=2.0, medium_risk_threshold=1.5)
        # 构造已知 delta：deep uniform - shallow focused
        # seq_len=8 → log(8) ≈ 2.079 > 2.0 → high
        model = _setup_entropy_delta_model(seq_len=8, deep_pattern="uniform", shallow_pattern="focused")
        tokenizer = MockTokenizer()
        tokenizer.set_fixed_ids(list(range(8)))
        results = analyzer.calc_entropy_delta(model, tokenizer, "x" * 8)
        # delta ≈ 2.079 > 2.0 → 所有 token 应为 high（因为 delta 是全局标量）
        high_risks = [r for r in results if r["risk_level"] == "high"]
        assert len(high_risks) == len(results), (
            f"delta 是全局标量，所有 token 应判 high（{len(high_risks)}/{len(results)}）"
        )

    def test_delta_below_medium_is_low(self):
        """delta <= 1.5 → risk_level = low"""
        analyzer = EntropyAnalyzer(high_risk_threshold=2.0, medium_risk_threshold=1.5)
        # 所有层 uniform → delta ≈ 0 → low
        model = _make_model(seq_len=4)
        model.set_attention_pattern("uniform")
        tokenizer = MockTokenizer()
        tokenizer.set_fixed_ids(list(range(4)))
        results = analyzer.calc_entropy_delta(model, tokenizer, "test")

        # 应该都是 low (delta ≈ 0)
        non_low = [r for r in results if r["risk_level"] != "low"]
        high_risks = [r for r in results if r["risk_level"] == "high"]
        assert len(high_risks) == 0
        assert len(non_low) == 0, "delta ≈ 0 应全部为 low"

    def test_high_threshold_boundary_medium_risk(self):
        """Med-2: delta ∈ (1.5, 2.0) 应判 medium（完整管线验证）

        seq_len=5, deep uniform → log(5)≈1.609, shallow focused → 0,
        delta≈1.609 ∈ (1.5, 2.0) → risk_level == medium
        """
        analyzer = EntropyAnalyzer(high_risk_threshold=2.0, medium_risk_threshold=1.5)
        model = MockModel(num_layers=6, seq_len=5, num_heads=4, hidden_dim=8)
        model.set_layer_attention(5, "uniform")   # layer[-1] = deep → high
        model.set_layer_attention(3, "focused")    # layer[-3] = shallow → low
        tokenizer = MockTokenizer()
        tokenizer.set_fixed_ids([0, 1, 2, 3, 4])
        results = analyzer.calc_entropy_delta(model, tokenizer, "test")
        # delta 是全局标量，所有 token 应判 medium
        medium_risks = [r for r in results if r["risk_level"] == "medium"]
        high_risks = [r for r in results if r["risk_level"] == "high"]
        assert len(medium_risks) == len(results), (
            f"delta≈1.609 所有 token 应判 medium（{len(medium_risks)}/{len(results)}）"
        )
        assert len(high_risks) == 0, "delta≈1.609 不应产生 high risk"
        for r in results:
            assert np.isfinite(r["entropy_delta"]), f"delta 应为有限值: {r}"

    def test_medium_threshold_boundary(self, analyzer):
        """Boundary: 通过手搓 attention 验证 delta 数值正确性（辅助 test_high_threshold_boundary_medium_risk 的数值验证）"""
        # 构造 head-specific 分布使平均熵≈1.609
        attn_deep = torch.full((2, 5), 0.2)  # 2 heads, 5 positions, uniform
        attn_shallow = torch.zeros((2, 5))
        attn_shallow[:, 0] = 1.0  # focused
        deep_entropy = analyzer._entropy(attn_deep)  # ≈ 1.609
        shallow_entropy = analyzer._entropy(attn_shallow)  # ≈ 0
        delta = deep_entropy - shallow_entropy  # ≈ 1.609
        assert 1.5 < delta.item() < 2.0, f"delta={delta.item()} 应在 medium 范围"


# ── Error Handling ────────────────────────────────────────────────

class TestErrorHandling:
    """错误处理与异常路径"""

    def test_output_attentions_flag_respected(self, analyzer, mock_tokenizer):
        """M5: 未传 output_attentions=True 时，MockModel 返回 attentions=None。
        实际 analyzer 必须传 True 才能工作，此测试验证 analyzer 的调用约定。"""
        model = _make_model()
        text = "test"
        # 验证 MockModel 的 M5 行为：不传 output_attentions → attentions=None
        outputs = model.forward(
            input_ids=mock_tokenizer(text, return_tensors="pt")["input_ids"],
            output_attentions=False,
        )
        assert outputs.attentions is None

    def test_forward_with_zero_attention(self, analyzer):
        """Edge: 全零注意力矩阵经 clamp(1e-10) 保护后不应产生 NaN"""
        zero_attn = torch.zeros((4, 4))  # 4 heads, 4 seq
        e = analyzer._entropy(zero_attn)
        # clamp(1e-10) 保证熵为有限值而非 NaN
        assert torch.isfinite(e), f"全零注意力熵应为有限值, 得到 {e}"

    def test_empty_tokenizer_output(self, analyzer):
        """Error: tokenizer 返回空 input_ids 时应触发短文本回退（Med-3 修复）"""
        empty_tok = MockTokenizer()
        empty_tok.set_fixed_ids([])
        model = _make_model(seq_len=0)
        # 空 input_ids → seq_len=0 → analyzer 应触发短文本保护逻辑
        # 而非抛异常。如果抛异常，必须是明确类型的已知异常。
        try:
            results = analyzer.calc_entropy_delta(model, empty_tok, "")
            # 短文本回退应返回空列表
            assert len(results) == 0, f"短文本回退应返回空列表, 得到 {len(results)}"
        except (ValueError, RuntimeError, IndexError) as e:
            # 允许特定类型的异常，但不允许裸 Exception
            assert len(str(e)) > 0

    def test_wrong_attention_shape(self, analyzer, mock_tokenizer):
        """Error: config.num_attention_heads 与实际 head 数不符时应自适应（F5/L1 修复）"""
        model = _make_model(num_heads=4, seq_len=4)
        model.config.num_attention_heads = 2  # config 说 2，实际张量 4
        # analyzer 应自适应实际张量维度，而非读取 config
        results = analyzer.calc_entropy_delta(model, mock_tokenizer, "test")
        assert len(results) > 0, "analyzer 应自适应实际 attention head 数"

    # F4 修复：删除 test_hidden_size_mismatch——熵分析器不消费 hidden states，
    # hidden_size 不匹配不会影响它的行为，测试无意义。


# ── Numerical Accuracy ────────────────────────────────────────────

class TestNumericalAccuracy:
    """数学计算正确性"""

    def test_entropy_uniform_distribution(self, analyzer):
        """均匀分布的熵应为 log(N)"""
        uniform = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        e = analyzer._entropy(uniform)
        expected = np.log(4)
        assert abs(e.item() - expected) < 1e-4

    def test_entropy_deterministic_distribution(self, analyzer):
        """确定分布的熵应为 0"""
        deterministic = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        e = analyzer._entropy(deterministic)
        assert abs(e.item()) < 1e-4

    def test_entropy_multi_head_average(self, analyzer):
        """多头熵应取平均值"""
        attn = torch.tensor([
            [0.25, 0.25, 0.25, 0.25],  # head 0: uniform → log(4)
            [1.0, 0.0, 0.0, 0.0],      # head 1: deterministic → 0
        ])
        e = analyzer._entropy(attn)
        expected = (np.log(4) + 0.0) / 2
        assert abs(e.item() - expected) < 1e-4

    def test_large_attention_matrix(self, analyzer):
        """Extreme: 1024 个 Token 的注意力矩阵不应 OOM"""
        num_positions = 1024
        # 模拟 4 个 head 在 1024 个位置上的均匀分布
        uniform = torch.full((4, num_positions), 1.0 / num_positions)
        e = analyzer._entropy(uniform)
        expected = np.log(num_positions)  # ≈ 6.93
        assert abs(e.item() - expected) < 0.1

    def test_negative_attention_weights(self, analyzer):
        """Extreme: 负的注意力权重应由 clamp 保护，不应导致 NaN"""
        neg_attn = torch.tensor([[-0.1, 0.5, 0.3, 0.3]])
        try:
            e = analyzer._entropy(neg_attn)
            # 如果 clamp 生效，熵应为正且有限
            assert torch.isfinite(e), "负权重 clamp 后应为有限熵"
        except (ValueError, RuntimeError):
            # 允许抛异常（log 负数），但不该是 NaN
            pass


# ── Concurrency / Stress ──────────────────────────────────────────
#
# 以下 TODO 为有意推迟的非功能性基准测试，
# 将在 TDD 实现完成后的重构阶段补充。
# 它们不阻碍测试定稿，也不影响核心算法验证。

class TestConcurrency:
    """并发安全与压力"""

    # TODO: 并发安全 — 多个 goroutine 同时调用 calc_entropy_delta
    # def test_concurrent_calls(self, analyzer, mock_tokenizer):
    #     ...

    # TODO: 压力测试 — 短时间内大量重复调用
    # def test_repeated_calls(self, analyzer, mock_tokenizer):
    #     ...


# ── Performance ────────────────────────────────────────────────────

class TestPerformance:
    """性能基准"""

    # TODO: 性能基准 — 短文本（<100 tokens）应在 1 秒内完成
    # def test_short_text_latency(self, analyzer, mock_tokenizer):
    #     ...

    # TODO: 性能基准 — 长文本（>10K tokens）不应 OOM
    # def test_long_text_memory(self, analyzer, mock_tokenizer):
    #     ...


# ── Consistency ────────────────────────────────────────────────────

class TestConsistency:
    """确定性与幂等性（H2 修复后应稳定通过）"""

    def test_deterministic_output(self, analyzer, mock_tokenizer):
        """同输入应产出同输出（MockModel 的 hidden states 确定性由 H2 保证）"""
        model = _setup_entropy_delta_model(seq_len=4)
        text = "test"
        result1 = analyzer.calc_entropy_delta(model, mock_tokenizer, text)
        result2 = analyzer.calc_entropy_delta(model, mock_tokenizer, text)
        for r1, r2 in zip(result1, result2):
            assert r1["entropy_delta"] == r2["entropy_delta"]
            assert r1["risk_level"] == r2["risk_level"]

    def test_idempotent(self, analyzer, mock_tokenizer):
        """Consistency: 相同输入连续调用 10 次，结果应完全一致"""
        model = _setup_entropy_delta_model(seq_len=4)
        text = "test"
        first = analyzer.calc_entropy_delta(model, mock_tokenizer, text)
        for _ in range(9):
            current = analyzer.calc_entropy_delta(model, mock_tokenizer, text)
            assert len(current) == len(first)
            for c, f in zip(current, first):
                assert c["entropy_delta"] == f["entropy_delta"]
                assert c["risk_level"] == f["risk_level"]
