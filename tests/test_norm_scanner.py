"""
测试骨架 — norm-scanner（隐藏态范数扫描模块）

AI 生成基础测试 + 已补充的 must-have 边界用例（基于 GLM 5.2 二审反馈）。

注意（L5）：metadata（model_name/total_tokens/analysis_time_ms）属于
PromptLinter 编排层的装配职责，不在本模块测试范围内。
"""

import numpy as np
import torch
import pytest

from model_loader_mock import MockModel, MockTokenizer
from prompt_linter.norm_scanner import NormScanner


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def scanner():
    return NormScanner(chunk_size=4, weak_percentile=15)


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


def _make_model(num_layers=6, seq_len=16, num_heads=4, hidden_dim=8):
    """创建标准 MockModel 辅助函数（L3 修复：统一构造入口）"""
    return MockModel(
        num_layers=num_layers,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
    )


# ── Happy Path ─────────────────────────────────────────────────────

class TestHappyPath:
    """正常输入下的基本功能验证"""

    def test_basic_scan(self, scanner, mock_tokenizer):
        """Happy Path: 正常文本应返回块级分析结果"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        assert len(results) > 0
        for c in results:
            assert "chunk_index" in c
            assert "start_token" in c
            assert "end_token" in c
            assert "text_snippet" in c
            assert "norm_score" in c
            assert "is_weak" in c

    def test_chunk_size_respected(self, scanner, mock_tokenizer):
        """Happy Path: 分块大小应与配置一致，且块连续不重叠（L4 修复）"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        # chunk_size=4, 16 tokens → 4 个块，每块结尾 ≤ 4
        assert len(results) == 4, f"期望 4 个块，得到 {len(results)}"
        for c in results:
            assert c["end_token"] - c["start_token"] <= 4
        # 块应连续不重叠（L4 补充断言）
        for i in range(1, len(results)):
            assert results[i]["start_token"] == results[i - 1]["end_token"], (
                f"块 {i} 起始位置 {results[i]['start_token']} "
                f"不等于前一块结束 {results[i-1]['end_token']}"
            )

    def test_norm_score_is_float(self, scanner, mock_tokenizer):
        """Happy Path: 范数分数应为浮点数"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 4)
        for c in results:
            assert isinstance(c["norm_score"], float)

    def test_is_weak_is_bool(self, scanner, mock_tokenizer):
        """Happy Path: is_weak 应为布尔值"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 4)
        for c in results:
            assert isinstance(c["is_weak"], bool)

    def test_text_snippet_non_empty(self, scanner, mock_tokenizer):
        """Happy Path: 文本片段不应为空"""
        model = _make_model()
        results = scanner.scan_signal_strength(
            model, mock_tokenizer, "hello world test prompt linter demo"
        )
        for c in results:
            assert len(c["text_snippet"]) > 0

    def test_weak_chunk_detection(self, scanner, mock_tokenizer):
        """Happy Path: 指定弱信号块应被正确标记"""
        model = _make_model(seq_len=12)
        model.set_norm_weak_chunks([1], chunk_size=4)
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 12)
        weak_chunks = [c for c in results if c["is_weak"]]
        assert len(weak_chunks) > 0
        # 弱信号块的分数应低于正常块
        for wc in weak_chunks:
            for nc in results:
                if not nc["is_weak"]:
                    assert wc["norm_score"] < nc["norm_score"]


# ── Default Configuration ────────────────────────────────────────

class TestDefaultConfig:
    """默认构造参数验证（M1 修复）"""

    def test_default_chunk_size_is_128(self):
        """默认 chunk_size 应为 128"""
        s = NormScanner()
        assert s.chunk_size == 128

    def test_default_weak_percentile_is_15(self):
        """默认 weak_percentile 应为 15"""
        s = NormScanner()
        assert s.weak_percentile == 15


# ── Chunk Size Configuration ──────────────────────────────────────

class TestChunkSize:
    """分块大小配置"""

    def test_custom_chunk_size(self, scanner, mock_tokenizer):
        """自定义 chunk_size 应生效"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 16, chunk_size=8)
        assert len(results) == 2

    def test_small_chunk_size(self, scanner, mock_tokenizer):
        """小 chunk_size 应产生更多块"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 16, chunk_size=2)
        assert len(results) == 8

    def test_chunk_size_larger_than_input(self, scanner, mock_tokenizer):
        """chunk_size 超过输入长度应自动调整"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 3)
        assert len(results) >= 1

    def test_chunk_size_one(self, scanner, mock_tokenizer):
        """Boundary: chunk_size=1 时每个 Token 一个块"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 8, chunk_size=1)
        assert len(results) == 8
        for c in results:
            assert c["end_token"] - c["start_token"] == 1

    def test_zero_chunk_size(self, scanner, mock_tokenizer):
        """Error: chunk_size=0 应被防御（自动调整为最小值或抛异常）"""
        model = _make_model()
        try:
            results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 8, chunk_size=0)
            # 如果没抛异常，应返回有效结果（可能自动调整为 1）
            assert len(results) >= 1
        except (ValueError, ZeroDivisionError):
            pass  # 允许抛已知异常

    def test_negative_chunk_size(self, scanner, mock_tokenizer):
        """Error: chunk_size 为负数应被防御"""
        model = _make_model()
        try:
            results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 8, chunk_size=-5)
            # 如果没抛异常，应返回有效结果
            assert len(results) >= 1
        except (ValueError, RuntimeError):
            pass


# ── Weak Percentile Configuration ─────────────────────────────────

class TestWeakPercentile:
    """弱信号百分位配置"""

    def test_high_percentile_more_weak(self, mock_tokenizer):
        """高百分位应标记更多弱信号块"""
        strict = NormScanner(chunk_size=4, weak_percentile=50)
        model = _make_model(seq_len=12)
        model.set_norm_weak_chunks([1], chunk_size=4)
        results = strict.scan_signal_strength(model, mock_tokenizer, "x" * 12)
        weak_count = sum(1 for c in results if c["is_weak"])
        assert weak_count >= 1

    def test_low_percentile_fewer_weak(self, mock_tokenizer):
        """低百分位应标记更少弱信号块"""
        lenient = NormScanner(chunk_size=4, weak_percentile=1)
        model = _make_model()
        results = lenient.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        weak_count = sum(1 for c in results if c["is_weak"])
        assert weak_count <= 1

    def test_zero_percentile(self, mock_tokenizer):
        """Boundary: weak_percentile=0 时不应标记任何块为弱信号"""
        zero = NormScanner(chunk_size=4, weak_percentile=0)
        model = _make_model()
        results = zero.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        weak_count = sum(1 for c in results if c["is_weak"])
        assert weak_count == 0, "percentile=0 时无块应低于 0 分位数"

    def test_hundred_percentile(self, mock_tokenizer):
        """Boundary: weak_percentile=100 时阈值=最大值；严格 < 下最大值块不算弱（F1 修复）"""
        all_weak = NormScanner(chunk_size=4, weak_percentile=100)
        model = _make_model()
        results = all_weak.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        weak_count = sum(1 for c in results if c["is_weak"])
        # 严格 < 约定：最大值块本身不算弱 → len-1（假定范数无重复）
        assert weak_count == len(results) - 1, (
            f"严格 < 约定下 percentile=100 应有 len-1 个弱块，实际 {weak_count}"
        )

    def test_negative_percentile(self, mock_tokenizer):
        """Error: 负百分位应被防御（默认 15 或抛异常）"""
        try:
            neg = NormScanner(chunk_size=4, weak_percentile=-10)
            model = _make_model()
            results = neg.scan_signal_strength(model, mock_tokenizer, "x" * 16)
            # 如果没抛异常，至少不崩溃
            assert len(results) > 0
        except (ValueError, RuntimeError):
            pass


# ── Edge Cases ────────────────────────────────────────────────────

class TestEdgeCases:
    """边界输入"""

    def test_single_token_input(self, scanner, mock_tokenizer):
        """单个 Token 的极短输入"""
        model = _make_model(seq_len=1)
        results = scanner.scan_signal_strength(model, mock_tokenizer, "a")
        assert len(results) >= 1
        for c in results:
            assert isinstance(c["norm_score"], float)
            assert isinstance(c["is_weak"], bool)

    def test_less_than_one_chunk(self, scanner, mock_tokenizer):
        """输入不足一个 chunk 时应自动调整（原 TODO）"""
        model = _make_model(seq_len=10)
        results = scanner.scan_signal_strength(model, mock_tokenizer, "a" * 3)
        assert len(results) == 1
        assert isinstance(results[0]["norm_score"], float)

    def test_insufficient_layers_fallback(self, scanner, mock_tokenizer):
        """层数 < 3 时应安全回退（与 entropy test_two_layer_model 对齐，F10 修复）。

        只有 2 层时取 hidden_states[-3] 会 IndexError（tuple 负索引越界），
        scanner 应在层数前守护并回退。
        """
        model = _make_model(num_layers=2, seq_len=4)
        results = scanner.scan_signal_strength(model, mock_tokenizer, "test")
        assert len(results) >= 1
        for c in results:
            assert "chunk_index" in c
            assert "norm_score" in c
            assert "is_weak" in c
            assert isinstance(c["norm_score"], float)
            assert isinstance(c["is_weak"], bool)

    def test_whitespace_input(self, scanner, mock_tokenizer):
        """Edge: 纯空白输入应正常处理"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "   \t\n  ")
        assert len(results) >= 1
        for c in results:
            assert isinstance(c["norm_score"], float)

    def test_unicode_input(self, scanner, mock_tokenizer):
        """Edge: 中文/emoji 混合输入不应崩溃"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "你好 👋 测试 test 123")
        assert len(results) >= 1
        for c in results:
            assert isinstance(c["norm_score"], float)

    def test_exact_chunk_boundary(self, scanner, mock_tokenizer):
        """Edge: 输入长度恰好等于 chunk_size 的整数倍"""
        model = _make_model()
        # chunk_size=4, 输入 12 tokens → 恰好 3 个完整块
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 12)
        assert len(results) == 3
        for c in results:
            assert c["end_token"] - c["start_token"] == 4


# ── Error Handling ────────────────────────────────────────────────

class TestErrorHandling:
    """错误处理与异常路径"""

    def test_output_hidden_states_flag_respected(self, scanner, mock_tokenizer):
        """M5: 未传 output_hidden_states=True 时，MockModel 返回 hidden_states=None。"""
        model = _make_model()
        text = "test"
        outputs = model.forward(
            input_ids=mock_tokenizer(text, return_tensors="pt")["input_ids"],
            output_hidden_states=False,
        )
        assert outputs.hidden_states is None

    def test_all_zero_hidden_states(self, scanner, mock_tokenizer):
        """Edge: 全零 hidden states → 范数全 0、分位数 0，不应崩溃（F2 修复）"""
        model = _make_model(seq_len=4)
        model.set_hidden_pattern("zero")
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 4)
        for c in results:
            assert c["norm_score"] == 0.0, f"全零 hidden 的 norm 应为 0: {c['norm_score']}"
            assert isinstance(c["is_weak"], bool)

    def test_empty_tokenizer_output(self, scanner):
        """Error: tokenizer 返回空 input_ids 时的行为（Med-3 修复，L2 修复：与 entropy 侧一致）"""
        empty_tok = MockTokenizer()
        empty_tok.set_fixed_ids([])
        model = _make_model(seq_len=0)
        try:
            results = scanner.scan_signal_strength(model, empty_tok, "")
            assert len(results) == 0
        except (ValueError, RuntimeError, IndexError) as e:
            # 允许特定类型的已知异常，不允许裸 Exception
            assert len(str(e)) > 0, "异常应有错误信息"

    def test_hidden_state_dim_mismatch(self, scanner, mock_tokenizer):
        """Error: config.hidden_size 与实际张量维度不一致时应自适应（F4/L1 修复）"""
        model = _make_model(hidden_dim=8)
        model.config.hidden_size = 16  # config=16，实际张量=8
        # scanner 应自适应实际张量维度，而非读取 config
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 8)
        assert len(results) > 0, "scanner 应自适应实际 hidden dim"
        # 若 scanner 自适配实际维度，则不应崩溃（此时需用其他方式验证它没读 config）


# ── Numerical Accuracy ────────────────────────────────────────────

class TestNumericalAccuracy:
    """数学计算正确性"""

    def test_norm_computation(self):
        """L2 范数 = sqrt(sum(x_i^2))"""
        v = torch.tensor([[3.0, 4.0]])  # 3-4-5 三角形 → 范数应为 5
        norms = torch.norm(v, dim=-1)
        assert abs(norms[0].item() - 5.0) < 1e-4

    def test_all_chunks_have_norm_scores(self, scanner, mock_tokenizer):
        """所有块都应有 norm_score"""
        model = _make_model()
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 16)
        for c in results:
            assert c["norm_score"] >= 0  # L2 范数非负

    def test_extreme_norm_values(self, scanner, mock_tokenizer):
        """Extreme: hidden state 范数 ~1e6 时不应溢出（F3 修复）"""
        model = _make_model(hidden_dim=8)
        model.set_hidden_pattern("huge")
        results = scanner.scan_signal_strength(model, mock_tokenizer, "x" * 8)
        for c in results:
            assert np.isfinite(c["norm_score"]), f"norm 应为有限值: {c['norm_score']}"
            assert c["norm_score"] > 1e5, f"huge 模式 norm 应 > 1e5: {c['norm_score']}"


# ── Concurrency / Stress ──────────────────────────────────────────
#
# 以下 TODO 为有意推迟的非功能性基准测试，
# 将在 TDD 实现完成后的重构阶段补充。
# 它们不阻碍测试定稿，也不影响核心算法验证。

class TestConcurrency:
    """并发安全与压力"""

    # TODO: 并发安全 — 同时扫描多个文本
    # def test_concurrent_scans(self, scanner, mock_tokenizer):
    #     ...

    # TODO: 压力测试 — 单个超长文本（>32K tokens）的性能与稳定性
    # def test_very_long_text(self, scanner, mock_tokenizer):
    #     ...

    # TODO: 压力测试 — 短时间内大量重复扫描
    # def test_repeated_scans(self, scanner, mock_tokenizer):
    #     ...


# ── Consistency ────────────────────────────────────────────────────

class TestConsistency:
    """确定性与幂等性（H2 修复后应稳定通过）"""

    def test_deterministic_output(self, scanner, mock_tokenizer):
        """同输入应产出同输出（H2 修复：使用种子化 Generator）"""
        model = _make_model()
        text = "test prompt xyz"
        result1 = scanner.scan_signal_strength(model, mock_tokenizer, text)
        result2 = scanner.scan_signal_strength(model, mock_tokenizer, text)
        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert c1["norm_score"] == c2["norm_score"], (
                f"确定性被破坏: {c1['norm_score']} != {c2['norm_score']}"
            )
            assert c1["is_weak"] == c2["is_weak"]

    def test_idempotent(self, scanner, mock_tokenizer):
        """Consistency: 相同输入连续调用 10 次，结果应完全一致（H2 保证）"""
        model = _make_model()
        text = "test idempotent scan"
        first = scanner.scan_signal_strength(model, mock_tokenizer, text)
        for _ in range(9):
            current = scanner.scan_signal_strength(model, mock_tokenizer, text)
            assert len(current) == len(first)
            for c, f in zip(current, first):
                assert c["norm_score"] == f["norm_score"]
                assert c["is_weak"] == f["is_weak"]
