"""
Auto-generated mock for model-loader based on API_CONTRACT.yaml
Do not edit manually. Run /mock-gen to regenerate.
Behavior overrides: see MOCK_BEHAVIOR.md

为 entropy-analyzer 和 norm-scanner 提供可注入的轻量模型桩。
MockModel 和 MockTokenizer 模拟 HuggingFace 模型/分词器的最小接口，
支持可控的 attention 矩阵和 hidden states 输出，使算法测试无需加载真实模型。
"""

import struct
import zlib
import torch
import torch.nn as nn
from typing import Optional


class MockConfig:
    """模拟 HuggingFace 模型配置"""

    def __init__(self, num_hidden_layers: int = 6, hidden_size: int = 8, num_attention_heads: int = 4):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


class MockOutput:
    """模拟 HuggingFace model forward 输出，包含 attentions 和 hidden_states"""

    def __init__(
        self,
        attentions: Optional[tuple[torch.Tensor]] = None,
        hidden_states: Optional[tuple[torch.Tensor]] = None,
    ):
        self.attentions = attentions
        self.hidden_states = hidden_states


class MockModel(nn.Module):
    """模拟 HuggingFace 因果语言模型。

    核心改进（基于二审反馈）：
    - 支持**逐层**注意力模式设定（H1），不再所有层相同
    - hidden states 使用确定性生成器（H2），同一输入永远同一输出
    - 校验 output_attentions / output_hidden_states 开关（M5），
      缺失开关时返回 None，模拟真实模型行为

    用法：
        model = MockModel(num_layers=6, seq_len=10, num_heads=4, hidden_dim=8)
        # 设置特定层的注意力模式
        model.set_layer_attention(5, "focused")   # 最后一层低熵
        model.set_layer_attention(3, "uniform")   # 倒数第三层高熵
        # 或全部层使用同一模式（全局默认）
        model.set_attention_pattern("uniform")
        # 指定弱信号块
        model.set_norm_weak_chunks([1])
        outputs = model(input_ids=inputs,
                        output_attentions=True,
                        output_hidden_states=True)
    """

    def __init__(
        self,
        num_layers: int = 6,
        seq_len: int = 10,
        num_heads: int = 4,
        hidden_dim: int = 8,
    ):
        super().__init__()
        self.config = MockConfig(num_hidden_layers=num_layers, hidden_size=hidden_dim)
        self._num_layers = num_layers
        self._seq_len = seq_len
        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        # 逐层注意力模式：None 表示使用全局默认
        self._layer_patterns: dict[int, str] = {}
        self._global_pattern: str = "uniform"
        self._norm_weak_chunks: list[int] = []
        # 用于确定性生成的固定种子（派生自 layer_idx）
        self._rng = torch.Generator()
        # hidden 模式: "normal" | "zero" | "huge"（F2/F3 修复）
        self._hidden_pattern: str = "normal"

    def set_hidden_pattern(self, pattern: str):
        """设置 hidden states 生成模式（F2/F3 修复）。

        - "normal": 种子化 randn*0.1（默认）
        - "zero": 全零张量（测试表征坍塌检测）
        - "huge": 常数 1e6（测试溢出防御）
        """
        assert pattern in ("normal", "zero", "huge"), f"未知 hidden pattern: {pattern}"
        self._hidden_pattern = pattern

    def set_attention_pattern(self, pattern: str):
        """设置全局注意力模式（所有层生效，除非被逐层覆盖）。

        可选值：
        - "uniform": 均匀分布 → 高熵（所有位置等概率）
        - "focused": 集中在位置 0 → 低熵（确定性最强）
        """
        self._global_pattern = pattern

    def set_layer_attention(self, layer_idx: int, pattern: str):
        """设置特定层的注意力模式，覆盖全局默认（H1 修复）。

        Args:
            layer_idx: 层序号（从 0 开始）
            pattern: "uniform" | "focused"
        """
        self._layer_patterns[layer_idx] = pattern

    def _get_pattern(self, layer_idx: int) -> str:
        """获取指定层的实际注意力模式"""
        return self._layer_patterns.get(layer_idx, self._global_pattern)

    def set_norm_weak_chunks(self, chunk_indices: list[int], chunk_size: int = 4):
        """设置哪些 chunk 的 hidden state 范数应偏低"""
        self._norm_weak_chunks = chunk_indices
        self._norm_chunk_size = chunk_size

    def _make_attention(self, layer_idx: int, batch: int, seq: int) -> torch.Tensor:
        """为指定层构造注意力矩阵"""
        num_heads = self._num_heads
        pattern = self._get_pattern(layer_idx)

        if pattern == "focused":
            attn = torch.zeros((batch, num_heads, seq, seq))
            attn[:, :, :, 0] = 1.0
        else:  # "uniform" 或未知模式
            attn = torch.full((batch, num_heads, seq, seq), 1.0 / seq)
        return attn

    def _make_hidden_state(self, layer_idx: int, batch: int, seq: int) -> torch.Tensor:
        """为指定层构造确定性 hidden state（H2 修复）。

        使用固定 seed（layer_idx 派生），确保同一输入永远同一输出，
        且不影响全局 RNG。支持零/超大模式（F2/F3 修复）。
        """
        hidden_dim = self._hidden_dim
        pattern = self._hidden_pattern

        if pattern == "zero":
            return torch.zeros(batch, seq, hidden_dim)

        if pattern == "huge":
            return torch.ones(batch, seq, hidden_dim) * 1e6

        # "normal" 模式
        seed = 42 + layer_idx * 7
        self._rng.manual_seed(seed)
        h = torch.randn(batch, seq, hidden_dim, generator=self._rng) * 0.1

        # 如果某 chunk 被标记为弱信号，将其范数压到接近 0
        if self._norm_weak_chunks:
            chunk_size = getattr(self, "_norm_chunk_size", 4)
            for ci in self._norm_weak_chunks:
                start = ci * chunk_size
                end = min(start + chunk_size, seq)
                self._rng.manual_seed(seed + 1000 + ci)
                h[:, start:end, :] = torch.randn(
                    batch, end - start, hidden_dim, generator=self._rng
                ) * 0.01

        return h

    def forward(self, input_ids=None, output_attentions=False, output_hidden_states=False, **kwargs):
        """前向传播（M5 修复：校验 output_* 开关）。

        未收到 output_attentions=True 时返回 attentions=None，
        未收到 output_hidden_states=True 时返回 hidden_states=None。
        """
        if input_ids is not None:
            batch, seq = input_ids.shape
        else:
            batch, seq = 1, self._seq_len

        # Mock 不模拟截断（M-1 修复：截断属于分析器/编排层职责，Mock 如实反映输入 seq）
        # 见 PromptLinter.analyze() 中的截断逻辑

        num_layers = self._num_layers

        # 根据 output_* 开关决定是否构造对应输出（M5）
        attentions = None
        hidden_states = None

        if output_attentions:
            attentions = []
            for layer_idx in range(num_layers):
                attentions.append(
                    self._make_attention(layer_idx, batch, seq)
                )

        if output_hidden_states:
            hidden_states = []
            for layer_idx in range(num_layers):
                hidden_states.append(
                    self._make_hidden_state(layer_idx, batch, seq)
                )

        return MockOutput(
            attentions=tuple(attentions) if attentions is not None else None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )


class MockTokenizer:
    """模拟 HuggingFace Tokenizer。

    核心改进（基于二审反馈 M4）：
    - 使用 zlib.crc32 确定性 hash 替代 Python hash()，跨进程稳定
    - convert_ids_to_tokens 反映真实 token id（不再返回假 tok_0..tok_n）
    - 支持固定 ids 注入（用于已知输出测试）
    """

    def __init__(self, vocab_size: int = 500, max_length: int = 32768):
        self.vocab_size = vocab_size
        self.model_max_length = max_length
        self._fixed_ids: Optional[list[int]] = None

    def set_fixed_ids(self, ids: list[int]):
        """设置固定的 input_ids 输出（用于确定性测试）"""
        self._fixed_ids = ids

    def __call__(self, text: str, return_tensors=None, **kwargs):
        if self._fixed_ids:
            ids = self._fixed_ids
        else:
            # 使用 zlib.crc32 替代 Python hash()（M4 修复）
            # crc32 跨进程、跨运行始终一致
            ids = []
            for c in text:
                h = zlib.crc32(c.encode("utf-8")) & 0x7FFFFFFF
                ids.append(h % self.vocab_size)
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        # 反映真实的 token id（M4 修复）
        return [f"id_{tid}" for tid in ids]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def decode(self, token_ids: list[int]) -> str:
        """模拟 decode：逐个 id 查 token 后拼接。"""
        tokens = [f"id_{tid}" for tid in token_ids]
        return self.convert_tokens_to_string(tokens)


class MockModelLoader:
    """模拟 ModelLoader，提供 load_model / load_tokenizer 接口。

    实现 API_CONTRACT.yaml 中 model-loader service 的契约。
    """

    def __init__(self, model_name: str = "mock/Qwen2.5-0.5B"):
        self._model_name = model_name
        self._model: Optional[MockModel] = None
        self._tokenizer: Optional[MockTokenizer] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> tuple[MockModel, MockTokenizer]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        self._model = MockModel(num_layers=6, seq_len=10, num_heads=4, hidden_dim=8)
        self._tokenizer = MockTokenizer()
        return self._model, self._tokenizer

    def load_model(self) -> dict:
        model, _ = self.load()
        return {
            "status": "loaded",
            "model_name": self._model_name,
            "model_size": "0.5B",
        }

    def load_tokenizer(self) -> dict:
        _, tokenizer = self.load()
        return {
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
        }

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def unload(self):
        self._model = None
        self._tokenizer = None
