"""
隐藏态范数扫描模块 (Norm Scanner)

核心算法：提取模型倒数第三层的 Hidden State，计算每个 Token 向量的 L2 范数。
按文本块（Chunk）聚合，找出低于全局 15% 分位数的"表征洼地"。

物理意义：范数极低的块，在模型内部被"压缩坍塌"，无论 Attention 是否关注它，
它都已失去语义激活强度。
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 默认参数
DEFAULT_CHUNK_SIZE = 128
WEAK_PERCENTILE = 15


class NormScanner:
    """隐藏态范数扫描器。

    分析长文本中每个文本块的信号强度，识别被模型"压缩坍塌"的语义洼地。
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        weak_percentile: float = WEAK_PERCENTILE,
    ):
        self.chunk_size = chunk_size
        self.weak_percentile = weak_percentile

    def scan_signal_strength(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        chunk_size: int | None = None,
    ) -> list[dict]:
        """扫描隐藏态范数，定位表征洼地。

        Args:
            model: 已加载的 HuggingFace 模型（需 output_hidden_states=True）
            tokenizer: 对应的 tokenizer
            text: 输入长文本
            chunk_size: 块大小（Token 数），默认使用实例的 chunk_size

        Returns:
            list[dict]: 每个文本块的分析结果
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        # 防御：负数和零 chunk_size 兜底
        if chunk_size <= 0:
            chunk_size = 1

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]

        if seq_len == 0:
            return []

        # 短文本：chunk_size 超过 seq_len 时，整段作为一个块
        if chunk_size > seq_len:
            chunk_size = seq_len

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
            )

        # 层数不足 3 时无法取 [-3]，直接返回空块列表
        if len(outputs.hidden_states) < 3:
            logger.warning("模型层数不足 3，无法取 layer[-3]，返回空结果")
            return []

        # 提取倒数第三层的 hidden states
        hidden_states = outputs.hidden_states
        target_layer = hidden_states[-3]  # [batch, seq_len, hidden_dim]
        hidden = target_layer[0]  # [seq_len, hidden_dim]

        # 计算每个 Token 的 L2 范数
        norms = torch.norm(hidden, dim=-1)  # [seq_len]
        norms_np = norms.cpu().numpy()

        # 解码 Token 列表用于文本片段预览
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # 按块聚合
        chunks = []
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_norms = norms_np[start:end]
            # 文本片段
            token_slice = all_tokens[start:end]
            text_snippet = tokenizer.convert_tokens_to_string(token_slice)

            chunks.append({
                "chunk_index": i,
                "start_token": int(start),
                "end_token": int(end),
                "text_snippet": text_snippet[:100],
                "norm_score": float(np.mean(chunk_norms).round(4)),
                "is_weak": False,
            })

        # 计算全局弱信号分位数（严格小于，见 ADR-002 约定）
        if chunks:
            all_means = np.array([c["norm_score"] for c in chunks])
            threshold = np.percentile(all_means, self.weak_percentile)
            for c in chunks:
                c["is_weak"] = bool(c["norm_score"] < threshold)

        return chunks
