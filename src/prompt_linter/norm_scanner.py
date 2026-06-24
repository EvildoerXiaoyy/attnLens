"""
隐藏态范数扫描模块 (Norm Scanner)

核心算法：提取模型倒数第三层的 Hidden State，计算每个 Token 向量的 L2 范数。
按文本块（Chunk）聚合，找出低于全局 15% 分位数的"表征洼地"。

物理意义：范数极低的块，在模型内部被"压缩坍塌"，无论 Attention 是否关注它，
它都已失去语义激活强度。

Pipeline: validate → compute → build → mark
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

    # ── 公开接口 ──────────────────────────────────────────────────

    def scan_signal_strength(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
        chunk_size: int | None = None,
    ) -> list[dict]:
        """扫描隐藏态范数，定位表征洼地。

        Pipeline:
          1. _resolve_chunk_size — 校验并确定分块大小
          2. _compute_norms — 模型推理，提取 L2 范数
          3. _build_chunks — 按块聚合为结果列表
          4. _mark_weak_chunks — 标记弱信号块

        Args:
            model: 已加载的 HuggingFace 模型（需 output_hidden_states=True）
            tokenizer: 对应的 tokenizer
            text: 输入长文本
            chunk_size: 块大小（Token 数），默认使用实例的 chunk_size

        Returns:
            list[dict]: 每个文本块的分析结果
        """
        chunk_size = self._resolve_chunk_size(chunk_size)

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]

        if seq_len == 0:
            return []
        if chunk_size > seq_len:
            chunk_size = seq_len

        norms = self._compute_norms(model, tokenizer, text, seq_len)
        if norms is None:
            return []  # 层数不足

        chunks = self._build_chunks(norms, input_ids, seq_len, chunk_size, tokenizer)
        self._mark_weak_chunks(chunks)
        return chunks

    # ── 输入校验 ───────────────────────────────────────────────────

    def _resolve_chunk_size(self, chunk_size: int | None) -> int:
        """确定实际使用的 chunk_size：参数 → 实例默认 → 最小兜底。"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        return max(chunk_size, 1)  # 负数和零兜底到 1

    # ── 模型推理与范数提取 ────────────────────────────────────────

    def _compute_norms(
        self, model: torch.nn.Module, tokenizer, text: str, seq_len: int
    ) -> np.ndarray | None:
        """执行推理，提取倒数第三层 hidden states 的 L2 范数。

        Returns:
            numpy 数组 [seq_len]，层数不足时返回 None
        """
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
            )

        # ADR-002：层数不足 3 时无法取 layer[-3]
        if len(outputs.hidden_states) < 3:
            logger.warning("模型层数不足 3，无法取 layer[-3]，返回空结果")
            return None

        hidden = outputs.hidden_states[-3][0]  # [seq_len, hidden_dim]
        norms = torch.norm(hidden, dim=-1)  # [seq_len]
        return norms.cpu().numpy()

    # ── 分块聚合 ───────────────────────────────────────────────────

    def _build_chunks(
        self,
        norms: np.ndarray,
        input_ids: torch.Tensor,
        seq_len: int,
        chunk_size: int,
        tokenizer,
    ) -> list[dict]:
        """将范数序列按块聚合为分析结果。"""
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        chunks = []
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            chunk_norms = norms[start:end]

            # 文本片段预览
            token_slice = all_tokens[start:end]
            text_snippet = tokenizer.convert_tokens_to_string(token_slice)

            chunks.append({
                "chunk_index": i,
                "start_token": start,
                "end_token": end,
                "text_snippet": text_snippet[:100],
                "norm_score": float(np.mean(chunk_norms).round(4)),
                "is_weak": False,
            })

        return chunks

    # ── 百分位标记 ─────────────────────────────────────────────────

    def _mark_weak_chunks(self, chunks: list[dict]):
        """标记低于全局弱信号分位数的块（严格小于，见 ADR-002 约定）。"""
        if not chunks:
            return
        all_means = np.array([c["norm_score"] for c in chunks])
        threshold = np.percentile(all_means, self.weak_percentile)
        for c in chunks:
            c["is_weak"] = bool(c["norm_score"] < threshold)
