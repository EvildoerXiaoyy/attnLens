"""
注意力熵变率分析模块 (Entropy Analyzer)

核心算法：计算最后一层（深层）与倒数第三层（浅层）在最后一个 Token（决策出口）
处对历史 Token 注意力分布的熵差。

物理意义：如果深层比浅层熵值突然飙升，说明该处的句法依赖断裂，模型在"硬猜"。

规约说明（M7 修复）：
熵变率算法计算的是「最后一个 Token 的注意力分布熵」在深层与浅层之间的差值。
运算过程：
  1. 取最后一层和倒数第三层 attention 矩阵中最后一个 Token 的分布 [:, -1, :]
  2. 对每层：拆分为 num_heads 个分布，各算熵后取平均 → 该层标量
  3. delta = deep_layer_entropy - shallow_layer_entropy（单个标量）
结果列表与输入 Token 序列等长，但每项携带相同的 delta 值。
delta 作为整个 Prompt 在"决策出口处"句法清晰度的量化信号。
"""

import logging

import torch

logger = logging.getLogger(__name__)

# 默认阈值
HIGH_RISK_THRESHOLD = 2.0
MEDIUM_RISK_THRESHOLD = 1.5


class EntropyAnalyzer:
    """注意力熵变率分析器。

    通过比较模型浅层和深层的注意力分布熵值，定位 Prompt 中可能导致
    模型"逻辑死结"的 Token 位置。
    """

    def __init__(
        self,
        high_risk_threshold: float = HIGH_RISK_THRESHOLD,
        medium_risk_threshold: float = MEDIUM_RISK_THRESHOLD,
    ):
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold

    @staticmethod
    def _entropy(attn_matrix: torch.Tensor) -> torch.Tensor:
        """计算注意力分布的熵。

        Args:
            attn_matrix: 注意力权重矩阵 [num_heads, seq_len]

        Returns:
            每个 Token 的平均熵值 [] — 对 head 维取平均后的标量
        """
        # 防止 log(0)
        attn_clamped = torch.clamp(attn_matrix, min=1e-10)
        # 计算每个 head 的熵: -sum(p * log(p))
        entropy_per_head = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)  # [num_heads]
        # 对所有 head 取平均
        return entropy_per_head.mean(dim=0)  # []

    def _classify(self, delta: float) -> str:
        """根据 delta 值分类风险等级。"""
        if delta > self.high_risk_threshold:
            return "high"
        if delta > self.medium_risk_threshold:
            return "medium"
        return "low"

    def calc_entropy_delta(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
    ) -> list[dict]:
        """计算注意力熵变率。

        Args:
            model: 已加载的 HuggingFace 模型（需 output_attentions=True）
            tokenizer: 对应的 tokenizer
            text: 输入文本

        Returns:
            list[dict]: 每个 Token 的分析结果，格式:
                {
                    "token": str,           # Token 原文
                    "token_id": int,        # Token ID
                    "entropy_delta": float, # 熵差值
                    "risk_level": str,      # "high" | "medium" | "low"
                }
        """
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]
        num_layers = model.config.num_hidden_layers

        # 短文本保护：至少需要 3 层 + 2 个 Token 才能取 layer[-1] 和 layer[-3]
        if seq_len < 2 or num_layers < 3:
            logger.warning("输入过短（%d tokens）或层数不足（%d layers），跳过熵分析", seq_len, num_layers)
            return self._build_fallback_results(tokenizer, input_ids)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
            )

        attentions = outputs.attentions  # tuple of (batch, heads, seq, seq)
        deep_attn = attentions[-1]  # 最后一层
        shallow_attn = attentions[-3]  # 倒数第三层

        # 取最后一个 Token 对所有 Token 的注意力分布
        deep_last = deep_attn[0, :, -1, :]  # [num_heads, seq_len]
        shallow_last = shallow_attn[0, :, -1, :]  # [num_heads, seq_len]

        # 计算每个 head 的熵并取平均
        deep_entropy = self._entropy(deep_last)  # []
        shallow_entropy = self._entropy(shallow_last)  # []

        delta = deep_entropy - shallow_entropy
        delta_val = delta.item()
        risk = self._classify(delta_val)

        # 解码 Token 并构建结果
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        results = []
        for tid, token_str in zip(input_ids.tolist(), tokens):
            results.append({
                "token": token_str,
                "token_id": tid,
                "entropy_delta": round(delta_val, 4),
                "risk_level": risk,
            })

        return results

    def _build_fallback_results(
        self, tokenizer, input_ids: torch.Tensor
    ) -> list[dict]:
        """输入过短时返回安全默认值。"""
        ids_list = input_ids.tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids_list)
        results = []
        for tid, token_str in zip(ids_list, tokens):
            results.append({
                "token": token_str,
                "token_id": tid,
                "entropy_delta": 0.0,
                "risk_level": "low",
            })
        return results
