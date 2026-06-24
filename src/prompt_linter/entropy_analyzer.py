"""
注意力熵变率分析模块 (Entropy Analyzer)

核心算法：计算每个 Token 位置在深层与浅层之间的注意力熵差，
使用 z-score 异常检测定位"逻辑死结"位置。

物理意义：如果某个位置深层比浅层熵值显著高于相邻位置，
说明该处的句法依赖断裂，模型在"硬猜"。

改进（基于病理 Prompt 测试反馈）：
- 从「只看最后一个 Token」改为「逐位置熵差」
- 从「固定阈值」改为「z-score 异常检测」，
  自适应不同输入长度和模型状态
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 默认阈值
HIGH_RISK_THRESHOLD = 2.0
MEDIUM_RISK_THRESHOLD = 1.5
# z-score 异常检测阈值
Z_HIGH = 2.0
Z_MEDIUM = 1.3

# 风险等级常量
RISK_HIGH = "high"
RISK_MEDIUM = "medium"
RISK_LOW = "low"


class EntropyAnalyzer:
    """注意力熵变率分析器。

    通过比较模型浅层和深层的注意力分布熵值，定位 Prompt 中可能导致
    模型"逻辑死结"的 Token 位置。
    """

    def __init__(
        self,
        high_risk_threshold: float = HIGH_RISK_THRESHOLD,
        medium_risk_threshold: float = MEDIUM_RISK_THRESHOLD,
        z_high: float = Z_HIGH,
        z_medium: float = Z_MEDIUM,
    ):
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.z_high = z_high
        self.z_medium = z_medium

    @staticmethod
    def _entropy(attn_matrix: torch.Tensor) -> torch.Tensor:
        """计算注意力分布的熵。

        Args:
            attn_matrix: 注意力权重矩阵 [num_heads, seq_len] 或 [num_heads]

        Returns:
            对 head 维取平均后的标量熵值
        """
        attn_clamped = torch.clamp(attn_matrix, min=1e-10)
        entropy_per_head = -torch.sum(
            attn_clamped * torch.log(attn_clamped), dim=-1
        )  # [num_heads]
        return entropy_per_head.mean(dim=0)  # []

    def _classify(self, delta: float) -> str:
        """根据绝对 delta 值分类风险等级（保留原逻辑）。"""
        if delta > self.high_risk_threshold:
            return RISK_HIGH
        if delta > self.medium_risk_threshold:
            return RISK_MEDIUM
        return RISK_LOW

    @staticmethod
    def _classify_zscore(z: float) -> str:
        """根据 z-score 绝对值分类风险等级（双向异常检测）。"""
        if abs(z) > Z_HIGH:
            return RISK_HIGH
        if abs(z) > Z_MEDIUM:
            return RISK_MEDIUM
        return RISK_LOW

    def calc_entropy_delta(
        self,
        model: torch.nn.Module,
        tokenizer,
        text: str,
    ) -> list[dict]:
        """计算注意力熵变率（逐位置 + z-score 异常检测）。

        Args:
            model: 已加载的 HuggingFace 模型（需 output_attentions=True）
            tokenizer: 对应的 tokenizer
            text: 输入文本

        Returns:
            list[dict]: 每个 Token 的分析结果
        """
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"][0]
        seq_len = input_ids.shape[0]
        num_layers = model.config.num_hidden_layers

        # 短文本保护
        if seq_len < 2 or num_layers < 3:
            logger.warning("输入过短（%d tokens）或层数不足（%d layers），跳过熵分析", seq_len, num_layers)
            return self._build_fallback_results(tokenizer, input_ids)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                output_hidden_states=False,
            )

        attentions = outputs.attentions
        deep_attn = attentions[-1][0]    # [heads, seq, seq]
        shallow_attn = attentions[-3][0]  # [heads, seq, seq]

        # 逐位置计算熵差
        deltas = []
        for i in range(seq_len):
            de = self._entropy(deep_attn[:, i, :])     # 位置 i 的深层注意力熵
            se = self._entropy(shallow_attn[:, i, :])  # 位置 i 的浅层注意力熵
            deltas.append(round((de - se).item(), 4))

        # z-score 异常检测
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas)) if float(np.std(deltas)) > 1e-8 else 1.0

        # 解码 Token
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        token_texts = [
            tokenizer.decode([tid]) for tid in input_ids.tolist()
        ]

        results = []
        for i, (tid, token_str, token_display) in enumerate(
            zip(input_ids.tolist(), tokens, token_texts)
        ):
            d = deltas[i]
            z = (d - mean_d) / std_d

            # 综合判断：取 z-score 和绝对阈值中更高的风险等级
            risk_z = self._classify_zscore(z)
            risk_abs = self._classify(d)
            risk_level = max(risk_z, risk_abs, key=lambda x: ["low", "medium", "high"].index(x))

            results.append({
                "token": token_str,
                "token_text": token_display.strip(),
                "token_id": tid,
                "entropy_delta": d,
                "z_score": round(z, 4),
                "risk_level": risk_level,
            })

        return results

    def _build_fallback_results(
        self, tokenizer, input_ids: torch.Tensor
    ) -> list[dict]:
        """输入过短时返回安全默认值。"""
        ids_list = input_ids.tolist()
        tokens = tokenizer.convert_ids_to_tokens(ids_list)
        token_texts = [tokenizer.decode([tid]).strip() for tid in ids_list]
        return [
            {
                "token": tok,
                "token_text": txt,
                "token_id": tid,
                "entropy_delta": 0.0,
                "z_score": 0.0,
                "risk_level": RISK_LOW,
            }
            for tid, tok, txt in zip(ids_list, tokens, token_texts)
        ]
