"""
PromptLinter — Prompt 注意力诊断器主入口

统一编排模型加载、熵变率分析、范数扫描三个模块。
对外提供 analyze() 接口供 UI 层调用。
"""

import logging
import time
from typing import Optional

from .model_loader import ModelLoader, ModelLoadError, OOMError
from .entropy_analyzer import EntropyAnalyzer, RISK_HIGH, RISK_MEDIUM, RISK_LOW
from .norm_scanner import NormScanner

logger = logging.getLogger(__name__)


class InputEmptyError(ValueError):
    """输入为空"""


class InputTooLongError(ValueError):
    """输入超过模型最大长度"""


class AnalysisError(RuntimeError):
    """分析过程内部错误"""


class PromptLinter:
    """Prompt 注意力诊断器主类。"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        entropy_high_threshold: float = 2.0,
        entropy_medium_threshold: float = 1.5,
        norm_chunk_size: int = 128,
        norm_weak_percentile: float = 15,
    ):
        self._model_loader = ModelLoader(model_name=model_name)
        self._entropy_analyzer = EntropyAnalyzer(
            high_risk_threshold=entropy_high_threshold,
            medium_risk_threshold=entropy_medium_threshold,
        )
        self._norm_scanner = NormScanner(
            chunk_size=norm_chunk_size,
            weak_percentile=norm_weak_percentile,
        )
        self._model = None
        self._tokenizer = None

    def _ensure_model_loaded(self):
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = self._model_loader.load()

    def analyze(
        self,
        text: str,
        chunk_size: Optional[int] = None,
    ) -> dict:
        """统一分析入口。

        对输入文本同时执行注意力熵变率检测和隐藏态范数扫描。

        Args:
            text: 输入文本
            chunk_size: 范数扫描的块大小，默认使用实例配置

        Returns:
            dict: { token_risks, chunk_risks, metadata }

        Raises:
            InputEmptyError: 输入为空
            InputTooLongError: 输入超过模型最大长度
            AnalysisError: 分析过程内部错误
        """
        if not text or not text.strip():
            raise InputEmptyError("输入文本不能为空")

        start_time = time.time()
        self._ensure_model_loaded()

        # 检查输入长度（使用模型的 max_position_embeddings 而非 tokenizer 的 model_max_length）
        encoded = self._tokenizer(text, return_tensors="pt")
        total_tokens = encoded["input_ids"].shape[1]
        max_length = self._model.config.max_position_embeddings
        if total_tokens > max_length:
            raise InputTooLongError(
                f"输入长度（{total_tokens} tokens）超过模型最大长度（{max_length} tokens）"
            )

        try:
            token_risks = self._entropy_analyzer.calc_entropy_delta(
                self._model, self._tokenizer, text
            )
            chunk_risks = self._norm_scanner.scan_signal_strength(
                self._model, self._tokenizer, text, chunk_size=chunk_size
            )
        except Exception as e:
            raise AnalysisError(f"分析过程出错: {e}") from e

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "token_risks": token_risks,
            "chunk_risks": chunk_risks,
            "metadata": {
                "model_name": self._model_loader.model_name,
                "total_tokens": int(total_tokens),
                "analysis_time_ms": round(elapsed_ms, 2),
            },
        }

    @property
    def is_ready(self) -> bool:
        return self._model_loader.is_loaded()

    def unload_model(self):
        self._model = None
        self._tokenizer = None
        self._model_loader.unload()

    def run_ablation_on_70B(
        self, text: str, target_model: str = "", endpoint: str = ""
    ) -> dict:
        """【预留】未来付费版接口 — 在主模型上运行段落消融测试。"""
        raise NotImplementedError(
            "段落消融测试是付费版功能，当前版本仅提供占位桩。"
        )
