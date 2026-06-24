"""
模型加载模块 (Model Loader)

职责：加载 Qwen2.5-0.5B 模型并配置 output_attentions / output_hidden_states。
提供全局唯一的 model 和 tokenizer 实例。
支持 HF_ENDPOINT 环境变量配置镜像源。
"""

import logging
import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"


class ModelLoadError(Exception):
    """模型加载失败时抛出"""


class OOMError(Exception):
    """内存不足时抛出"""


class ModelLoader:
    """管理代理模型的加载与生命周期。

    使用懒加载（lazy loading），首次调用 load() 时下载并加载模型。
    支持 HF_ENDPOINT 环境变量配置 HuggingFace 镜像地址。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._model: Optional[torch.nn.Module] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def load(self) -> tuple[torch.nn.Module, AutoTokenizer]:
        """加载模型和 tokenizer（如果尚未加载则下载）。

        Returns:
            (model, tokenizer) 元组

        Raises:
            ModelLoadError: 下载或加载失败
            OOMError: 内存不足
        """
        if self._model is not None and self._tokenizer is not None:
            logger.info("模型已加载，使用缓存实例")
            return self._model, self._tokenizer

        try:
            logger.info("正在加载 tokenizer: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )

            logger.info("正在加载模型: %s", self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,  # CPU 上使用 float32
                low_cpu_mem_usage=True,
            )

            # 设置为评估模式
            self._model.eval()

            logger.info("模型加载完成: %s", self._model_name)
            return self._model, self._tokenizer

        except torch.cuda.OutOfMemoryError as e:
            raise OOMError(f"GPU 显存不足: {e}") from e
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OOMError(f"内存不足: {e}") from e
            raise ModelLoadError(f"模型加载运行时错误: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"模型加载失败: {e}") from e

    def is_loaded(self) -> bool:
        """模型是否已加载"""
        return self._model is not None and self._tokenizer is not None

    def load_model(self) -> dict:
        """实现 API_CONTRACT 定义的 load_model 方法"""
        model, _ = self.load()
        return {
            "status": "loaded",
            "model_name": self._model_name,
            "model_size": "0.5B",
        }

    def load_tokenizer(self) -> dict:
        """实现 API_CONTRACT 定义的 load_tokenizer 方法"""
        _, tokenizer = self.load()
        return {
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.model_max_length,
        }

    def unload(self):
        """卸载模型，释放内存"""
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型已卸载")
