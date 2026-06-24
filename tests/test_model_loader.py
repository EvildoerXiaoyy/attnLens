"""
测试骨架 — model-loader（模型加载模块）

TDD 驱动：mock HuggingFace from_pretrained 以隔离网络依赖。
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from prompt_linter.model_loader import ModelLoader, ModelLoadError, OOMError


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_hf():
    """Mock HuggingFace from_pretrained 方法"""
    with patch("prompt_linter.model_loader.AutoModelForCausalLM") as mock_model:
        with patch("prompt_linter.model_loader.AutoTokenizer") as mock_tok:
            mock_model.from_pretrained.return_value = MagicMock()
            mock_tok.from_pretrained.return_value = MagicMock()
            mock_tok.from_pretrained.return_value.model_max_length = 32768
            yield (mock_model, mock_tok)


# ── Happy Path ─────────────────────────────────────────────────────

class TestHappyPath:
    """正常使用流程"""

    def test_init_with_default_model(self):
        """默认模型名应为 Qwen/Qwen2.5-0.5B"""
        loader = ModelLoader()
        assert loader.model_name == "Qwen/Qwen2.5-0.5B"

    def test_init_with_custom_model(self):
        """可自定义模型名"""
        loader = ModelLoader(model_name="custom/model")
        assert loader.model_name == "custom/model"

    def test_is_ready_false_before_load(self):
        """加载前 is_loaded 应为 False"""
        loader = ModelLoader()
        assert loader.is_loaded() is False

    def test_load_returns_model_and_tokenizer(self, mock_hf):
        """load() 应返回 (model, tokenizer) 元组"""
        loader = ModelLoader()
        model, tokenizer = loader.load()
        assert model is not None
        assert tokenizer is not None

    def test_load_model_returns_correct_structure(self, mock_hf):
        """load_model() 应返回含 status/model_name/model_size 的 dict"""
        loader = ModelLoader()
        result = loader.load_model()
        assert isinstance(result, dict)
        assert "status" in result
        assert "model_name" in result
        assert "model_size" in result
        assert result["model_name"] == "Qwen/Qwen2.5-0.5B"
        assert result["model_size"] == "0.5B"

    def test_load_tokenizer_returns_correct_structure(self, mock_hf):
        """load_tokenizer() 应返回含 vocab_size/max_length 的 dict"""
        loader = ModelLoader()
        result = loader.load_tokenizer()
        assert isinstance(result, dict)
        assert "vocab_size" in result
        assert "max_length" in result

    def test_is_loaded_true_after_load(self, mock_hf):
        """加载后 is_loaded 应为 True"""
        loader = ModelLoader()
        loader.load()
        assert loader.is_loaded() is True

    def test_unload_sets_not_loaded(self, mock_hf):
        """unload 后 is_loaded 应为 False"""
        loader = ModelLoader()
        loader.load()
        loader.unload()
        assert loader.is_loaded() is False

    def test_unload_before_load_does_not_crash(self):
        """未加载时 unload 不应崩溃"""
        loader = ModelLoader()
        loader.unload()  # should not raise

    def test_load_is_lazy_called_once(self, mock_hf):
        """load() 是懒加载，第二次调用不应重新下载"""
        mock_model_cls, mock_tok_cls = mock_hf
        loader = ModelLoader()
        loader.load()
        loader.load()  # 第二次调用
        assert mock_model_cls.from_pretrained.call_count == 1
        assert mock_tok_cls.from_pretrained.call_count == 1


# ── Model Loading ─────────────────────────────────────────────────

class TestModelLoading:
    """模型加载细节"""

    def test_model_set_to_eval_mode(self, mock_hf):
        """加载后模型应设为 eval 模式"""
        loader = ModelLoader()
        model, _ = loader.load()
        model.eval.assert_called_once()

    def test_model_loaded_with_trust_remote_code(self, mock_hf):
        """from_pretrained 应传 trust_remote_code=True"""
        mock_model_cls, _ = mock_hf
        loader = ModelLoader()
        loader.load()
        _, kwargs = mock_model_cls.from_pretrained.call_args
        assert kwargs.get("trust_remote_code") is True

    def test_model_loaded_with_float32(self, mock_hf):
        """CPU 上应使用 float32"""
        mock_model_cls, _ = mock_hf
        loader = ModelLoader()
        loader.load()
        _, kwargs = mock_model_cls.from_pretrained.call_args
        import torch
        assert kwargs.get("torch_dtype") == torch.float32

    def test_tokenizer_loaded_with_trust_remote_code(self, mock_hf):
        """tokenizer 加载应传 trust_remote_code=True"""
        _, mock_tok_cls = mock_hf
        loader = ModelLoader()
        loader.load()
        _, kwargs = mock_tok_cls.from_pretrained.call_args
        assert kwargs.get("trust_remote_code") is True


# ── Environment Configuration ─────────────────────────────────────

class TestEnvConfig:
    """环境变量配置"""

    def test_hf_endpoint_env_var_respected(self, mock_hf):
        """HF_ENDPOINT 环境变量应传递给 from_pretrained"""
        mock_model_cls, _ = mock_hf
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}):
            loader = ModelLoader()
            loader.load()
            # from_pretrained 内部会读取 HF_ENDPOINT，这里验证调用成功
            mock_model_cls.from_pretrained.assert_called_once()
            args, _ = mock_model_cls.from_pretrained.call_args
            assert args[0] == "Qwen/Qwen2.5-0.5B"


# ── Error Handling ────────────────────────────────────────────────

class TestErrorHandling:
    """错误处理"""

    def test_model_load_error_on_failure(self):
        """from_pretrained 失败时应抛出 ModelLoadError"""
        with patch("prompt_linter.model_loader.AutoModelForCausalLM.from_pretrained",
                   side_effect=Exception("download failed")):
            with patch("prompt_linter.model_loader.AutoTokenizer.from_pretrained",
                       MagicMock()):
                loader = ModelLoader()
                with pytest.raises(ModelLoadError):
                    loader.load()

    def test_oom_error_on_oom(self):
        """CUDA OOM 时应抛出 OOMError"""
        import torch
        with patch("prompt_linter.model_loader.AutoModelForCausalLM.from_pretrained",
                   side_effect=torch.cuda.OutOfMemoryError("CUDA OOM")):
            with patch("prompt_linter.model_loader.AutoTokenizer.from_pretrained",
                       MagicMock()):
                loader = ModelLoader()
                with pytest.raises(OOMError):
                    loader.load()


# ── Consistency ───────────────────────────────────────────────────

class TestConsistency:
    """确定性"""

    def test_same_loader_returns_same_instance(self, mock_hf):
        """同一个 loader 实例多次 load 应返回相同对象"""
        loader = ModelLoader()
        m1, t1 = loader.load()
        m2, t2 = loader.load()
        assert m1 is m2  # 同一模型对象
        assert t1 is t2  # 同一 tokenizer 对象
