# Mock Behavior — model-loader

- **Generated for:** entropy-analyzer, norm-scanner（并行 TDD 循环）
- **Based on:** API_CONTRACT.yaml (v0.1-demo)
- **Last updated:** 2026-07-09（修复 H1, H2, M4, M5）

## 默认行为

### MockModel

| 配置方法 | 效果 | 对应修复 |
|---------|------|---------|
| `set_attention_pattern("uniform")` | 所有层均匀注意力分布 → 高熵（全局默认） | — |
| `set_attention_pattern("focused")` | 所有层注意力集中在位置 0 → 低熵 | — |
| `set_layer_attention(5, "focused")` | 仅第 5 层设为 focused，其他层用全局模式 | H1 |
| `set_layer_attention(3, "uniform")` | 仅第 3 层设为 uniform，其他层用全局模式 | H1 |
| `set_norm_weak_chunks([1], chunk_size=4)` | 第 1 个 chunk 的 hidden state 范数被压到接近 0 | — |
| 默认（不调用 setter） | 所有层 uniform 注意力 + 确定性 hidden states | H2 |

**forward 行为（M5 修复）：**
- `output_attentions=True` → 返回 attentions（tuple[Tensor]）
- `output_attentions=False`（缺省）→ 返回 `attentions=None`
- `output_hidden_states=True` → 返回 hidden_states（tuple[Tensor]）
- `output_hidden_states=False`（缺省）→ 返回 `hidden_states=None`

**确定性（H2 修复）：**
- hidden states 使用种子化 `torch.Generator`（`seed = 42 + layer_idx * 7`），同一输入永远同一输出
- 弱信号 chunk 使用独立种子（`seed + 1000 + chunk_idx`）

### MockTokenizer

| 配置方法 | 效果 | 对应修复 |
|---------|------|---------|
| `set_fixed_ids([101, 205, ...])` | 固定返回指定的 token ids | — |
| 默认（不调用） | 使用 zlib.crc32 确定性 hash 生成伪随机 ids | M4 |

**确定性（M4 修复）：**
- 使用 `zlib.crc32(text_char) & 0x7FFFFFFF` 替代 `hash()`，跨进程、跨运行始终一致
- `convert_ids_to_tokens(ids)` 返回 `f"id_{tid}"`，真实反映 token id（不再返回 `tok_0..tok_n`）

### MockModelLoader

| 方法 | 返回值 |
|------|--------|
| `load()` | `(MockModel, MockTokenizer)` |
| `load_model()` | `{status: "loaded", model_name: "mock/Qwen2.5-0.5B", model_size: "0.5B"}` |
| `load_tokenizer()` | `{vocab_size: 500, max_length: 32768}` |
| `is_loaded()` | `False`（首次调用前）/ `True`（调用 load 后） |

## 已知限制

- MockModel 不模拟真实 transformer 的层间依赖（每层的 attention 和 hidden state 独立生成）
- MockModel 不支持 KV cache 或 past_key_values
- MockTokenizer 不做真实的分词（subword tokenization），仅用 crc32 模拟
- 不模拟 `ModelLoadError` 或 `OOMError`（可手动在测试中注入异常）

## 覆盖真实依赖时的注意事项

当真实的 model-loader 模块通过门禁后，应：
1. 替换 `MockModelLoader` → `ModelLoader`
2. 替换 `MockModel` → `AutoModelForCausalLM.from_pretrained(...)`
3. 替换 `MockTokenizer` → `AutoTokenizer.from_pretrained(...)`
4. 在集成测试中验证 Mock attention 模式（uniform/focused）与真实模型浅/深层行为的大致一致性
