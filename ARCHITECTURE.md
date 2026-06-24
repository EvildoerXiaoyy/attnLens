# Prompt 注意力诊断器 (Linter) — 架构蓝图

## 项目定位

一款基于**同源小模型代理**的静态代码检查器（ESLint），通过对 Prompt 文本进行物理级信号扫描（熵与范数），在发送给昂贵大模型之前，量化预测"文本结构脆弱性"与"表征坍塌风险"。

## 核心理念

> **显微镜（测量仪）** 定位，非生成式 AI，非侵入式诊断。

- 不修改用户文本，只输出量化热力图/分数
- **绝对确定性**：同一输入永远输出同一指标
- **白盒归因**：明确指出第 N 个 Token 的数据流不畅

## 系统组件（四模块）

```
┌──────────────────────────────────────────────────────────────────┐
│                       UI Layer (Streamlit)                        │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Tab 1: Prompt 分析页    │  Tab 2: 长文本扫描页           │   │
│  │  - 输入框 + 分析按钮     │  - 文本区 (~10K tokens)        │   │
│  │  - Token 级热力图(Plotly) │  - 块级灰色扫描结果           │   │
│  │  - 高风险 Token 高亮     │  - 数值分数 + 建议           │   │
│  └───────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ 调用
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    PromptLinter (分析引擎)                         │
│                                                                  │
│  ┌────────────────────────┐  ┌────────────────────────────┐     │
│  │  EntropyAnalyzer        │  │  NormScanner               │     │
│  │  - 注意力熵变率计算     │  │  - 隐藏态范数提取          │     │
│  │  - KL 散度/熵差分析     │  │  - 块聚合 + 百分位比较     │     │
│  │  - Token 级风险评分     │  │  - Chunk 级风险评分        │     │
│  └───────────┬────────────┘  └────────────┬───────────────┘     │
│              │             并行             │                      │
│              └──────────┬──────────────────┘                      │
│                         ▼                                        │
│           ┌──────────────────────────┐                           │
│           │    Model Loader          │                           │
│           │  - Qwen2.5-0.5B 加载     │                           │
│           │  - Tokenizer 管理        │                           │
│           │  - Attention/Hidden 输出  │                           │
│           └──────────────────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

## 组件关系

所有组件间通过**同步函数调用**通信，全部在单一 Python 进程内运行，无网络 RPC 或消息队列。

```
UI Layer (Streamlit)
  │
  ├─ analyze(prompt) → {
  │     token_risks: [{token, entropy_delta, risk_level}],
  │     chunk_risks: [{chunk_index, text, norm_score, is_weak}]
  │   }
  │
  ├─ model_loader.get_model() → model
  ├─ model_loader.get_tokenizer() → tokenizer
  │
  ├─ entropy_analyzer.calc_entropy_delta(model, tokenizer, prompt) → token_risks
  └─ norm_scanner.scan_signal_strength(model, tokenizer, text, chunk_size) → chunk_risks
```

## 数据流

### Flow 1: Prompt 熵分析

```
用户输入 Prompt
  → Streamlit 文本输入框
  → PromptLinter.analyze(prompt)
    → Tokenizer 编码
    → Model forward pass (output_attentions=True)
    → 提取 layer[-1] 和 layer[-3] 的 attention 矩阵
    → 取最后一个 Token 对所有历史 Token 的注意力分布 [:, -1, :]
    → 计算每个头的熵，平均到每个层得到一个标量
    → delta = entropy(layer[-1]) - entropy(layer[-3])（单个标量，作用于整个 Prompt）
    → delta > 2.0 → 高风险 (红色)
    → delta > 1.5 → 中风险 (黄色)
  → 返回 token_risks[]
  → Plotly 柱状图渲染
  → 原文高亮标注
  → Streamlit 展示
```

### Flow 2: 长文本范数扫描

```
用户输入长文本
  → Streamlit 文本区
  → PromptLinter.analyze(text)
    → Tokenizer 编码
    → Model forward pass (output_hidden_states=True)
    → 提取 layer[-3] 的 hidden states
    → 计算每个 Token 的 L2 范数
    → 按 chunk_size=128 聚合求平均
    → 计算全局 15% 分位数
    → 标记低于分位数的 chunk 为"弱信号"
  → 返回 chunk_risks[]
  → 灰色块渲染
  → Streamlit 展示
```

## 外部依赖

| 依赖 | 用途 | 失败影响 | 降级策略 |
|------|------|---------|---------|
| HuggingFace Transformers | 模型加载与推理 | 应用无法启动 | 捕获 ImportError，显示安装指引 |
| PyTorch | 张量计算 | 应用无法启动 | 捕获 ImportError，显示安装指引 |
| Qwen2.5-0.5B（HuggingFace） | 代理模型 | 模型加载失败 | 捕获 OSError，提示网络/缓存问题 |
| Streamlit | Web UI 框架 | UI 无法渲染 | N/A（框架层） |
| Plotly | 热力图可视化 | 图表无法渲染 | 回退到 Matplotlib 或纯文本 |

> **关键**：模型下载完成后，运行时**完全离线**，无需网络连接。

## 容错策略

| 场景 | 策略 |
|------|------|
| 模型加载失败 | try-catch OSError → UI 显示"模型加载失败，请检查网络连接和磁盘空间" |
| PyTorch OOM | try-catch RuntimeError → UI 提示"输入过长，请分段分析" |
| 输入为空 | 前端校验，拦截空输入 |
| 输入过长（>模型 max_length） | 截断或分片处理，UI 提示 |
| Tokenizer 编码失败 | try-catch → UI 显示编码错误详情 |

## 关键约束

- **运行环境**：Mac M 系列芯片 / 纯 CPU，无需 GPU
- **内存**：约 1-2GB（加载 0.5B 模型）
- **延迟**：单次分析 < 10 秒（视输入长度）
- **网络**：仅模型下载时需要，运行时可离线
- **模型参数量**：0.5B（Qwen2.5-0.5B）
- **最大输入长度**：Qwen2.5-0.5B 上下文窗口（通常 32K tokens）

## 边界声明（写在 UI 和 README 中）

> 1. **0.5B ≠ 70B**：高熵不代表 70B 会犯错，仅代表该处文本结构不符合 Transformer 浅层认知规律。
> 2. **RAG 归因限制**：范数扫描检测的是"拼接后上下文中的表征洼地"，无法检测向量数据库中的检索漏召。
> 3. **仅限静态分析**：本工具不做推理生成，不涉及模型知识库，无法判断 Prompt 内容的事实正确性。
