# Prompt 注意力诊断器 (attnLens)

> **实验状态：方向性终止** — 详见 [`docs/experiment-report.md`](docs/experiment-report.md)

一款基于 Qwen2.5-0.5B 代理模型的 Prompt 文本静态分析工具。通过提取 Transformer 内部注意力熵变率和隐藏态范数，量化预测文本"结构脆弱性"与"表征坍塌风险"。

## 核心结论

**学术有价值，产品无出路。** 0.5B 模型的浅层物理量无法可靠区分"逻辑谬误"与"复杂指令"，信噪比不足以做自动化诊断。详见[实验报告](docs/experiment-report.md)。

## 项目资产

| 资产 | 路径 |
|------|------|
| 架构文档 | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| 形式化契约 | [`API_CONTRACT.yaml`](API_CONTRACT.yaml) |
| 需求文档 | [`PRD.md`](PRD.md) |
| 架构决策 | [`docs/adr/`](docs/adr/) |
| 测试套件 | [`tests/`](tests/) — 86 个测试全部通过 |
| 实验报告 | [`docs/experiment-report.md`](docs/experiment-report.md) |
| 漂移报告 | [`REconcile-20260709.md`](REconcile-20260709.md) |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -q

# 启动 Streamlit UI
streamlit run ui/app.py
```

首次分析会自动下载 Qwen2.5-0.5B 模型（约 1GB），之后可离线运行。

## 测试

86 个测试覆盖两个核心算法模块：

```bash
pytest tests/ -v --tb=short
```

| 模块 | 测试数 | 说明 |
|------|--------|------|
| model-loader | 18 | 懒加载、生命周期、HF_ENDPOINT 镜像、异常处理 |
| entropy-analyzer | 34 | 逐位置熵差、z-score 异常检测、阈值边界、确定性 |
| norm-scanner | 34 | L2 范数扫描、mean+std 双阈值、分块边界、百分位算子 |

## 项目历程

1. **阶段一（整体规划）**：架构蓝图、PRD、ADR、基线签字
2. **阶段二（测试设计）**：5 轮 GLM 5.2 外部审查，34 项问题全部修复
3. **阶段二（TDD 实现）**：model-loader → entropy-analyzer → norm-scanner → ui-layer
4. **阶段三（实测验证）**：用 10 条病理 Prompt 测试 → 信噪比不足 → **终止**

> 本项目全程使用 **纯 Vibe Coding** 完成，基于 [Reasonix Workflow](https://github.com/EvildoerXiaoyy/reasonix-skills/tree/main) 四层开发工作流（整体规划 → 模块并行 TDD → 集成 → 回顾）驱动，AI Agent 自主完成架构设计、测试编写、代码实现、外部审查全流程。

## 技术栈

- **代理模型**: Qwen2.5-0.5B (HuggingFace Transformers)
- **推理**: PyTorch (CPU / Mac M 系列)
- **UI**: Streamlit + Plotly
- **测试**: pytest + MockModel (确定性 RNG, 逐层注意力)
