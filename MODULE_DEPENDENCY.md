# 模块依赖顺序

> 由 /arch-workflow 生成，作为阶段二模块循环的串行/并行依据。
> ✅ = 门禁已通过（所有测试通过 + /codereview --assistant 无 blocker）。

## 依赖图

```
┌──────────────┐
│ model-loader │  ← 无依赖，最先开发
└──────┬───────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌──────────────┐  ┌──────────────┐
│entropy-      │  │ norm-        │  ← 无相互依赖，可并行开发
│ analyzer     │  │ scanner      │
└──────┬───────┘  └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌──────────────┐
        │  ui-layer    │  ← 依赖两个分析模块
        └──────────────┘
```

## 依赖顺序

```
1. model-loader ✅
2. entropy-analyzer ✅ (depends on model-loader)
   norm-scanner     ✅ (depends on model-loader)  ← 可与 entropy 并行
3. ui-layer ✅ (depends on entropy-analyzer, norm-scanner)
```

## 模块说明

| 模块 | 职责 | 文件 |
|------|------|------|
| model-loader | 加载 Qwen2.5-0.5B，管理 Tokenizer，提供 model/tokenizer 实例 | `src/prompt_linter/model_loader.py` |
| entropy-analyzer | 注意力熵变率计算，Token 级风险评分 | `src/prompt_linter/entropy_analyzer.py` |
| norm-scanner | 隐藏态范数提取，块聚合，百分位比较 | `src/prompt_linter/norm_scanner.py` |
| ui-layer | Streamlit 页面布局，Plotly 渲染，用户交互 | `ui/app.py` |
