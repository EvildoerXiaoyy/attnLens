# TODOs

## 分析耗时预估

- **What**: 分析前根据输入 tokens 数量和模型推理速度估算等待时间，在 UI 中显示"预计等待 X 秒"。
- **Why**: 用户在分析超长文本时不至于以为程序卡死。
- **Context**: /grill-me 已确认 spinner 方案，此 TODO 是进一步 UX 优化。
- **Depends on**: 需要一次 warm-up 推理测得基准速度。

## 分析报告导出

- **What**: 在 UI 中添加下载按钮，将分析结果导出为 JSON 或 HTML 报告。
- **Why**: 方便用户留存和分享诊断结果。
- **Effort**: 约 10-20 行代码（json.dumps + streamlit download_button）。
- **Context**: /plan-eng-review 审查中提出的增量改进。
