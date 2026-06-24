# Task Spec — 修复 GLM 5.2 二审发现的测试缺陷

## Goal
修复 tests/mocks/model_loader_mock.py、tests/test_entropy_analyzer.py、tests/test_norm_scanner.py 中二审指出的 2 High + 7 Medium + 5 Low 问题，然后补充 TODO 占位符为真实测试内容。

## Scope
- tests/mocks/model_loader_mock.py
- tests/test_entropy_analyzer.py
- tests/test_norm_scanner.py
- tests/conftest.py（新建）
- API_CONTRACT.yaml（如需调整措辞）

## Non-goals
- 不涉及生产代码（src/prompt_linter/）
- 不涉及 UI 层（ui/app.py）
- 不涉及架构文档修改

## Success Criteria
1. MockModel 支持逐层可变的注意力模式（H1）
2. MockModel hidden states 使用确定性生成（H2）
3. 默认构造参数被测试覆盖（M1）
4. 假绿测试被修复为真实断言（M2, L4）
5. 截断测试完成（M3）
6. MockTokenizer 确定性且反映真实 token id（M4）
7. 校验型 Mock 验证 output_* 开关（M5）
8. 阈值→风险等级表驱动测试（M6）
9. 规约歧义显式记录在测试 docstring（M7）
10. sys.path 抽到 conftest.py（L1）
11. MockModel 构造统一（L3）
12. metadata 测试覆盖或文档声明范围（L5）
13. 27 个 TODO 中 10 个 must-have 补充为真实测试

## Verification Gate
pytest 全部通过
