# Task Spec — 语义一致性专项修复 (F1-F11)

## Goal
Implement all 11 findings from semantic consistency review (P0-P4).

## Scope
- tests/mocks/model_loader_mock.py (add set_hidden_pattern, num_attention_heads)
- tests/test_entropy_analyzer.py (F4/F5/F6/F7/F8/F9/F11)
- tests/test_norm_scanner.py (F1/F2/F3/F4/F10)
- docs/adr/ADR-002-core-detection-algorithm.md (F1 operator)

## Success Criteria
1. F1: percentile strict < operator, ADR-002 updated
2. F2: test_all_zero_hidden_states injects real zeros
3. F3: test_extreme_norm_values injects 1e6 scale
4. F4: mismatch tests construct real mismatch
5. F5: wrong_attention_shape uses real mismatch
6. F6: high_threshold_boundary verifies strict > with exact threshold
7. F7: medium risk verified through full pipeline
8. F8: whitespace/negative/zero threshold tests fixed
9. F9: test_two_layer_model comment corrected
10. F10: test_insufficient_layers_fallback comment corrected
11. F11: test_single_char_input isolates short-text path

## Verification Gate
All files parse. No remaining `except Exception: pass`.
