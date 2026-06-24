# Task Spec — 二审 4 Med + 6 Low 修复

## Goal
Fix the 4 Medium + 6 Low issues identified in GLM 5.2 second review of test design v2.

## Scope
- tests/mocks/model_loader_mock.py (add truncation simulation)
- tests/test_entropy_analyzer.py (Med-1/2/3, L1/L4/L6)
- tests/test_norm_scanner.py (Med-3, L2)
- ARCHITECTURE.md (L3)
- MOCK_BEHAVIOR.md (Med-4 documentation fix)

## Success Criteria
1. Med-1: MockModel supports max_length truncation; test asserts len(results) <= max_length
2. Med-2: Full-pipe test verifying risk_level=="medium" via calc_entropy_delta
3. Med-3: No bare `except Exception: pass` — use specific exception types
4. Med-4: Trade-off doc accurately says "Mock tokenizer" not "real tokenizer"
5. L1: Order test uses fixed_ids + token_id match, not self-referential token names
6. L2: Norm insufficient-layers test asserts consistent fallback behavior
7. L3: ARCHITECTURE.md per-token semantics clarified
8. L4: test_basic_entropy_analysis checks delta finiteness + risk_level validity
9. L5: Document metadata scope (module-level vs service-level contract)
10. L6: Note private method coupling (accept as-is)

## Verification Gate
All test files parse correctly. No bare `except Exception: pass` remains.
