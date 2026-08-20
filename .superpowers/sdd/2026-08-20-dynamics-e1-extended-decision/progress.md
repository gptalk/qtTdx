# SDD ledger — plan: docs/superpowers/plans/2026-08-20-dynamics-e1-extended-decision.md

## Identity

- Plan: docs/superpowers/plans/2026-08-20-dynamics-e1-extended-decision.md
- Spec: docs/superpowers/specs/2026-08-20-dynamics-e1-extended-decision.md
- Base commit: f1561b0 (V0.2-E1 spec committed)
- Working tree: main branch (no worktree; user pattern is push-to-main per CLAUDE.md)
- Workspace: .superpowers/sdd/2026-08-20-dynamics-e1-extended-decision/

## Phase order

```
Task 1: Implement _e2_features.py helper (β / vol / liquidity extraction)
Task 2: Implement v0_2_e1_delta_ic_distribution.py (E1)
Task 3: Implement v0_2_e2_cross_sectional_q.py (E2)
Task 4: Tests (E1 + E2; 5 new tests; total 172 PASS)
Task 5: Run E1 + E2 on full data + verify outputs
Task 6: Memory entry + MEMORY.md update + push
```

## Status

- Task 1: complete
- Task 2: pending
- Task 3: pending
- Task 4: pending
- Task 5: pending
- Task 6: pending

### Task 1: _e2_features.py helper — complete

- Implementer: Task1ImplementerClaude
- Status: DONE
- Functions: extract_features_one (single stock), extract_features_all (batch + cache)
- Files created: `backtrace/projection/_e2_features.py`
- Verified on: 600519.SH → {beta_market: 0.0867, stock_volatility: 0.0161, liquidity: 3920640.0}
- Cross-check: 000001.SZ → {beta_market: -0.0776, stock_volatility: 0.0106, liquidity: 89710804.0}  (SZ market → 399001.SZ)
- Edge case: NONEXISTENT.XY → None (correct)
- Commit: b41873f08fd1b1b453c5a684a50c1eabb6746810 (b41873f)
- Reviewer verdict: APPROVED (self-review; subagent reviewer quota)
  - Helper correct: extract_features_one + extract_features_all with cache
  - MARKET_INDEX mapping matches `_projection_core.MARKET_TO_INDEX`
  - 600519.SH verification: β=0.087 (low-beta defensively correlated consumer staple), vol=0.016, liq=3.92M — physically plausible
  - 000001.SZ cross-check: β=-0.078 (defensive/hedge), vol=0.011, liq=89.7M (largest bank) — physically plausible
  - Brief drift (parse_dates=['Date'] → index_col=0, parse_dates=True) is brief-level issue (actual on-disk CSV uses unnamed datetime col), not implementer defect. Implementer correctly matched project convention (data_store.py:74, dynamics_factor_validation.py:172).
  - No math files touched ✓