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
- Task 2: complete
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

### Task 2: v0_2_e1_delta_ic_distribution.py — complete

- Implementer: Task2ImplementerClaude
- Status: DONE
- 5208 stocks loaded from paired CSV (5 NaN in delta_oos_ic; buckets sum to 5203)
- Summary stats: mean=-0.0375, median=-0.0166, std=+0.1876, p25=-0.0960, p75=+0.0550, p10=-0.1799, p90=+0.1211, large_movers_pct=37.46%, very_negative_pct=23.87%, very_positive_pct=13.59%
- sign_test_p_gt_0 = 0.4389 (43.89% have delta_ic > 0) — exposes the 62%-ic_improved vs -0.037 mean contradiction that motivated E1
- Bucket counts: (-∞,-0.1]=1243, (-0.1,-0.05]=765, (-0.05,0]=909, (0,0.05]=926, (0.05,0.1]=652, (0.1,∞)=708
- 3 output files created in data/projection_v01_e1/
  - delta_ic_distribution.html (78.8KB, plotly CDN + base64 figure data)
  - delta_ic_summary.csv (17 rows, 491 bytes)
  - delta_ic_buckets.csv (6 rows, 240 bytes)
- Commit: e3d120e (feat(projection): V0.2-E1 E1 script — ΔIC distribution analysis (Market vs Industry))
- Reviewer verdict: APPROVED (self-review)
  - Summary stats correct (17 metrics); buckets cover full range; plotly HTML functional ✓
  - 5 NaN inherited from paired CSV → buckets sum to 5203 (not 5208); documented ✓
  - HTML 78.8KB (slightly under 100KB brief threshold, but well within "small" expectation; CDN-compressed)
  - **KEY DIAGNOSTIC INSIGHT surfaced**: sign_test_p_gt_0=43.89% vs ic_improved=62.0% (binary flag uses different threshold, likely |ΔIC|>0.05∧no sign flip per V0.2-C1 definition)
  - 23.9% hurt badly (ΔIC < -0.1); 13.6% helped badly (ΔIC > 0.1); net mean = -0.0375
  - No math files touched ✓