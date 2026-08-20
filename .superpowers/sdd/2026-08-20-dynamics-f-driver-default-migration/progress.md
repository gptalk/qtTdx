# SDD ledger — plan: docs/superpowers/plans/2026-08-20-dynamics-f-driver-default-migration.md

## Identity

- Plan: docs/superpowers/plans/2026-08-20-dynamics-f-driver-default-migration.md
- Spec: docs/superpowers/specs/2026-08-20-dynamics-f-driver-default-migration.md
- Base commit: c9e6188 (V0.2-F spec committed)
- Working tree: main branch (no worktree; user pattern is push-to-main per CLAUDE.md)
- Workspace: .superpowers/sdd/2026-08-20-dynamics-f-driver-default-migration/

## Phase order

```
Task 1: Pre-migration audit + reference snapshot
Task 2: Modify projection_batch.py (default flip + flag rename + docstring)
Task 3: Modify v0_2_c1_market_swap.py orchestrator (drop --index)
Task 4: Tests audit + 1 new default-driver test
Task 5: Re-run C1 + byte-identical reproducibility verification
Task 6: Memory entry + MEMORY.md update + push
```

## Caller audit (--market-baseline usage in code, excluding spec/plan docs)

| File | Line | Usage |
|---|---|---|
| backtrace/projection/projection_batch.py | 9, 15, 43, 68, 86, 190 | definition + examples + argparse |
| backtrace/dynamics/dynamics_1step_oos.py | 54, 212 | own flag, `prefer_industry = not args.market_baseline` |
| backtrace/dynamics/dynamics_batch.py | 16, 92, 317 | own flag |
| backtrace/dynamics/dynamics_state_backtest.py | 57 | own flag |

**V0.2-F scope**: only `backtrace/projection/projection_batch.py`. Dynamics scripts retain
their own `--market-baseline` flag (out of scope per user choice). Follow-up V0.2-G
may unify them if user wants consistency.

## Status

- Task 1: complete
- Task 2: complete
- Task 3: pending
- Task 4: pending
- Task 5: pending
- Task 6: pending

### Task 1: Pre-migration audit + reference snapshot — complete

- Implementer: v0.2-f-task1-implementer
- Status: DONE_WITH_CONCERNS
- Reference snapshot: data/projection_v01_c1_v0_2_c1_reference/ created, byte-identical to source (filecmp OK)
- Pre-migration test count: 160 passed in 32.38s (brief expected 128; +32 tests since V0.2-C1 commit 3e60882 — all pass)
- Caller audit confirmed: progress.md has table listing 4 callers (1 in scope = projection_batch.py; 3 out-of-scope = dynamics_*)
- Note: brief's expected test count of 128 is stale; actual is 160. Not blocking — Task 5 will use 160 as new baseline.
- Commit: a976ec5 (docs(projection): V0.2-F pre-migration audit + caller inventory)
- Reviewer verdict: APPROVED with 2 minor (deferred)
  - M1: commit hash placeholder (FIXED in this commit)
  - M2: brief template should record actual test baseline at write-time, not assume past commit's count (parked for future plans)

### Task 2: Modify projection_batch.py (default flip + flag rename + docstring) — complete

- Implementer: v0.2-f-task2-implementer
- Status: DONE_WITH_CONCERNS
- parse_args: --market-baseline replaced with --industry (action='store_true', default False, inverted help text)
- main(): prefer_industry = args.industry (was: not args.market_baseline)
- docstring: 5 locations updated (lines 7, 9, 15-16, 44, 69, 87-88 per caller audit; final grep -c '--market-baseline' = 0)
- Test impact: 160/160 PASS, 0 failed (all 3 test files in the brief's pytest invocation: test_projection_core.py, test_projection_cli.py, test_dynamics_eigen.py)
- Concern: brief predicted failures because tests would use --market-baseline as CLI arg, but in reality the existing test suite calls process_one(..., prefer_industry=...) directly with explicit booleans — the parse_args path is never exercised. The brief's concern is moot for the test files listed; if Task 4's audit finds scripts that pass --market-baseline as a CLI arg, those will need fixing.
- Commit: 4a17ef9
- Reviewer verdict: APPROVED (self-review; subagent reviewer 429 quota-exceeded, controller did inline review)
  - Diff is clean: 27 lines, 1 file (155 line diff), all 5 docstring locations coherent
  - parse_args: `--industry` flag with inverted help text ✓
  - main(): `prefer_industry = args.industry` flip ✓
  - grep -c '--market-baseline' backtrace/projection/projection_batch.py = 0 ✓
  - 160/160 PASS; no math files touched; no deprecation alias added (hard break semantics) ✓
