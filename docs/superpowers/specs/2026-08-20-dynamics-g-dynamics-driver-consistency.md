# V0.2-G — Dynamics Driver Consistency

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step, after user approves).

**Status:** Draft (awaiting user approval)
**Parent spec:** V0.2-F Driver-Default Migration ([docs/superpowers/specs/2026-08-20-dynamics-f-driver-default-migration.md](2026-08-20-dynamics-f-driver-default-migration.md))
**Type:** Code migration — 3 dynamics scripts align their `--market-baseline` flag with the V0.2-F rename

---

## §1 Context

V0.2-F migrated `projection_batch.py` to default per-exchange market-driver and renamed `--market-baseline` → `--industry` (inverted semantic). V0.2-F §8 explicitly listed "dynamics scripts (`dynamics_*.py`)" as out-of-scope to keep the V0.2-F PR focused.

The 3 dynamics scripts independently define their own `--market-baseline` flag with the same `prefer_industry = not args.market_baseline` pattern. After V0.2-F, this creates an inconsistency:

| Script | Default driver | Flag |
|---|---|---|
| `projection_batch.py` (V0.2-F done) | market | `--industry` opt-in to industry |
| `dynamics_1step_oos.py` | **industry** | `--market-baseline` opt-in to market |
| `dynamics_batch.py` | **industry** | `--market-baseline` opt-in to market |
| `dynamics_state_backtest.py` | **industry** | `--market-baseline` opt-in to market |

Running `dynamics_batch.py` without flags today still uses industry-driver — the opposite default from `projection_batch.py`. This is a footgun: a user running the same stock through both pipelines gets different driver behavior depending on which script they use.

V0.2-G aligns all 3 dynamics scripts to the V0.2-F convention (default = market, `--industry` opt-in).

---

## §2 Current Behavior (preserved for reference)

Each of the 3 dynamics scripts has the same pattern as `projection_batch.py` did pre-V0.2-F:

```python
# argparse
p.add_argument('--market-baseline', action='store_true',
               help='全部回退大盘基线(覆盖默认行业基线)。')

# resolution
prefer_industry = not args.market_baseline
```

`prefer_industry` is then passed downstream to `load_pair()` or `process_one()`, identical to `projection_batch.py`'s pre-V0.2-F pattern.

`dynamics_batch.py` line 16 also has a docstring reference to `--market-baseline`. The other 2 scripts' docstrings do not reference it (verified via grep at task scope audit time).

---

## §3 New Behavior

### 3.1 Argument surface (3 scripts × identical change)

| Flag | Old | New |
|---|---|---|
| `--market-baseline` | default = False (industry) | **deleted** |
| `--industry` | (didn't exist) | default = False (market) |

`prefer_industry = args.industry` (was `not args.market_baseline`).

The flag-naming convention is identical to `projection_batch.py` post-V0.2-F. Users running any script without flags now get per-exchange market behavior consistently.

### 3.2 Hard break semantics (consistent with V0.2-F)

Per V0.2-F precedent and V0.2-G user choice:

- **No deprecation period.** `--market-baseline` is deleted outright, not aliased.
- **No warning message.** Users running the old flag get `unrecognized arguments: --market-baseline` from argparse.
- **No fallback heuristic.** The old `--market-baseline` argument has zero equivalent in the new API.

This is a hard break, by user choice, to enforce consistency across all 4 scripts (projection + 3 dynamics).

### 3.3 Verification gate

After code change, run full-market OOS for `dynamics_batch.py` (the most representative of the 3 dynamics scripts; it covers batch processing of all stocks). Compare its output (CSV manifest + per-stock CSV/HTML) to a pre-migration snapshot:

- **Acceptance:** byte-identical or floating-point ε-identical for data files (CSV + TXT). HTML may differ by plotly random UUID (1-line cosmetic diff), same as V0.2-F Task 5.
- **Failure action:** STOP. Revert the 3 commit chain. Investigate before re-merge.

The other 2 dynamics scripts (`dynamics_1step_oos.py`, `dynamics_state_backtest.py`) are smaller / focused on different slices; they don't need full-market re-runs because the math is identical to `dynamics_batch.py` (same `_dynamics_core`).

---

## §4 Scope

| In scope | Out of scope |
|---|---|
| `backtrace/dynamics/dynamics_1step_oos.py`: parse_args, main, docstring (~3-4 locations per grep audit) | Modify math in `_dynamics_core.py`, `_projection_core.py`, `prediction_ode.py`, `parameter_fit.py` |
| `backtrace/dynamics/dynamics_batch.py`: parse_args, main, docstring (~4-5 locations per grep audit) | `projection_batch.py` (already done in V0.2-F) |
| `backtrace/dynamics/dynamics_state_backtest.py`: parse_args, main, docstring (~3-4 locations per grep audit) | `gp_factor_mining/*` |
| Tests in `tests/test_dynamics_*.py`: 1 new default-driver test per script (3 total) | `ablation_fit.py`, `_solve_ols` |
| Snapshot of `dynamics_batch.py` output for reproducibility check | Other dynamics scripts (none beyond the 3) |
| Memory entry `projection-v02-g-dynamics-driver-consistency.md` | |
| Push to origin/main | |

---

## §5 Implementation Detail

### 5.1 Per-script change (apply to all 3)

**parse_args()**: replace `--market-baseline` with `--industry`. Use the same help-text template as V0.2-F `projection_batch.py`:

```python
p.add_argument(
    '--industry', action='store_true',
    help=(
        'Per-stock 申万二级 industry index(881xxx.SH/SZ)。'
        '默认走 per-exchange market 基线(SH→000001.SH / SZ→399001.SZ)。'
        'V0.2-G: 与 projection_batch.py 一致, 默认 market-driver.'
    ),
)
```

**main()**: flip `prefer_industry = not args.market_baseline` → `prefer_industry = args.industry`.

```python
prefer_industry = args.industry   # V0.2-G: 与 projection_batch.py 一致; --industry opt-in
```

**docstring**: replace `--market-baseline` with `--industry` in CLI tables and examples. Mirror V0.2-F's projection_batch.py docstring style.

### 5.2 Test additions (1 new test per script)

For each of the 3 dynamics scripts, append a `test_default_driver_is_market_<script_short_name>` to the appropriate test file:

```python
def test_default_driver_is_market_<script_short_name>(monkeypatch):
    """V0.2-G: dynamics_<short_name> 默认 driver 与 projection_batch.py 一致
    (per-exchange market, --industry opt-in)."""
    monkeypatch.setattr(sys, 'argv', [
        'dynamics_<short_name>.py', '--code', '002475.SZ', '--days', '30',
    ])
    import importlib
    import dynamics_<short_name> as d_mod
    importlib.reload(d_mod)
    args = d_mod.parse_args()
    assert args.industry is False  # market default
    assert not hasattr(args, 'market_baseline')  # hard break
```

3 test additions: 
- `tests/test_dynamics_eigen.py` for `dynamics_1step_oos.py`
- (one file per script — confirm during execution)
- Pattern mirrors V0.2-F `test_default_driver_is_market`

### 5.3 Reproducibility check

Pre-migration snapshot of `dynamics_batch.py` output. Run `dynamics_batch.py --f-self-mode rolling` on full market before any code change; save the manifest CSV + per-stock sample outputs to a gitignored directory. Post-migration re-run; assert byte-identical for data files.

Snapshot location: `data/dynamics_v02_g_v0_2_g_reference/` (sibling of the C1 reference directory).

### 5.4 Memory entry

`projection-v02-g-dynamics-driver-consistency.md` records:
- 3 scripts migrated, consistent with V0.2-F
- Hard break semantics applied (consistent)
- Reproducibility gate: PASSED (or FAILED with diagnosis)
- Caller audit: 3 callers migrated, 0 remaining `--market-baseline` in dynamics/ scope
- Link to V0.2-F memory + V0.2-E decision

---

## §6 Risks

| Risk | Mitigation |
|---|---|
| 1 dynamics script regresses (math changes by accident) | Reproducibility gate on `dynamics_batch.py` full-market re-run |
| Test pre-existing relies on industry default → breaks post-flip | Same as V0.2-F: tests use `process_one(prefer_industry=...)` directly; audit confirms before commit |
| User running legacy pipeline scripts doesn't realize the change | Memory entry + MEMORY.md index; CLI help text mentions V0.2-G migration |
| docs/api.md still references `--market-baseline` for projection_batch.py | Out-of-scope for V0.2-G; spec §8 follow-up; cosmetic only |

---

## §7 Deliverables

1. 3 dynamics scripts modified (parse_args + main + docstring)
2. 3 new default-driver tests in `tests/test_dynamics_*.py`
3. Pre-migration snapshot of `dynamics_batch.py` output (gitignored)
4. Post-migration full-market re-run + byte-identical (or UUID-only HTML diff) verification
5. Memory entry `projection-v02-g-dynamics-driver-consistency.md`
6. MEMORY.md index updated
7. Push to origin/main

---

## §8 Out of Scope

| Out-of-scope | Where |
|---|---|
| `projection_batch.py` (already done in V0.2-F) | n/a |
| Two-tier driver (market + industry joint) | V0.2-C.2 (if requested) |
| Cross-stock q × industry β residual | V0.2-D.2 (if requested) |
| V0.2-B shrinkage | (now even less urgent post V0.2-F + V0.2-G) |
| Modify math: `_dynamics_core.py`, `_projection_core.py`, `prediction_ode.py`, `parameter_fit.py`, `ablation_fit.py` | (forbidden) |
| docs/api.md updates for projection_batch section | (deferred; spec §8 of V0.2-F) |

---

## §9 Self-Review Checklist

- [x] Placeholder scan: no TBD / TODO
- [x] Internal consistency: §3 default behavior matches §5.1 code change; §3.3 verification gate matches §5.3 reproducibility check
- [x] Scope check: §4 explicit about 3 in-scope scripts and math files frozen
- [x] Ambiguity check: flag rename identical to V0.2-F, no semantic drift
- [x] Hard break semantics: §3.2 + §6 consistent
- [x] No modifications to math files (declared in §8)
- [x] Tests count: 3 new tests (1 per script) + existing tests preserved

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*