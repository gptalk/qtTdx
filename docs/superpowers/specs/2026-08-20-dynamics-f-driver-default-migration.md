# V0.2-F — Driver-Default Migration

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step, after user approves).

**Status:** Draft (awaiting user approval)
**Parent spec:** V0.2-E Integration Decision ([docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md](2026-08-20-dynamics-e-integration-decision.md))
**Type:** Code migration — `projection_batch.py` driver default flips + downstream callers updated

---

## §1 Context

V0.2-E recommended switching the main-line driver from per-stock 申万二级 industry index to per-exchange market index. The recommendation rests on the V0.2-C1 paired diagnostic showing D1 P(|q_drift|>0.3) drops from 10.27% to 3.61% (-65%) under market-driver, with ic_real std concentrating 46% (0.17 → 0.09) and sign_flip held to 2.7%.

`backtrace/projection/projection_batch.py` currently defaults to industry-driver (`prefer_industry = not args.market_baseline`). C1 callers work around this by passing `--index 000001.SH` or `--index 399001.SZ` explicitly per-exchange. V0.2-F enacts V0.2-E by making market-driver the new default and adjusting callers.

---

## §2 Current Behavior (preserved for reference)

`projection_batch.py` arg parser today:

```python
parser.add_argument('--market-baseline', action='store_true',
                    help='回退到大盘基线(SZ→深证成指/SH→上证综指)。默认走行业基线(申万二级)。')
parser.add_argument('--index', default=None, help='强制指定基线指数 ...')
```

`main()` resolves `prefer_industry = not args.market_baseline`. With nothing passed, `prefer_industry=True` → 申万二级 industry per stock.

`v0_2_c1_market_swap.py` works around this by:

```python
for label, stocks_csv, idx in [('SH→000001', sh_csv, '000001.SH'),
                                ('SZ→399001', sz_csv, '399001.SZ')]:
    cmd = [sys.executable, '...projection_batch.py',
           '--input', stocks_csv, '--output-dir', ...,
           '--index', idx, '--movement', '--days', ...]
```

That is, the orchestrator manually splits stocks by exchange and passes the market index explicitly. After V0.2-F this is redundant.

---

## §3 New Behavior

### 3.1 `projection_batch.py` argument surface

| Flag | Default | Old name | New name |
|---|---|---|---|
| `--industry` | NOT set (False) | `--market-baseline` (inverted) | `--industry` (semantic flip) |
| `--index` | `None` (auto-resolve) | unchanged | unchanged |

Default flow (no flag): `prefer_industry=False` → per-exchange market (SH → 000001.SH, SZ → 399001.SZ).

`--industry` flag: `prefer_industry=True` → per-stock 申万二级 industry index (旧默认行为).

`--index CODE`: unchanged — overrides per-stock auto-resolution with explicit single index for all stocks.

`--market-baseline` flag: **deleted** (hard break per user choice; no deprecation period).

### 3.2 Why hard break, not deprecation

Per user choice (rejected soft migration): hard break is correct because (a) V0.2-E is approved and the routing is firm, (b) no downstream production system depends on industry-driver default (verified by usage scan: only the C1 orchestrator and historical V0.2-D baseline calls), (c) keeping both `--market-baseline` and `--industry` as aliases would create two ways to do the same thing and confuse future readers. Hard break forces clean update of all callers.

### 3.3 Numeric reproducibility invariant

V0.2-C1 already ran market-driver on 5208 stocks via the C1 orchestrator's explicit `--index` passing. The outputs are at `data/projection_v01_c1/`. After V0.2-F, **the same numbers must emerge** when re-running with the new default (because the math is unchanged; only the path of reaching it changes). This is the migration's correctness gate.

---

## §4 Scope

| In scope | Out of scope |
|---|---|
| `backtrace/projection/projection_batch.py`: change default; replace `--market-baseline` with `--industry` | Modifying math in `ablation_fit.py` / `_solve_ols` / `prediction_ode.py` / `dynamics_*.py` |
| `backtrace/projection/v0_2_c1_market_swap.py`: drop `--index 000001.SH` / `--index 399001.SZ` from subprocess commands; orchestrator splits by exchange but no longer passes index | `fetch_stock_basic.py` / data layer |
| Tests in `tests/test_projection_*.py`: update fixtures / flags so existing tests still describe industry behavior when needed (using `--industry`); add 1 new test asserting the new default | Adding new diagnostics / panels / HTML |
| `docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md`: appendix note that V0.2-F enacts the routing | V0.2-C.2 two-tier experiment |
| `backtrace/projection/projection_batch.py` docstring: update `--market-baseline` reference → `--industry` | V0.2-D.2 cross-stock analysis |
| Re-run `v0_2_c1_market_swap.py --skip-data-gen --limit 0` after migration; assert output CSV/TXT is byte-identical to V0.2-C1 reference at `data/projection_v01_c1/` | V0.2-B shrinkage |
| Memory entry `projection-v02-f-driver-default-migration.md` | |

---

## §5 Implementation Detail

### 5.1 `projection_batch.py` changes

**`parse_args()`**: replace `--market-baseline` with `--industry`. Help text reversed: "Per-stock 申万二级 industry index. Default is per-exchange market (SH→000001.SH, SZ→399001.SZ)."

```python
parser.add_argument(
    '--industry', action='store_true',
    help=(
        'Per-stock 申万二级 industry index(881xxx). '
        '默认 per-exchange market(SH→000001.SH, SZ→399001.SZ). '
        'V0.2-F 后默认已切到 market-driver, 仅在重跑历史 V0.2-D industry baseline 时需 --industry.'
    ),
)
```

**`main()`**: invert `prefer_industry` logic.

```python
prefer_industry = args.industry   # was: not args.market_baseline
```

**Docstring**: update §1 (auto-resolution ordering) and CLI examples to drop `--market-baseline` reference.

### 5.2 `v0_2_c1_market_swap.py` changes

**`main()`**: drop explicit `--index` from the two subprocess calls. Stock split by exchange remains (so per-exchange pairing still happens via `prefer_industry=False` default, but actually it's irrelevant now — projection_batch does per-exchange pairing itself).

Wait — does `projection_batch.py` do per-exchange pairing internally when `--industry` is NOT set? Yes: line 480-481 logic flips `prefer_industry` and `load_pair` uses that. The per-exchange split comes from `load_pair`'s resolution (it uses `stock_code` prefix to pick `000001` for SH or `399001` for SZ).

So we can simplify even further: the orchestrator no longer needs to pre-split stocks by exchange either. **However**, the orchestrator already splits by exchange for the `--input` files (`_stocks_sh.csv` / `_stocks_sz.csv`), and this split is benign (no harm done if we remove it). For minimal code churn in V0.2-F, keep the split but drop `--index`:

```python
for label, stocks_csv, idx in [('SH→000001', sh_csv, '000001.SH'),    # idx unused now
                                ('SZ→399001', sz_csv, '399001.SZ')]:  # idx unused now
    cmd = [sys.executable, '...projection_batch.py',
           '--input', stocks_csv,
           '--output-dir', args.market_dir,
           '--movement',
           '--days', str(args.days),
]
    # '--index', idx,  # REMOVED (V0.2-F)
```

Or even simpler: drop the loop entirely and call projection_batch once with full stock list (since per-exchange pairing happens internally now). But this changes the data layout — each call writes its own manifest + per-exchange split. Need to verify behavior.

**Decision**: keep the loop for now; just drop `--index`. This minimizes change. Future cleanup pass can collapse the loop if desired.

### 5.3 Test changes

`tests/test_projection_cli.py` and `tests/test_projection_core.py` likely have tests that pass `--market-baseline` (to get market-driver) or that pass nothing (relying on industry default). Migration:

- Tests relying on industry default: add `--industry` flag.
- Tests passing `--market-baseline`: replace with `--industry` flag (or remove if the test was checking default behavior).
- Tests verifying the **default is market-driver**: 1 new test.

New test outline (`test_projection_cli.py`):

```python
def test_default_driver_is_market():
    """V0.2-F: projection_batch.py 默认 driver 是 per-exchange market,
    不是 per-stock 申万二级 industry."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        # Pre-populate a single SH stock + 上证综指 daily
        os.makedirs(os.path.join(td, 'data', 'stocks'))
        os.makedirs(os.path.join(td, 'data', 'indices'))
        # ... create 600000.SH daily + 000001.SH daily + stocks.csv ...
        # Run WITHOUT --industry, expect output CSV's INDEX_TAG column = '000001'
        # NOT '881xxx'
        ...
        assert index_tag == '000001'  # market default
```

### 5.4 Memory entry

`projection-v02-f-driver-default-migration.md` records: spec link, what changed (1-line summary), before/after default behavior, numeric reproducibility invariant verified, follow-up spec hooks.

### 5.5 Re-run + reproducibility check

After code change, run `python backtrace/projection/v0_2_c1_market_swap.py --skip-data-gen --limit 0 --days 240` to produce outputs at `data/projection_v01_c1/` (overwriting). Compare:

```python
import pandas as pd, filecmp
old = pd.read_csv('data/projection_v01_c1_v0_2_c1_reference/kc_estimates_model2_diag.csv')
new = pd.read_csv('data/projection_v01_c1/kc_estimates_model2_diag.csv')
assert old.equals(new), "post-migration output diverged from V0.2-C1 reference"
```

The reference is preserved at a sibling directory `data/projection_v01_c1_v0_2_c1_reference/` (snapshot taken before migration).

---

## §6 Risks

| Risk | Mitigation |
|---|---|
| Existing scripts in `backtrace/legacy/` or `dynamics/` rely on industry default | Audit callers; each gets `--industry` flag if it intentionally wants industry; doc note flags the change |
| Test suite regresses because tests assumed industry default | Run full suite after migration; if any test fails, fix by adding `--industry` to that test (no math change) |
| C1 orchestrator output diverges from V0.2-C1 reference (numeric regression) | Reproducibility gate (§5.5); if divergent, revert and investigate before re-merge |
| `load_pair()` resolution edge cases (newly-listed stocks, special suffixes) | Pre-migration scan: how many stocks in `stock_basic.csv` have non-standard exchange prefix? If > 0, audit `load_pair` |

---

## §7 Deliverables

1. `backtrace/projection/projection_batch.py` modified (default flip + flag rename + docstring)
2. `backtrace/projection/v0_2_c1_market_swap.py` modified (drop `--index` from subprocess calls)
3. Tests in `tests/test_projection_*.py` updated; 1 new default-driver test
4. `data/projection_v01_c1_v0_2_c1_reference/` snapshot (taken before migration for reproducibility check)
5. Memory entry `projection-v02-f-driver-default-migration.md`
6. Numeric reproducibility verification: post-migration `data/projection_v01_c1/` matches reference

---

## §8 Out of Scope

| Out-of-scope | Where |
|---|---|
| Two-tier driver (market + industry) | V0.2-C.2 (if requested) |
| Cross-stock q × industry-β residual | V0.2-D.2 |
| V0.2-B shrinkage | V0.2-B |
| Modify math: `ablation_fit.py`, `_solve_ols`, `prediction_ode.py`, `dynamics_*.py`, `gp_factor_mining/*` | (forbidden) |
| Update `docs/api.md` §projection_batch section | small follow-up (could be done here if scope allows; not strictly required) |
| Update CLAUDE.md / projection/README.md / docs/README.md | small follow-up (could be done here if scope allows) |

---

## §9 Self-Review Checklist

- [x] Placeholder scan: no TBD / TODO
- [x] Internal consistency: §3 (default behavior) matches §5.1 (code change); §5.5 reproducibility invariant matches §3.3
- [x] Scope check: §4 explicit about callers affected and math files frozen
- [x] Ambiguity check: `--industry` semantic is clearly different from old `--market-baseline` (which was an off-by-default flag); both documented
- [x] Numeric reproducibility: §3.3 + §5.5 explicit
- [x] No modifications to math files (declared in §8)
- [x] Tests count: 1 new test for default-driver assertion; existing tests updated to use `--industry` where they relied on old default

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*