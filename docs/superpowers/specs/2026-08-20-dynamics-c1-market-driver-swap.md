# V0.2-C1 — Market Driver Swap Experiment

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step).

**Date:** 2026-08-20
**Status:** Draft (awaiting user approval)
**Parent plan:** V0.2-D (commit `c1226f0`, READY_TO_MERGE)
**Parent verdict:** V0.2-D H1 (regime shift q_drift) plausible; H2/H3 ruled out

---

## §1 Context

V0.2-D's diagnostic surface on **5211 stocks** (industry-driver) revealed:

```
D1 |q_drift|      median=0.120  p75=0.203  P(|x|>0.3)=10.3%  → H1 plausible
D2 |corr_x_beta_d| median=0.034  p75=0.059  P(|x|>0.3)= 0.0%  → H2 ruled out
D3 corr_F_d        median=0.006  p75=0.016  P( x >0.2)= 0.0%  → H3 ruled out
```

**V0.2-C1 (this spec)** answers the natural follow-up question:
> *Is H1 (q regime drift) a property of the dynamics equation itself,
> or is it induced by the choice of driver?*

In V0.2-D's current state, every stock is paired with its **申万二级 industry index**
(881xxx.SH/SZ) — e.g., 000002.SZ (万科) pairs with 881418.SH (房地产开发).
This is the **industry-driver baseline** (C0).

V0.2-C1 swaps the external driver to **market index** per exchange:
- SH stocks (6xxxxx.SH, 9xxxxx.SH, 5xxxxx.SH) → 上证综指 **000001.SH**
- SZ stocks (0xxxxx.SZ, 3xxxxx.SZ) → 深证成指 **399001.SZ**

This is the **market-driver experiment** (C1). One variable changes; everything else
(数学定义、样本边界、参数估计、OOS 划分、placebo、评价指标、决策阈值) stays identical.

---

## §2 Research Question

**If we replace the industry driver with the market driver (per exchange), does
H1 (q regime drift) persist, attenuate, or invert?**

Two non-exclusive sub-hypotheses:

| # | Hypothesis | Diagnostic surface (in C1) |
|---|---|---|
| **H1a** | H1 is **driver-invariant**: q drift is structural to the free-q formulation; persists with both drivers | C1 D1 distribution ≈ C0 D1 distribution |
| **H1b** | H1 is **driver-induced**: industry-driver over-fits because industry dynamics are noisier than market dynamics; market-driver eliminates or attenuates q drift | C1 D1 distribution << C0 D1 distribution (median \|q_drift\| drops, P(\|x\|>0.3) drops) |

V0.2-C1 is **a paired-comparison experiment** between C0 (industry) and C1 (market)
on the same 5211 stocks. It is **NOT** a verdict on which driver is "best" — that
routing decision lives in V0.2-E or the user.

---

## §3 Method — Strict Isolation

**Write-dead rule (per user spec):**
> C1 仅将 Model 2 的 external driver 从 Industry 替换为 Market,除 driver 数据源外,
> 其余数学定义、样本边界、参数估计、OOS 划分、placebo、评价指标及决策阈值全部保持不变。

| Component | C0 (V0.2-D baseline) | C1 (V0.2-C1) |
|---|---|---|
| External driver M | 申万二级 industry (per stock) | Market per exchange (SH→000001, SZ→399001) |
| Stock list | 5211 stocks | **Same** 5211 stocks (paired) |
| Math (4 models, 36-col schema) | V0.2-D | **Identical** |
| Train/test split (70/30) | V0.2-D | **Identical** |
| OOS prediction (Layer 1 only, θ_train → X_test) | V0.2-D | **Identical** |
| Placebo (permuted regressors) | V0.2-D | **Identical** |
| Identification status / fit_quality | V0.2-D | **Identical** |
| Decision thresholds (D1/D2/D3) | V0.2-D | **Identical** |
| Output dir | `data/projection_v01_d/` | `data/projection_v01_c1/` |

**Variables held constant across C0 and C1:**
- All 5211 stocks (paired)
- All 4 models (M0/M1/M2/M3)
- All 36 schema columns for M2/M3 (Group A/B/C/D)
- All decision gates (D1/D2/D3) and report-only thresholds
- All CLI args (only `--movement-dir` and `--output-dir` differ)
- All test fixtures and patterns

---

## §4 Output Schema — Paired Comparison CSV

V0.2-C1 adds ONE new output: a **paired comparison CSV** that joins C0 and C1 per stock
for Model 2 only. C0 and C1 individual CSVs are also kept (so V0.2-D's diagnostic
surface can be re-derived on C1 alone).

### 4.1 `data/projection_v01_c1/kc_estimates_model2_diag.csv` (5211 × 36, identical schema to V0.2-D)

Same as `data/projection_v01_d/kc_estimates_model2_diag.csv`. Same columns, same math.
Only the data behind the columns differs (market-driver).

### 4.2 `data/projection_v01_c1/{panel5, distributions, summary}` (V0.2-D equivalents)

Same as V0.2-D outputs, recomputed for C1. Diagnostic only.

### 4.3 **NEW** `data/projection_v01_c1/c0_c1_paired_compare.csv` (5211 rows, ~22 cols)

Per-stock paired comparison of key metrics between C0 and C1:

```
code, name,
delta_oos_ic        = ic_real_C1 - ic_real_C0
delta_q_drift       = |q_drift_C1| - |q_drift_C0|        (signed by direction)
delta_q_hat         = q_hat_C1 - q_hat_C0
delta_test_fit_r2   = test_fit_r2_C1 - test_fit_r2_C0
delta_oos_r2        = oos_r2_C1 - oos_r2_C0
delta_cond          = condition_number_C1 - condition_number_C0
sign_flipped        = (sign(ic_real_C0) != sign(ic_real_C1))   ← H1b fingerprint
q_drift_attenuated  = (|q_drift_C1| < 0.5 * |q_drift_C0|)     ← H1b attenuation
q_drift_amplified   = (|q_drift_C1| > 1.5 * |q_drift_C0|)     ← H1a invariant
ic_improved         = (|delta_oos_ic| > 0.05 AND sign not flipped)
ic_worsened         = (|delta_oos_ic| < -0.05)
```

(No PASS/FAIL — these are diagnostic flags for the user to interpret.)

### 4.4 `data/projection_v01_c1/c0_c1_compare_summary.txt` (UTF-8 Chinese report)

Per-gate comparison: C0 distribution vs C1 distribution. Same diagnostic-only style
as V0.2-D's `v0_2_d_summary.txt`.

```
============================================================
V0.2-C1 — Market Driver Swap (Paired Comparison)
============================================================
Run date:  2026-08-20

NOTE: This is a diagnostic report. No PASS/FAIL verdicts.
Interpretation routes to V0.2-E or user.

--- D1 |q_drift| ---
                  C0 (industry)      C1 (market)
  median          +0.1199            +???
  p75             +0.2034            +???
  P(>0.3)         +0.1028            +???

--- D2 |corr_x_beta_d| ---
  median          +0.0345            +???
  p75             +0.0591            +???
  P(>0.3)         +0.0000            +???

--- D3 corr_F_d ---
  median          +0.0056            +???
  p75             +0.0158            +???
  P(>0.2)         +0.0000            +???
...
```

(Numbers for C1 will be filled by the run; C0 numbers are from V0.2-D's
already-computed `v0_2_d_distributions.csv`.)

---

## §5 Decision Routing (per user's A/B/C/D)

After the C1 run, the user interprets the paired comparison:

| Scenario | Diagnostic signature | Routing |
|---|---|---|
| **A. Market >> Industry** | `delta_oos_ic > +0.05` AND `sign_flipped == False` for >60% of stocks AND `P_C1(\|q_drift\|>0.3) < 5%` | **Switch main line to stock ↔ market**; deepen C1 with rolling OOS or per-exchange subsets |
| **B. Industry >> Market** | `delta_oos_ic < -0.05` for >60% of stocks AND `P_C1(\|q_drift\|>0.3) > 15%` | **Keep main line as stock ↔ industry** (V0.2-D's H1 stands); defer Two-tier as V0.2-B prerequisite |
| **C. Both bad** | Both C0 and C1 have OOS IC ≈ 0 or negative; both have P(\|q_drift\|>0.3) > 5% | **Model 2 free-q is structurally unstable regardless of driver**; route to V0.2-B (q stability / shrinkage B1/B2/B3) |
| **D. Both good** | Both C0 and C1 have OOS IC > 0.3 | **Either driver works**; route to V0.2-C.4 (industry heterogeneity) to find which sub-population benefits from which driver |

These are **diagnostic signatures**, NOT verdict thresholds. The user reads them
and routes — V0.2-C1 does not choose.

---

## §6 Implementation Plan (for the writing-plans skill)

V0.2-C1 implementation is straightforward because the math is identical to V0.2-D.
The new work is:

| Phase | Deliverable |
|---|---|
| **C1-0** | Generate market-driver movement CSVs (5211 stocks × per-exchange) |
| **C1-1** | Run V0.2-D pipeline on `data/projection_market/` (no code change to `ablation_fit.py`) |
| **C1-2** | New `compute_c0_c1_paired_compare(c0_csv, c1_csv, output_csv)` — writes paired compare CSV |
| **C1-3** | New `write_c0_c1_compare_summary_txt(paired_csv, c0_dist, c1_dist, output_txt)` |
| **C1-4** | New CLI `v0_2_c1_market_swap.py` orchestrator: orchestrate C1-0/1/2/3 |
| **C1-5** | 1 test for paired-compare helper; 1 CLI smoke test |
| **C1-6** | Full-market C1 run (5211 stocks × 4 models × market driver) + final review |

Key constraint: **`backtrace/projection/ablation_fit.py` MUST NOT be modified**.
The C1 pipeline reuses V0.2-D's `fit_one_split`, `write_ablation_csvs`,
`summarize_ablation`, `build_panel5_html`, `compute_v0_2_d_distributions`,
`write_v0_2_d_summary_txt` — all unchanged. The only new code is the paired-comparison
helper and the CLI orchestrator.

---

## §7 CLI

```bash
# Generate market-driver movement files (per-exchange)
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py \
    --index 000001.SH --movement --input data/projection_market_sh.csv --output-dir data/projection_market
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py \
    --index 399001.SZ --movement --input data/projection_market_sz.csv --output-dir data/projection_market

# Run V0.2-C1 (delegates to V0.2-D pipeline + adds paired comparison)
PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_c1_market_swap.py --limit 0
```

CLI outputs (in `data/projection_v01_c1/`):
```
kc_estimates_model0.csv         (5211 × 18)
kc_estimates_model1.csv         (5211 × 18)
kc_estimates_model2_diag.csv    (5211 × 36)
kc_estimates_model3.csv         (5211 × 36)
kc_ablation_summary.csv         (4 × 11)
panel5_drift_vs_collinearity.html
v0_2_d_distributions.csv        (C1 alone; C0 read from V0.2-D)
v0_2_d_summary.txt              (C1 alone)
c0_c1_paired_compare.csv        (5211 × ~22, NEW)
c0_c1_compare_summary.txt       (UTF-8 Chinese, NEW)
```

---

## §8 Tests (per spec §12 of V0.2-D pattern)

| # | Test | Verifies |
|---|---|---|
| 1 | `test_paired_compare_columns_present` | `c0_c1_paired_compare.csv` has all 22 columns |
| 2 | `test_paired_compare_sign_flipped_correct` | `sign_flipped` is `True` exactly when `sign(ic_real_C0) != sign(ic_real_C1)` |
| 3 | `test_paired_compare_attenuation_correct` | `q_drift_attenuated` matches `\||q_drift_C1\|| < 0.5 * \||q_drift_C0\||` |
| 4 | `test_c1_cli_smoke` | CLI runs end-to-end with 3 synthetic stocks; all 6 C1 outputs + paired compare exist |

(No new tests for math — V0.2-D's 9 tests already cover all math; C1 reuses identical math.)

---

## §9 Explicitly Out of Scope

| Out-of-scope | Rationale | Deferred to |
|---|---|---|
| Two-tier M (market + industry) | Add 2 driver variables; can't isolate H1 cause | V0.2-C.2 (if requested) |
| Industry-relative alpha (a_S - a_I) | Changes target semantics, not a pure driver swap | V0.2-C.3 (if requested) |
| Industry heterogeneity (per-industry q_drift) | Useful AFTER C1 if D scenario (both drivers work) | V0.2-C.4 (if requested) |
| V0.2-B shrinkage / Lasso / Elastic Net | Only meaningful AFTER C1 routing | V0.2-B (after C1) |
| Modify `_solve_ols` / `prediction_ode.py` / `dynamics_*.py` / `gp_factor_mining/*` | Math is frozen; C1 reuses identical math | (forbidden) |
| Verdict PASS/FAIL on A/B/C/D | Diagnostic only; routing is V0.2-E or user | V0.2-E |

---

## §10 Risks

| Risk | Mitigation |
|---|---|
| Stock list mismatch between C0 and C1 (paired comparison breaks) | Use the same `stocks.csv` for both, with per-exchange filter; verify 5211 stocks in both dirs |
| `projection_batch.py --index 000001.SH` for SZ stocks is "wrong" per exchange | Per-exchange split: SH stocks → 000001.SH, SZ stocks → 399001.SZ |
| Movement file format drift between C0 and C1 | `_read_movement` uses column-name substitution via `index_tag`; format invariant |
| `data/projection_market/` pollutes git | Output dir is gitignored; spec also flags it |
| `data/stocks/` cache missing market index | All market indices are 000001.SH and 399001.SZ — must exist in `data/indices/`; check before C1-0 |

---

## §11 Deliverables

1. `backtrace/projection/v0_2_c1_market_swap.py` (new CLI orchestrator, ~80 lines)
2. `backtrace/projection/ablation_fit.py` (NO modification — reuses V0.2-D functions)
3. `tests/test_dynamics_eigen.py` (+2 tests: paired-compare correctness + CLI smoke)
4. `data/projection_market/movement_*_*.csv` (5211 market-driver movement files, gitignored)
5. `data/projection_v01_c1/` (full-market C1 outputs, gitignored)
6. Memory entry `projection-v02-c1-market-driver-swap.md` (after C1 full-market run)

---

## §12 Self-Review Checklist

- [x] Placeholder scan: no TBD/TODO
- [x] Internal consistency: §3 strict isolation matches §4 schema; §5 routing uses §4 metrics
- [x] Scope check: §9 forbids Two-tier, industry-relative alpha, industry heterogeneity
- [x] Ambiguity check: each new column in §4.3 has unique math definition
- [x] Decision routing: §5 is diagnostic, not PASS/FAIL
- [x] CLI parity: §7 mirrors V0.2-D's CLI structure
- [x] Test count: §8 has 2 new tests (math reuse means no math tests)
- [x] Write-dead line preserved: §3 verbatim from user
- [x] No modification to `_solve_ols` / `prediction_ode.py` / `dynamics_*.py` / `gp_factor_mining/*`

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*
