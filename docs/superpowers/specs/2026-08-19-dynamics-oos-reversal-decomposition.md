# V0.2-D — Dynamics OOS Reversal Decomposition

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step).

**Date:** 2026-08-19
**Status:** Draft (awaiting user approval)
**Parent plan:** V0.1 (commit `328b30c`)
**Verdict basis:** V0.1 PARTIAL — Model 2 train R² = 0.171 ↑9×, OOS IC = -0.506 ↓ reversed

---

## §1 Context

V0 audit (`parameter_fit.py`, commit `75a7b2b`) confirmed OLS solver is not numerically sick
(`cond(X)` median 7.65, well-conditioned), but R² median = 1.58% suggested **model
specification error**, not OLS pathology.

V0.1 (commits `7a2e98b`...`328b30c`) ran a strict 4-model ablation with placebo on
**5211 stocks** to test two corrections to the original dynamics:

```
Model 0: a_S − β·a_M = −k·d − c·u           (status quo, q=1, β̇=0)
Model 1: a_S − β·a_M − β̇·v_M = −k·d − c·u  (β-drift correction, q=1)
Model 2: a_S = q·β·a_M − k·d − c·u           (free q, no β-drift)
Model 3: a_S − β̇·v_M = q·β·a_M − k·d − c·u  (joint)
```

Verdict (pre-registered, locked thresholds):

```
Step 1 (β-drift ΔR² > 0.005): FAIL  (ΔR²_M1 = -0.0107)
Step 2 (|q̂−1| > 0.1):         PASS  (|q̂−1| = 0.880)
Step 3 (ΔIC placebo > 0.02):  PASS  (ΔIC_M3 = +0.417)
→ PARTIAL — q≠1 有用,但 β-drift 不显著;Model 2 优先
```

**Anomaly (this spec's trigger):** Model 2 shows the largest in-sample improvement
(R²: 0.019 → 0.171, **9×**) but its OOS Spearman IC is **strongly negative**
(−0.506). This is **anti-validation**: the model fits training well, but its
predictions on held-out data correlate negatively with the realized targets.

| metric | M0 | M2 | direction |
|---|---:|---:|---|
| median R² (full sample) | 0.019 | 0.171 | ↑ 9× |
| OOS IC (real) | +0.487 | −0.506 | ↓ reversed |
| OOS IC (null / placebo) | +0.005 | +0.032 | null is near zero |
| ΔIC (real − null) | +0.482 | −0.538 | ↓ reversed |
| median cond(X) | 6.79 | 18.16 | ↑ mildly ill-conditioned |

The median of `(IC_real − IC_null)` is −0.469 for M2 (CSV row
`median_delta_ic`), while `median(IC_real) − median(IC_null) = −0.538`. Both
numbers describe the same phenomenon via two valid aggregation rules
(median-of-differences vs difference-of-medians); the gap (0.07) is normal.

---

## §2 Research Question

**Why does Model 2's free-q estimation improve in-sample fit dramatically
(R² ↑ 9×) but reverse the OOS Spearman IC sign (positive → strongly negative)?**

Four non-exclusive hypotheses:

| # | Hypothesis | Diagnostic |
|---|---|---|
| H1 | **Regime shift**: q is not a stable structural parameter; it drifts when the train→test window crosses a market regime boundary | `q_drift`, `q_train_fit` vs `q_test_fit` distribution |
| H2 | **Parameter collinearity**: free q is absorbing estimation noise from `β·a_M` because the design matrix has structural correlation | `corr(Xβ, Xd)`, `corr(Xβ, Xu)`, `cond(X)` |
| H3 | **Residual structure (missing term)**: `−k·d − c·u` does not fully absorb the d/u dynamics, leaving systematic residual that the free q overfits to | `corr(F_self, d)`, `corr(F_self, u)` |
| H4 | **Cross-stock heterogeneity**: M2's behavior on average is misleading; the OOS reversal may be concentrated in a subset (e.g., high-β stocks, illiquid names) | per-stock scatter, Panel 5 |

V0.2-D **does not select** among these. It produces the diagnostic surface for the
user to interpret and route to V0.2-E (integrated decision) or V0.2-C / V0.2-B.

---

## §3 Diagnostic Model — Strict Layering

**Three strictly separated computations, no leakage:**

```
Layer 1 (train fit):       θ_train  = argmin ||X_train·θ − Y_train||²
Layer 2 (test fit):        θ_test   = argmin ||X_test·θ − Y_test||²     ← diagnostic only
Layer 3 (OOS prediction):  Ŷ_oos    = X_test · θ_train                  ← only valid generalization
```

**Boundary rules (写死,spec §7 详述):**
- `θ_test` MUST NOT be used for OOS prediction, model selection, or verdict logic.
- `θ_test` exists solely to quantify parameter drift between train and test windows.
- `oos_r2` and `oos_ic` are computed ONLY from `θ_train` applied to `X_test`.
- `test_fit_r2` is the R² of `X_test · θ_test` vs `Y_test` (diagnostic; expected high if model is consistent across windows).

---

## §4 Parameter Stability (Group A)

For each stock, refit OLS on each window and save **separate** drift quantities.

### Fields (per stock, 9 new columns)

| field | definition |
|---|---|
| `q_train_fit` | q̂ from OLS on (X_train, Y_train) |
| `k_train_fit` | k̂ from OLS on (X_train, Y_train) |
| `c_train_fit` | ĉ from OLS on (X_train, Y_train) |
| `q_test_fit` | q̂ from OLS on (X_test, Y_test) — **diagnostic only** |
| `k_test_fit` | k̂ from OLS on (X_test, Y_test) — **diagnostic only** |
| `c_test_fit` | ĉ from OLS on (X_test, Y_test) — **diagnostic only** |
| `q_drift` | `q_test_fit − q_train_fit` |
| `k_drift` | `k_test_fit − k_train_fit` |
| `c_drift` | `c_test_fit − c_train_fit` |

### Explicitly FORBIDDEN (spec §3 boundary, §14 out-of-scope)

- ❌ `param_drift_l2 = sqrt((Δq)² + (Δk)² + (Δc)²)` — **q is dimensionless, k/c have units of inverse-time; L2 mix is unit-incoherent and would distort cross-stock comparison.**
- ❌ Standardized parameter drift `D_θ = sqrt((Δq/SE_q)² + ...)` — deferred to V0.2-B shrinkage phase if user chooses to pursue.
- ❌ Any "drift stability score" combining the three drifts.

**Rationale:** Save the three drifts separately. Diagnostic interpretation per drift;
aggregation deferred to V0.2-E or user.

---

## §5 X-X Collinearity (Group C)

**Critical correction from V0.2-D brainstorm:** the original draft proposed
`corr(q̂_s, X_s(t))` — **statistically undefined** (scalar × time series). Replaced
with the correct per-stock cross-sectional correlation among regressors.

### Model 2 design matrix (per stock)

```
X = [X_β  |  X_d  |  X_u]    shape (n_valid, 3)
   = [β·a_M | −d | −u]
```

### Fields (per stock, 3 new columns + existing 1)

| field | definition | source |
|---|---|---|
| `corr_x_beta_d` | Pearson corr(β·a_M, −d) over valid rows | new |
| `corr_x_beta_u` | Pearson corr(β·a_M, −u) over valid rows | new |
| `corr_x_d_u` | Pearson corr(−d, −u) over valid rows | new |
| `condition_number` | cond(X) — already exists | unchanged |

These three correlations form the **OLS identifiability surface** of Model 2. If
`|corr_x_beta_d|` is large systemically, free q is confounded with the k·d term —
the regressors are not orthogonal and free q has freedom to absorb slack.

**Out of scope** for V0.2-D: cross-stock correlation `corr_s(q_s, E_β,s)` where
`E_β,s = std_t(β_t·a_M,t)` — this is a *second-layer* cross-sectional analysis,
potentially useful for H4 but not needed for V0.2-D's primary diagnostic. Parked.

---

## §6 Residual Structure (Group D)

After fitting on train, compute residual time series and correlate with each regressor.

### Residual definition

```
F_self(t) = Y_train(t) − X_train(t) · θ_train     shape (n_train,)
```

### Fields (per stock, 3 new columns)

| field | definition |
|---|---|
| `corr_F_beta_aM` | Pearson corr(F_self, β·a_M) over train rows |
| `corr_F_d` | Pearson corr(F_self, −d) over train rows |
| `corr_F_u` | Pearson corr(F_self, −u) over train rows |

### Diagnostic interpretation

If `corr_F_d` is **systemically > 0.2** across the universe, the current
dynamics terms `−k·d − c·u` do not fully absorb the d-evolution. This suggests a
**missing dynamics term** (e.g., `d_{t-1}`, `ḋ`, `|d|·d`). V0.2-D only reports; the
re-spec decision is deferred to V0.2-E.

If `corr_F_u` is systemically large, the u-coupling is mis-specified.

If all three are near zero (residuals are white noise), the dynamics is correctly
specified **for the in-sample window** — the OOS reversal must then come from H1
(regime shift) or H4 (heterogeneity), not H3.

---

## §7 OOS Boundary (Strict Rules)

**These rules are write-dead per spec §3:**

1. `θ_test_fit` (i.e., `q_test_fit`, `k_test_fit`, `c_test_fit`) **MUST NOT** appear in:
   - `oos_r2` computation
   - `oos_ic` (= `ic_real`) computation
   - HTML Panel 5 (uses `oos_ic` only via coloring, never as axis)
   - Decision verdict (D1/D2/D3 do not use test-fit parameters)
   - Any downstream consumer in `parameter_fit.py`, `prediction_ode.py`, `dynamics_*.py`

2. `oos_r2` MUST be computed as:
   ```
   Ŷ_oos = X_test · θ_train
   oos_r2 = 1 − SS_res(Ŷ_oos, Y_test) / SS_tot(Y_test)
   ```
   `SS_tot` uses the test-set baseline; this is the **only valid** OOS R².

3. `test_fit_r2` MUST be computed as:
   ```
   Ŷ_test_fit = X_test · θ_test_fit
   test_fit_r2 = 1 − SS_res(Ŷ_test_fit, Y_test) / SS_tot(Y_test)
   ```
   This measures **within-test consistency** of the refit, not OOS generalization.
   A high `test_fit_r2` paired with low `oos_r2` is the **fingerprint** of H1/H2
   (parameter instability across windows).

4. `train_fit_r2` = the existing `r2` column (semantic rename in CSV header / doc
   only; numerical value unchanged).

---

## §8 CSV Schema

Extend the existing 18-column schema (from V0.1, file
`data/projection_v01/kc_estimates_model{0,1,2,3}.csv`) to **36 columns** for
Models 2 and 3 only (Models 0 and 1 keep the original 18; the new fields are
populated only for models with free parameters).

### Schema (33 columns)

```
[18 existing]
code, name, index_code, index_tag, stock_tag,
n_train, n_test,
condition_number, regressor_corr, r2,            ← r2 改名 train_fit_r2
identification_status, fit_quality,
q_hat, k_hat, c_hat,
f_self_loss, ic_real, ic_null

[9 new — Group A]
q_train_fit, k_train_fit, c_train_fit,
q_test_fit, k_test_fit, c_test_fit,
q_drift, k_drift, c_drift

[3 new — Group B]
train_fit_r2, test_fit_r2, oos_r2

[3 new — Group C]
corr_x_beta_d, corr_x_beta_u, corr_x_d_u

[3 new — Group D]
corr_F_beta_aM, corr_F_d, corr_F_u
```

Models 0/1: `*_train_fit` / `*_test_fit` / `*_drift` = NaN (no free q to drift);
`*_corr_*` still populated (X-X collinearity and F structure are model-agnostic).

---

## §9 Panel 5 — Drift × Collinearity Scatter

**Correction from V0.2-D brainstorm:** Panel 5 originally proposed
`q_drift × corr(q̂, β·a_M)`. The y-axis was statistically undefined (scalar ×
time-series). Replaced with:

### Panel 5 — Model 2 only

- **x-axis:** `corr_x_beta_d` (Pearson corr of β·a_M and −d within Model 2's
  design matrix)
- **y-axis:** `q_drift = q_test_fit − q_train_fit`
- **color:** `oos_ic` (= `ic_real`), continuous RdBu_r colormap, midpoint 0
- **shape:** none (single panel)
- **markersize:** 4 pt, opacity 0.6

### 4 quadrants (interpretation)

| quadrant | meaning |
|---|---|
| **upper-right** (high \|corr\|, large q_drift) | H1+H2: regime shift amplified by collinearity. Highest-risk stocks. |
| **upper-left** (low \|corr\|, large q_drift) | H1 only: regime shift dominates, OLS identifiability is fine. |
| **lower-right** (high \|corr\|, small q_drift) | H2 only: structural collinearity but stable parameters — Lasso/Elastic Net might recover signal (deferred to V0.2-B). |
| **lower-left** (low \|corr\|, small q_drift) | H1+H2+H3 unlikely; check H4 (heterogeneity) or measurement noise. |

The plot is **diagnostic only** — no PASS/FAIL annotation. User interprets.

---

## §10 Decision Gates — Distribution Reporting Only

V0.2-D **does not produce PASS/FAIL verdicts**. Each gate is a **distribution
report** (median / p25 / p75 / tail probability) for the user to interpret.

| gate | metric | distribution reported | threshold (reporting only) |
|---|---|---|---|
| **D1** | `q_drift` | median, p25, p75, P(\|q_drift\| > 0.3) | 0.3 |
| **D2** | `corr_x_beta_d` | median, p25, p75, P(\|corr\| > 0.3) | 0.3 |
| **D3** | `corr_F_d` | median, p25, p75, P(corr > 0.2) | 0.2 |

All three distributions go into `data/projection_v01_d/v0_2_d_distributions.csv`.

**V0.2-D verdict text (`v0_2_d_summary.txt`) states only the observed distributions,
e.g.:**

```
D1 q drift:           median=..., p25=..., p75=..., P(>0.3)=...
D2 Xβ-Xd collinearity: median=..., p25=..., p75=..., P(>0.3)=...
D3 residual-d corr:   median=..., p25=..., p75=..., P(>0.2)=...
```

**Interpretation rules** (which hypothesis dominates) are deferred to **V0.2-E**
(an integrated decision spec the user will author separately if the user wants
formal routing) or back to the user.

---

## §11 Parked Audit Fix — ΔIC Triple Statistic

**Bug parked from V0.1 final review (minor):** the verdict text used one ΔIC
statistic (difference of medians) while the summary CSV only stored one other
(median of differences), with the third (delta vs M0) also present. Three
statistics, two storage locations, one missing — verdict was not fully
reproducible from CSV.

**V0.2-D Phase 0 — audit hygiene (low-cost fix):**

Modify `summarize_ablation` to write **all three** ΔIC statistics explicitly:

```
median_delta_ic                (A: median_s(IC_real_s − IC_null_s))   ← already stored
diff_of_medians_delta_ic       (B: median(IC_real) − median(IC_null)) ← NEW
delta_ic_vs_m0                 (C: median_s(IC_real_new,s − IC_real_M0,s)) ← already stored
```

The verdict text and any future consumer (recommendation TXT, HTML Panel 3)
must read from CSV, not recompute inline.

**Cost:** ~10 lines in `summarize_ablation`, no new tests, no schema break
(adding one new row to the 4×10 → 4×11 metric matrix).

---

## §12 Tests

Append to `tests/test_dynamics_eigen.py`:

### Required (Group A)
1. `test_fit_split_returns_train_test_params` — synthetic Model 2 data; verify
   `q_train_fit ≠ q_test_fit` after fitting on each window.
2. `test_param_drift_handles_unit_incoherence` — verify k_drift, c_drift, q_drift
   are reported **separately** (no `param_drift_l2` aggregation in code).
3. `test_oos_uses_train_params_only` — given a `theta_train`, verify
   `oos_r2 = R²(Y_test, X_test · theta_train)` even if `theta_test` would
   yield higher R².

### Required (Group C)
4. `test_x_x_correlation_per_stock_scalar` — verify output shape is scalar per
   stock (not a time series), and that the original V0.2-D draft
   `corr(q_hat, β·a_M)` would have raised an error.
5. `test_x_x_correlation_matches_cond_x` — sanity: large `corr_x_beta_d`
   generally coincides with large `condition_number`.

### Required (Group D)
6. `test_residual_correlation_white_noise_zero` — synthetic Model 2 with
   Gaussian noise → expect `corr_F_*` < 0.05.
7. `test_residual_correlation_missing_term_detects` — synthetic data where
   dynamics is missing a `d²` term → expect `corr_F_d` > 0.3.

### Required (Panel 5)
8. `test_panel5_uses_x_x_corr_not_q_x_corr` — render the HTML and verify the
   x-axis label is `corr_x_beta_d`, NOT `corr(q, β·a_M)`.

### Required (audit fix)
9. `test_summarize_ablation_writes_three_delta_ic` — verify CSV has all 3
   `*delta_ic*` rows, and verdict numbers are reproducible from CSV.

### Forbidden test patterns (per V0.1 lessons)
- ❌ `Move_Delta_Vol_stk` literal — must use `{stock_tag}` substitution via
  `_read_movement`.
- ❌ `assert not NaN` on aggregated metrics — must check finite, not zero.
- ❌ Single-stock assertions on cross-sectional statistics.

---

## §13 CLI / Outputs

### CLI extension

The existing `dynamics_v0_1_ablation.py` (or equivalent; whatever ran V0.1's
full-market) gains **one new flag**:

```bash
PYTHONIOENCODING=utf-8 python backtrace/projection/v0_2_d_decompose.py \
    --model 2 --include-diagnostics
```

(The default V0.2-D run produces only Models 2 and 3 CSVs with the 33-column
schema; `--all` adds Models 0 and 1 with the X-X / residual columns populated
but no `_train_fit` / `_test_fit` / `_drift` columns.)

### Outputs (in `data/projection_v01_d/`)

```
kc_estimates_model2_diag.csv         5211 × 33 (Model 2 with full diagnostics)
kc_estimates_model3_diag.csv         5211 × 33 (Model 3 with full diagnostics)
x_x_collinearity_distribution.html   per-stock histogram of (corr_x_beta_d, corr_x_beta_u, corr_x_d_u)
residual_correlation_panel.html      per-stock histogram of (corr_F_*) + scatter F_d vs F_u
panel5_drift_vs_collinearity.html    the §9 scatter for Model 2
v0_2_d_distributions.csv             D1 / D2 / D3 distribution tables
v0_2_d_summary.txt                   UTF-8 Chinese distribution report
```

### Audit hygiene output (Phase 0)

```
data/projection_v01/kc_ablation_summary.csv   — overwrite, now 4×11
data/projection_v01/kc_ablation_recommendation.txt  — regenerate, numbers read from CSV
```

---

## §14 Explicitly Out of Scope

The following are **forbidden** in V0.2-D's implementation, tests, and verdict:

| out-of-scope | rationale | deferred to |
|---|---|---|
| Lasso / Elastic Net shrinkage | belongs to stability/shrinkage phase | V0.2-B |
| Industry vs market driver (M) swap | belongs to driver-validity phase | V0.2-C |
| New dynamics terms (`d²`, `ḋ`, `|d|·d`) | belongs to re-specification phase | V0.2-E |
| Prediction target change (`a_S` → different target) | belongs to spec-revision phase | V0.2-E |
| Trading strategy change | forbidden by spec §14 | (never) |
| V6 rerun | V0.2-D does not modify the alpha pipeline | V0.2-F (if at all) |
| `param_drift_l2` or other unit-coherent aggregations | forbids dimensionless/physical mixing | V0.2-B if at all |
| Cross-stock `corr_s(q_s, E_β,s)` | parked; not needed for V0.2-D primary diagnostic | V0.2-D.2 (if requested) |
| Standardized parameter drift `D_θ` | belongs to shrinkage phase | V0.2-B |
| Verdict PASS/FAIL | diagnostic only; routing is V0.2-E or user | V0.2-E |

---

## §15 Phases (proposed decomposition; final shape in plan)

| phase | name | key deliverable |
|---|---|---|
| **P0** | Audit hygiene | `summarize_ablation` writes 3 ΔIC stats; recommendation TXT reproducible from CSV |
| **P1** | Parameter stability | Group A fields + tests 1–3 |
| **P2** | X-X collinearity | Group C fields + tests 4–5 |
| **P3** | Residual structure | Group D fields + tests 6–7 |
| **P4** | Panel 5 + summary | HTML scatter, distribution CSV, summary TXT + test 8 |
| **P5** | Audit verification | test 9 (verdict reproducibility) + full-market run |

---

## §16 Risks

| risk | mitigation |
|---|---|
| Test fixture column-naming typo (recurring V0.1 issue) | spec §12 explicitly forbids literal `Move_Delta_Vol_stk`; tests must use `_read_movement` substitution |
| CSV schema break (consumers reading `r2` field) | semantically `r2` is now `train_fit_r2`; add `train_fit_r2` as a new column for clarity, keep `r2` for backward compat with a doc note |
| `oos_r2` SS_tot near zero (flat test window) | guard: NaN if SS_tot < ε; same as V0's R² NaN guard pattern |
| `corr_F_*` when train has < 20 valid rows | skip (same V0 fallback) |
| `q_drift` / `k_drift` / `c_drift` undefined when either window < 20 valid | NaN per row |
| Panel 5 with 5211 points becomes cluttered | opacity 0.6 + markersize 4 + plotly WebGL; no overplotting annotations |
| Boundary rule violations (`θ_test` leaking into OOS prediction) | spec §7 explicit; review rubric must grep for `θ_test` / `q_test_fit` references in OOS / verdict / Panel 5 code paths |

---

## §17 Deliverables (after plan + SDD)

1. `backtrace/projection/v0_2_d_decompose.py` (new CLI; the diagnostic extension)
2. `backtrace/projection/ablation_fit.py` (extended with Group A/B/C/D functions; Phase 0 audit fix)
3. `tests/test_dynamics_eigen.py` (9 new tests, append)
4. `data/projection_v01/kc_ablation_summary.csv` (regenerated, 4×11)
5. `data/projection_v01/kc_ablation_recommendation.txt` (regenerated, all numbers from CSV)
6. `data/projection_v01_d/kc_estimates_model{2,3}_diag.csv` (5211 × 33)
7. `data/projection_v01_d/{x_x_collinearity_distribution, residual_correlation_panel, panel5_drift_vs_collinearity}.html`
8. `data/projection_v01_d/v0_2_d_distributions.csv`
9. `data/projection_v01_d/v0_2_d_summary.txt`
10. Memory entry `projection-v02-d-oos-reversal-decomposition.md`

---

**Spec self-review checklist (to run before user review):**

- [ ] Placeholder scan: no "TBD" / "TODO" / "fill in later"
- [ ] Internal consistency: §4 forbids `param_drift_l2`; §17 deliverables don't reference it; review rubric must enforce
- [ ] Scope check: §14 lists all forbidden items
- [ ] Ambiguity check: each field has a unique Python name + math formula
- [ ] Test fixture names: §12 forbids literal `Move_Delta_Vol_stk`
- [ ] Boundary rules: §7 explicit; no ambiguous "test parameter" reference
- [ ] Audit fix: §11 is concrete and complete (3 ΔIC stats, no schema break)
- [ ] Panel 5: §9 axis labels match §5/§4 field names exactly
- [ ] Decision gates: §10 is diagnostic only, no PASS/FAIL

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*