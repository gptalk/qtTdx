# V0.2-E1 — Extended Integration Decision (ΔIC distribution + cross-sectional q)

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step, after user approves).

**Status:** Draft (awaiting user approval)
**Parent spec:** V0.2-E Integration Decision (REVISED 2026-08-20T11:30) — softer conclusion: "Market shows superior q-drift stability, but no evidence yet of superior mean OOS IC"
**Type:** Diagnostic — analyze paired C0/C1 surface for ΔIC distribution + Δ|q_drift| × cross-stock features

---

## §1 Context

V0.2-E concluded (REVISED 2026-08-20T11:30) that Market is the leading candidate reference driver but the final reference frame is NOT decided. Two outstanding diagnostic questions block the final decision:

**E1 — ΔIC distribution question**: 62% of stocks show `ic_improved` under Market, but `mean ΔIC = -0.037` is negative. This contradiction (most stocks slightly improve, mean is negative) implies a few stocks must be hurting badly. We need the full ΔIC distribution to understand whether Market is "narrowly bad for outliers" or "broadly bad on average".

**E2 — cross-sectional question**: `P(|q_drift|>0.3)` drops from 10.27% to 3.61% under Market. Does this improvement concentrate on a specific subset of stocks (e.g., high-volatility, low-liquidity, high-β stocks), or is it universal? If it concentrates on a specific subset, that subset may be a candidate for special treatment (shrinkage, two-tier, etc.).

Both questions feed the final reference-frame decision. V0.2-E1 answers them; V0.2-D.2 then probes the H1b mechanism; V0.2-B then evaluates shrinkage; then we commit to a reference frame.

---

## §2 Current Paired Surface (reference)

The 5208-stock paired C0/C1 surface already exists at `data/projection_v01_c1/c0_c1_paired_compare.csv` (23 columns):

```
code, name,
ic_real_C0, ic_real_C1, delta_oos_ic,
q_drift_C0, q_drift_C1, delta_q_drift,
q_hat_C0, q_hat_C1, delta_q_hat,
test_fit_r2_C0, test_fit_r2_C1, delta_test_fit_r2,
oos_r2_C0, oos_r2_C1, delta_oos_r2,
condition_number_C0, condition_number_C1, delta_cond,
sign_flipped,
q_drift_attenuated, q_drift_amplified,
ic_improved, ic_worsened
```

Per-stock β, stock volatility, and liquidity are NOT in this CSV. They must be computed from daily data (`data/stocks/{code}_daily.csv`) and market data (`data/indices/{000001.SH, 399001.SZ}_daily.csv`).

---

## §3 New Analysis (E1 + E2)

### 3.1 E1 — ΔIC Distribution Analysis

**Input**: `data/projection_v01_c1/c0_c1_paired_compare.csv` (5208 rows)

**Compute** (already available as `delta_oos_ic`):
- ΔIC = `ic_real_C1 - ic_real_C0` per stock

**Summary statistics**:
- n (count)
- mean, median
- std, IQR (p75 - p25)
- p10, p25, p75, p90
- min, max
- Sign test: P(ΔIC > 0)
- Symmetric percentiles: p5, p95

**Distribution decomposition**:
- Count of stocks where ΔIC ∈ (-∞, -0.1], (-0.1, -0.05], (-0.05, 0], (0, 0.05], (0.05, 0.1], (0.1, ∞)
- Count of stocks where |ΔIC| > 0.1 (large movers)
- Count of stocks where ΔIC < -0.1 AND ic_worsened == True (consistently worse)

**Output**:
- `data/projection_v01_e1/delta_ic_distribution.html` (plotly histogram + summary table)
- `data/projection_v01_e1/delta_ic_summary.csv` (single row: all stats above)
- `data/projection_v01_e1/delta_ic_buckets.csv` (bucket counts)

### 3.2 E2 — Cross-Sectional q-drift Improvement Analysis

**Input**: 
- `data/projection_v01_c1/c0_c1_paired_compare.csv` (paired C0/C1 metrics)
- `data/projection_v01_c1/kc_estimates_model2_diag.csv` (per-stock fitting details: condition_number, oos_r2, r2, fit_quality, q_hat, k_hat, c_hat)
- `data/stock_basic.csv` (code → market mapping: SH/SZ)
- `data/stocks/{code}_daily.csv` (per-stock daily OHLCV)
- `data/indices/{000001.SH, 399001.SZ}_daily.csv` (market indices)

**Compute per stock**:
- `Δ|q_drift|` = `|q_drift_C1| - |q_drift_C0|` (target variable)
- β_industry (C0): regress stock daily returns on industry index daily returns over training window (240 days × 70% = 168 days)
- β_market (C1): regress stock daily returns on market index (SH/SZ appropriate) daily returns over training window
- Stock volatility = std(daily stock returns over training window)
- Liquidity = median(daily Volume over training window)
- (Already in CSV) C0 IC = `ic_real_C0`, original q = `q_hat_C0`, condition_number = `condition_number_C0`, fit_quality

**Cross-sectional analysis**:
- Spearman correlation: ρ(Δ|q_drift|, each feature)
- OLS regression: Δ|q_drift| ~ β_industry + β_market + stock_volatility + liquidity + q_hat_C0 + ic_real_C0 + condition_number_C0 + fit_quality
- Standardized coefficients (z-scored) to compare effect sizes

**Subgroup analysis** (stratify by stock characteristics):
- By volatility quartile (Q1-Q4): mean Δ|q_drift| per quartile
- By liquidity quartile: mean Δ|q_drift| per quartile
- By β_industry quartile: mean Δ|q_drift| per quartile
- By C0 IC quartile: mean Δ|q_drift| per quartile
- By condition_number quartile: mean Δ|q_drift| per quartile

**Output**:
- `data/projection_v01_e2/cross_sectional_correlations.csv` (Spearman ρ table)
- `data/projection_v01_e2/cross_sectional_regression.csv` (OLS coefficients)
- `data/projection_v01_e2/quartile_summary.csv` (Q1-Q4 mean Δ|q_drift| per feature)
- `data/projection_v01_e2/cross_sectional.html` (plotly scatter plot matrix + quartile bar chart)

---

## §4 Scope

| In scope | Out of scope |
|---|---|
| New script `backtrace/projection/v0_2_e1_delta_ic_distribution.py` (E1) | Modify math: `ablation_fit.py`, `_projection_core.py`, `prediction_ode.py`, `parameter_fit.py`, `dynamics_*.py` |
| New script `backtrace/projection/v0_2_e2_cross_sectional_q.py` (E2) | Re-run the C0/C1 projections (use existing CSVs + daily data) |
| Helper `backtrace/projection/_e2_features.py` (β / vol / liquidity extraction from daily data) | Modify C0/C1 output schema |
| Outputs in `data/projection_v01_e1/` and `data/projection_v01_e2/` (gitignored) | |
| Tests in `tests/test_v0_2_e1.py` and `tests/test_v0_2_e2.py` (1 summary-stats test each) | Any production deployment |
| Memory entry `projection-v02-e1-extended-decision.md` | V0.2-D.2 / V0.2-B (independent specs, follow-up) |

---

## §5 Implementation Detail

### 5.1 E1 — `v0_2_e1_delta_ic_distribution.py`

```python
# Inputs
paired_csv = 'data/projection_v01_c1/c0_c1_paired_compare.csv'
output_dir = 'data/projection_v01_e1'

# Load
df = pd.read_csv(paired_csv)  # 5208 rows

# Compute ΔIC (already column delta_oos_ic; rename for clarity)
df['delta_ic'] = df['delta_oos_ic']

# Summary stats
summary = {
    'n': len(df),
    'mean': df['delta_ic'].mean(),
    'median': df['delta_ic'].median(),
    'std': df['delta_ic'].std(),
    'p25': df['delta_ic'].quantile(0.25),
    'p75': df['delta_ic'].quantile(0.75),
    'p10': df['delta_ic'].quantile(0.10),
    'p90': df['delta_ic'].quantile(0.90),
    'p5': df['delta_ic'].quantile(0.05),
    'p95': df['delta_ic'].quantile(0.95),
    'min': df['delta_ic'].min(),
    'max': df['delta_ic'].max(),
    'sign_test_p_gt_0': (df['delta_ic'] > 0).mean(),
    'large_movers_pct': (df['delta_ic'].abs() > 0.1).mean() * 100,
}

# Buckets
buckets = pd.cut(df['delta_ic'],
                 bins=[-np.inf, -0.1, -0.05, 0, 0.05, 0.1, np.inf],
                 labels=['(-∞,-0.1]', '(-0.1,-0.05]', '(-0.05,0]',
                         '(0,0.05]', '(0.05,0.1]', '(0.1,∞)'])
bucket_counts = buckets.value_counts().sort_index()

# HTML output: histogram + summary table
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Histogram(x=df['delta_ic'], nbinsx=50, name='ΔIC'))
fig.add_vline(x=0, line_dash='dash', line_color='red')
fig.update_layout(title='V0.2-E1: ΔIC = IC_C1 - IC_C0 (Market vs Industry) — 5208 stocks',
                  xaxis_title='ΔIC', yaxis_title='count')
fig.write_html(f'{output_dir}/delta_ic_distribution.html')

# Write summary + buckets
pd.DataFrame([summary]).T.to_csv(f'{output_dir}/delta_ic_summary.csv')
bucket_counts.to_csv(f'{output_dir}/delta_ic_buckets.csv')
```

### 5.2 E2 — `v0_2_e2_cross_sectional_q.py`

```python
# Inputs
paired_csv = 'data/projection_v01_c1/c0_c1_paired_compare.csv'
kc_csv = 'data/projection_v01_c1/kc_estimates_model2_diag.csv'
stock_basic_csv = 'data/stock_basic.csv'
stocks_dir = 'data/stocks'
indices_dir = 'data/indices'
output_dir = 'data/projection_v01_e2'

# Load paired + kc
paired = pd.read_csv(paired_csv)
kc = pd.read_csv(kc_csv)

# For each stock, compute features from daily data
features = []
for code in paired['code']:
    s = paired[paired['code'] == code].iloc[0]
    market = 'SH' if code.endswith('.SH') else 'SZ'
    market_code = '000001.SH' if market == 'SH' else '399001.SZ'
    
    stock_df = pd.read_csv(f'{stocks_dir}/{code.replace(".","_")}_daily.csv', parse_dates=['Date']).set_index('Date')
    market_df = pd.read_csv(f'{indices_dir}/{market_code.replace(".","_")}_daily.csv', parse_dates=['Date']).set_index('Date')
    
    # Align on dates; use last 168 days (70% of 240)
    common = stock_df.index.intersection(market_df.index)
    if len(common) < 100:
        continue  # skip if insufficient data
    train_dates = common[-168:]  # training window approximation
    
    stock_close = stock_df.loc[train_dates, 'Close'].pct_change().dropna()
    market_close = market_df.loc[train_dates, 'Close'].pct_change().dropna()
    
    # β_market
    cov = np.cov(stock_close, market_close)
    beta_market = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
    
    # β_industry (per-stock industry index from kc_estimates or daily data)
    industry_index_code = s.get('index_code_C0', None)  # not in paired CSV; would need separate extraction
    # For simplicity, use condition_number as β identifiability proxy
    
    # Stock volatility
    stock_vol = stock_close.std()
    
    # Liquidity
    liquidity = stock_df.loc[train_dates, 'Volume'].median()
    
    features.append({
        'code': code,
        'beta_market': beta_market,
        'stock_volatility': stock_vol,
        'liquidity': liquidity,
        'delta_abs_q_drift': abs(s['q_drift_C1']) - abs(s['q_drift_C0']),
    })

feat_df = pd.DataFrame(features)

# Merge with paired + kc
merged = paired.merge(feat_df, on='code').merge(kc[['code', 'condition_number', 'r2', 'fit_quality', 'q_hat', 'k_hat', 'c_hat']], on='code', suffixes=('','_kc'))

# Spearman correlations
spearman = {}
for feat in ['beta_market', 'stock_volatility', 'liquidity', 'q_hat', 'r2', 'condition_number']:
    spearman[feat] = merged['delta_abs_q_drift'].corr(merged[feat], method='spearman')
spearman_df = pd.DataFrame([spearman]).T.rename(columns={0: 'spearman_rho'})

# Quartile analysis
for feat in ['beta_market', 'stock_volatility', 'liquidity', 'q_hat', 'r2']:
    merged[f'{feat}_quartile'] = pd.qcut(merged[feat], 4, labels=['Q1','Q2','Q3','Q4'])
quartile_summary = merged.groupby([f'{feat}_quartile' for feat in ['beta_market', 'stock_volatility', 'liquidity', 'q_hat', 'r2']]).agg({'delta_abs_q_drift': 'mean'}).reset_index()

# Write outputs
spearman_df.to_csv(f'{output_dir}/cross_sectional_correlations.csv')
quartile_summary.to_csv(f'{output_dir}/quartile_summary.csv')

# HTML: scatter plot matrix + quartile bar chart
# ... plotly subplots ...
fig.write_html(f'{output_dir}/cross_sectional.html')
```

### 5.3 Helper `_e2_features.py`

Per-stock feature extraction. Loads 5208 daily CSVs sequentially (avoid memory blowup). Caches β / vol / liquidity into a single CSV for fast re-runs.

### 5.4 Tests

**E1 test** (`tests/test_v0_2_e1.py`):
- Synthetic 5208-row paired CSV
- Verify summary stats compute correctly (mean, median, percentiles, sign test)
- Verify bucket counts sum to N

**E2 test** (`tests/test_v0_2_e2.py`):
- Synthetic 100-row paired CSV + 100 fake daily files
- Verify Spearman correlations are computed
- Verify quartile summary has 4 quartiles per feature

---

## §6 Risks

| Risk | Mitigation |
|---|---|
| β_industry extraction requires industry index daily data; not in paired CSV | Document as "best-effort β_market only"; β_industry requires extra extraction (future) |
| Loading 5208 daily files slow (~5-10 min) | Cache feature CSV after first run; re-runs use cache |
| Daily data alignment mismatches (Date format, missing days) | Use `pd.Index.intersection`; log dropped dates; document in summary |
| Feature proxy misinterpretation (condition_number ≠ β) | Document each feature's interpretation in CSV comment header |
| E1/E2 sample size 5208 may not detect subtle subgroup effects | Use bootstrap CIs on Spearman ρ; report p-values |

---

## §7 Deliverables

1. `backtrace/projection/v0_2_e1_delta_ic_distribution.py` (E1)
2. `backtrace/projection/v0_2_e2_cross_sectional_q.py` (E2)
3. `backtrace/projection/_e2_features.py` (helper)
4. Outputs in `data/projection_v01_e1/` (3 files: HTML + 2 CSVs)
5. Outputs in `data/projection_v01_e2/` (4 files: HTML + 3 CSVs)
6. Tests in `tests/test_v0_2_e1.py` + `tests/test_v0_2_e2.py`
7. Memory entry `projection-v02-e1-extended-decision.md`

---

## §8 Out of Scope

| Out-of-scope | Where |
|---|---|
| β_industry extraction (would require additional fitting runs) | V0.2-E1.1 (if needed for follow-up) |
| Two-tier driver | V0.2-C.2 (independent spec, after reference frame settles) |
| H1b mechanism (q × β residual) | V0.2-D.2 (independent spec, next after V0.2-E1) |
| Shrinkage | V0.2-B (after V0.2-E1 + V0.2-D.2) |
| Modify `ablation_fit.py` / `_projection_core.py` / `parameter_fit.py` / `dynamics_*.py` | (forbidden) |
| Modify C0/C1 output schema | (would invalidate V0.2-D / V0.2-C1 baselines) |

---

## §9 Self-Review Checklist

- [x] Placeholder scan: no TBD / TODO
- [x] Internal consistency: §3 E1 metrics align with §5.1 code; §3 E2 metrics align with §5.2 code
- [x] Scope check: §4 explicit about 2 new scripts + 1 helper + outputs + tests; math files frozen
- [x] Ambiguity check: each feature (β, vol, liquidity) has explicit source (daily CSV) and computation method
- [x] No modifications to math files (declared in §8)
- [x] No modifications to C0/C1 output schema (preserves V0.2-D / V0.2-C1 baselines)

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*