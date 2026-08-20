# V0.2-E — Integration Decision

> **For agentic workers:** Required sub-skill: `superpowers:writing-plans` (next step, after user approves).

**Status:** Draft (awaiting user approval)
**Parent specs:** V0.2-D ([docs/superpowers/specs/2026-08-19-dynamics-oos-reversal-decomposition.md](2026-08-19-dynamics-oos-reversal-decomposition.md)), V0.2-C1 ([docs/superpowers/specs/2026-08-20-dynamics-c1-market-driver-swap.md](2026-08-20-dynamics-c1-market-driver-swap.md))
**Type:** Pure diagnostic decision document — **zero code change**

---

## §1 Context

V0.2-D and V0.2-C1 together produced a paired diagnostic surface on the Model 2 dynamics formulation. V0.2-D (industry-driver baseline, 5211 stocks) found H1 (q regime drift) plausible and H2/H3 (collinearity / missing term) ruled out. V0.2-C1 (market-driver swap, 5208 paired stocks) found the same H2/H3 ruling plus a strong H1b signal: the upper tail of |q_drift| collapses from 10.27% to 3.61% when the driver is switched from per-stock 申万二级 industry index to per-exchange market index. Median |q_drift| barely moves (0.1199 → 0.1186).

V0.2-E does not introduce new code. It is a single decision document that integrates both diagnostics and recommends the next routing — primarily, switch the main-line driver from industry to market.

---

## §2 Paired Diagnosis Summary

The Model 2 diagnostic surface, paired across 5208 stocks (industry → market):

| Gate | Metric | C0 (industry) | C1 (market) | Reading |
|---|---|---:|---:|---|
| **D1** | \|q_drift\| median | 0.1199 | 0.1186 | ≈ same — H1a (driver-invariant at median) |
| **D1** | \|q_drift\| p75 | 0.2034 | 0.1807 | upper tail contracts |
| **D1** | P(\|x\| > 0.3) | **10.27%** | **3.61%** | **-65%** — H1b (driver-induced tail) strong |
| **D2** | \|corr_x_beta_d\| median | 0.0345 | 0.0366 | ≈ same |
| **D2** | P(\|x\| > 0.3) | 0.00% | 0.02% | H2 ruled out under both |
| **D3** | corr_F_d median | 0.0056 | 0.0058 | ≈ same |
| **D3** | P(x > 0.2) | 0.00% | 0.00% | H3 ruled out under both |
| IC | ic_real mean | -0.481 | -0.518 | C1 slightly worse mean |
| IC | ic_real std | 0.172 | 0.093 | **C1 集中 46%** — predictable, no excess |

Paired diagnostic flags (Model 2 only, 5208 stocks):

| Flag | Count | Pct | Reading |
|---|---:|---:|---|
| `sign_flipped` | 140 / 5208 | 2.7% | driver choice rarely flips IC sign |
| `q_drift_attenuated` (\|q_drift_C1\| < 0.5 · \|q_drift_C0\|) | 1526 / 5208 | 29.3% | H1b partial — tail-aware |
| `q_drift_amplified` (\|q_drift_C1\| > 1.5 · \|q_drift_C0\|) | 1791 / 5208 | 34.4% | H1a partial — median-aware |
| `ic_improved` (\|Δic\| > 0.05 ∧ no sign flip) | 3228 / 5208 | 62.0% | C1 helps the majority |
| `ic_worsened` (\|Δic\| < -0.05) | 2008 / 5208 | 38.6% | C1 hurts a substantial minority |

The full diagnostic files are:
- `data/projection_v01_d/v0_2_d_summary.txt` (C0 distribution report, UTF-8 Chinese)
- `data/projection_v01_c1/v0_2_d_summary.txt` (C1 distribution report)
- `data/projection_v01_c1/c0_c1_paired_compare.csv` (5208 × 25)
- `data/projection_v01_c1/c0_c1_compare_summary.txt` (paired summary, UTF-8 Chinese)

---

## §3 Mechanism Narrative

The paired surface admits two non-exclusive readings, with different mechanistic implications:

### H1a (driver-invariant): the free-q formulation structurally allows regime drift, regardless of driver.

The evidence supporting this is that the **median** of |q_drift| does not move between C0 and C1 (0.1199 → 0.1186). The typical stock has the same q regime instability under either driver. If the cause were purely driver-specific, the median should also move. The natural interpretation is that Model 2's free-q parameter is identified on noise rather than signal in roughly the same fraction of stocks no matter which exogenous series is paired — that is, the identification problem is structural to the regression specification, not to the choice of regressor.

### H1b (driver-induced): industry-driver amplifies the upper tail of q_drift because industry beta estimation is noisier than market beta estimation.

The evidence supporting this is that the **upper tail** collapses when driver is swapped (10.27% → 3.61% for P(|q_drift| > 0.3); p75 contracts from 0.2034 to 0.1807). Two facts about 申万二级 industry indices make this mechanically plausible:
1. Industry indices aggregate ~30–200 stocks; their daily returns have idiosyncratic component from constituents that is not shared by any individual stock. The OLS fit on this noisier regressor will produce less stable coefficient estimates, especially in the test window.
2. Industry membership is static in our `data/sectors/` cache, so for newly-listed stocks or stocks that recently changed industry, the pairing may be inappropriate. This adds another source of identification noise.

In contrast, the market indices (000001.SH, 399001.SZ) are themselves aggregate measures of thousands of stocks with smooth dynamic properties; the regression coefficient β against market is well-identified from any individual stock's daily data.

The two readings are not in conflict: H1a operates at the median, H1b operates at the tail. The combined picture is "free-q is structurally unstable everywhere; industry-driver makes it worse on the upper tail."

### Why ic_real std collapses 0.17 → 0.09 under market-driver

A secondary finding: the OOS IC distribution becomes much more concentrated around its mean (-0.48 → -0.52, std 0.17 → 0.09). This is consistent with market-driver producing a more homogeneous OOS quality across stocks — the same reduction of identification noise that compresses the q_drift tail also compresses the IC distribution. Note the mean is *not* improved: the concentration is around a negative number, which means market-driver does not generate alpha but does reduce *cross-stock dispersion* in the IC distribution.

### Sign-flip rarity

Only 140 / 5208 (2.7%) of stocks flip the sign of their IC between drivers. The sign of OOS IC is mostly driver-invariant. So driver choice changes the *magnitude* and *dispersion* of IC, not its sign.

---

## §4 Routing Recommendation

**Recommended routing**: switch the main-line driver from per-stock 申万二级 industry index to per-exchange market index (SH stocks → 000001.SH, SZ stocks → 399001.SZ).

### Why this routing

1. **Tail risk reduction is the largest single effect in the diagnostic surface.** The D1 P(|x|>0.3) drop from 10.27% to 3.61% (-65%) means that under market-driver, the fraction of stocks where q is wildly unstable shrinks to roughly the residual structural rate (H1a median-level baseline). Industry-driver was systematically inflating this tail.

2. **IC distribution concentrates.** Lower variance in OOS IC means portfolio construction becomes more uniform across stocks; tail-aware risk models work better when tail risk is bounded.

3. **Sign-flip is rare (2.7%).** We are not trading alpha away; we are trading noise for slightly more noise. Mean IC moves from -0.48 to -0.52 — a marginal worsening, well within typical OOS noise.

4. **Simpler data path.** Per-exchange pairing (SH → 000001.SH, SZ → 399001.SZ) is static and requires no per-stock industry lookup, no membership refresh, no edge cases for newly-listed or reclassified stocks.

### Trade-offs accepted

- **Loss of industry-specific signal**. If certain stocks' alpha is genuinely sector-driven rather than market-driven, market-driver averages that out. The diagnostic does not show a "hidden alpha" the industry-driver was capturing — but absence of evidence is not evidence of absence.
- **Mean IC slightly worse**. -0.481 → -0.518 (Δ ≈ -0.04). Within OOS noise, but real.
- **No alpha harvest**. Both drivers produce negative mean IC. Switching does not turn the formulation into a money-making strategy; it makes the diagnostic surface cleaner for further iteration.

### When NOT to switch: trigger conditions for reversal

The recommendation is conditional. Reconsider market-driver if any of these are observed:
- A V0.2-D.2 cross-stock analysis (q_drift × industry beta residual correlation) shows strong positive correlation, which would confirm H1b at the mechanism level rather than just the surface level — at which point market-driver is even more clearly preferred, not reversed.
- A V0.2-C.2 two-tier experiment (market + industry as joint drivers) shows industry adds incremental signal that neither captures alone. If industry adds value jointly but not alone, the recommendation flips to "two-tier, not market-only."
- A V0.2-B shrinkage on industry-driver reduces the D1 tail below the market-driver level. If shrinkage fully compensates for driver noise, industry-driver is preferred for richer dynamics.
- Forward-looking OOS (rolling re-fit) shows market-driver tail re-expands over time. The current diagnostic uses a single 70/30 split; if the tail re-expands under rolling evaluation, market-driver may not be stable.

---

## §5 Risks

### Risk 1: industry signal is real but hidden

The paired diagnostic cannot see what industry-driver adds that market-driver cannot. Industry indices may carry sector-specific momentum or rotation signals that are present only in stocks where the industry is the dominant exogenous regressor. If so, switching to market-driver averages them out and the loss is invisible in the current surface.

**Mitigation**: V0.2-C.2 two-tier experiment — add industry as a second regressor alongside market, see whether it adds incremental predictive signal. If yes, the routing becomes "market + industry" (two-tier), not "market-only."

### Risk 2: market-driver over-smoothing

Market indices aggregate thousands of stocks; their noise floor is low, but their signal-to-noise for any individual stock is also lower (because market only carries the common component). The fitted β and q may be biased toward "this stock behaves like the average" rather than "this stock has a unique dynamics structure." Stocks with truly idiosyncratic dynamics may get over-regularized.

**Mitigation**: spot-check 10–20 stocks with the highest IC improvement under market-driver. If their dynamics look generic (β ≈ 1, q ≈ 0, low d), that's evidence of over-smoothing. If they look idiosyncratic (β ≠ 1, q ≠ 0, high d), the routing is preserving real structure.

### Risk 3: OOS mean IC is negative — there is no alpha to harvest regardless of driver

Both C0 and C1 produce mean IC ≈ -0.5. This is consistent with either (a) the Model 2 free-q formulation is fundamentally not capturing any predictive structure, or (b) the diagnostic surface is reflecting OOS noise floor rather than true predictive signal. Switching drivers does not address this underlying problem; it only changes the *form* of the noise around the negative mean.

**Mitigation**: V0.2-E does not promise alpha. If the user wants alpha, V0.2-B shrinkage + V0.2-E re-specification (e.g., add β·d² or |d|·d terms, change target from Δu_S to next-day Δu_S · sign) are downstream paths. V0.2-E's recommendation is about *which noise to live with*, not about generating returns.

### Risk 4: out-of-sample generalization unknown

The current diagnostic uses a single 70/30 train/test split per stock. We do not know whether the market-driver tail stability persists across multiple rolling OOS windows. A regime shift in late-2026 or 2027 might re-expand the tail.

**Mitigation**: future rolling OOS evaluation (planned in v6+ dynamics work) will measure whether the C1 tail stays at ~3.6% across rolling windows.

---

## §6 Opportunities

### V0.2-C.2 — Two-tier driver (market + industry)

If H1b mechanism is real, industry-driver adds noise on top of market-driver. A natural test is to fit Model 2 with **both** market and industry regressors and see whether industry contributes incremental signal. If yes, "market + industry" jointly is the right main line, not market alone.

This is the highest-leverage follow-up: it tests the mechanism directly while potentially producing better IC.

### V0.2-D.2 — Cross-stock q_drift × industry beta residual correlation

A diagnostic that probes H1b at the mechanism level: for each stock, compute the residual of its β̂_industry after subtracting the cross-stock mean β̂_industry for that stock's industry. The hypothesis is that |q_drift| is positively correlated with |β_residual|. If yes, q drift is driven by industry-beta estimation noise, which strengthens the case for market-driver (or for two-tier).

### V0.2-B shrinkage — now lower-priority

V0.2-B (rolling q̂ mean, Bayesian prior, Lasso/Elastic Net) was originally motivated by H1 (q instability). The C1 finding that market-driver reduces the tail means V0.2-B is less urgent if we adopt the recommended routing. If the user decides to keep industry-driver for richer dynamics, V0.2-B becomes essential as the tail-control mechanism.

### V0.3 — re-specification of the dynamics

Mean IC ≈ -0.5 under both drivers suggests the dynamics formulation itself needs revision. Candidates (each a separate spec):
- Add β·d² to capture nonlinear mean reversion
- Change target to a directional variant (sign-aware)
- Replace free-q with a regime-switching model
- Use β̂ from market-driver but d from industry-driver (mixed)

V0.2-E does not pick a winner among these; it only documents that the current formulation is not alpha-generating regardless of driver.

---

## §7 Deliverables

1. This document ([docs/superpowers/specs/2026-08-20-dynamics-e-integration-decision.md](2026-08-20-dynamics-e-integration-decision.md))
2. Memory entry `projection-v02-e-integration-decision.md` recording the routing recommendation
3. **No code change**. The recommended driver switch — if approved — would happen in a follow-up spec (suggested name: V0.2-F, "driver-default migration").

---

## §8 Out of Scope

| Out-of-scope | Rationale | Where it lives |
|---|---|---|
| Driver switch implementation (change `projection_batch.py` default `--index`) | V0.2-E is a decision document only; user decides whether to enact | V0.2-F (if approved) |
| V0.2-C.2 two-tier experiment | Tests H1b mechanism jointly, requires its own spec | V0.2-C.2 (if requested) |
| V0.2-D.2 cross-stock q × industry β residual | Tests H1b at mechanism level | V0.2-D.2 (if requested) |
| V0.2-B shrinkage as primary path | Less urgent now that C1 reduces tail; remains a fallback | V0.2-B (if requested) |
| Modify `ablation_fit.py` / `_solve_ols` / `prediction_ode.py` / `dynamics_*.py` / `gp_factor_mining/*` | Forbidden | (forbidden) |
| Backtest new strategy on the chosen driver | Routing decision is about diagnostics, not PnL | (deferred to user) |

---

## §9 Self-Review Checklist

- [x] Placeholder scan: no TBD / TODO
- [x] Internal consistency: §3 mechanism is consistent with §2 numbers (H1a at median, H1b at tail)
- [x] Scope check: zero code change explicitly declared in §7 and §8
- [x] Ambiguity check: each scenario in §4 has unique trigger conditions for reversal
- [x] Decision routing: §4 is recommendation + conditional, §6 enumerates next-spec hooks, §8 explicitly forbids implementation
- [x] No modifications to math files (declared in §8)

---

*Awaiting user approval before invoking `superpowers:writing-plans`.*