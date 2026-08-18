# v5.10 — Full-Market OOS Prediction Quality Distribution (HTML)

**Date:** 2026-08-18
**Status:** Draft
**Base:** v5.9 (HEAD `d0ebe25`)
**Author:** Brainstorming output (user opened v4.3 plan; "按计划执行，不用问我")

## 1. Goal

把 v5.9 单股 OOS 预测可视化升级到 **全市场 N 只股票**,产出 portfolio-level 评估 dashboard:
- Hit-rate / RMSE 分布(跨股票)
- 排名表(哪只预测准 / 哪只跑偏)
- Top-5 detail 小图(便于深挖)
- 行业 / 交易所聚合(可选,echo v4.3 思路)

**业务读法**:
- 头部: "全 A 5000 只 hit-rate 中位数 0.62, 模型整体可信"
- 排名: "000001.SZ hit=0.81 RMSE=0.005 (top 5%) — 模型可用"
- 尾部: "600519.SH hit=0.34 RMSE=0.020 (bottom 5%) — 需重拟合"

## 2. Why now

v5.9 = 单股 4-row HTML(明天预测准不准)。
v5.10 = 全市场分布 dashboard(整个组合的预测质量)。

v4.3 已经做过 (k̂, ĉ) 在全市场的分布 — 现在做 OOS 预测质量的全市场分布,同样的"full-market"思路,业务相同的 portfolio 视角。

业务从未看到模型在 N 只上的整体表现 — 没法判断"模型整体是否可用"。

## 3. Design

### 3.1 架构

**复用**(0 新依赖):
- `load_oos_predictions(stock_code, days, ...)` from `dynamics_oos_viz.py` (v5.9)
- `compute_movement_projection` + `load_pair` from `_projection_core.py`
- `predict_next_state` (Plan v3 API) from `_dynamics_core.py`
- `plotly.subplots.make_subplots(2, 2, ...)` — 4 subplot 分布 dashboard

**新增**:
- 1 新 CLI: `backtrace/dynamics/dynamics_oos_batch.py`
- 1 新函数: `compute_oos_metrics(stock_code, days, ...)` — 单股 → metrics dict
- 1 新函数: `aggregate_oos_metrics(metrics_list)` — 跨股票聚合
- 1 新函数: `build_full_market_oos_html(per_stock_metrics, output_path)` — 4-subplot dashboard
- 1 新函数: `build_top5_small_multiples(top5_data, output_path)` — 5 mini 4-row charts
- 1 新 test: `test_cli_oos_batch_mode`

### 3.2 Per-stock metrics

```python
def compute_oos_metrics(stock_code: str, days: int = 250, **kwargs) -> dict:
    """load_oos_predictions → compute hit_rate / RMSE / MAE / direction_accuracy.

    Returns dict with keys:
        code: str
        n_oos: int
        hit_rate: float        # fraction of days where pred_state == actual_state
        rmse: float            # sqrt(mean(error^2))
        mae: float             # mean(|error|)
        direction_accuracy: float  # fraction of days where sign(pred) == sign(actual)
        k_used: float
        c_used: float
    """
```

### 3.3 Aggregation

```python
def aggregate_oos_metrics(metrics_list: list[dict]) -> dict:
    """Cross-stock distribution.

    Returns dict with:
        n_stocks: int
        median_hit_rate: float
        p25_hit_rate: float
        p75_hit_rate: float
        median_rmse: float
        median_mae: float
        median_direction_acc: float
        ranked: list[dict]  # sorted by hit_rate desc
    """
```

### 3.4 4-subplot dashboard (`build_full_market_oos_html`)

Layout 2×2:
- (1,1) Hit-rate histogram (40 bins, 0-1 range, with median + p25/p75 markers)
- (1,2) RMSE histogram (40 bins, with median marker)
- (2,1) Hit-rate vs RMSE scatter (each dot = 1 stock, color by hit_rate)
- (2,2) Cumulative distribution (hit-rate CDF, with median marker)

Title: "Full-Market OOS Prediction Quality — N stocks, median days=250"

### 3.5 Top-5 small multiples (`build_top5_small_multiples`)

For top-5 stocks (by hit_rate):
- 5 mini 4-row HTML (reusing `build_oos_prediction_html` from v5.9)
- Single HTML with `subplot_titles` listing 5 codes

OR: 1 HTML with 5 sections, each section = embed of v5.9's HTML

Decision: **5 mini figures in 1 HTML** (more compact, single file).

### 3.6 CLI

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_batch.py \
    --days 250 \
    --top-n 5 \
    --output backtrace/outputs/dynsys_oos_full_market.html
```

**Flags**:
- `--days INT` (default 250)
- `--limit INT` (default 0 = all in local cache, else first N from manifest)
- `--prefer-industry` (default True)
- `--top-n INT` (default 5, for small multiples)
- `--output PATH` (default `backtrace/outputs/dynsys_oos_full_market.html`)

**Performance consideration**: 5000 stocks × ~250 days = 1.25M predict_next_state calls. Each call ~microseconds. Total ~minutes. Acceptable.

### 3.7 Edge cases

- 0 stocks (no local cache) → ValueError friendly
- 1 stock → 1-point histograms (degenerate but valid)
- Skipped stocks (data errors) → log + continue, not fail

## 4. Files

| File | Type | Lines |
|---|---|---|
| `backtrace/dynamics/dynamics_oos_batch.py` | new | ~280 |
| `tests/test_dynamics_eigen.py` | modify | +60 |
| `backtrace/dynamics/README.md` | modify | +18 |
| `docs/superpowers/specs/2026-08-18-dynamics-v5-10-full-market-oos-distribution.md` | new | (this) |
| `docs/superpowers/plans/2026-08-18-dynamics-v5-10-full-market-oos-distribution.md` | new | (TBD) |

Total: ~660 lines, 5 files

## 5. Test

1 new test `test_cli_oos_batch_mode` — same F3 inverted tolerance.

合计: 76 → 77 tests pass (1 new test, M1 tsfresh shadow tolerated as skip)。

## 6. 验证

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_batch.py \
    --days 250 --top-n 5 --limit 50 \
    --output backtrace/outputs/dynsys_oos_full_market.html
```

期望: 1 HTML with 2×2 distribution dashboard + top-5 mini 4-row charts.

## 7. 兼容性

- 0 modifications to 11 protected files
- 0 new dependencies (plotly 已装)
- v5.9 `dynamics_oos_viz.py` 0 改动 (复用)
- v5.8 `dynamics_state_timeline.py` 0 改动
- 1 新 CLI + 4 新函数 + 1 新 test + 1 新 README §4.1.9

## 8. Risk

| Risk | Mitigation |
|---|---|
| 全市场 5000 只耗时 | --limit 默认 0, 但 --limit 50 冒烟; 业务默认 limit 0 跑 20-40 分钟 |
| 单股 predict 失败 | try/except, log + skip |
| Plotly HTML 大小 | N=5000 → ~50MB HTML (acceptable), top-5 small multiples 限制大小 |
| M1 tsfresh shadow | F3 容忍 as skip,与 v5.8.1 / v5.9 一致 |

## 9. v5.10 vs v5.9 关系

| 版本 | 模态 | 数据源 | 输出 | 业务用例 |
|---|---|---|---|---|
| v5.9 | 静态 (HTML) | 1 stock × 250 days | 4-row prediction | 单股模型质量 |
| **v5.10** | **静态 (HTML)** | **N stocks × 250 days** | **2×2 distribution + top-5** | **组合模型质量** |

业务上 v5.9 + v5.10 互补:
- v5.9 = "001 怎么样"
- v5.10 = "组合里所有股票怎么样, 哪些好哪些差"

## 10. Future / out of scope

- v5.11: state transition heatmap (state_i × state_j → 概率)
- v5.12: actual M1 root-cause fix (lazy-import tsfresh)
- v5.13: per-industry OOS quality (cross-section analysis, v4.3 style)
- parameter_fit integration (k_used/c_used default 0 → fitted values)

本次 v5.10 只做全市场分布 + top-5 detail, 其他按需。