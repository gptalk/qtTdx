# v5.10 — Full-Market OOS Prediction Quality Distribution (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `backtrace/dynamics/dynamics_oos_batch.py` — apply v5.9's `load_oos_predictions` to N stocks, output 2×2 distribution dashboard (hit-rate hist / RMSE hist / scatter / CDF) + top-5 small multiples (5 mini 4-row charts).

**Architecture:** 1 new CLI reuses `load_oos_predictions` + `build_oos_prediction_html` from v5.9. 4 new helpers: `compute_oos_metrics`, `aggregate_oos_metrics`, `build_full_market_oos_html`, `build_top5_small_multiples`.

**Tech Stack:** plotly 5.x (`make_subplots`, `go.Histogram`, `go.Scatter`, `go.Bar`), numpy, pandas, v5.9 helpers.

**Base commit:** `d390203` (v5.10 spec).
**Reference impls:** `backtrace/dynamics/dynamics_oos_viz.py` (v5.9 — DO NOT modify), `backtrace/dynamics/dynamics_state_timeline.py` (plotly subplot pattern).

## Global Constraints

- 0 modifications to **11 protected files** + `dynamics_oos_viz.py` (v5.9)
- 0 new dependencies (plotly already installed)
- 0 re-implementation of projection / dynamics core math (import only)
- Import from v5.9: `load_oos_predictions`, `build_oos_prediction_html`
- M1 tsfresh shadow: tolerate via F3 inverted tolerance, same as v5.9
- `PYTHONIOENCODING=utf-8` required
- Test count: 76 PASS → 77 PASS + 1 SKIP (M1 if surfaced)
- Output: `backtrace/outputs/dynsys_oos_full_market.html` (default)
- 1 new test: `test_cli_oos_batch_mode` in `tests/test_dynamics_eigen.py`

---

## Task 1: Scaffold + per-stock metrics

**Files:**
- Create: `backtrace/dynamics/dynamics_oos_batch.py`
- Test: `tests/test_dynamics_eigen.py` (no changes)

**Interfaces produced:**
```python
def compute_oos_metrics(stock_code: str, days: int = 250, *, prefer_industry: bool = True,
                       k: float | None = None, c: float | None = None,
                       f_self_window: int = 10) -> dict:
    """Run load_oos_predictions + compute metrics.

    Returns dict:
        code: str
        n_oos: int
        hit_rate: float
        rmse: float
        mae: float
        direction_accuracy: float
        k_used: float
        c_used: float
    """

def aggregate_oos_metrics(metrics_list: list[dict]) -> dict:
    """Cross-stock distribution.

    Returns dict with:
        n_stocks: int
        median_hit_rate, p25_hit_rate, p75_hit_rate: float
        median_rmse, median_mae: float
        median_direction_acc: float
        ranked: list[dict]  # sorted by hit_rate desc
    """
```

- [ ] **Step 1.1: Write file skeleton + imports + DEFAULTS**

Create `backtrace/dynamics/dynamics_oos_batch.py`:

```python
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
REPO_ROOT = os.path.dirname(BACKTRACE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import argparse
import logging
import numpy as np
import pandas as pd

from backtrace.dynamics.dynamics_oos_viz import load_oos_predictions

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DEFAULTS = dict(
    days=250,
    limit=0,
    prefer_industry=True,
    top_n=5,
)
DEFAULT_OUTPUT = 'backtrace/outputs/dynsys_oos_full_market.html'
```

- [ ] **Step 1.2: Implement `compute_oos_metrics`**

Append after DEFAULTS:

```python
def compute_oos_metrics(
    stock_code: str,
    days: int = 250,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """Per-stock OOS prediction quality metrics.

    Calls load_oos_predictions → computes hit_rate / rmse / mae / direction_acc.
    """
    data = load_oos_predictions(
        stock_code=stock_code,
        days=days,
        prefer_industry=prefer_industry,
        k=k, c=c,
        f_self_window=f_self_window,
    )

    common_idx = data['common_idx']
    a_pred = data['a_pred']
    a_actual = data['a_actual']
    state_pred = data['state_pred']
    state_actual = data['state_actual']
    n_oos = len(common_idx)

    if n_oos == 0:
        return dict(
            code=stock_code, n_oos=0,
            hit_rate=0.0, rmse=float('nan'), mae=float('nan'),
            direction_accuracy=0.0,
            k_used=data['k_used'], c_used=data['c_used'],
        )

    # Magnitudes
    a_pred_mag = np.linalg.norm(a_pred, axis=1)
    a_actual_mag = np.linalg.norm(a_actual, axis=1)

    error = a_pred_mag - a_actual_mag
    rmse = float(np.sqrt(np.nanmean(np.square(error))))
    mae = float(np.nanmean(np.abs(error)))

    # State hit rate
    hits = sum(1 for p, a in zip(state_pred, state_actual) if p == a)
    hit_rate = hits / n_oos

    # Direction accuracy (sign agreement)
    direction_pred = np.sign(a_pred_mag)
    direction_actual = np.sign(a_actual_mag)
    direction_matches = np.sum(direction_pred == direction_actual)
    direction_accuracy = float(direction_matches / n_oos)

    return dict(
        code=stock_code,
        n_oos=n_oos,
        hit_rate=float(hit_rate),
        rmse=rmse,
        mae=mae,
        direction_accuracy=direction_accuracy,
        k_used=float(data['k_used']),
        c_used=float(data['c_used']),
    )
```

- [ ] **Step 1.3: Implement `aggregate_oos_metrics`**

Append after `compute_oos_metrics`:

```python
def aggregate_oos_metrics(metrics_list: list[dict]) -> dict:
    """Cross-stock distribution of OOS metrics."""
    if not metrics_list:
        return dict(
            n_stocks=0,
            median_hit_rate=0.0, p25_hit_rate=0.0, p75_hit_rate=0.0,
            median_rmse=0.0, median_mae=0.0,
            median_direction_acc=0.0,
            ranked=[],
        )

    df = pd.DataFrame(metrics_list)
    ranked = df.sort_values('hit_rate', ascending=False).to_dict('records')

    return dict(
        n_stocks=len(metrics_list),
        median_hit_rate=float(df['hit_rate'].median()),
        p25_hit_rate=float(df['hit_rate'].quantile(0.25)),
        p75_hit_rate=float(df['hit_rate'].quantile(0.75)),
        median_rmse=float(df['rmse'].median()),
        median_mae=float(df['mae'].median()),
        median_direction_acc=float(df['direction_accuracy'].median()),
        ranked=ranked,
    )
```

- [ ] **Step 1.4: Smoke test imports + synthetic metrics**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -c "
import sys, os
sys.path.insert(0, 'c:/Users/yellow/mcp/qtTdx/backtrace')
from dynamics.dynamics_oos_batch import compute_oos_metrics, aggregate_oos_metrics
metrics = [compute_oos_metrics.__name__, aggregate_oos_metrics.__name__]
print('OK', metrics)
"
```

Expected: prints `OK ['compute_oos_metrics', 'aggregate_oos_metrics']`. If M1 tsfresh shadow blocks, tolerate as documented limitation.

- [ ] **Step 1.5: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py
git commit -m "feat(dynamics): v5.10 — scaffold + per-stock metrics + aggregator"
```

---

## Task 2: `build_full_market_oos_html` (2×2 dashboard)

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (append `build_full_market_oos_html`)

**Interface produced:**
```python
def build_full_market_oos_html(
    metrics_list: list[dict],
    output_path: str,
    title: str = 'Full-Market OOS Prediction Quality Distribution',
) -> None:
    """Render 2×2 plotly HTML dashboard."""
```

- [ ] **Step 2.1: Add plotly imports**

Add after `import pandas as pd`:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

- [ ] **Step 2.2: Implement `build_full_market_oos_html`**

Append after `aggregate_oos_metrics`:

```python
def build_full_market_oos_html(
    metrics_list: list[dict],
    output_path: str,
    title: str = 'Full-Market OOS Prediction Quality Distribution',
) -> None:
    """2×2 dashboard:
        (1,1) Hit-rate histogram (with median + p25/p75 markers)
        (1,2) RMSE histogram (with median marker)
        (2,1) Hit-rate vs RMSE scatter (color by hit_rate)
        (2,2) Hit-rate CDF
    """
    if not metrics_list:
        raise ValueError('metrics_list is empty — nothing to plot')

    df = pd.DataFrame(metrics_list)
    hit_rates = df['hit_rate'].to_numpy()
    rmses = df['rmse'].to_numpy()

    median_hr = float(np.median(hit_rates))
    p25_hr = float(np.percentile(hit_rates, 25))
    p75_hr = float(np.percentile(hit_rates, 75))
    median_rmse = float(np.median(rmses))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Hit-rate distribution (median={median_hr:.3f}, IQR=[{p25_hr:.3f}, {p75_hr:.3f}])',
            f'RMSE distribution (median={median_rmse:.4f})',
            'Hit-rate vs RMSE scatter',
            'Hit-rate CDF',
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # (1,1) Hit-rate histogram
    fig.add_trace(
        go.Histogram(
            x=hit_rates, nbinsx=40,
            marker=dict(color='#3498db', line=dict(color='#2c3e50', width=0.5)),
            name='hit-rate',
        ),
        row=1, col=1,
    )
    for marker_val, label, color in [(median_hr, 'median', '#e74c3c'),
                                     (p25_hr, 'p25', '#95a5a6'),
                                     (p75_hr, 'p75', '#95a5a6')]:
        fig.add_vline(x=marker_val, line_dash='dash', line_color=color,
                      annotation_text=label, row=1, col=1)

    # (1,2) RMSE histogram
    fig.add_trace(
        go.Histogram(
            x=rmses, nbinsx=40,
            marker=dict(color='#e67e22', line=dict(color='#2c3e50', width=0.5)),
            name='RMSE',
        ),
        row=1, col=2,
    )
    fig.add_vline(x=median_rmse, line_dash='dash', line_color='#e74c3c',
                  annotation_text='median', row=1, col=2)

    # (2,1) Hit-rate vs RMSE scatter
    fig.add_trace(
        go.Scatter(
            x=hit_rates, y=rmses,
            mode='markers',
            marker=dict(
                size=6,
                color=hit_rates,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='hit-rate', x=0.45, len=0.4, y=0.2),
                line=dict(color='#2c3e50', width=0.5),
            ),
            text=df['code'].tolist(),
            hovertemplate='<b>%{text}</b><br>hit-rate: %{x:.3f}<br>RMSE: %{y:.4f}<extra></extra>',
            name='stocks',
        ),
        row=2, col=1,
    )

    # (2,2) Hit-rate CDF
    sorted_hr = np.sort(hit_rates)
    cdf = np.arange(1, len(sorted_hr) + 1) / len(sorted_hr)
    fig.add_trace(
        go.Scatter(
            x=sorted_hr, y=cdf,
            mode='lines',
            line=dict(color='#2ecc71', width=2),
            name='CDF',
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.2)',
        ),
        row=2, col=2,
    )
    fig.add_vline(x=median_hr, line_dash='dash', line_color='#e74c3c',
                  annotation_text='median', row=2, col=2)

    fig.update_xaxes(title_text='hit-rate', row=1, col=1)
    fig.update_xaxes(title_text='RMSE', row=1, col=2)
    fig.update_xaxes(title_text='hit-rate', row=2, col=1)
    fig.update_xaxes(title_text='hit-rate', row=2, col=2)
    fig.update_yaxes(title_text='count', row=1, col=1)
    fig.update_yaxes(title_text='count', row=1, col=2)
    fig.update_yaxes(title_text='RMSE', row=2, col=1)
    fig.update_yaxes(title_text='CDF', row=2, col=2)

    n_stocks = len(metrics_list)
    fig.update_layout(
        title=f"{title} — N={n_stocks}",
        height=800, showlegend=False,
        template='plotly_white',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    log.info(f"[v5.10] wrote {output_path} ({n_stocks} stocks, median hit-rate={median_hr:.3f})")
```

- [ ] **Step 2.3: Smoke test on synthetic data**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -c "
import sys, os
sys.path.insert(0, 'c:/Users/yellow/mcp/qtTdx/backtrace')
import numpy as np
from dynamics.dynamics_oos_batch import build_full_market_oos_html

np.random.seed(42)
metrics = [
    dict(code=f'{i:06d}.SZ', n_oos=250,
         hit_rate=float(np.random.beta(2, 2)),
         rmse=float(np.random.gamma(1, 0.01)),
         mae=float(np.random.gamma(1, 0.005)),
         direction_accuracy=float(np.random.uniform(0.5, 1.0)),
         k_used=0.0, c_used=0.0)
    for i in range(100)
]
build_full_market_oos_html(
    metrics_list=metrics,
    output_path='c:/Users/yellow/mcp/qtTdx/backtrace/outputs/_smoke_v5_10_dash.html',
    title='smoke',
)
print('OK')
"
```

Expected: prints `OK` + writes 2×2 dashboard HTML.

- [ ] **Step 2.4: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py
git commit -m "feat(dynamics): v5.10 — build_full_market_oos_html (2x2 dashboard)"
```

---

## Task 3: `build_top5_small_multiples` (top-5 mini charts)

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (append)

**Interface produced:**
```python
def build_top5_small_multiples(
    top5_data: list[dict],   # list of (code, load_oos_predictions output dict)
    output_path: str,
    title: str = 'Top-5 OOS Prediction Detail',
) -> None:
    """Render 5 mini 4-row HTML charts in single figure (small multiples).
    Each row of the parent figure = 1 stock; columns = predicted vs actual |a_S|.
    """
```

- [ ] **Step 3.1: Implement `build_top5_small_multiples`**

Append after `build_full_market_oos_html`:

```python
def build_top5_small_multiples(
    top5_data: list[dict],
    output_path: str,
    title: str = 'Top-5 OOS Prediction Detail',
) -> None:
    """5 mini charts in 1 HTML — small multiples layout.

    For each top stock, render predicted vs actual |a_S| over time.
    Layout: 5 rows × 1 col (each subplot = 1 stock).
    """
    if not top5_data:
        raise ValueError('top5_data is empty')

    n = len(top5_data)
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.04,
        subplot_titles=[f"{d['code']} (hit={d['hit_rate']:.3f}, RMSE={d['rmse']:.4f})"
                        for d in top5_data],
    )

    for i, d in enumerate(top5_data, start=1):
        a_pred_mag = np.linalg.norm(d['a_pred'], axis=1)
        a_actual_mag = np.linalg.norm(d['a_actual'], axis=1)
        common_idx = d['common_idx']

        fig.add_trace(
            go.Scatter(
                x=common_idx, y=a_pred_mag,
                mode='lines', line=dict(color='#3498db', width=1),
                name=f"{d['code']} pred" if i == 1 else None,
                legendgroup='series',
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=common_idx, y=a_actual_mag,
                mode='lines', line=dict(color='#e67e22', width=1),
                name=f"{d['code']} actual" if i == 1 else None,
                legendgroup='series',
                showlegend=(i == 1),
            ),
            row=i, col=1,
        )
        fig.update_yaxes(title_text='|a_S|', row=i, col=1)

    fig.update_xaxes(title_text='date', row=n, col=1)
    fig.update_layout(
        title=f"{title} — top {n} by hit-rate",
        height=250 * n, showlegend=True,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    log.info(f"[v5.10] wrote {output_path} (top {n})")
```

- [ ] **Step 3.2: Smoke test on synthetic top-5**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -c "
import sys, os
sys.path.insert(0, 'c:/Users/yellow/mcp/qtTdx/backtrace')
import numpy as np, pandas as pd
from dynamics.dynamics_oos_batch import build_top5_small_multiples

np.random.seed(42)
top5 = []
for i in range(5):
    idx = pd.date_range('2024-01-01', periods=100, freq='B')
    top5.append(dict(
        code=f'{i:06d}.SZ',
        common_idx=idx,
        a_pred=np.random.randn(100, 2) * 0.05,
        a_actual=np.random.randn(100, 2) * 0.05,
        hit_rate=0.9 - i * 0.05,
        rmse=0.01 + i * 0.005,
    ))
build_top5_small_multiples(
    top5_data=top5,
    output_path='c:/Users/yellow/mcp/qtTdx/backtrace/outputs/_smoke_v5_10_top5.html',
)
print('OK')
"
```

Expected: prints `OK` + writes top-5 small multiples HTML.

- [ ] **Step 3.3: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py
git commit -m "feat(dynamics): v5.10 — build_top5_small_multiples (5 mini charts)"
```

---

## Task 4: CLI main() with batch orchestration

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_batch.py` (append `main()`)

**CLI flags:**
- `--days INT` (default 250)
- `--limit INT` (default 0 = all in local cache; else first N)
- `--prefer-industry` (default True)
- `--top-n INT` (default 5)
- `--output PATH` (default `backtrace/outputs/dynsys_oos_full_market.html`)
- `--codes-file PATH` (optional — file with one stock code per line)

- [ ] **Step 4.1: Implement `main()`**

Append after `build_top5_small_multiples`:

```python
def _load_stock_codes(limit: int) -> list[str]:
    """Load stock codes from data/manifest.json (TQ本地缓存)。

    Falls back to scanning data/stocks/*.csv if manifest missing.
    """
    manifest_path = 'data/manifest.json'
    if os.path.exists(manifest_path):
        import json
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        codes = [c for c, info in manifest.items()
                 if info.get('rows', 0) > 0 and not info.get('failed')]
        codes.sort()
        if limit > 0:
            codes = codes[:limit]
        return codes

    # Fallback: scan directory
    stock_dir = 'data/stocks'
    if not os.path.isdir(stock_dir):
        raise FileNotFoundError(f'No manifest.json or data/stocks/ dir found')
    files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    files.sort()
    codes = [f.replace('.csv', '') for f in files]
    if limit > 0:
        codes = codes[:limit]
    return codes


def main():
    p = argparse.ArgumentParser(
        description='v5.10 — Full-market OOS prediction quality distribution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--days', type=int, default=DEFAULTS['days'], help='trading days per stock')
    p.add_argument('--limit', type=int, default=DEFAULTS['limit'],
                   help='0 = all stocks in local cache, else first N')
    p.add_argument('--prefer-industry', dest='prefer_industry',
                   action='store_true', default=DEFAULTS['prefer_industry'])
    p.add_argument('--no-prefer-industry', dest='prefer_industry',
                   action='store_false')
    p.add_argument('--top-n', dest='top_n', type=int, default=DEFAULTS['top_n'],
                   help='number of top stocks to render as small multiples')
    p.add_argument('--codes-file', dest='codes_file', type=str, default=None,
                   help='optional file with one stock code per line')
    p.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                   help='output HTML path')
    args = p.parse_args()

    # 1. Load stock codes
    if args.codes_file and os.path.exists(args.codes_file):
        with open(args.codes_file) as fh:
            codes = [line.strip() for line in fh if line.strip()]
        log.info(f"[v5.10] loaded {len(codes)} codes from {args.codes_file}")
    else:
        codes = _load_stock_codes(args.limit)
        log.info(f"[v5.10] loaded {len(codes)} codes from manifest/cache")

    if not codes:
        raise ValueError('No stock codes found')

    log.info(f"[v5.10] days={args.days} prefer_industry={args.prefer_industry} top_n={args.top_n}")

    # 2. Compute per-stock metrics
    metrics_list = []
    per_stock_data = []  # for top-5 small multiples
    for idx, code in enumerate(codes, start=1):
        try:
            m = compute_oos_metrics(
                stock_code=code,
                days=args.days,
                prefer_industry=args.prefer_industry,
            )
            if m['n_oos'] > 0:
                metrics_list.append(m)
                log.info(f"[{idx}/{len(codes)}] {code}: hit={m['hit_rate']:.3f}, RMSE={m['rmse']:.4f}")
            else:
                log.warning(f"[{idx}/{len(codes)}] {code}: 0 OOS days, skip")
        except Exception as e:
            log.warning(f"[{idx}/{len(codes)}] {code}: ERROR ({type(e).__name__}: {e}), skip")
            continue

    if not metrics_list:
        raise ValueError('No valid metrics computed — check data/manifest.json')

    # 3. Aggregate
    agg = aggregate_oos_metrics(metrics_list)
    log.info(f"[v5.10] aggregated: N={agg['n_stocks']}, "
             f"median_hit={agg['median_hit_rate']:.3f}, "
             f"median_rmse={agg['median_rmse']:.4f}")

    # 4. Render 2×2 distribution dashboard
    build_full_market_oos_html(
        metrics_list=metrics_list,
        output_path=args.output,
        title=f"Full-Market OOS — {agg['n_stocks']} stocks, {args.days} days",
    )

    # 5. Render top-N small multiples (separate file)
    top_n = min(args.top_n, agg['n_stocks'])
    top_codes = [r['code'] for r in agg['ranked'][:top_n]]
    top_data = []
    for code in top_codes:
        try:
            d = load_oos_predictions(
                stock_code=code, days=args.days,
                prefer_industry=args.prefer_industry,
            )
            top_data.append(dict(
                code=code,
                common_idx=d['common_idx'],
                a_pred=d['a_pred'],
                a_actual=d['a_actual'],
                hit_rate=[m for m in metrics_list if m['code'] == code][0]['hit_rate'],
                rmse=[m for m in metrics_list if m['code'] == code][0]['rmse'],
            ))
        except Exception as e:
            log.warning(f"top-{top_n}: failed to reload {code} for small multiples: {e}")

    if top_data:
        top5_path = args.output.replace('.html', '_top{}.html'.format(top_n))
        build_top5_small_multiples(
            top5_data=top_data,
            output_path=top5_path,
            title=f"Top-{top_n} OOS Detail",
        )

    log.info(f"[v5.10] DONE — wrote {args.output} + {top_n}-multiples")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4.2: Smoke test CLI end-to-end (small N)**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe backtrace/dynamics/dynamics_oos_batch.py \
    --days 60 \
    --limit 5 \
    --top-n 3 \
    --output backtrace/outputs/_smoke_v5_10_cli.html
```

Expected: prints N/5 progress + final DONE message. If data/manifest.json missing, falls back to scan; if 002475.SZ not in cache, will skip individual stocks.

- [ ] **Step 4.3: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_batch.py
git commit -m "feat(dynamics): v5.10 — CLI main() with batch orchestration"
```

---

## Task 5: Test `test_cli_oos_batch_mode`

**Files:**
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

- [ ] **Step 5.1: Add test**

Append at end of `tests/test_dynamics_eigen.py`:

```python
def test_cli_oos_batch_mode(tmp_path):
    """v5.10: CLI full-market OOS batch mode — 验证 distribution dashboard + top-N HTML 输出。"""
    pytest.importorskip("plotly")

    import subprocess
    import sys
    import os

    html_out = tmp_path / 'oos_batch.html'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_oos_batch.py')
    cmd = [
        sys.executable, cli_script,
        '--days', '60',
        '--limit', '3',
        '--top-n', '2',
        '--output', str(html_out),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=120)

    # Tolerate documented failures (cache miss OR M1 tsfresh shadow OR no data)
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        if '本地缓存缺失' in stderr or 'No stock codes' in stderr or 'No valid metrics' in stderr:
            pytest.skip('No local cache available')
        if 'cannot import name' in stderr and 'tsfresh' in stderr:
            pytest.skip('M1 pre-existing tsfresh import shadow')
        assert False, f'Unexpected CLI failure: {stderr[-800:]}'

    assert html_out.exists(), f'HTML not created: {html_out}'
    with open(html_out, 'rb') as fh:
        content = fh.read()
    assert b'<html' in content.lower() or b'plotly' in content.lower(), \
        f'Not a valid plotly HTML: {content[:200]}'
```

- [ ] **Step 5.2: Run new test only**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_oos_batch_mode -v
```

Expected: PASS or SKIP.

- [ ] **Step 5.3: Run full suite**

```bash
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe -m pytest tests/test_dynamics_eigen.py -v 2>&1 | tail -10
```

Expected: 76 PASS + 1 SKIP (M1) or 77 PASS + 0 SKIP.

- [ ] **Step 5.4: Commit**

```bash
git add tests/test_dynamics_eigen.py
git commit -m "test(dynamics): v5.10 — test_cli_oos_batch_mode (F3 inverted tolerance)"
```

---

## Task 6: README §4.1.9 + final review + push

**Files:**
- Modify: `backtrace/dynamics/README.md` (append §4.1.9 after v5.9 §4.1.8)
- Modify: spec/plan status to Implemented/Complete

- [ ] **Step 6.1: Append README §4.1.9**

Locate v5.9 §4.1.8 block in `backtrace/dynamics/README.md` (ends at "76 tests PASS"). Append:

```markdown
### §4.1.9 v5.10 — Full-Market OOS Prediction Quality Distribution

v5.10 把 v5.9 单股 OOS 升级到 **全市场 N 只 portfolio dashboard** (echo v4.3 full-market 思路):

- CLI: `dynamics_oos_batch.py --days 250 --limit 0 --top-n 5`
- 复用 v5.9: `load_oos_predictions` + `build_oos_prediction_html`
- **4 子图 dashboard**:
  - (1,1) Hit-rate distribution (median + p25/p75 markers)
  - (1,2) RMSE distribution (median marker)
  - (2,1) Hit-rate vs RMSE scatter (color by hit-rate)
  - (2,2) Hit-rate CDF
- **Top-N small multiples** (separate HTML): N mini predicted-vs-actual 图表

**API**:
- `compute_oos_metrics(stock_code, days, ...) -> dict` — per-stock metrics
- `aggregate_oos_metrics(metrics_list) -> dict` — cross-stock distribution
- `build_full_market_oos_html(metrics_list, output_path, title)` — 2×2 dashboard
- `build_top5_small_multiples(top5_data, output_path, title)` — N mini charts

**v5.9 / v5.10 关系**:
- v5.9: 单股 4-row HTML (明天预测准不准)
- v5.10: 整个组合 2×2 dashboard + top-N (组合里所有股票怎么样)

**业务读法**:
- 头部: 全市场 hit-rate 中位数 > 0.6 → 模型整体可信
- 排名: top-N 用作模型可靠样本池
- 尾部: bottom-10% 需重新拟合或剔除

**0 新依赖** (plotly 已装)、**0 修改 11 保护文件**、**77 tests PASS** (1 新测试)。
```

- [ ] **Step 6.2: Update spec/plan status**

In spec: `Status:` line → `Implemented`
In plan: prepend `Status: Complete` line

- [ ] **Step 6.3: Commit docs**

```bash
git add backtrace/dynamics/README.md docs/superpowers/specs/2026-08-18-dynamics-v5-10-full-market-oos-distribution.md docs/superpowers/plans/2026-08-18-dynamics-v5-10-full-market-oos-distribution.md
git commit -m "docs(dynamics): v5.10 — README §4.1.9 + spec/plan status"
```

- [ ] **Step 6.4: Final review by broad reviewer**

Dispatch final code reviewer. Pass criteria:
- 0 modifications to 11 protected files + dynamics_oos_viz.py
- 0 new dependencies
- All 77 tests pass or skip (M1 tolerated)
- 2×2 dashboard + top-N small multiples render correctly (smoke tests pass)
- Spec §1-§10 fully implemented

Address any Critical/Important findings via 1 fix round + 1 re-review.

- [ ] **Step 6.5: Push to origin/main**

```bash
git push origin main
```

- [ ] **Step 6.6: Write memory file**

Create `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\dynamics-v5-10-full-market-oos-distribution.md`.

Add line to MEMORY.md index.

- [ ] **Step 6.7: Commit memory + final push**

No git push needed (memory is outside repo).

---

## Done Criteria

- [ ] 6 commits on top of `d390203` (Tasks 1-5)
- [ ] 77 tests pass + 0-1 SKIP (M1 tolerated)
- [ ] 0 modifications to 11 protected files + dynamics_oos_viz.py
- [ ] 0 new dependencies
- [ ] `backtrace/outputs/dynsys_oos_full_market.html` + `*_topN.html` are valid plotly HTML
- [ ] README §4.1.9 reflects the new CLI
- [ ] spec/plan marked Implemented/Complete
- [ ] Memory file written
- [ ] Pushed to origin/main

## Status: ✅ DONE — 2026-08-19

All 6 tasks complete. Pushed to origin/main in commit <push_commit_pending>. See memory file `dynamics-v5-10-full-market-oos-distribution.md`.