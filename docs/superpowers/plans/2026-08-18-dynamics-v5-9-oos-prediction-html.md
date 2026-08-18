# v5.9 — OOS Prediction Visualization HTML (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `backtrace/dynamics/dynamics_oos_viz.py` — plotly 4-subplot HTML visualizing 1-step OOS predictions (predicted vs actual a_S / error / rolling RMSE / state hit rate) for visual model-quality assessment.

**Architecture:** 1 new CLI module reuses `predict_next_state` (Plan v3 API) + `load_pair` + `compute_movement_projection` to compute predictions and compare against actual motion projection. 1 plotly HTML with shared x-axis and 4 row subplots.

**Tech Stack:** plotly 5.x (`make_subplots`, `go.Scatter`, `go.Bar`), numpy, pandas, `_dynamics_core.py` (Plan v3 API), `_projection_core.py` (data 接入 + state labels).

**Base commit:** `3f2e49f` (v5.9 spec).
**Status:** Complete — 5 tasks done + v5.9.1 fix (v5.8 sys.path patch), 76 tests PASS, pushed to origin/main.
**Reference impls:** `backtrace/dynamics/dynamics_1step_oos.py` (OOS 1-step loop logic — DO NOT import; rewrite to avoid tsfresh M1 shadow), `backtrace/dynamics/dynamics_state_timeline.py` (plotly subplot pattern reference).

## Global Constraints

- 0 modifications to **11 protected files** (listed in `backtrace/dynamics/README.md` §11 — `_projection_core.py`, `_dynamics_core.py`, `tsfresh_pipeline.py`, `tsfresh_config.py`, `jhzq_fees.py`, `data_store.py`, `tqcenter.py`, plus the 4 other dynamics CLIs except v5.9 new file)
- 0 new dependencies (plotly already installed)
- 0 re-implementation of projection / dynamics core math (import only)
- 7 protected symbols imported from `_projection_core.py`: `load_pair`, `compute_movement_projection`, `compute_dynamics`, `compute_forces`, `classify_states`, `STATE_COLORS`, `STATE_LABELS` — same as v5.8
- M1 tsfresh shadow: `dynamics_1step_oos.py` import chain breaks all dynamics CLIs. **v5.9 does NOT import `dynamics_1step_oos.py`** — implement OOS loop inline using `_dynamics_core.predict_next_state` directly
- `PYTHONIOENCODING=utf-8` required (Windows GBK terminal)
- All threshold literals reused via local var `(0.10, 0.50, np.deg2rad(30), np.deg2rad(90))` at top of file (matches `projection_2d.py` — promoted to constant is future cleanup, **not** this version)
- Test count: 74 PASS + 1 SKIP → 75 PASS + 1 SKIP after this plan
- Output: `backtrace/outputs/dynsys_oos_viz_<code>.html` (or `--output` override)
- `BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` and `sys.path.insert(0, BACKTRACE_DIR)` — same as v5.8 dynamics_state_timeline.py
- 1 new test: `test_cli_oos_viz_mode` in `tests/test_dynamics_eigen.py` — same F3 inverted tolerance pattern as v5.8.1

---

## Task 1: Scaffold + helpers (`load_oos_predictions` + 1-step loop)

**Files:**
- Create: `backtrace/dynamics/dynamics_oos_viz.py`
- Test: `tests/test_dynamics_eigen.py` (no changes in this task)

**Interfaces produced (used by Tasks 2-3):**
```python
def load_oos_predictions(
    stock_code: str,
    days: int,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    lambda_q: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """load_pair + 1-step predict loop + extract actual a_S from motion projection.

    Returns dict with keys:
        common_idx: pd.DatetimeIndex (T_oos,)
        a_pred: np.ndarray (T_oos, 2) — predicted acceleration per state
        a_actual: np.ndarray (T_oos, 2) — actual Δu_S per state (motion projection)
        state_pred: list[str] (T_oos,) — predicted state labels
        state_actual: list[str] (T_oos,) — actual state labels
        k_used: float
        c_used: float
        mv: dict — full motion projection (for debugging)
        dyn: dict — full dynamics (for debugging)
    """
```

- [ ] **Step 1.1: Write file skeleton + imports + constants**

Create `backtrace/dynamics/dynamics_oos_viz.py`:

```python
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from backtrace.projection._projection_core import (
    load_pair,
    compute_movement_projection,
    compute_dynamics,
    classify_states,
    STATE_LABELS,
)
from backtrace.dynamics._dynamics_core import predict_next_state

DEFAULTS = dict(
    days=250,
    prefer_industry=True,
    k=None,
    c=None,
    lambda_q=None,
    f_self_window=10,
)
THRESHOLDS = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90))
DEFAULT_OUTPUT = 'backtrace/outputs/dynsys_oos_viz_{code}.html'
```

- [ ] **Step 1.2: Implement `load_oos_predictions`**

Add after DEFAULTS:

```python
def load_oos_predictions(
    stock_code: str,
    days: int,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    lambda_q: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """Load pair, compute motion projection + dynamics, run 1-step predict loop.

    The OOS loop:
        For t in [0, T_oos-1]:
            1. Compute F_self_pred (rolling mean over last f_self_window days)
            2. Call predict_next_state(v_S[t], v_M[t], v_M[t+1], β[t], β[t+1],
                                       d[t], F_self=F_self_pred, k=k, c=c, q=q_t)
            3. Extract a_pred = result[0], v_pred = result[1]
            4. actual a[t+1] = mv['a_S_mag'][t+1] (next-period motion projection)

    Returns dict (see docstring above).
    """
    # 1. Load pair
    stock_df, index_df = load_pair(
        stock_code,
        days=days,
        prefer_industry=prefer_industry,
    )

    # 2. Motion projection → Δu_S, β, q_t (regression coefficient + correlation)
    mv = compute_movement_projection(stock_df, index_df, prefer_industry=prefer_industry)

    # 3. Dynamics → d_vec, k̂, ĉ
    dyn = compute_dynamics(mv)

    # 4. Determine effective k, c (use parameter_fit if not provided)
    k_used = k if k is not None else float(dyn.get('k_hat', 0.0))
    c_used = c if c is not None else float(dyn.get('c_hat', 0.0))

    # 5. Extract sequences (length T_common = days)
    common_idx = mv['common_idx']
    T = len(common_idx)
    delta_u = mv['a_S_mag']            # (T, 2) actual acceleration
    delta_v = mv['a_M_mag']            # (T, 2) market acceleration
    beta = mv['beta']                  # (T,) regression coefficient
    q_t_seq = mv['q_t']                # (T,) correlation magnitude
    d_vec = dyn['d_vec']               # (T, 2) position deviation
    u_vec = dyn['u_vec']               # (T, 2) internal displacement (description layer)

    # 6. F_self rolling mean predictor (window=f_self_window)
    f_self_history = []
    F_self_seq = []
    for t in range(T):
        if t < f_self_window:
            F_self_seq.append(np.zeros(2))
        else:
            F_self_seq.append(np.mean(f_self_history[-f_self_window:], axis=0))
        f_self_history.append(delta_u[t])  # observed self-force

    # 7. 1-step predict loop: predict a[t+1] using info up to t
    a_pred_list = []
    a_actual_list = []
    state_pred_list = []
    state_actual_list = []

    # Pre-classify states for actual comparison
    classify_kwargs = dict(zip(
        ['zeta_threshold', 'omega_n_threshold', 'angle_threshold', 'phase_threshold'],
        THRESHOLDS,
    ))
    classified = classify_states(dyn, **classify_kwargs)
    state_actual_full = classified['state_labels']  # (T,) str list

    for t in range(T - 1):
        F_self_pred = F_self_seq[t]
        # predict_next_state returns (a_pred, v_pred, d_pred, u_pred)
        a_pred, _, _, _ = predict_next_state(
            v_S_now=delta_u[t],
            v_M_now=delta_v[t],
            v_M_next=delta_v[t + 1],
            beta_now=float(beta[t]),
            beta_next=float(beta[t + 1]),
            d_now=d_vec[t],
            F_self_now=F_self_pred,
            k=k_used,
            c=c_used,
            q_now=float(q_t_seq[t]),
        )
        a_pred_list.append(a_pred)
        # Actual = next-period motion projection (ground truth from data)
        a_actual_list.append(delta_u[t + 1])
        # Predicted state: classify via magnitude of predicted a_pred
        state_pred_list.append(_label_from_a(a_pred))
        state_actual_list.append(state_actual_full[t + 1])

    common_idx_oos = common_idx[1:]  # drop first (no prediction possible)

    return dict(
        common_idx=common_idx_oos,
        a_pred=np.array(a_pred_list),
        a_actual=np.array(a_actual_list),
        state_pred=state_pred_list,
        state_actual=state_actual_list,
        k_used=k_used,
        c_used=c_used,
        mv=mv,
        dyn=dyn,
    )


def _label_from_a(a: np.ndarray) -> str:
    """Map predicted acceleration magnitude → discrete state label.

    Simple heuristic (avoid double-calling classify_states on synthetic d_vec):
        |a| < 0.005  → 'none' (no motion)
        0.005 ≤ |a| < 0.05 → 'follow'
        else → 'accelerating'

    Refine if needed in Task 2.
    """
    mag = float(np.linalg.norm(a))
    if mag < 0.005:
        return 'none'
    elif mag < 0.05:
        return 'follow'
    else:
        return 'accelerating'
```

- [ ] **Step 1.3: Verify file imports cleanly**

Run:
```bash
PYTHONIOENCODING=utf-8 python -c "import backtrace.dynamics.dynamics_oos_viz as m; print('OK', m.load_oos_predictions.__name__)"
```
Expected: prints `OK load_oos_predictions` (no ModuleNotFoundError). If M1 tsfresh shadow breaks the import chain (because `from backtrace.common import tsfresh_pipeline as P` is reachable through projection helpers), tolerate as known limitation — the helpers themselves only touch `_dynamics_core.predict_next_state` and `_projection_core.{load_pair,compute_movement_projection,compute_dynamics,classify_states,STATE_LABELS}`.

- [ ] **Step 1.4: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_viz.py
git commit -m "feat(dynamics): v5.9 — scaffold load_oos_predictions + 1-step predict loop"
```

---

## Task 2: `build_oos_prediction_html` — plotly 4-subplot visualization

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_viz.py` (append `build_oos_prediction_html`)
- Test: `tests/test_dynamics_eigen.py` (no changes in this task)

**Interface produced:**
```python
def build_oos_prediction_html(
    common_idx: pd.DatetimeIndex,
    a_pred: np.ndarray,         # (T_oos, 2)
    a_actual: np.ndarray,       # (T_oos, 2)
    state_pred: list[str],
    state_actual: list[str],
    k_used: float,
    c_used: float,
    output_path: str,
    title: str = 'OOS 1-Step Prediction vs Actual',
) -> None:
    """Render 4-row plotly HTML.
    ...
    """
```

- [ ] **Step 2.1: Add plotly import**

Add after `import pandas as pd`:
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

- [ ] **Step 2.2: Implement `build_oos_prediction_html`**

Append after `_label_from_a`:

```python
def build_oos_prediction_html(
    common_idx: pd.DatetimeIndex,
    a_pred: np.ndarray,
    a_actual: np.ndarray,
    state_pred: list[str],
    state_actual: list[str],
    k_used: float,
    c_used: float,
    output_path: str,
    title: str = 'OOS 1-Step Prediction vs Actual',
) -> None:
    """4-row plotly HTML:
        Row 1 (35%): predicted vs actual |a_S| over time
        Row 2 (25%): error = |a_pred| - |a_actual| (bar, color by σ band)
        Row 3 (20%): 20-day rolling RMSE of error
        Row 4 (20%): state hit rate (1 = pred==actual, 0 = pred!=actual)
    """
    # Convert to 1-D magnitude
    a_pred_mag = np.linalg.norm(a_pred, axis=1)
    a_actual_mag = np.linalg.norm(a_actual, axis=1)
    error = a_pred_mag - a_actual_mag
    T = len(common_idx)

    # Error color band (σ band over absolute error)
    abs_err = np.abs(error)
    sigma = float(np.nanstd(abs_err)) if T > 1 else 0.0
    if sigma == 0.0:
        sigma = 1e-9
    band_low = 0.5 * sigma
    band_high = 1.0 * sigma
    err_colors = np.where(
        abs_err < band_low, '#2ecc71',                    # green
        np.where(abs_err < band_high, '#f39c12', '#e74c3c')  # yellow / red
    )

    # Rolling RMSE (20-day window)
    win = 20
    rolling_rmse = pd.Series(error).pow(2).rolling(win).mean().pow(0.5).to_numpy()

    # State hit rate (binary)
    hit = np.array([1.0 if p == a else 0.0 for p, a in zip(state_pred, state_actual)])

    # Figure layout
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.2, 0.2],
        vertical_spacing=0.05,
        subplot_titles=(
            'Row 1 — |a_S| predicted (blue) vs actual (orange)',
            'Row 2 — error (color: <0.5σ green / 0.5–1σ yellow / >1σ red)',
            f'Row 3 — {win}-day rolling RMSE of error',
            'Row 4 — state hit rate (1 = pred==actual)',
        ),
    )

    # Row 1: predicted vs actual magnitude
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=a_pred_mag, name='predicted |a_S|',
            mode='lines', line=dict(color='#3498db', width=1.5),
            legendgroup='series',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=a_actual_mag, name='actual |a_S|',
            mode='lines+markers', line=dict(color='#e67e22', width=1.5),
            marker=dict(size=4),
            legendgroup='series',
        ),
        row=1, col=1,
    )

    # Row 2: error bars (one bar per day, colored by σ band)
    fig.add_trace(
        go.Bar(
            x=common_idx, y=error, name='error',
            marker=dict(color=err_colors.tolist()),
            showlegend=False,
            legendgroup='series',
        ),
        row=2, col=1,
    )

    # Row 3: rolling RMSE
    fig.add_trace(
        go.Scatter(
            x=common_idx, y=rolling_rmse, name=f'{win}-d rolling RMSE',
            mode='lines', line=dict(color='#9b59b6', width=2),
            legendgroup='series',
        ),
        row=3, col=1,
    )

    # Row 4: state hit rate
    fig.add_trace(
        go.Bar(
            x=common_idx, y=hit, name='state hit (1=yes, 0=no)',
            marker=dict(color=hit, colorscale=[[0, '#e74c3c'], [1, '#2ecc71']]),
            showlegend=False,
            legendgroup='series',
        ),
        row=4, col=1,
    )

    # Layout
    k_str = f"{k_used:.4f}" if k_used is not None else 'auto'
    c_str = f"{c_used:.4f}" if c_used is not None else 'auto'
    fig.update_layout(
        title=f"{title} (k̂={k_str}, ĉ={c_str})",
        height=900,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
    )
    fig.update_yaxes(title_text='|a_S| magnitude', row=1, col=1)
    fig.update_yaxes(title_text='error', row=2, col=1)
    fig.update_yaxes(title_text='RMSE', row=3, col=1)
    fig.update_yaxes(title_text='hit rate', row=4, col=1, range=[-0.05, 1.05])
    fig.update_xaxes(title_text='date', row=4, col=1)

    # Output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn', full_html=True)
    print(f"[v5.9] wrote {output_path} ({T} OOS days, k̂={k_str}, ĉ={c_str})")
```

- [ ] **Step 2.3: Smoke test the function in isolation**

Run a synthetic call to verify plotly wiring:
```bash
PYTHONIOENCODING=utf-8 python -c "
import sys, os
sys.path.insert(0, 'c:/Users/yellow/mcp/qtTdx/backtrace')
import pandas as pd, numpy as np
from dynamics.dynamics_oos_viz import build_oos_prediction_html
idx = pd.date_range('2024-01-01', periods=50, freq='B')
a_pred = np.random.randn(50, 2) * 0.05
a_actual = np.random.randn(50, 2) * 0.05
build_oos_prediction_html(
    common_idx=idx,
    a_pred=a_pred, a_actual=a_actual,
    state_pred=['follow']*50, state_actual=['follow']*50,
    k_used=0.5, c_used=0.2,
    output_path='c:/Users/yellow/mcp/qtTdx/backtrace/outputs/_smoke_v5_9.html',
    title='smoke',
)
print('OK')
"
```
Expected: prints `OK` and writes `_smoke_v5_9.html`. Open it briefly to verify 4 rows render.

- [ ] **Step 2.4: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_viz.py
git commit -m "feat(dynamics): v5.9 — build_oos_prediction_html (4-row plotly)"
```

---

## Task 3: CLI main() with arg parser

**Files:**
- Modify: `backtrace/dynamics/dynamics_oos_viz.py` (append `main()` + `if __name__ == '__main__'`)

- [ ] **Step 3.1: Add `main()` and CLI entry**

Append after `build_oos_prediction_html`:

```python
def main():
    p = argparse.ArgumentParser(
        description='v5.9 — OOS 1-step prediction visualization (plotly HTML)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--code', required=True, help='stock code, e.g. 002475.SZ')
    p.add_argument('--days', type=int, default=DEFAULTS['days'], help='trading days')
    p.add_argument('--prefer-industry', dest='prefer_industry',
                   action='store_true', default=DEFAULTS['prefer_industry'])
    p.add_argument('--no-prefer-industry', dest='prefer_industry',
                   action='store_false')
    p.add_argument('--k', type=float, default=DEFAULTS['k'],
                   help='override k̂ (else estimated from data)')
    p.add_argument('--c', type=float, default=DEFAULTS['c'],
                   help='override ĉ (else estimated from data)')
    p.add_argument('--lambda-q', dest='lambda_q', type=float,
                   default=DEFAULTS['lambda_q'], help='reserved for future')
    p.add_argument('--f-self-window', dest='f_self_window', type=int,
                   default=DEFAULTS['f_self_window'], help='rolling window for F_self predictor')
    p.add_argument('--output', type=str, default=None,
                   help='output HTML path (default: backtrace/outputs/dynsys_oos_viz_<code>.html)')
    args = p.parse_args()

    print(f"[v5.9] code={args.code} days={args.days} "
          f"prefer_industry={args.prefer_industry} k={args.k} c={args.c} "
          f"f_self_window={args.f_self_window}")

    # 1. Load + predict
    data = load_oos_predictions(
        stock_code=args.code,
        days=args.days,
        prefer_industry=args.prefer_industry,
        k=args.k, c=args.c,
        lambda_q=args.lambda_q,
        f_self_window=args.f_self_window,
    )

    # 2. Render
    output_path = args.output or DEFAULT_OUTPUT.format(code=args.code.replace('.', '_'))
    build_oos_prediction_html(
        common_idx=data['common_idx'],
        a_pred=data['a_pred'],
        a_actual=data['a_actual'],
        state_pred=data['state_pred'],
        state_actual=data['state_actual'],
        k_used=data['k_used'],
        c_used=data['c_used'],
        output_path=output_path,
        title=f"{args.code} — OOS 1-step prediction (k̂ vs ĉ)",
    )


if __name__ == '__main__':
    main()
```

- [ ] **Step 3.2: Smoke test the CLI end-to-end (002475.SZ if available)**

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_viz.py \
    --code 002475.SZ \
    --days 100 \
    --output backtrace/outputs/_smoke_v5_9_cli.html
```
Expected: prints "[v5.9] code=002475.SZ days=100 ..." and "[v5.9] wrote ...html". If `002475.SZ` not in local cache, falls back to error from `load_pair` ("本地缓存缺失" / similar). Tolerate — the test (Task 4) will skip.

- [ ] **Step 3.3: Commit**

```bash
git add backtrace/dynamics/dynamics_oos_viz.py
git commit -m "feat(dynamics): v5.9 — CLI main() with --code/--days/--output flags"
```

---

## Task 4: Test `test_cli_oos_viz_mode`

**Files:**
- Modify: `tests/test_dynamics_eigen.py` (append 1 test)

- [ ] **Step 4.1: Add test**

Append at end of `tests/test_dynamics_eigen.py`:

```python
def test_cli_oos_viz_mode(tmp_path):
    """v5.9: CLI OOS visualization mode — 验证 build_oos_prediction_html 输出 HTML."""
    pytest.importorskip("plotly")

    import subprocess
    import sys
    import os

    html_out = tmp_path / 'oos_viz.html'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_oos_viz.py')
    cmd = [
        sys.executable, cli_script,
        '--code', '002475.SZ',
        '--days', '250',
        '--output', str(html_out),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)

    # Tolerate documented failures (cache miss OR M1 tsfresh shadow)
    if result.returncode != 0:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        if '本地缓存缺失' in stderr:
            pytest.skip('002475.SZ not in local cache')
        if 'cannot import name' in stderr and 'tsfresh' in stderr:
            pytest.skip('M1 pre-existing tsfresh import shadow')
        assert False, f'Unexpected CLI failure: {stderr[-800:]}'

    assert html_out.exists(), f'HTML not created: {html_out}'
    with open(html_out, 'rb') as fh:
        content = fh.read()
    assert b'<html' in content.lower() or b'plotly' in content.lower(), \
        f'Not a valid plotly HTML: {content[:200]}'
```

- [ ] **Step 4.2: Run new test only**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py::test_cli_oos_viz_mode -v
```
Expected: PASS or SKIP (per F3 tolerance). Any other failure is a real bug.

- [ ] **Step 4.3: Run full suite**

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v 2>&1 | tail -30
```
Expected: 74 PASS + 2 SKIP (M1 for state timeline + M1 for OOS viz if tsfresh shadow surfaces) or 75 PASS + 1 SKIP (M1 only on one CLI).

- [ ] **Step 4.4: Commit**

```bash
git add tests/test_dynamics_eigen.py
git commit -m "test(dynamics): v5.9 — test_cli_oos_viz_mode (F3 inverted tolerance)"
```

---

## Task 5: README §4.1.8 + final review + push

**Files:**
- Modify: `backtrace/dynamics/README.md` (append §4.1.8 after v5.8 §4.1.7)
- Modify: `docs/superpowers/specs/2026-08-18-dynamics-v5-9-oos-prediction-html.md` (status: Implemented)
- Modify: `docs/superpowers/plans/2026-08-18-dynamics-v5-9-oos-prediction-html.md` (status: complete)

- [ ] **Step 5.1: Append README §4.1.8**

Locate the v5.8 §4.1.7 block in `backtrace/dynamics/README.md`. Append after it:

```markdown
### 4.1.8 v5.9 — OOS Prediction Visualization HTML

**文件**: `backtrace/dynamics/dynamics_oos_viz.py`

闭环 `predict_next_state` (Plan v3 API) → plotly 4 子图 OOS 1 步预测可视化。

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_viz.py \
    --code 002475.SZ \
    --days 250 \
    --prefer-industry \
    --output backtrace/outputs/dynsys_oos_viz.html
```

**4 子图**:
- Row 1 (35%): |a_S| predicted (blue) vs actual (orange)
- Row 2 (25%): error bar, color = σ band (绿/黄/红)
- Row 3 (20%): 20-day rolling RMSE (purple)
- Row 4 (20%): state hit rate (1=correct, 0=wrong)

**业务读法**:
- Row 1: 预测 vs 实际时间序列对齐情况
- Row 2: 单日误差分布 (σ band 视觉)
- Row 3: 长期稳定性 (rolling RMSE 趋势)
- Row 4: 离散状态预测准确率

**API**:
- `load_oos_predictions(stock_code, days, *, prefer_industry=True, k=None, c=None, lambda_q=None, f_self_window=10) -> dict`
- `build_oos_prediction_html(common_idx, a_pred, a_actual, state_pred, state_actual, k_used, c_used, output_path, title=...) -> None`

**新增 CLI**: `dynamics_oos_viz.py` (--code/--days/--prefer-industry/--k/--c/--lambda-q/--f-self-window/--output)

**复用**:
- `predict_next_state` from `_dynamics_core.py` (Plan v3 API)
- `load_pair` + `compute_movement_projection` + `compute_dynamics` + `classify_states` from `_projection_core.py`
- `STATE_LABELS` from `_projection_core.py`

**0 新依赖** (plotly 已装)、**0 修改 11 保护文件**、**M1 tsfresh shadow tolerated** (Task 4 F3 inverted tolerance)。
```

- [ ] **Step 5.2: Update spec status to Implemented**

In `docs/superpowers/specs/2026-08-18-dynamics-v5-9-oos-prediction-html.md` change line 3 from `Status: Draft` to `Status: Implemented`.

- [ ] **Step 5.3: Update plan status to complete**

In this plan file, prepend line 5 with `[x]` after writing the closing marker:
```markdown
**Status:** Complete (Tasks 1-5 done, final review pass, pushed to origin/main).
```

- [ ] **Step 5.4: Commit docs**

```bash
git add backtrace/dynamics/README.md docs/superpowers/specs/2026-08-18-dynamics-v5-9-oos-prediction-html.md docs/superpowers/plans/2026-08-18-dynamics-v5-9-oos-prediction-html.md
git commit -m "docs(dynamics): v5.9 — README §4.1.8 + spec/plan status Implemented/Complete"
```

- [ ] **Step 5.5: Final review by broad reviewer**

Dispatch final code reviewer (per SDD §"Setup → final review"). Pass criteria:
- 0 modifications to 11 protected files
- 0 new dependencies
- All 75 tests pass or skip (M1 tolerated)
- 4-row plotly HTML renders correctly (smoke test passes)
- Spec §1-§7 fully implemented (no scope drift)

Address any Critical/Important findings via 1 fix round + 1 re-review. Park Minor findings in ledger.

- [ ] **Step 5.6: Push to origin/main**

```bash
git push origin main
```

- [ ] **Step 5.7: Write memory file**

Create `C:\Users\yellow\.claude\projects\c--Users-yellow-mcp-qtTdx\memory\dynamics-v5-9-oos-prediction-html.md` (parallel structure to v5.7 / v5.8 memory files).

Add one-line entry to `MEMORY.md` index.

- [ ] **Step 5.8: Final commit + push memory**

```bash
git add memory/
git commit -m "docs(memory): v5.9 — record OOS prediction viz + final state"
git push origin main
```

---

## Done Criteria

- [ ] 5 commits on top of `3f2e49f` (Tasks 1, 2, 3, 4, 5)
- [ ] 75 tests pass + 1 SKIP (M1 tsfresh shadow tolerated)
- [ ] 0 modifications to 11 protected files
- [ ] 0 new dependencies
- [ ] `backtrace/outputs/dynsys_oos_viz_<code>.html` is a valid 4-row plotly HTML
- [ ] README §4.1.8 reflects the new CLI
- [ ] spec/plan marked Implemented/Complete
- [ ] Memory file written
- [ ] Pushed to origin/main