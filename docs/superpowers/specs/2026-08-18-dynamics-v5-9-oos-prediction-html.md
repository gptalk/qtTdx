# v5.9 — OOS Prediction Visualization HTML

**Date:** 2026-08-18
**Status:** Implemented (v5.9 — 5 commits + v5.9.1 fix = 76 tests PASS)
**Base:** v5.8.1 (HEAD `ae9bf38`)
**Author:** Brainstorming output (user authorized: "按计划和推荐执行，不用问我")

## 1. Goal

给 `dynamics_1step_oos.py` 的 OOS 1 步预测输出加 **plotly 时间序列可视化**:
- 业务首次看到预测 vs 实际的时间序列
- 直观评估模型质量:哪天预测准 / 哪天跑偏
- 4 个子图:predicted a_S / actual a_S / error / 残差分布

**业务读法**:
- Top: "周三预测 a_S=0.02,实际 a_S=0.05,误差 0.03 — 偏小"
- Bottom: "预测误差分布显示 80% 误差 < 0.01,模型整体可用"

## 2. Why now

v5.8 = 时间域状态时间线(现在是什么)。
v5.9 = 1 步预测 vs 实际(明天的预测准不准)。

`dynamics_1step_oos.py` 已有 OOS 1 步预测逻辑,但**输出只有 CSV**。业务从未看到模型质量曲线。

模型可信度是所有量化研究的**最大信任基础**。v5.9 把这个基础可视化:
- 哪段时期预测准 → 业务可依赖
- 哪段时期预测跑偏 → 业务要知道

## 3. Design

### 3.1 架构

**复用**(0 新依赖):
- `predict_next_state(v_S_now, v_M_now, v_M_next, ...)` from `_dynamics_core.py` (Plan v3 API)
- `dynamics_1step_oos.py` 的 OOS 1 步预测逻辑(import or refactor)
- `compute_movement_projection` (proxy for actual Δu_S)
- `load_pair` from `_projection_core.py` (data 接入)
- `plotly.subplots.make_subplots(4, 1, shared_xaxes=True)`

**新增**:
- 1 新 CLI: `backtrace/dynamics/dynamics_oos_viz.py`
- 1 新函数: `build_oos_prediction_html(stock_df, index_df, ...)`
- 1 新数据接入函数: `load_oos_predictions(stock_code, days, ...)`
- 1 新 test: `test_cli_oos_viz_mode`

### 3.2 函数 `build_oos_prediction_html`

```python
def build_oos_prediction_html(
    common_idx: pd.DatetimeIndex,
    a_pred: np.ndarray,         # (T_oos,) predicted a_S
    a_actual: np.ndarray,       # (T_oos,) actual a_S (from motion projection)
    state_pred: list[str],      # (T_oos,) predicted state labels
    state_actual: list[str],    # (T_oos,) actual state labels
    k_used: float,              # (k̂, ĉ) used
    c_used: float,
    output_path: str,
    title: str = 'OOS 1-Step Prediction vs Actual',
) -> None:
    """Render 4-row plotly HTML: predicted a_S / actual a_S / error / hit rate.

    Args:
        common_idx: oos 期间交易日的 DatetimeIndex (T_oos,)
        a_pred: predicted a_S (T_oos,)
        a_actual: actual a_S (T_oos,)
        state_pred: predicted state labels (T_oos,)
        state_actual: actual state labels (T_oos,)
        k_used / c_used: 锁定参数(显示在 title)
        output_path: HTML 输出路径
        title: figure 标题
    """
```

**4 subplot layout**:
- Row 1: predicted a_S (line) + actual a_S (line, marker)
- Row 2: error = a_pred - a_actual (bar, color = sign)
- Row 3: rolling RMSE (line, 20-day window)
- Row 4: state hit rate (bar, "pred == actual" / "pred != actual")

### 3.3 数据接入

```python
def load_oos_predictions(
    stock_code: str,
    days: int,
    pipeline,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    lambda_q: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """load_pair → compute_movement_projection → 提取 actual a_S / 用 predict_next_state 跑预测。

    Returns:
        dict with keys: common_idx, a_pred, a_actual, state_pred, state_actual,
                       k_used, c_used, mv, dyn。
    """
```

**最简实现**: 1 步预测用最近 1 期的 (k̂, ĉ) 锁定参数,跑预测;actual 是下一期的 mv['a_S_mag']。

完整实现可参考 `dynamics_1step_oos.py` 内部已有逻辑,但**不依赖**其 CLI 入口(避免 tsfresh shadow M1)。

### 3.4 CLI

新文件 `backtrace/dynamics/dynamics_oos_viz.py`:

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_viz.py \
    --code 002475.SZ \
    --days 250 \
    --prefer-industry \
    --output backtrace/outputs/dynsys_oos_viz.html
```

**Flags**:
- `--code STR` (required)
- `--days INT` (default 250)
- `--prefer-industry` (default True)
- `--k FLOAT` (default None,自适应窗口)
- `--c FLOAT` (default None,自适应窗口)
- `--lambda-q FLOAT` (default None,自适应)
- `--f-self-window INT` (default 10)
- `--output PATH` (default `backtrace/outputs/dynsys_oos_viz.html`)

### 3.5 视觉规格

**Row 1 — predicted vs actual a_S**:
- 2 lines: `a_pred` (蓝色) + `a_actual` (橙色 marker)
- y-axis: a_S magnitude
- x-axis: date
- Hover: 显示 date, a_pred, a_actual, 误差

**Row 2 — error**:
- bar = `a_pred - a_actual`
- 颜色:绿(误差 < 0.5σ) / 黄(0.5σ-1σ) / 红(>1σ)
- y-axis: 误差 magnitude

**Row 3 — rolling RMSE**:
- 20-day rolling RMSE of error
- y-axis: RMSE magnitude
- 用于看预测稳定性随时间变化

**Row 4 — state hit rate**:
- daily bar: 1 (pred == actual) / 0 (pred != actual)
- y-axis: 0 or 1
- 直观看到哪天状态预测错

**Layout**:
- shared_xaxes=True
- row_heights=[0.35, 0.25, 0.2, 0.2]
- title: `<STOCK> @ <INDEX> — OOS 1-step prediction`

### 3.6 颜色 / 字体

- pred/actual: 蓝/橙(避免 4-color palette 混淆)
- error bar: 绿/黄/红
- rolling RMSE: 紫
- state hit rate: 绿/红

### 3.7 Edge cases

- 0 days → 0-row DataFrame → `ValueError`
- 1 day → 1-point chart (no rolling RMSE)
- α error / NaN → skip NaN in rolling
- 预测崩溃 → caller 友好报错

## 4. Files

| File | Type | Lines |
|---|---|---|
| `backtrace/dynamics/dynamics_oos_viz.py` | new | ~270 |
| `tests/test_dynamics_eigen.py` | modify | +44 |
| `backtrace/dynamics/README.md` | modify | +15 |
| `docs/superpowers/specs/2026-08-18-dynamics-v5-9-oos-prediction-html.md` | new | (this) |
| `docs/superpowers/plans/2026-08-18-dynamics-v5-9-oos-prediction-html.md` | new | (TBD) |

Total: ~570 lines, 5 files

## 5. 测试

新增 1 test:

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

合计:74 → 75 tests pass (1 new test, M1 tolerated as skip)。

## 6. 验证

### 行为

- 输入 1 stock × 1 industry → 输出 4-row plotly HTML
- Row 1: pred vs actual a_S 时间序列
- Row 2: 误差 bar
- Row 3: rolling RMSE
- Row 4: state hit rate binary

### CLI

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_oos_viz.py \
    --code 002475.SZ \
    --days 250 \
    --prefer-industry \
    --output backtrace/outputs/dynsys_oos_viz.html
```

## 7. 兼容性 / 不破坏

- 0 modifications to 11 protected files
- 0 新依赖 (plotly 已装)
- 现有 9 个 CLI 0 改动
- v5.8 `dynamics_state_timeline.py` 0 改动
- 1 新 CLI + 1 新函数 + 1 新 test + 1 新 README §4.1.8

## 8. Risk

| Risk | Mitigation |
|---|---|
| 1 步预测计算开销 | 250 天 ≈ 250 次 predict_next_state call,毫秒级 |
| 滚动 RMSE 空窗口(<20) | 自动跳过,NaN |
| 预测 vs actual 同一数据自证风险 | 用 predict_next_state 真正预测,不是用 actual 当 pred |
| M1 tsfresh shadow | F3 模式容忍 as skip,与 v5.8.1 一致 |

## 9. v5.9 vs v5.8 关系

| 版本 | 模态 | 数据源 | 业务用例 |
|---|---|---|---|
| v5.8 | 静态 (HTML) | load_pair + compute_dynamics | 现在是什么状态 |
| **v5.9** | **静态 (HTML)** | **load_pair + predict_next_state** | **明天预测准不准** |

业务上 v5.8 + v5.9 互补:
- v5.8 = "现状 dashboard"
- v5.9 = "预测质量 dashboard"

## 10. Future / out of scope

- v5.10: state transition heatmap (state_i × state_j → 概率)
- v5.11: multi-stock OOS comparison (1 HTML, N stocks)
- v5.12: actual M1 root-cause fix (lazy-import tsfresh)

本次 v5.9 只做单 stock + 4-row prediction viz,其他按需。
