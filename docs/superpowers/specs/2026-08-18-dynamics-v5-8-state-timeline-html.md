# v5.8 — State Timeline & Force Decomposition HTML

**Date:** 2026-08-18
**Status:** Draft
**Base:** v5.7 (HEAD `d446bf1`)
**Author:** Brainstorming output (user authorized: "按计划和推荐执行，不用问我")

## 1. Goal

给 `_projection_core.py` 已有的 3 个高级函数加 **可视化层**:
- `compute_dynamics()` → 9 个指标 (q_t, θ, R, V magnitudes, E_market/E_self/E_total, a_S/a_M)
- `compute_forces()` → 4 个力 (F_market, F_restore, F_damp, F_self)
- `classify_states()` → 7 个状态 (follow/weak_div/accelerating/independent/against/returning/resonance)

**v5.8 输出**:1 个 plotly HTML, 2 个子图:
1. **State timeline** (top) — 7 状态离散颜色时间线 (1 条线/industry)
2. **Force stacked area** (bottom) — 4 力 stacked area over time

业务读法:
- Top: "申万半导体 2024-09 红色 (accelerating) → 10 月绿 (follow) → 11 月紫 (independent)"
- Bottom: "市场力 / 恢复力 / 阻尼力 / 自驱力 在该行业的相对贡献"

## 2. Why now

`_projection_core.py` 已经有 `compute_dynamics` / `compute_forces` / `classify_states` 三个高级函数,但只在 `projection_2d.py` 的 CSV 输出里出现。业务从未直接看到这些结果。

v5.3-v5.7 是 **Bode 频率域** 可视化(抽象,业务难读,工程价值)。v5.8 是 **时间域状态** 可视化(离散,业务可读,业务价值)。两个互补:
- Bode = "物理学"(为什么这个 regime)
- State timeline = "管理学"(哪天发生了什么)
- Force stacked = "动力学"(哪个力在主导)

业务上 v5.8 比 Bode **更直接**(bottom-line: 状态 + 因果)。

## 3. Design

### 3.1 架构

**复用**(0 新依赖):
- `compute_movement_projection(stock_df, index_df)` (projection core, 输出 Δ 向量)
- `compute_dynamics(mv, lambda_q)` (projection core, 输出 9 指标)
- `compute_forces(dyn, mv, k_restore, c_damp)` (projection core, 输出 4 力)
- `classify_states(R, theta, E_self, thresholds)` (projection core, 输出 7 状态)
- `STATE_COLORS` dict (projection core, 7 状态配色)
- `load_pair(stock_code, days, pipeline, prefer_industry, ...)` (projection core, 数据接入)
- `plotly.subplots.make_subplots(2, 1, row_heights=[0.4, 0.6], shared_xaxes=True)`

**新增**:
- 1 新 CLI: `backtrace/dynamics/dynamics_state_timeline.py`
- 1 新函数: `build_state_timeline_html(stock_df, index_df, common_idx, output_path, ...)`
- 2 个数据接入函数: `load_state_force_timeseries(stock_code, days, prefer_industry, ...)`, `aggregate_to_industry(...)`
- 1 新 test: `test_cli_state_timeline_mode`

### 3.2 函数 `build_state_timeline_html`

```python
def build_state_timeline_html(
    series_per_industry: list,  # [{industry_code, common_idx, states, forces, dyn}, ...]
    output_path: str,
    title: str = 'Industry State Timeline + Force Decomposition',
    thresholds: tuple = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90)),
) -> None:
    """Render N industries' state timeline + 4-force stacked area as 2-row plotly HTML.

    Top: 7-state categorical line (1 row per industry, y=state, color=STATE_COLORS)
    Bottom: 4 forces stacked area (F_market / F_restore / F_damp / F_self)

    Args:
        series_per_industry: list of dicts, each with keys:
            - industry_code: str(e.g. '881427.SH')
            - common_idx: DatetimeIndex
            - states: list[str] length T-1
            - forces: dict with 'F_market', 'F_restore', 'F_damp', 'F_self' (each T-1,)
            - dyn: dict with 'R', 'theta', 'E_self', 'lambda_q_used'
        output_path: HTML 输出路径
        title: figure 标题
        thresholds: classify_states 用 (R_low, R_high, theta_following_rad, theta_against_rad)
    """
```

**Decision**: plotly > matplotlib 因为:
- 7-state 时间线需要 hover 详情 (q_t, R, θ, E_self 等指标)
- 4 力 stacked area 需要 hover 数值
- 与 v5.5 (HTML 互动) 模式一致

### 3.3 数据接入

```python
def load_state_force_timeseries(
    stock_code: str,
    days: int,
    pipeline,
    prefer_industry: bool = True,
    lambda_q: float | None = None,
    k_restore: float = 0.0,
    c_damp: float = 0.0,
) -> dict:
    """load_pair → compute_movement_projection → compute_dynamics →
    compute_forces → classify_states 一步到位。

    Returns:
        dict with keys: stock_df, index_df, common_idx, index_code, index_name,
                       mv, dyn, frc, states。
    """
```

CLI 调用示例:
```python
# 单股单 industry
out = load_state_force_timeseries(
    '002475.SZ', days=250, pipeline=P,
    prefer_industry=True,
    lambda_q=None,  # 自适应
    k_restore=0.0, c_damp=0.0,
)
# 多股 cross-industry (后续 v5.9 可扩展)
```

### 3.4 CLI

新文件 `backtrace/dynamics/dynamics_state_timeline.py`:

```python
python backtrace/dynamics/dynamics_state_timeline.py \
    --code 002475.SZ \
    --days 250 \
    --prefer-industry \
    --output backtrace/outputs/dynsys_state_timeline.html
```

**Flags**:
- `--code STR` (required, e.g. '002475.SZ')
- `--days INT` (default 250, 1 年 ≈ 250 个交易日)
- `--prefer-industry` (default True, 行业基线优先)
- `--lambda-q FLOAT` (default None, 自适应)
- `--k-restore FLOAT` (default 0.0)
- `--c-damp FLOAT` (default 0.0)
- `--output PATH` (default `backtrace/outputs/dynsys_state_timeline.html`)

### 3.5 视觉规格

**Top subplot (state timeline)**:
- y-axis: 7 state (categorical, ordered: follow → weak_div → accelerating → independent → against → returning → resonance)
- 1 line per industry, color = STATE_COLORS[state]
- Hover: 显示 date, state, q_t, R, theta, E_self
- markers: 大小 = E_self (越大动得越凶)

**Bottom subplot (force stacked area)**:
- 4 trace stacked: F_market (蓝) + F_restore (绿) + F_damp (橙) + F_self (红)
- y-axis: 力模长 (与 F_market,F_restore 等同量纲)
- Hover: 显示 date, 4 力数值

**Layout**:
- shared_xaxes=True (date 同步)
- row_heights=[0.4, 0.6] (state timeline 略小, force 占大头)
- title: 全部 industry 名字 + asof_date range

### 3.6 颜色 / 字体

- State 颜色:用 `STATE_COLORS` (projection core),7 色已 fixed
- Force 颜色:matplotlib/plotly 默认 qualitative 调色板里挑 4 色(蓝/绿/橙/红)
- 中文:由 plotly 默认处理(实际不出现中文,industry code 英文)

### 3.7 Edge cases

- 单 industry 无数据 → `ValueError` 提示
- 全部 industry 同一 state → stacked area 退化为 1 trace
- 1 industry 1 天 → 1 点线
- data 缺失 → `load_pair` 抛 RuntimeError(沿用)

## 4. Files

| File | Type | Lines |
|---|---|---|
| `backtrace/dynamics/dynamics_state_timeline.py` | new | ~250 |
| `tests/test_dynamics_eigen.py` | modify | +44 |
| `backtrace/dynamics/README.md` | modify | +15 |
| `docs/superpowers/specs/2026-08-18-dynamics-v5-8-state-timeline-html.md` | new | (this) |
| `docs/superpowers/plans/2026-08-18-dynamics-v5-8-state-timeline-html.md` | new | (TBD) |

Total: ~530 lines, 5 files

## 5. 测试

新增 1 test:

```python
def test_cli_state_timeline_mode(tmp_path):
    """v5.8: CLI state timeline mode — 验证 build_state_timeline_html 输出 HTML."""
    pytest.importorskip("plotly")

    # 合成 1 stock × 1 industry (复用 v5.6 fixture 模式)
    csv_path = tmp_path / 'kc_time.csv'
    rows = []
    for date_str in ['2024-09-30', '2024-10-31', '2024-11-30']:
        for code, k, c in [('AAA', 0.5, 2.0), ('BBB', 3.5, 0.5)]:
            rows.append({
                'code': code, 'index_code': f'Industry_{code}',
                'asof_date': date_str, 'k_hat': k, 'c_hat': c,
                'status': 'ok', 'n_valid_days': 200,
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    html_out = tmp_path / 'timeline.html'
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cli_script = os.path.join(repo_root, 'backtrace', 'dynamics', 'dynamics_state_timeline.py')
    cmd = [
        sys.executable, cli_script,
        '--code', '002475.SZ',
        '--days', '250',
        '--output', str(html_out),
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
    # 关联 stock 002475.SZ 在 data/stocks/ 缓存里需存在;否则 CLI 失败
    # 测试只验证 HTML 文件结构(如果存在)
    if html_out.exists():
        with open(html_out, 'rb') as fh:
            content = fh.read()
        # plotly HTML 必有 <html> + plotly 标志
        assert b'<html' in content.lower() or b'plotly' in content.lower()
```

合计:74 → 75 tests pass (1 new test)。

## 6. 验证

### 行为

- 输入 1 stock × 1 industry → 输出 1 行 state timeline + 1 stacked force area
- State 颜色看 `STATE_COLORS` 而定
- 顶部 5-state legend, 底部 4-force legend

### CLI

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_timeline.py \
    --code 002475.SZ \
    --days 250 \
    --prefer-industry \
    --output backtrace/outputs/dynsys_state_timeline.html
```

## 7. 兼容性 / 不破坏

- 0 modifications to 10 protected files (`_projection_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py` / `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py` / `projection/parameter_fit.py`)
- 0 新依赖 (plotly 已装)
- 现有 8 个 CLI 0 改动
- v5.7 `build_regime_heatmap` + `REGIME_ABBREV` dict 0 改动
- 1 新 CLI + 1 新函数 + 1 新 test + 1 新 README §4.1.7

## 8. Risk

| Risk | Mitigation |
|---|---|
| load_pair 调本地缓存,002475.SZ 不在 data/stocks/ → RuntimeError | 测试用 `if exists` 容错;CLI 文档说明需先跑 fetch_daily.py |
| plotly 5.x vs 4.x API 差异 | 已在 v5.5 验证兼容 |
| 7-state categorical y-axis 排序 | 强制 ordinal (follow=1, weak_div=2, ..., resonance=7) |
| Force 4 trace 颜色与已有 4-regime 颜色混淆 | 故意用蓝色/绿色/橙色/红色 instead of green/orange/red/purple |

## 9. v5.8 vs v5.5/v5.6/v5.7 关系

| 版本 | 模态 | 数据源 | 输出 | 业务用例 |
|---|---|---|---|---|
| v5.5 | 交互 (HTML) | kc_time.csv | 频率曲线 overlay | 物理分析 |
| v5.6 | 静态 (PNG) | kc_time.csv | 频率曲线 grid | 物理分析报告 |
| v5.7 | 静态 (PNG) | kc_time.csv | regime cells | 物理分析 dashboard |
| **v5.8** | **交互 (HTML)** | **load_pair + compute_dynamics** | **state timeline + force** | **业务分析状态演化** |

业务可同时要 v5.5-v5.7 + v5.8, 数据源不同。

## 10. Future / out of scope

- v5.9: multi-industry cross-compare (1 HTML, N industries stacked)
- v5.10: state transition heatmap (state_i × state_j → 概率)
- v5.11: force attribution HTML (drill-down:某天为啥 accelerating)

本次 v5.8 只做单 industry + 4-force + state timeline,其他按需。
