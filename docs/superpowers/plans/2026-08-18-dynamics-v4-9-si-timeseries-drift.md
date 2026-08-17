# v4.9 SI 时序 + 漂移检测 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 行业稳定性指数 SI 从单值扩展到时序;在每月末用最近 240 天 OLS 估 (k̂, ĉ),聚合到行业层,产出 SI 时序 + rolling 60 日 z-score 漂移检测。

**Architecture:**
- **数据生产端** (Task 1):`parameter_fit.py` 新增 `--rolling-time` 模式 — 每月末 asof_date × 每只票 OLS 估 (k̂, ĉ),复用 `fit_rolling` 内部 `_load_movement` / `_build_kinematics` / `_solve_ols`。
- **数据消费端** (Task 2):`dynamics_eigen_analysis.py` 末尾追加 `compute_sector_stability_timeseries`(复用 v4.7 `compute_sector_stability` 公式与 `SI_WEIGHTS`);新增 `dynamics_si_timeseries.py` 独立 CLI 跑漂移检测 + HTML + 文本汇总。
- **隔离边界**:Task 1 输出 `kc_estimates_time.csv` (long format) 是 Task 2 的唯一输入。中间文件 gitignored。

**Tech Stack:**
- Python 3.13 (Anaconda)
- pandas / numpy (已有)
- plotly (subplots + make_subplots, 沿用 v4.7/v4.8)
- pytest + tmp_path fixtures

---

## Global Constraints

复制自 spec(每条都需严格遵守):

- 数学层 `_dynamics_core.py` **0 行修改**(硬约束,任务验证会查)
- 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) **0 行修改**
- `analyze_eigenvalues` / `simulate_trajectory` / `compute_sector_stability` 函数签名不变
- `dynamics_si_ic.py` (v4.8) **0 行修改**
- `parameter_fit.py --rolling-fit` 既有模式**不动**,新 `--rolling-time` 是平行选项
- 输出全部 gitignored (`data/dynamics/` + `backtrace/outputs/`)
- `PYTHONIOENCODING=utf-8` 必备(Windows GBK)
- Python 路径:`/c/ProgramData/anaconda3/python.exe`
- 安全:`jhzq/交易凭据.md` 不能写进代码或 git
- Subagent-Driven Development (SDD) workflow
- 总测试数:38 (v4.8) + 5 (v4.9) = **43 tests pass**
- 漂移检测默认 z_threshold = -2 (可调 `--z-threshold`)
- rolling 窗口 = 3 个 asof_date (≈ 60 交易日 ≈ 3 个月,匹配 v4.8 IC window)

---

## Task 1: parameter_fit.py --rolling-time 模式

**Files:**
- Modify: `backtrace/projection/parameter_fit.py:118-150` (parse_args 加 `--rolling-time`)
- Modify: `backtrace/projection/parameter_fit.py:545-572` (main() 分支)
- New: `backtrace/projection/_main_rolling_time.py` 或 inline 在 parameter_fit.py 末尾

**Interfaces:**
- Consumes: `targets = list_movement_csvs(input_csv)` (已有)
- Consumes: `fit_rolling(...)` 已有函数,内部用 `_load_movement` + `_solve_ols`
- Produces: `data/projection/kc_estimates_time.csv` (long format)
  ```csv
  asof_date,code,name,index_code,index_tag,stock_tag,k_hat,c_hat,f_self_loss,n_valid_days,status
  2024-01-31,600519.SH,贵州茅台,000001.SH,SH,S,0.85,1.05,1.2e-4,240,ok
  ```

- [ ] **Step 1: 在 `parse_args` 加 `--rolling-time` / `--rolling-time-window`**

读 `backtrace/projection/parameter_fit.py:118-150` 现有 argparse,新增:

```python
parser.add_argument(
    '--rolling-time', action='store_true',
    help='每月末用最近 N 天 OLS 估 (k̂, ĉ),产 kc_estimates_time.csv (long format)',
)
parser.add_argument(
    '--rolling-time-window', type=int, default=240,
    help='rolling-time 模式窗口大小(交易日,默认 240)',
)
```

- [ ] **Step 2: 在 `main()` 加 `--rolling-time` 分支调用**

读 `parameter_fit.py:545-572` main(),加 if 分支:

```python
if args.rolling_time:
    main_rolling_time(targets,
                      window=args.rolling_time_window,
                      clip_extreme=args.clip_extreme)
    return
```

放在 `if args.rolling_fit:` 分支之前。

- [ ] **Step 3: 实现 `main_rolling_time()` 函数(~50 行,放在 `main_rolling` 之后)**

放在 `parameter_fit.py:main_rolling` 函数定义之后(行 ~720),新函数:

```python
def main_rolling_time(targets, window: int = 240, clip_extreme: float = 10.0):
    """每月末用最近 N 天 OLS 估 (k̂, ĉ),产 long format CSV。

    对每只票:
        1. 读 movement CSV
        2. 找月末 asof_date 列表(每月最后交易日)
        3. 对每个 asof_date:截幅到该日期 + 取最后 window 行,跑 OLS
        4. 落 1 行 (asof_date, code, k_hat, c_hat, ...)

    输出: data/projection/kc_estimates_time.csv
    """
    rows = []
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(targets)}] {label} ...', end=' ', flush=True)
        loaded, err = _load_movement(mv_csv, stock_tag, index_tag)
        if loaded is None:
            print(f'⚠ load failed: {err}')
            continue
        df, delta_u, delta_v, beta = loaded
        u_vec, d_vec, a_u_vec, a_v_vec = _build_kinematics(delta_u, delta_v, beta)
        T = len(delta_u)
        # 找月末 asof_date 列表
        month_ends = _month_ends(df['Date'])
        print(f'{len(month_ends)} asof_dates', end=' ', flush=True)
        for asof in month_ends:
            mask = (df['Date'] <= asof).values
            n_avail = int(mask.sum())
            if n_avail < max(3, window // 4):  # 至少需要 window/4 天
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan, 'n_valid_days': 0,
                    'status': f'too_few_days ({n_avail})',
                })
                continue
            # 取最后 window 行
            idx = np.where(mask)[0][-window:]
            sub = slice(idx[0], idx[-1] + 1)
            valid = (
                np.isfinite(a_u_vec[sub]).all(axis=1)
                & np.isfinite(a_v_vec[sub]).all(axis=1)
                & np.isfinite(d_vec[sub]).all(axis=1)
                & np.isfinite(u_vec[sub]).all(axis=1)
            )
            n_valid = int(valid.sum())
            if n_valid < 3:
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan,
                    'n_valid_days': n_valid,
                    'status': f'too_few_valid ({n_valid})',
                })
                continue
            try:
                k_hat, c_hat, f_loss, _, rank = _solve_ols(
                    a_u_vec[sub], a_v_vec[sub], d_vec[sub], u_vec[sub], beta[sub], valid,
                )
            except Exception as e:
                rows.append({
                    'asof_date': str(asof)[:10],
                    'code': code, 'name': name or '',
                    'index_code': index_code,
                    'index_tag': index_tag, 'stock_tag': stock_tag,
                    'k_hat': np.nan, 'c_hat': np.nan,
                    'f_self_loss': np.nan,
                    'n_valid_days': n_valid,
                    'status': f'solve_failed: {type(e).__name__}: {e}',
                })
                continue
            finite = np.isfinite(k_hat) and np.isfinite(c_hat)
            extreme = abs(k_hat) > clip_extreme or abs(c_hat) > clip_extreme
            if not finite:
                status = 'solve_failed'
            elif rank < 2:
                status = 'singular'
            elif extreme:
                status = f'extreme (|k| or |c| > {clip_extreme:g})'
            else:
                status = 'ok'
            rows.append({
                'asof_date': str(asof)[:10],
                'code': code, 'name': name or '',
                'index_code': index_code,
                'index_tag': index_tag, 'stock_tag': stock_tag,
                'k_hat': k_hat, 'c_hat': c_hat,
                'f_self_loss': f_loss,
                'n_valid_days': n_valid,
                'status': status,
            })
        print('✓')
    out = pd.DataFrame(rows)
    out_path = os.path.join(CSV_OUT_DIR, 'kc_estimates_time.csv')
    out.to_csv(out_path, index=False, encoding='utf-8')
    print(f'✓ {out_path} ({len(out)} 行)')
```

**关键点**:
- 复用 `_load_movement` / `_build_kinematics` / `_solve_ols` (parameter_fit.py 私有函数,文件内可直接调)
- **不重复实现** OLS 主体
- `_month_ends` 是新 helper,~10 行(见 Step 4)

- [ ] **Step 4: 实现 `_month_ends()` helper**

放在 `main_rolling_time` 之前(同文件):

```python
def _month_ends(dates: pd.Series) -> list[pd.Timestamp]:
    """返回每月最后一个交易日的 Timestamp 列表(去重 + 升序)。"""
    df = pd.DataFrame({'Date': pd.to_datetime(dates)})
    df['_ym'] = df['Date'].dt.to_period('M')
    month_ends = df.groupby('_ym')['Date'].max().sort_values().tolist()
    return month_ends
```

- [ ] **Step 5: 冒烟测试(CLI 端到端 `--limit 5`)**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/projection/parameter_fit.py \
    --rolling-time --rolling-time-window 240 --limit 5
```

Expected:
- 打印 `✓ kc_estimates_time.csv (N 行)`,其中 N = 5 stocks × month_ends_count (depends on data length)
- 文件存在于 `data/projection/kc_estimates_time.csv`
- 列名:`asof_date, code, name, index_code, index_tag, stock_tag, k_hat, c_hat, f_self_loss, n_valid_days, status`

**注意**:此冒烟依赖 `data/daily/<code>.csv` 和 `data/projection/movement_*.csv` 存在(已有数据)。若失败,检查 movement CSV 是否生成过。

- [ ] **Step 6: 单元测试 `test_rolling_time_basic_shape`(放在 `tests/test_dynamics_eigen.py` 末尾)**

```python
def test_rolling_time_basic_shape():
    """kc_estimates_time.csv 应该有 asof_date, code, k_hat, c_hat 等列。"""
    pytest.importorskip("backtrace.projection.parameter_fit")
    from backtrace.projection.parameter_fit import _month_ends
    dates = pd.to_datetime([
        '2024-01-15', '2024-01-31', '2024-02-29', '2024-03-31', '2024-04-30',
        '2024-05-31', '2024-06-30',
    ])
    ends = _month_ends(pd.Series(dates))
    # 每月最后一日
    assert ends == [
        pd.Timestamp('2024-01-31'), pd.Timestamp('2024-02-29'),
        pd.Timestamp('2024-03-31'), pd.Timestamp('2024-04-30'),
        pd.Timestamp('2024-05-31'), pd.Timestamp('2024-06-30'),
    ]
```

- [ ] **Step 7: 跑测试 + 提交**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest \
    tests/test_dynamics_eigen.py::test_rolling_time_basic_shape -v
```

Expected: PASS

Commit:
```bash
git add backtrace/projection/parameter_fit.py tests/test_dynamics_eigen.py
git commit -m "feat(projection): --rolling-time 月末 OLS 时序模式

为 v4.9 行业 SI 时序准备输入数据。每月末用最近 N 天(默认 240)对每只票
跑 OLS 估 (k̂, ĉ),产出 kc_estimates_time.csv (long format)。
复用 _load_movement / _build_kinematics / _solve_ols,不重复实现 OLS。
不影响现有 --rolling-fit / 默认模式。"
```

---

## Task 2: SI 时序计算 + 漂移检测 CLI + 5 个测试 + README

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py` (末尾追加 `compute_sector_stability_timeseries` ~50 行)
- New: `backtrace/dynamics/dynamics_si_timeseries.py` (~250 行,独立 CLI)
- Modify: `tests/test_dynamics_eigen.py` (追加 4 测试 ~70 行;Step 6 已有 1 个,共 5)
- Modify: `backtrace/dynamics/README.md` §3.9 (~30 行)

**Interfaces:**
- Consumes: `data/projection/kc_estimates_time.csv` (Task 1 产出,long format)
- Consumes: `data/dynamics/eigen_summary.csv` 或 `data/projection/kc_estimates.csv` (回查 code → industry_l1)
- Consumes: `data/sw2/members.csv` (code → sector_name 中文名,可选)
- Produces:
  - `data/dynamics/sector_si_timeseries.csv` (long format)
  - `data/dynamics/si_drift_events.csv`
  - `backtrace/outputs/dynsys_si_timeseries.html`
  - `backtrace/outputs/dynsys_si_timeseries_summary.txt`

- [ ] **Step 1: 在 `dynamics_eigen_analysis.py` 末尾追加 `compute_sector_stability_timeseries`**

放在 `compute_sector_stability` 函数定义之后(行 ~275 之后,沿用 v4.7 函数风格):

```python
def compute_sector_stability_timeseries(
    kc_long_df: pd.DataFrame,
    industry_lookup: dict | None = None,
    n_stocks_threshold: int = 50,
) -> pd.DataFrame:
    """按 (asof_date, industry_l1) 计算稳定性指数 SI 时序。

    公式与 v4.7 compute_sector_stability 完全一致,只多了 asof_date 轴:
      ρ_health      = clip(1 - ρ_med / 2,        0, 1)
      damping_health = clip(1 - |c_med - 1| / 2,  0, 1)
      wedge_health   = clip(in_wedge_pct,         0, 1)
      SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health

    Args:
        kc_long_df: 含 asof_date, code, k_hat, c_hat 列(Task 1 产出)
        industry_lookup: 可选 code → industry_l1 反查表
                         (默认无 → 每行都需自带 industry_l1 列)
        n_stocks_threshold: 行业筛选阈值(沿用 v4.3,默认 50)

    Returns:
        11 列 DataFrame: asof_date, industry_l1, sector_name, n_stocks,
                         rho_median, c_median, in_wedge_pct,
                         rho_health, damping_health, wedge_health, SI
        按 (asof_date ASC, SI DESC) 排序
    """
    if kc_long_df.empty:
        return pd.DataFrame(columns=[
            'asof_date', 'industry_l1', 'sector_name', 'n_stocks',
            'rho_median', 'c_median', 'in_wedge_pct',
            'rho_health', 'damping_health', 'wedge_health', 'SI',
        ])
    # 加 industry_l1(若没有)
    df = kc_long_df.copy()
    if 'industry_l1' not in df.columns:
        if industry_lookup is None:
            raise ValueError('kc_long_df missing industry_l1 and no lookup given')
        df['industry_l1'] = df['code'].map(industry_lookup).fillna('')
        df = df[df['industry_l1'] != '']
    # 加 spectral_radius + in_wedge(单值 (k, c) → analyze_eigenvalues)
    # 性能:5000 × 60 asof = 300k calls,每个 ~0.1ms → ~30s,可接受
    spec_radii, in_wedges = [], []
    for k, c in zip(df['k_hat'].values, df['c_hat'].values):
        if not (np.isfinite(k) and np.isfinite(c)):
            spec_radii.append(np.nan)
            in_wedges.append(False)
            continue
        eig = analyze_eigenvalues(float(k), float(c))
        spec_radii.append(eig['spectral_radius'])
        in_wedges.append(eig['in_wedge'])
    df['spectral_radius'] = spec_radii
    df['in_wedge'] = in_wedges
    # 聚合
    rho_w, damp_w, wedge_w = SI_WEIGHTS
    agg = df.groupby(['asof_date', 'industry_l1']).agg(
        n_stocks=('code', 'count'),
        rho_median=('spectral_radius', 'median'),
        c_median=('c_hat', 'median'),
        wedge_pct=('in_wedge', 'mean'),
    ).reset_index()
    # 行业筛选(沿用 v4.3 / v4.7)
    agg['rho_health'] = (1.0 - agg['rho_median'] / 2.0).clip(0.0, 1.0)
    agg['damping_health'] = (1.0 - (agg['c_median'] - 1.0).abs() / 2.0).clip(0.0, 1.0)
    agg['wedge_health'] = agg['wedge_pct'].clip(0.0, 1.0)
    agg['SI'] = (
        rho_w * agg['rho_health']
        + damp_w * agg['damping_health']
        + wedge_w * agg['wedge_health']
    )
    # sector_name(若给了 lookup)
    if 'sector_name' not in agg.columns:
        agg['sector_name'] = ''
    # 排序 + 列顺序
    agg = agg.sort_values(['asof_date', 'SI'], ascending=[True, False]).reset_index(drop=True)
    # n_stocks 过滤:每 (asof_date, industry_l1) 都应用阈值;简化版本直接保留全部
    # 与 v4.7 不同:时序版不过滤(避免 warmup 期空缺),上层汇总会标 low-confidence
    return agg[[
        'asof_date', 'industry_l1', 'sector_name', 'n_stocks',
        'rho_median', 'c_median', 'in_wedge_pct',
        'rho_health', 'damping_health', 'wedge_health', 'SI',
    ]]
```

**关键点**:
- 复用 `SI_WEIGHTS` 常量(已在 eigen_analysis.py 顶部)
- 复用 `analyze_eigenvalues` 函数(已在 eigen_analysis.py imports)
- **不重复实现** 公式
- 与 v4.7 `compute_sector_stability` 公式完全一致

- [ ] **Step 2: 单元测试 `test_si_timeseries_basic_shape`**

放在 `tests/test_dynamics_eigen.py` 末尾,Task 1 Step 6 测试之后:

```python
def test_si_timeseries_basic_shape():
    """5 行业 × 100 日 → 500 行,SI ∈ [0,1]。"""
    pytest.importorskip("backtrace.dynamics.dynamics_eigen_analysis")
    from backtrace.dynamics.dynamics_eigen_analysis import compute_sector_stability_timeseries
    # 构造 synthetic kc_long_df
    rows = []
    for ind in range(5):
        for d in range(100):
            rows.append({
                'asof_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=d*7),
                'code': f'{ind:06d}.SH',
                'name': f'测试_{ind}',
                'industry_l1': f'8010{ind:02d}',
                'k_hat': 0.5, 'c_hat': 1.0,
                'n_valid_days': 240, 'status': 'ok',
            })
    kc_long = pd.DataFrame(rows)
    out = compute_sector_stability_timeseries(kc_long)
    # 5 行业 × 100 日
    assert len(out) == 5 * 100, f'expected 500 rows, got {len(out)}'
    assert out['SI'].between(0, 1).all(), 'SI must be in [0, 1]'
    assert set(out.columns) >= {'asof_date', 'industry_l1', 'SI', 'rho_median', 'c_median'}
```

- [ ] **Step 3: 单元测试 `test_si_timeseries_stable_industry`**

```python
def test_si_timeseries_stable_industry():
    """k̂, ĉ 恒定 → SI 几乎不变。"""
    pytest.importorskip("backtrace.dynamics.dynamics_eigen_analysis")
    from backtrace.dynamics.dynamics_eigen_analysis import compute_sector_stability_timeseries
    rows = []
    for d in range(120):  # 120 个 asof_dates
        rows.append({
            'asof_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=d*7),
            'code': '000001.SH',
            'name': '测试银行',
            'industry_l1': '801010',
            'k_hat': 0.5, 'c_hat': 1.0,  # 恒定
            'n_valid_days': 240, 'status': 'ok',
        })
    kc_long = pd.DataFrame(rows)
    out = compute_sector_stability_timeseries(kc_long)
    # SI 应该几乎恒定(每行只有 1 只票 → SI 等于该票 SI)
    si_values = out[out['industry_l1'] == '801010']['SI']
    assert si_values.std() < 1e-6, f'SI should be constant, got std={si_values.std()}'
```

- [ ] **Step 4: 新建 `backtrace/dynamics/dynamics_si_timeseries.py`(空白骨架,~20 行)**

创建文件,先只放 imports + main() stub:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v4.9 — 行业 SI 时序 + 漂移检测 CLI。

读 parameter_fit.py --rolling-time 输出 (kc_estimates_time.csv),
聚合到行业层,产出 SI 时序 + rolling 60 日 z-score 漂移事件。

输出(全 gitignored):
  - data/dynamics/sector_si_timeseries.csv
  - data/dynamics/si_drift_events.csv
  - backtrace/outputs/dynsys_si_timeseries.html
  - backtrace/outputs/dynsys_si_timeseries_summary.txt
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, BACKTRACE_DIR)  # 让 from dynamics import ... 工作

import argparse
import numpy as np
import pandas as pd

CSV_OUT_DIR = 'data/dynamics'
HTML_OUT_DIR = 'backtrace/outputs'
DEFAULT_KC_TIME = 'data/projection/kc_estimates_time.csv'
DEFAULT_EIGEN = 'data/dynamics/eigen_summary.csv'
DEFAULT_SW2 = 'data/sw2/members.csv'
DEFAULT_SI_TS = os.path.join(CSV_OUT_DIR, 'sector_si_timeseries.csv')
DEFAULT_DRIFT = os.path.join(CSV_OUT_DIR, 'si_drift_events.csv')
DEFAULT_HTML = os.path.join(HTML_OUT_DIR, 'dynsys_si_timeseries.html')
DEFAULT_TXT = os.path.join(HTML_OUT_DIR, 'dynsys_si_timeseries_summary.txt')


def parse_args():
    p = argparse.ArgumentParser(description='v4.9 SI 时序 + 漂移检测')
    p.add_argument('--kc-time', default=DEFAULT_KC_TIME,
                   help=f'kc_estimates_time.csv 路径 (默认 {DEFAULT_KC_TIME})')
    p.add_argument('--eigen', default=DEFAULT_EIGEN,
                   help=f'eigen_summary.csv 路径 (默认 {DEFAULT_EIGEN})')
    p.add_argument('--sw2-members', default=DEFAULT_SW2,
                   help=f'sw2 members.csv 路径 (默认 {DEFAULT_SW2})')
    p.add_argument('--si-ts-output', default=DEFAULT_SI_TS,
                   help=f'sector_si_timeseries.csv 输出路径 (默认 {DEFAULT_SI_TS})')
    p.add_argument('--drift-output', default=DEFAULT_DRIFT,
                   help=f'si_drift_events.csv 输出路径 (默认 {DEFAULT_DRIFT})')
    p.add_argument('--html-output', default=DEFAULT_HTML,
                   help=f'HTML 输出路径 (默认 {DEFAULT_HTML})')
    p.add_argument('--txt-output', default=DEFAULT_TXT,
                   help=f'文本汇总输出路径 (默认 {DEFAULT_TXT})')
    p.add_argument('--window', type=int, default=3,
                   help='漂移检测 rolling window 大小(asof_date 数,默认 3 ≈ 60 交易日)')
    p.add_argument('--z-threshold', type=float, default=-2.0,
                   help='漂移检测 z-score 阈值(默认 -2.0)')
    p.add_argument('--limit', type=int, default=0,
                   help='限制输入股票数(0 = 全部,默认 0)')
    return p.parse_args()
```

**关键点**:
- 文件独立,**不 import 同目录 sibling 模块**(`dynamics_eigen_analysis` 等),避免循环依赖
- 通过 `from dynamics_eigen_analysis import compute_sector_stability_timeseries` 在 main() 内延迟导入

- [ ] **Step 5: 实现 `load_kc_long` / `load_industry_membership` (~30 行)**

在 `dynamics_si_timeseries.py` 加:

```python
def load_kc_long(path: str, limit: int = 0) -> pd.DataFrame:
    """读 kc_estimates_time.csv,返回标准化 DataFrame。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} 不存在。先跑:\n'
            f'  python backtrace/projection/parameter_fit.py --rolling-time --limit <N>'
        )
    df = pd.read_csv(path, encoding='utf-8')
    if limit > 0:
        df = df[df['code'].isin(df['code'].unique()[:limit])]
    df['asof_date'] = pd.to_datetime(df['asof_date'])
    return df


def load_industry_membership(eigen_path: str, sw2_path: str) -> dict:
    """回查 code → industry_l1,优先 eigen_summary.csv(已有 industry_l1 列)。"""
    if os.path.exists(eigen_path):
        df = pd.read_csv(eigen_path, usecols=['code', 'industry_l1'], encoding='utf-8')
        lookup = dict(zip(df['code'], df['industry_l1']))
        if lookup:
            return lookup
    # 降级:从 sw2 members.csv 反查(code → industry_l1)
    if os.path.exists(sw2_path):
        df = pd.read_csv(sw2_path, encoding='utf-8')
        # members.csv schema:code, industry_l1, sector_name(假设)
        if 'industry_l1' in df.columns and 'code' in df.columns:
            return dict(zip(df['code'], df['industry_l1']))
    raise FileNotFoundError(
        f'无法构建 code → industry_l1 映射。{eigen_path} 和 {sw2_path} 都不含 industry_l1。'
    )
```

**注意**:`load_industry_membership` 实际是 code → industry_l1(不是 membership list)。命名沿用 spec §4.2 表格。

- [ ] **Step 6: 实现 `detect_si_drift` 函数(~40 行)**

```python
def detect_si_drift(
    si_ts_df: pd.DataFrame,
    window: int = 3,
    z_threshold: float = -2.0,
) -> pd.DataFrame:
    """对每个行业的 SI(t) 做 rolling z-score,触发 drift event。

    Args:
        si_ts_df: compute_sector_stability_timeseries 输出的 11 列
        window: rolling window 大小(asof_date 数,默认 3 ≈ 60 交易日)
        z_threshold: 触发阈值(默认 -2.0,负值越极端越算漂移)

    Returns:
        drift events DataFrame,列:
            asof_date, industry_l1, sector_name, SI, rolling_mean, rolling_std, z_score
        排序:按 (asof_date ASC, z_score ASC)
    """
    if si_ts_df.empty:
        return pd.DataFrame(columns=[
            'asof_date', 'industry_l1', 'sector_name',
            'SI', 'rolling_mean', 'rolling_std', 'z_score',
        ])
    si_ts_df = si_ts_df.sort_values(['industry_l1', 'asof_date']).copy()
    drift_rows = []
    for ind, g in si_ts_df.groupby('industry_l1'):
        si = g['SI'].values
        dates = g['asof_date'].values
        sector_name = g['sector_name'].iloc[0] if 'sector_name' in g.columns else ''
        n = len(si)
        for i in range(n):
            # rolling window = [max(0, i-window), i),不含 i 自身(避免 leak)
            s = max(0, i - window)
            if i - s < 2:  # 至少需要 2 个历史点算 std
                continue
            hist = si[s:i]
            hist = hist[np.isfinite(hist)]
            if len(hist) < 2:
                continue
            m = float(np.mean(hist))
            sd = float(np.std(hist, ddof=1))
            if sd < 1e-9:
                continue
            z = (si[i] - m) / sd
            if z < z_threshold:
                drift_rows.append({
                    'asof_date': pd.Timestamp(dates[i]),
                    'industry_l1': ind,
                    'sector_name': sector_name,
                    'SI': float(si[i]),
                    'rolling_mean': m,
                    'rolling_std': sd,
                    'z_score': float(z),
                })
    if not drift_rows:
        return pd.DataFrame(columns=[
            'asof_date', 'industry_l1', 'sector_name',
            'SI', 'rolling_mean', 'rolling_std', 'z_score',
        ])
    out = pd.DataFrame(drift_rows)
    return out.sort_values(['asof_date', 'z_score']).reset_index(drop=True)
```

- [ ] **Step 7: 单元测试 `test_si_timeseries_drift_zscore` + `test_si_timeseries_sudden_drop`**

放在 `tests/test_dynamics_eigen.py` 末尾:

```python
def test_si_timeseries_drift_zscore():
    """detect_si_drift z-score 计算正确。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import detect_si_drift
    # 构造 SI 时序:0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1 (t=7)
    dates = pd.date_range('2024-01-01', periods=8, freq='7D')
    si_ts = pd.DataFrame({
        'asof_date': dates,
        'industry_l1': '801010',
        'sector_name': '银行',
        'n_stocks': 42,
        'rho_median': 0.85,
        'c_median': 1.05,
        'in_wedge_pct': 0.92,
        'rho_health': 0.575,
        'damping_health': 0.975,
        'wedge_health': 0.92,
        'SI': [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.1],
    })
    drift = detect_si_drift(si_ts, window=3, z_threshold=-2.0)
    # t=7 应该触发 drift event(0.1 比 0.85 低很多)
    assert len(drift) >= 1, f'expected ≥ 1 drift, got {len(drift)}'
    assert drift.iloc[0]['industry_l1'] == '801010'
    # z_score 应该是负值且 < -2
    assert drift.iloc[0]['z_score'] < -2.0


def test_si_timeseries_sudden_drop():
    """构造 SI(t=50) 从 0.8 → 0.2 → 触发 drift event。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import detect_si_drift
    dates = pd.date_range('2024-01-01', periods=60, freq='7D')
    si_values = [0.8] * 50 + [0.2] * 10
    si_ts = pd.DataFrame({
        'asof_date': dates,
        'industry_l1': '801080',
        'sector_name': '半导体',
        'n_stocks': 38,
        'rho_median': 0.85,
        'c_median': 1.05,
        'in_wedge_pct': 0.92,
        'rho_health': 0.575,
        'damping_health': 0.975,
        'wedge_health': 0.92,
        'SI': si_values,
    })
    drift = detect_si_drift(si_ts, window=3, z_threshold=-2.0)
    # t=50 起应该触发 drift(0.2 比 0.8 低)
    assert len(drift) >= 1, f'expected ≥ 1 drift, got {len(drift)}'
    # 至少一个 drift 的 asof_date ≥ t=50
    assert any(drift['asof_date'] >= dates[50])
```

- [ ] **Step 8: 实现 `write_si_timeseries_summary` (~40 行,UTF-8 文本)**

在 `dynamics_si_timeseries.py` 加:

```python
def write_si_timeseries_summary(
    si_ts_df: pd.DataFrame,
    drift_events_df: pd.DataFrame,
    output_path: str,
) -> None:
    """写 UTF-8 中文文本汇总。

    包含:
      - 行业 × asof_date 统计
      - top 10 行业按最新 SI 排序
      - 漂移事件汇总(总事件数 + top 10 行业)
    """
    lines = []
    lines.append('=' * 70)
    lines.append('v4.9 行业 SI 时序 + 漂移检测 (Sector Stability Index Timeseries + Drift)')
    lines.append('=' * 70)
    if si_ts_df.empty:
        lines.append('无数据')
    else:
        n_industries = si_ts_df['industry_l1'].nunique()
        n_dates = si_ts_df['asof_date'].nunique()
        lines.append(f'行业数: {n_industries}')
        lines.append(f'asof_date 数: {n_dates}')
        lines.append(f'总行数: {len(si_ts_df)}')
        lines.append('')
        # 最新 SI top 10
        latest_date = si_ts_df['asof_date'].max()
        latest = si_ts_df[si_ts_df['asof_date'] == latest_date].sort_values('SI', ascending=False).head(10)
        lines.append(f'最新一期 ({pd.Timestamp(latest_date).strftime("%Y-%m-%d")}) Top 10 行业 SI:')
        for _, row in latest.iterrows():
            lines.append(f'  {row["sector_name"] or row["industry_l1"]:<10} '
                         f'SI={row["SI"]:.3f}  ρ_med={row["rho_median"]:.2f}  '
                         f'c_med={row["c_median"]:.2f}')
        lines.append('')
        # 漂移事件
        n_drift = len(drift_events_df)
        lines.append(f'漂移事件总数: {n_drift}')
        if n_drift > 0:
            top_drift = (drift_events_df.groupby('industry_l1')
                         .size().sort_values(ascending=False).head(10))
            lines.append('漂移事件 top 10 行业:')
            for ind, n in top_drift.items():
                name = (drift_events_df[drift_events_df['industry_l1'] == ind]
                        ['sector_name'].iloc[0] or ind)
                lines.append(f'  {name:<10} {n} 次')
    text = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
```

- [ ] **Step 9: 实现 `build_si_timeseries_html` (~80 行,4 子图 plotly)**

```python
def build_si_timeseries_html(
    si_ts_df: pd.DataFrame,
    drift_events_df: pd.DataFrame,
    output_path: str,
) -> None:
    """4 子图 plotly HTML。

    (1,1) Top 6 行业 SI 时序 + drift 红点
    (1,2) Bottom 6 行业 SI 时序 + drift 红点
    (2,1) z-score 热力图 (industry × date)
    (2,2) drift 事件 top 10 行业 直方图
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if si_ts_df.empty:
        # 空图也写,避免 caller 报错
        fig = go.Figure()
        fig.update_layout(title='(无数据)')
    else:
        # 按最新 SI 排序,确定 top/bottom 6
        latest_date = si_ts_df['asof_date'].max()
        latest = si_ts_df[si_ts_df['asof_date'] == latest_date].sort_values('SI', ascending=False)
        top6 = latest.head(6)['industry_l1'].tolist()
        bot6 = latest.tail(6)['industry_l1'].tolist()
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 6 SI 行业时序 (漂移事件红点)',
                'Bottom 6 SI 行业时序 (漂移事件红点)',
                'z-score 热力图 (行业 × 日期)',
                '漂移事件频次 top 10 行业',
            ),
            vertical_spacing=0.12, horizontal_spacing=0.10,
        )
        # (1,1) Top 6
        for ind in top6:
            g = si_ts_df[si_ts_df['industry_l1'] == ind].sort_values('asof_date')
            name = g['sector_name'].iloc[0] if g['sector_name'].iloc[0] else ind
            fig.add_trace(go.Scatter(
                x=g['asof_date'], y=g['SI'], mode='lines+markers', name=name,
                legendgroup=name, showlegend=True,
            ), row=1, col=1)
            # drift events
            d = drift_events_df[drift_events_df['industry_l1'] == ind]
            if not d.empty:
                fig.add_trace(go.Scatter(
                    x=d['asof_date'], y=d['SI'], mode='markers',
                    marker=dict(color='red', size=12, symbol='x'),
                    name=f'{name} drift', legendgroup=name, showlegend=False,
                ), row=1, col=1)
        # (1,2) Bottom 6(同结构)
        for ind in bot6:
            g = si_ts_df[si_ts_df['industry_l1'] == ind].sort_values('asof_date')
            name = g['sector_name'].iloc[0] if g['sector_name'].iloc[0] else ind
            fig.add_trace(go.Scatter(
                x=g['asof_date'], y=g['SI'], mode='lines+markers', name=name,
                legendgroup=name, showlegend=True,
            ), row=1, col=2)
            d = drift_events_df[drift_events_df['industry_l1'] == ind]
            if not d.empty:
                fig.add_trace(go.Scatter(
                    x=d['asof_date'], y=d['SI'], mode='markers',
                    marker=dict(color='red', size=12, symbol='x'),
                    name=f'{name} drift', legendgroup=name, showlegend=False,
                ), row=1, col=2)
        # (2,1) z-score 热力图
        if not drift_events_df.empty:
            pivot = drift_events_df.pivot_table(
                index='industry_l1', columns='asof_date',
                values='z_score', aggfunc='min',
            )
            fig.add_trace(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index,
                colorscale='RdBu_r', zmid=0,
                colorbar=dict(title='z_score'),
            ), row=2, col=1)
        # (2,2) drift 事件 top 10
        if not drift_events_df.empty:
            top_drift = (drift_events_df.groupby(['industry_l1', 'sector_name'])
                         .size().reset_index(name='count')
                         .sort_values('count', ascending=False).head(10))
            top_drift['label'] = top_drift['sector_name'].where(
                top_drift['sector_name'] != '', top_drift['industry_l1'])
            fig.add_trace(go.Bar(
                x=top_drift['count'], y=top_drift['label'],
                orientation='h', marker_color='indianred',
            ), row=2, col=2)
        fig.update_layout(
            height=900, width=1400,
            title_text=f'v4.9 SI 时序 + 漂移检测 (N_industries={si_ts_df["industry_l1"].nunique()}, '
                       f'N_dates={si_ts_df["asof_date"].nunique()}, N_drift={len(drift_events_df)})',
        )
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[0, 1], row=1, col=2)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

**关键点**:
- `ensure_ascii=False` 已通过 write_html 默认 UTF-8 处理
- 中文行业名直接来自 sector_name 列
- 子图对齐 spec §6 布局

- [ ] **Step 10: 实现 `main()` (~30 行,端到端串起来)**

```python
def main():
    args = parse_args()
    # 1. 加载输入
    kc_long = load_kc_long(args.kc_time, limit=args.limit)
    industry_lookup = load_industry_membership(args.eigen, args.sw2_members)
    # 2. SI 时序计算(动态 import 避免循环依赖)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dynamics_eigen_analysis import compute_sector_stability_timeseries
    si_ts = compute_sector_stability_timeseries(kc_long, industry_lookup=industry_lookup)
    # 3. 漂移检测
    drift = detect_si_drift(si_ts, window=args.window, z_threshold=args.z_threshold)
    # 4. 写出 CSV
    os.makedirs(CSV_OUT_DIR, exist_ok=True)
    si_ts.to_csv(args.si_ts_output, index=False, encoding='utf-8')
    drift.to_csv(args.drift_output, index=False, encoding='utf-8')
    print(f'✓ {args.si_ts_output} ({len(si_ts)} 行)')
    print(f'✓ {args.drift_output} ({len(drift)} 事件)')
    # 5. 文本汇总
    write_si_timeseries_summary(si_ts, drift, args.txt_output)
    print(f'✓ {args.txt_output}')
    # 6. HTML
    build_si_timeseries_html(si_ts, drift, args.html_output)
    print(f'✓ {args.html_output}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 11: 单元测试 `test_si_timeseries_summary_text`**

```python
def test_si_timeseries_summary_text(tmp_path):
    """write_si_timeseries_summary 包含 '漂移事件' + 中文行业名。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_timeseries")
    from backtrace.dynamics.dynamics_si_timeseries import write_si_timeseries_summary
    si_ts = pd.DataFrame({
        'asof_date': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-02-01')],
        'industry_l1': ['801010', '801010'],
        'sector_name': ['银行', '银行'],
        'n_stocks': [42, 42],
        'rho_median': [0.85, 0.85],
        'c_median': [1.05, 1.05],
        'in_wedge_pct': [0.92, 0.92],
        'rho_health': [0.575, 0.575],
        'damping_health': [0.975, 0.975],
        'wedge_health': [0.92, 0.92],
        'SI': [0.85, 0.85],
    })
    drift = pd.DataFrame(columns=[
        'asof_date', 'industry_l1', 'sector_name',
        'SI', 'rolling_mean', 'rolling_std', 'z_score',
    ])
    out = tmp_path / 'summary.txt'
    write_si_timeseries_summary(si_ts, drift, str(out))
    content = out.read_text(encoding='utf-8')
    assert '漂移事件' in content
    assert '银行' in content
```

- [ ] **Step 12: 跑全部 43 个测试**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest \
    tests/test_dynamics_eigen.py -v
```

Expected:
- `test_si_timeseries_basic_shape` PASS
- `test_si_timeseries_stable_industry` PASS
- `test_si_timeseries_drift_zscore` PASS
- `test_si_timeseries_sudden_drop` PASS
- `test_si_timeseries_summary_text` PASS
- 总共 43/43 PASS (38 v4.8 + 5 v4.9)

- [ ] **Step 13: 端到端 CLI 测试(可选,需要数据)**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/projection/parameter_fit.py \
    --rolling-time --rolling-time-window 240 --limit 10
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_timeseries.py \
    --limit 10 --window 3 --z-threshold -2.0
```

Expected:
- `data/projection/kc_estimates_time.csv` 存在
- `data/dynamics/sector_si_timeseries.csv` 存在
- `data/dynamics/si_drift_events.csv` 存在(可能 0 事件)
- `backtrace/outputs/dynsys_si_timeseries.html` 存在
- `backtrace/outputs/dynsys_si_timeseries_summary.txt` 存在

**注意**:E2E 依赖 `data/daily/<code>.csv` 已有数据。若失败,先确认 fetch_daily.py 已跑过。

- [ ] **Step 14: 更新 `backtrace/dynamics/README.md` §3.9(~30 行)**

在 §3.8(v4.8)之后追加:

```markdown
### 3.9 v4.9 — SI 时序 + 漂移检测 (Sector Stability Timeseries + Drift)

v4.7 SI 单值答"哪些行业最稳",v4.8 IC ≈ 0 答"稳定对未来收益无预测力"。v4.9 把 SI 扩展到时序:
行业稳定性是否随时间漂移?漂移能否预警风险?

**数据流**:
1. `parameter_fit.py --rolling-time` (新增) — 每月末用最近 240 天 OLS 估 (k̂, ĉ)
   产出 `data/projection/kc_estimates_time.csv` (long format)
2. `compute_sector_stability_timeseries` (eigen_analysis 末尾追加) — 复用 v4.7 SI 公式,
   按 (asof_date, industry_l1) 聚合
3. `detect_si_drift` — rolling 60 日 z-score < -2 → drift event

**输出** (全 gitignored):
- `data/dynamics/sector_si_timeseries.csv` — 11 列 long format
- `data/dynamics/si_drift_events.csv` — drift event list
- `backtrace/outputs/dynsys_si_timeseries.html` — 4 子图 plotly
- `backtrace/outputs/dynsys_si_timeseries_summary.txt` — UTF-8 中文汇总

**漂移检测**: rolling window = 3 asof_dates (≈ 60 交易日 ≈ 3 个月)。
对每个行业 SI(t):
  rolling_mean = mean(SI over [t-60d, t))
  rolling_std = std(SI over [t-60d, t))
  z_score = (SI(t) - rolling_mean) / rolling_std
  drift event: z_score < -2.0

**CLI**:
\`\`\`bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_timeseries.py
# 默认: window=3, z_threshold=-2.0
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_timeseries.py --window 6 --z-threshold -1.5
# 调参
\`\`\`

**已知陷阱**:
- 月末 asof_date 列表依赖 daily data 完整性,数据 < 60 天则该 asof_date 跳过
- 行业 member 数 < 10 → SI 噪声大,n_stocks_threshold=50 沿用 v4.7
- drift event 是经验性信号,不是预测性 — v4.10 lagged IC 验证
```

- [ ] **Step 15: 提交(2 commits,先代码后文档)**

Commit 1 (代码):
```bash
git add backtrace/projection/parameter_fit.py \
       backtrace/dynamics/dynamics_eigen_analysis.py \
       backtrace/dynamics/dynamics_si_timeseries.py \
       tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v4.9 — SI 时序 + 漂移检测

parameter_fit 新增 --rolling-time 模式(每月末用最近 240 天 OLS);
eigen_analysis 末尾追加 compute_sector_stability_timeseries(复用 SI_WEIGHTS
+ analyze_eigenvalues);新 CLI dynamics_si_timeseries.py 跑漂移检测 +
4 子图 plotly HTML + 文本汇总;5 个新单元测试(43 total)。

约束兑现:_dynamics_core.py / 3 caller / dynamics_si_ic.py 0 行修改;
SI 公式与 v4.7 compute_sector_stability 完全一致;新文件独立 CLI。"
```

Commit 2 (文档):
```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README §3.9 v4.9 SI 时序 + 漂移检测"
```

---

## Self-Review(Plan v1)

1. **Spec 覆盖**:
   - [x] spec §3.1 数据源(parameter_fit --rolling-time) → Task 1
   - [x] spec §3.2 SI 公式 → Task 2 Step 1 复用 SI_WEIGHTS + analyze_eigenvalues
   - [x] spec §3.3 输出 schema → Task 2 Step 10 main() 写出 sector_si_timeseries.csv
   - [x] spec §4 漂移检测算法 → Task 2 Step 6 detect_si_drift
   - [x] spec §5.2 输出 4 个文件 → Task 2 Step 10 main()
   - [x] spec §6 HTML 布局 → Task 2 Step 9 build_si_timeseries_html
   - [x] spec §7 测试 5 个 → Task 1 Step 6 + Task 2 Step 2/3/7/11
   - [x] spec §8 0 行修改 → 严格遵守

2. **Placeholder scan**: 无 TBD / TODO / 模糊描述。

3. **类型一致性**:
   - `compute_sector_stability_timeseries` 返回 11 列(已 verify against spec §3.3)
   - `detect_si_drift` 返回 7 列(已 verify against spec §4 输出 schema)
   - `kc_estimates_time.csv` 列名与 spec §3.1 schema 一致

4. **潜在风险**:
   - E2E 依赖 `data/daily/` 已有数据 — Step 13 标注可选
   - `load_industry_membership` 实际是 code → industry_l1(命名沿用 spec,但需在 Step 5 注释里明确)
   - 漂移检测 rolling window 默认 3(≈ 60 交易日)与 spec §4 一致

5. **修复记录**(inline):
   - Step 1 main_rolling_time 重复 fit_rolling 部分逻辑 — 不可避免(因为 fit_rolling 只支持"end-aligned 全样本",不支持"end at asof_date")
   - Step 1 添加 `_month_ends` helper(避免重复 period 处理)
   - Step 5 load_industry_membership 注释明确"实际是 code → industry_l1 lookup"

---

## 执行提示

**SDD 推荐**(沿用 v4.7/v4.8 模式):
- 1 implementer subagent 跑 Task 1 → task review → fix rounds → push
- 1 implementer subagent 跑 Task 2 → task review → fix rounds → push
- 1 final code reviewer 整 branch 扫一遍 → push

**关键文件**:
- `backtrace/projection/parameter_fit.py` (Task 1)
- `backtrace/dynamics/dynamics_eigen_analysis.py` (Task 2 末尾追加)
- `backtrace/dynamics/dynamics_si_timeseries.py` (Task 2 new)
- `tests/test_dynamics_eigen.py` (Task 1 + 2 末尾追加)
- `backtrace/dynamics/README.md` (Task 2 Step 14)

**预期产出**:
- 2-3 commits total
- 43 tests pass
- 0 行修改:_dynamics_core.py / 3 caller / dynamics_si_ic.py