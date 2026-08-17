# v4.10 时序 SI 的 lagged IC 评估 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 lagged IC(时序 SI(t) vs future forward return)测"今日 SI 能否预测未来收益排名",闭环 v4.8 的 contemporaneous IC ≈ 0 结论。

**Architecture:**
- **独立 CLI** `backtrace/dynamics/dynamics_si_lagged_ic.py` (~300 行)— 不 import 同目录 sibling 模块,沿用 v4.8 / v4.9 模式
- **复用 v4.8 IC 计算模式**:`scipy.stats.spearmanr` 跨截面 + 60 日窗口 / 20 步长 + per-day IC 取均值
- **关键差异(vs v4.8)**:输入 SI 是**时序的**(每行业 N 个 asof_date 的 SI,来自 v4.9),不是单值 SI;IC 计算时 SI 时序对齐需向后偏移 h 日

**Tech Stack:**
- Python 3.13 (Anaconda)
- pandas / numpy / scipy.stats.spearmanr(沿用 v4.8)
- plotly (make_subplots, 沿用 v4.9)
- pytest + tmp_path fixtures

---

## Global Constraints

复制自 spec(每条都需严格遵守):

- 数学层 `_dynamics_core.py` **0 行修改**(硬约束,任务验证会查)
- 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) **0 行修改**
- `dynamics_si_ic.py` (v4.8) **0 行修改**(新文件独立 CLI)
- `dynamics_si_timeseries.py` (v4.9) **0 行修改**(只读 sector_si_timeseries.csv)
- `compute_sector_stability_timeseries` (v4.9) **0 行修改**
- v4.7 `compute_sector_stability` **不动**
- 输出全部 gitignored (`data/dynamics/` + `backtrace/outputs/`)
- `PYTHONIOENCODING=utf-8` 必备(Windows GBK)
- Python 路径:`/c/ProgramData/anaconda3/python.exe`
- 安全:`jhzq/交易凭据.md` 不能写进代码或 git
- Subagent-Driven Development (SDD) workflow
- 总测试数:44 (v4.9) + 4 (v4.10) = **48 tests pass**
- rolling 窗口 = 60 交易日 / 步长 20 日(沿用 v4.8)
- horizons = {20, 60} 日

---

## Task 1: 时序 SI 的 lagged IC 评估 CLI + 4 测试 + README

**Files:**
- New: `backtrace/dynamics/dynamics_si_lagged_ic.py` (~300 行,独立 CLI)
- Modify: `tests/test_dynamics_eigen.py` (末尾追加 4 测试, ~70 行)
- Modify: `backtrace/dynamics/README.md` §3.10 (~30 行)

**Interfaces:**
- Consumes: `data/dynamics/sector_si_timeseries.csv` (v4.9 产出,11 列 long format)
- Consumes: `data/projection/kc_estimates.csv` 或 `data/dynamics/eigen_summary.csv` (回查 code → industry_l1)
- Consumes: `data/sw2/members.csv` (code → sector_name 中文名)
- Consumes: `data/daily/<code>.csv` (forward return 计算)
- Produces:
  - `data/dynamics/si_lagged_ic_summary.csv` (2 horizons × 6 列)
  - `data/dynamics/si_lagged_ic_timeseries.csv` (per-window detail)
  - `backtrace/outputs/dynsys_si_lagged_ic.html` (3 子图)
  - `backtrace/outputs/dynsys_si_lagged_ic_summary.txt` (UTF-8 中文)

- [ ] **Step 1: 新建 `dynamics_si_lagged_ic.py` 骨架(~50 行,imports + argparse + main() stub)**

创建文件:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v4.10 — 时序 SI 的 lagged IC 评估 CLI。

读 v4.9 sector_si_timeseries.csv(每行业时序 SI)+ 各行业 forward return,
用 lagged Spearman IC 测"今日 SI 能否预测未来收益排名"。
区别于 v4.8: lagged IC 是真正预测性测试(SI 领先 forward 收益 h 日)。

输出(全 gitignored):
  - data/dynamics/si_lagged_ic_summary.csv
  - data/dynamics/si_lagged_ic_timeseries.csv
  - backtrace/outputs/dynsys_si_lagged_ic.html
  - backtrace/outputs/dynsys_si_lagged_ic_summary.txt
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

import argparse
import numpy as np
import pandas as pd

CSV_OUT_DIR = 'data/dynamics'
HTML_OUT_DIR = 'backtrace/outputs'
DEFAULT_SI_TS = os.path.join(CSV_OUT_DIR, 'sector_si_timeseries.csv')
DEFAULT_KC = 'data/projection/kc_estimates.csv'
DEFAULT_SW2 = 'data/sw2/members.csv'
DEFAULT_DAILY_DIR = 'data/daily'
DEFAULT_V8_SUMMARY = os.path.join(CSV_OUT_DIR, 'si_ic_summary.csv')
DEFAULT_IC_SUMMARY = os.path.join(CSV_OUT_DIR, 'si_lagged_ic_summary.csv')
DEFAULT_IC_TS = os.path.join(CSV_OUT_DIR, 'si_lagged_ic_timeseries.csv')
DEFAULT_HTML = os.path.join(HTML_OUT_DIR, 'dynsys_si_lagged_ic.html')
DEFAULT_TXT = os.path.join(HTML_OUT_DIR, 'dynsys_si_lagged_ic_summary.txt')


def parse_args():
    p = argparse.ArgumentParser(description='v4.10 时序 SI 的 lagged IC 评估')
    p.add_argument('--si-timeseries', default=DEFAULT_SI_TS,
                   help=f'v4.9 sector_si_timeseries.csv 路径 (默认 {DEFAULT_SI_TS})')
    p.add_argument('--kc-estimates', default=DEFAULT_KC,
                   help=f'kc_estimates.csv 路径 (默认 {DEFAULT_KC})')
    p.add_argument('--sw2-members', default=DEFAULT_SW2,
                   help=f'sw2 members.csv 路径 (默认 {DEFAULT_SW2})')
    p.add_argument('--daily-dir', default=DEFAULT_DAILY_DIR,
                   help=f'日线 CSV 目录 (默认 {DEFAULT_DAILY_DIR})')
    p.add_argument('--v8-summary', default=DEFAULT_V8_SUMMARY,
                   help=f'v4.8 si_ic_summary.csv 路径 (默认 {DEFAULT_V8_SUMMARY},用于对比子图)')
    p.add_argument('--ic-summary-output', default=DEFAULT_IC_SUMMARY,
                   help=f'si_lagged_ic_summary.csv 输出路径 (默认 {DEFAULT_IC_SUMMARY})')
    p.add_argument('--ic-timeseries-output', default=DEFAULT_IC_TS,
                   help=f'si_lagged_ic_timeseries.csv 输出路径 (默认 {DEFAULT_IC_TS})')
    p.add_argument('--html-output', default=DEFAULT_HTML,
                   help=f'HTML 输出路径 (默认 {DEFAULT_HTML})')
    p.add_argument('--txt-output', default=DEFAULT_TXT,
                   help=f'文本汇总输出路径 (默认 {DEFAULT_TXT})')
    p.add_argument('--window', type=int, default=60,
                   help='rolling window 大小(交易日,默认 60)')
    p.add_argument('--step', type=int, default=20,
                   help='rolling step 大小(交易日,默认 20)')
    p.add_argument('--horizons', default='20,60',
                   help='forward horizons 日(逗号分隔,默认 20,60)')
    p.add_argument('--limit', type=int, default=0,
                   help='限制行业数(0 = 全部,默认 0;冒烟测试用)')
    return p.parse_args()
```

**关键点**:
- 不 import 同目录 sibling 模块(避免循环依赖 + 与 v4.8/v4.9 模式一致)
- 不在此步实现 main() 业务逻辑,只骨架

- [ ] **Step 2: 实现 `load_sector_si_timeseries` 函数(~25 行)**

放在 `parse_args` 之后:

```python
def load_sector_si_timeseries(path: str) -> pd.DataFrame:
    """读 v4.9 sector_si_timeseries.csv(11 列 long format)。

    Returns:
        DataFrame with columns: asof_date, industry_l1, sector_name, n_stocks,
                                 rho_median, c_median, in_wedge_pct,
                                 rho_health, damping_health, wedge_health, SI
        asof_date 转 datetime,按 (industry_l1, asof_date) 排序
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} 不存在。先跑 v4.9 CLI:\n'
            f'  python backtrace/dynamics/dynamics_si_timeseries.py --limit <N>'
        )
    df = pd.read_csv(path, encoding='utf-8')
    df['asof_date'] = pd.to_datetime(df['asof_date'])
    return df.sort_values(['industry_l1', 'asof_date']).reset_index(drop=True)


def load_industry_membership(kc_path: str, sw2_path: str) -> dict:
    """回查 code → industry_l1(沿用 v4.8 实现风格)。"""
    if os.path.exists(kc_path):
        df = pd.read_csv(kc_path, usecols=['code', 'industry_l1'], encoding='utf-8')
        lookup = dict(zip(df['code'], df['industry_l1']))
        if lookup:
            return lookup
    raise FileNotFoundError(
        f'无法构建 code → industry_l1 映射。{kc_path} 不含 industry_l1 列。'
    )
```

- [ ] **Step 3: 实现 `compute_industry_forward_returns`(~50 行,沿用 v4.8 算法)**

```python
def compute_industry_forward_returns(
    members_by_industry: dict,    # {industry_l1: [code1, code2, ...]}
    daily_dir: str,
    eval_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """对每个行业在每个 eval_date 算 forward horizon 日收益(中位数收盘价法)。

    Returns:
        DataFrame with columns: asof_date, industry_l1, forward_return
        NaN 表示该行业在该 eval_date 缺失 / 数据不足
    """
    rows = []
    for ind, codes in members_by_industry.items():
        valid_codes = []
        median_prices_by_date = {}
        for code in codes:
            csv = os.path.join(daily_dir, f'{code}.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv, encoding='utf-8')
            if 'close' not in df.columns or 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            median_prices_by_date[code] = df['close']
            valid_codes.append(code)
        if len(valid_codes) < 3:
            continue
        # 拼接所有成员的 close 价,算每天的中位数
        all_dates = sorted(set().union(*[s.index for s in median_prices_by_date.values()]))
        if not all_dates:
            continue
        median_df = pd.DataFrame(index=all_dates)
        for code in valid_codes:
            median_df[code] = median_prices_by_date[code]
        median_close = median_df.median(axis=1)
        # 对每个 eval_date 算 forward return
        for t in eval_dates:
            t_idx = median_close.index.get_indexer([t], method='ffill')[0]
            if t_idx < 0 or t_idx + horizon >= len(median_close):
                continue
            p_now = median_close.iloc[t_idx]
            p_next = median_close.iloc[t_idx + horizon]
            if pd.isna(p_now) or pd.isna(p_next) or p_now <= 0:
                continue
            rows.append({
                'asof_date': t,
                'industry_l1': ind,
                'forward_return': float((p_next - p_now) / p_now),
            })
    return pd.DataFrame(rows)
```

**关键点**:
- 中位数法抗极端值(沿用 v4.8 §3.1)
- `n_industries < 5` 的行业跳过
- NaN 通过 DataFrame 自然传播,后续 `dropna()` 处理

- [ ] **Step 4: 实现 `compute_lagged_cross_sectional_ic`(~40 行)**

**核心:lagged 对齐**(spec §3.4)。

```python
def compute_lagged_cross_sectional_ic(
    si_ts_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """对每个 eval_date t:Spearman(SI 在 t-h 的排名, forward_return 在 t 的排名)。

    Args:
        si_ts_df: 11 列时序 SI(v4.9 产出)
        forward_returns_df: 3 列 (asof_date, industry_l1, forward_return)
        horizon: lagged 偏移天数(20 或 60)

    Returns:
        DataFrame with columns: asof_date, horizon, ic, p_value, n_industries
        NaN 表示该日 IC 不可计算(< 5 行业或 std=0)
    """
    from scipy.stats import spearmanr
    # 把 forward return 按 industry × date pivot 成宽表
    pivot = forward_returns_df.pivot_table(
        index='asof_date', columns='industry_l1', values='forward_return',
    )
    # 把 SI 时序按 industry × date pivot
    si_pivot = si_ts_df.pivot_table(
        index='asof_date', columns='industry_l1', values='SI',
    )
    rows = []
    for t in pivot.index:
        # lagged 对齐:SI 在 (t - horizon) 时刻的排名
        si_target_date = t - pd.Timedelta(days=horizon)
        # 找 si_pivot 中 ≤ si_target_date 的最新 asof_date
        si_avail = si_pivot.index[si_pivot.index <= si_target_date]
        if len(si_avail) == 0:
            continue
        si_date = si_avail.max()
        si_row = si_pivot.loc[si_date].dropna()
        fwd_row = pivot.loc[t].dropna()
        # 对齐行业
        common = si_row.index.intersection(fwd_row.index)
        if len(common) < 5:
            continue
        si_vals = si_row[common].values
        fwd_vals = fwd_row[common].values
        # 跨截面 Spearman(避免常数方差)
        if np.std(si_vals) < 1e-9 or np.std(fwd_vals) < 1e-9:
            continue
        corr, p = spearmanr(si_vals, fwd_vals)
        if not np.isfinite(corr):
            continue
        rows.append({
            'asof_date': t,
            'horizon': horizon,
            'ic': float(corr),
            'p_value': float(p),
            'n_industries': len(common),
        })
    return pd.DataFrame(rows)
```

**关键点**:
- SI 时序对齐:用 `asof_date <= t - horizon` 的最新 SI(因为 asof_date 是月度,可能不恰好等于 `t - horizon`)
- `n_industries >= 5` 才计算(避免 spearmanr 在 < 3 时退化)
- `np.std < 1e-9` 跳过常数序列

- [ ] **Step 5: 实现 `rolling_lagged_ic` 函数(~30 行)**

```python
def rolling_lagged_ic(
    daily_ic_df: pd.DataFrame,
    window: int = 60,
    step: int = 20,
) -> pd.DataFrame:
    """对每日 lagged IC 做滚动窗口平均。

    Args:
        daily_ic_df: Step 4 输出 (asof_date, horizon, ic, p_value, n_industries)
        window: rolling window(默认 60 日)
        step: rolling step(默认 20 日)

    Returns:
        DataFrame with columns: window_end_date, horizon, ic, p_value, n_industries
        排序: (window_end_date ASC, horizon ASC)
    """
    if daily_ic_df.empty:
        return pd.DataFrame(columns=[
            'window_end_date', 'horizon', 'ic', 'p_value', 'n_industries',
        ])
    daily_ic_df = daily_ic_df.sort_values('asof_date').reset_index(drop=True)
    rows = []
    for h, g in daily_ic_df.groupby('horizon'):
        g = g.sort_values('asof_date').reset_index(drop=True)
        dates = g['asof_date'].values
        ics = g['ic'].values
        ps = g['p_value'].values
        ns = g['n_industries'].values
        n = len(g)
        # 滚动窗口 (闭区间 [i-step, i],步长 step)
        for i in range(window - 1, n, step):
            window_ics = ics[i - window + 1: i + 1]
            window_ps = ps[i - window + 1: i + 1]
            # 跳过含 NaN 的窗口
            mask = np.isfinite(window_ics) & np.isfinite(window_ps)
            if mask.sum() < window // 2:
                continue
            rows.append({
                'window_end_date': pd.Timestamp(dates[i]),
                'horizon': int(h),
                'ic': float(np.nanmean(window_ics)),
                'p_value': float(np.nanmean(window_ps)),
                'n_industries': int(np.mean(ns[i - window + 1: i + 1])),
            })
    return pd.DataFrame(rows).sort_values(['window_end_date', 'horizon']).reset_index(drop=True)
```

- [ ] **Step 6: 实现 `write_si_lagged_ic_summary`(~25 行,UTF-8 文本)**

```python
def write_si_lagged_ic_summary(
    timeseries_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """写跨期汇总 + UTF-8 中文文本。

    Returns:
        summary_df (2 horizons × 6 列) 同时也写出 CSV
    """
    if timeseries_df.empty:
        summary = pd.DataFrame(columns=[
            'horizon', 'ic_mean', 'ic_std', 'ic_ir', 'p_value_mean', 'n_windows',
        ])
    else:
        summary = timeseries_df.groupby('horizon').agg(
            ic_mean=('ic', 'mean'),
            ic_std=('ic', 'std'),
            ic_ir=('ic', lambda s: s.mean() / s.std() if s.std() > 0 else 0.0),
            p_value_mean=('p_value', 'mean'),
            n_windows=('ic', 'count'),
        ).reset_index()
        summary['ic_mean'] = summary['ic_mean'].round(4)
        summary['ic_std'] = summary['ic_std'].round(4)
        summary['ic_ir'] = summary['ic_ir'].round(4)
        summary['p_value_mean'] = summary['p_value_mean'].round(4)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    summary.to_csv(output_path, index=False, encoding='utf-8')
    # 文本汇总
    lines = [
        '=' * 70,
        'v4.10 时序 SI 的 lagged IC 评估',
        '=' * 70,
        f'窗口数: {len(timeseries_df)}',
        f'Horizons: {sorted(timeseries_df["horizon"].unique()) if not timeseries_df.empty else "无"}',
        '',
    ]
    if not summary.empty:
        for _, row in summary.iterrows():
            verdict = ('预测性(显著)' if row['ic_ir'] > 0.5 and row['p_value_mean'] < 0.05
                       else '弱预测' if row['ic_ir'] > 0.2
                       else '描述性(不显著)')
            lines.append(
                f"horizon={int(row['horizon'])}d: ic_mean={row['ic_mean']:+.4f} "
                f"ic_std={row['ic_std']:.4f} ic_ir={row['ic_ir']:+.4f} "
                f"p_mean={row['p_value_mean']:.4f} n={int(row['n_windows'])} "
                f"-> {verdict}"
            )
    text = '\n'.join(lines) + '\n'
    txt_path = output_path.replace('.csv', '_summary.txt')
    # 注意:这里约定 output_path 是 summary csv 路径,txt 路径由 caller 决定
    # 实际 caller 会传 txt_path 单独处理
    return summary, text
```

**简化**:把文本写出单独抽到 main() 末尾,这里只返回 summary + text 字符串,避免路径耦合。修改签名:

```python
def write_si_lagged_ic_summary(
    timeseries_df: pd.DataFrame,
    output_csv_path: str,
) -> tuple[pd.DataFrame, str]:
    """返回 (summary_df, text_str)。CSV 写出到 output_csv_path。"""
    ...
```

- [ ] **Step 7: 单元测试 `test_si_lagged_ic_synthetic_perfect` + `test_si_lagged_ic_synthetic_random`**

放在 `tests/test_dynamics_eigen.py` 末尾:

```python
def test_si_lagged_ic_synthetic_perfect():
    """5 行业 × 100 日,SI(t) 与 forward(t+20) 完美正相关 → lagged IC > 0.5。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    # SI 时序:5 行业 × 100 日,行业 i 的 SI 始终 = (i+1)/5
    si_rows = []
    for ind in range(5):
        for d in range(100):
            si_rows.append({
                'asof_date': dates[d],
                'industry_l1': f'8010{ind:02d}',
                'sector_name': f'测试_{ind}',
                'n_stocks': 42,
                'rho_median': 0.85, 'c_median': 1.05,
                'in_wedge_pct': 0.92, 'rho_health': 0.575,
                'damping_health': 0.975, 'wedge_health': 0.92,
                'SI': (ind + 1) / 5.0,
            })
    si_ts = pd.DataFrame(si_rows)
    # forward returns:行业 i 的 forward_return 始终 = (i+1)/5(完美正相关)
    fwd_rows = []
    for ind in range(5):
        for d in range(80):  # 100-20=80 个 eval_dates
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{ind:02d}',
                'forward_return': (ind + 1) / 5.0,
            })
    fwd = pd.DataFrame(fwd_rows)
    # 算 lagged IC,horizon=20
    daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    assert len(daily_ic) > 0, '应有至少 1 个 IC'
    assert daily_ic['ic'].mean() > 0.5, f'完美正相关应 IC > 0.5,got {daily_ic["ic"].mean()}'


def test_si_lagged_ic_synthetic_random():
    """5 行业 × 100 日,SI 与 forward 完全独立 → |lagged IC| < 0.3。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    si_rows = []
    for ind in range(5):
        for d in range(100):
            si_rows.append({
                'asof_date': dates[d],
                'industry_l1': f'8010{ind:02d}',
                'sector_name': f'测试_{ind}',
                'n_stocks': 42,
                'rho_median': 0.85, 'c_median': 1.05,
                'in_wedge_pct': 0.92, 'rho_health': 0.575,
                'damping_health': 0.975, 'wedge_health': 0.92,
                'SI': np.random.rand(),
            })
    si_ts = pd.DataFrame(si_rows)
    fwd_rows = []
    for ind in range(5):
        for d in range(80):
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{ind:02d}',
                'forward_return': np.random.rand(),
            })
    fwd = pd.DataFrame(fwd_rows)
    daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    if len(daily_ic) > 0:
        mean_abs_ic = daily_ic['ic'].abs().mean()
        assert mean_abs_ic < 0.3, f'随机应 |IC| < 0.3,got {mean_abs_ic}'
```

- [ ] **Step 8: 实现 `build_si_lagged_ic_html`(~80 行,3 子图 plotly)**

```python
def build_si_lagged_ic_html(
    timeseries_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    v8_summary_path: str,
    output_path: str,
) -> None:
    """3 子图 plotly HTML。

    (1,1) Lagged IC 时序(20d / 60d 双线)+ IC=0 红虚线
    (1,2) v4.10 lagged vs v4.8 contemporaneous IC 对比(若 v4.8 CSV 存在)
    (2,1, 全宽) IC 统计汇总表
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Lagged IC 时序 (20d / 60d) + IC=0 红虚线',
            'v4.10 lagged vs v4.8 contemporaneous IC 对比',
            'IC 统计汇总',
        ),
        specs=[[{}, {}], [{'colspan': 2}, None]],
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )
    # (1,1) Lagged IC 时序
    if not timeseries_df.empty:
        for h in sorted(timeseries_df['horizon'].unique()):
            g = timeseries_df[timeseries_df['horizon'] == h].sort_values('window_end_date')
            fig.add_trace(go.Scatter(
                x=g['window_end_date'], y=g['ic'],
                mode='lines+markers',
                name=f'horizon={int(h)}d',
                error_y=dict(type='data', array=g['p_value'], visible=False),
            ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)
    fig.update_yaxes(range=[-0.5, 0.5], row=1, col=1, title_text='IC')
    # (1,2) v4.10 vs v4.8 对比
    v8_loaded = False
    if os.path.exists(v8_summary_path):
        try:
            v8 = pd.read_csv(v8_summary_path, encoding='utf-8')
            if not v8.empty and not summary_df.empty:
                merged = summary_df.merge(
                    v8, on='horizon', suffixes=('_v10', '_v8'),
                )
                if not merged.empty:
                    fig.add_trace(go.Scatter(
                        x=merged['ic_mean_v8'], y=merged['ic_mean_v10'],
                        mode='markers+text', text=merged['horizon'].astype(int).astype(str) + 'd',
                        textposition='top center',
                        marker=dict(size=14, color='indianred'),
                        name='horizon',
                    ), row=1, col=2)
                    # y=x 参考线
                    lim = max(abs(merged['ic_mean_v10'].min()), abs(merged['ic_mean_v8'].max()), 0.1)
                    fig.add_trace(go.Scatter(
                        x=[-lim, lim], y=[-lim, lim],
                        mode='lines', line=dict(color='gray', dash='dash'),
                        name='y=x 参考线', showlegend=False,
                    ), row=1, col=2)
                    v8_loaded = True
        except Exception:
            pass
    if not v8_loaded:
        fig.add_annotation(
            text='v4.8 si_ic_summary.csv 未生成,跳过对比',
            xref='x2 domain', yref='y2 domain',
            x=0.5, y=0.5, showarrow=False,
            row=1, col=2,
        )
    fig.update_xaxes(title_text='v4.8 IC_mean', row=1, col=2)
    fig.update_yaxes(title_text='v4.10 lagged IC_mean', row=1, col=2)
    # (2,1) 统计表
    if not summary_df.empty:
        table_text = '<br>'.join(
            f"h={int(r['horizon'])}d: ic_mean={r['ic_mean']:+.4f} "
            f"ic_ir={r['ic_ir']:+.3f} p={r['p_value_mean']:.3f} n={int(r['n_windows'])}"
            for _, r in summary_df.iterrows()
        )
        fig.add_annotation(
            text=table_text, xref='x3 domain', yref='y3 domain',
            x=0.05, y=0.5, showarrow=False, align='left',
            font=dict(family='monospace', size=12),
            row=2, col=1,
        )
    fig.update_layout(
        height=900, width=1400,
        title_text=f'v4.10 时序 SI 的 lagged IC 评估 (N_windows={len(timeseries_df)})',
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

**关键点**:
- v4.8 CSV 不存在时子图 (1,2) 退化(annotation 提示,不报错)
- 中文文本通过 `<br>` 在 annotation 中展示

- [ ] **Step 9: 单元测试 `test_si_lagged_ic_temporal_shift` + `test_si_lagged_ic_summary_schema`**

```python
def test_si_lagged_ic_temporal_shift(tmp_path):
    """验证时间偏移正确: t=0 SI 排名 + t=20 forward 收益排名相关。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import compute_lagged_cross_sectional_ic
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    # SI 在 t=0..99 行业排名:[高, 中, 低, 中高, 中低]
    si_vals_per_ind = [0.9, 0.7, 0.3, 0.8, 0.4]
    si_rows = []
    for ind, s in enumerate(si_vals_per_ind):
        for d in range(100):
            si_rows.append({
                'asof_date': dates[d],
                'industry_l1': f'8010{ind:02d}',
                'sector_name': f'测试_{ind}',
                'n_stocks': 42,
                'rho_median': 0.85, 'c_median': 1.05,
                'in_wedge_pct': 0.92, 'rho_health': 0.575,
                'damping_health': 0.975, 'wedge_health': 0.92,
                'SI': s,
            })
    si_ts = pd.DataFrame(si_rows)
    # forward return 在 t=20..99:与 SI 在 t=0..79 完全正相关
    fwd_vals_per_ind = si_vals_per_ind  # 同样的排名
    fwd_rows = []
    for ind, fv in enumerate(fwd_vals_per_ind):
        for d in range(80):  # eval_dates 100-20=80
            fwd_rows.append({
                'asof_date': dates[d + 20],
                'industry_l1': f'8010{ind:02d}',
                'forward_return': fv,
            })
    fwd = pd.DataFrame(fwd_rows)
    daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=20)
    assert len(daily_ic) > 0
    # 完美正相关 → IC ≈ 1.0
    assert daily_ic['ic'].mean() > 0.9


def test_si_lagged_ic_summary_schema(tmp_path):
    """write_si_lagged_ic_summary 写出 2 horizons × 6 列。"""
    pytest.importorskip("backtrace.dynamics.dynamics_si_lagged_ic")
    from backtrace.dynamics.dynamics_si_lagged_ic import write_si_lagged_ic_summary
    # 构造 timeseries:2 horizons × 5 windows
    rows = []
    for h in [20, 60]:
        for w in range(5):
            rows.append({
                'window_end_date': pd.Timestamp('2024-01-01') + pd.DateOffset(days=w*20),
                'horizon': h,
                'ic': 0.05 + w * 0.01,
                'p_value': 0.3 - w * 0.05,
                'n_industries': 25,
            })
    ts = pd.DataFrame(rows)
    out = tmp_path / 'ic_summary.csv'
    summary, text = write_si_lagged_ic_summary(ts, str(out))
    assert len(summary) == 2, f'expected 2 horizons, got {len(summary)}'
    assert set(summary.columns) >= {
        'horizon', 'ic_mean', 'ic_std', 'ic_ir', 'p_value_mean', 'n_windows',
    }
    assert out.exists()
    # 文本包含 horizons
    assert 'horizon=' in text
```

- [ ] **Step 10: 实现 `main()`(~50 行,端到端)**

```python
def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(',')]
    # 1. 加载 SI 时序
    si_ts = load_sector_si_timeseries(args.si_timeseries)
    # 限制行业(冒烟)
    if args.limit > 0:
        top_ind = (si_ts.groupby('industry_l1')['SI'].last()
                   .sort_values(ascending=False).head(args.limit).index)
        si_ts = si_ts[si_ts['industry_l1'].isin(top_ind)]
    # 2. 回查 industry membership(code → industry_l1)
    industry_lookup = load_industry_membership(args.kc_estimates, args.sw2_members)
    # 反向:industry_l1 → [code1, code2, ...]
    members_by_ind = {}
    for code, ind in industry_lookup.items():
        members_by_ind.setdefault(ind, []).append(code)
    members_by_ind = {ind: codes for ind, codes in members_by_ind.items()
                      if ind in si_ts['industry_l1'].unique()}
    # 3. eval_dates:取 SI 时序中所有 asof_date + horizon
    eval_dates = si_ts['asof_date'].unique()
    # 4. 对每个 horizon 算 daily lagged IC + rolling
    all_ts = []
    for h in horizons:
        fwd = compute_industry_forward_returns(members_by_ind, args.daily_dir, eval_dates, h)
        daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=h)
        rolled = rolling_lagged_ic(daily_ic, window=args.window, step=args.step)
        all_ts.append(rolled)
    timeseries = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()
    # 5. 写出 summary + timeseries CSV
    summary, text = write_si_lagged_ic_summary(timeseries, args.ic_summary_output)
    timeseries.to_csv(args.ic_timeseries_output, index=False, encoding='utf-8')
    print(f'✓ {args.ic_summary_output} ({len(summary)} horizons)')
    print(f'✓ {args.ic_timeseries_output} ({len(timeseries)} 窗口)')
    # 6. 写出文本汇总
    with open(args.txt_output, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'✓ {args.txt_output}')
    # 7. HTML
    build_si_lagged_ic_html(timeseries, summary, args.v8_summary, args.html_output)
    print(f'✓ {args.html_output}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 11: 跑全部 48 个测试**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest \
    tests/test_dynamics_eigen.py -v
```

Expected:
- `test_si_lagged_ic_synthetic_perfect` PASS
- `test_si_lagged_ic_synthetic_random` PASS
- `test_si_lagged_ic_temporal_shift` PASS
- `test_si_lagged_ic_summary_schema` PASS
- 总共 48/48 PASS (44 v4.9 + 4 v4.10)

- [ ] **Step 12: 端到端 CLI 测试(可选,需要数据)**

Run:
```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_lagged_ic.py \
    --limit 10 --window 30 --step 10 --horizons 20,60
```

Expected:
- 4 个新文件存在
- `si_lagged_ic_summary.csv` 2 行
- `dynsys_si_lagged_ic.html` 3 子图正常渲染

**注意**:E2E 依赖 `data/dynamics/sector_si_timeseries.csv` (v4.9 产出) + `data/daily/<code>.csv` 已有数据。若失败,先确认 v4.9 已跑过。

- [ ] **Step 13: 更新 `backtrace/dynamics/README.md` §3.10(~30 行)**

在 §3.9(v4.9)之后追加:

```markdown
### 3.10 v4.10 — 时序 SI 的 lagged IC 评估

v4.8 contemporaneous IC ≈ 0 揭示 SI 不是预测性指标。v4.10 闭环:用 lagged IC(时序 SI(t) vs future forward return)测"今日 SI 能否预测未来收益排名"。

**关键差异(vs v4.8)**:
- v4.8 contemporaneous:`Spearman(SI_i(t), r_i(t, h))` 同时点 SI vs forward return — 描述性
- v4.10 lagged:`Spearman(SI_i(t), r_i(t+h, h))` 不同时点 — 真正预测性测试

**数据流**:
1. 输入: `data/dynamics/sector_si_timeseries.csv` (v4.9 产出,11 列 long format)
2. 输入: `data/daily/<code>.csv` 算各行业 forward return(同 v4.8 中位数法)
3. lagged 对齐:对每个 eval_date t,取 SI 在 (t - horizon) 的排名,forward return 在 t 的排名
4. 跨截面 Spearman + 60 日 rolling window / 20 日 step
5. 输出 4 个文件(全 gitignored)

**输出**:
- `data/dynamics/si_lagged_ic_summary.csv` — 跨期汇总 2 horizons × 6 列
- `data/dynamics/si_lagged_ic_timeseries.csv` — per-window detail
- `backtrace/outputs/dynsys_si_lagged_ic.html` — 3 子图 plotly
  - (1,1) Lagged IC 时序 + IC=0 红虚线
  - (1,2) v4.10 lagged vs v4.8 contemporaneous 对比(若 v4.8 CSV 存在)
  - (2,1) IC 统计汇总表
- `backtrace/outputs/dynsys_si_lagged_ic_summary.txt` — UTF-8 中文汇总

**CLI**:
\`\`\`bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py
# 默认: window=60, step=20, horizons=20,60
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py --window 30 --step 10
# 短窗口实验
\`\`\`

**已知陷阱**:
- SI 时序是月度 asof_date,horizon=20/60 日可能不对齐 — 用 `asof_date <= t - horizon` 的最新 SI
- 行业 member 数 < 5 → 该 eval_date 跳过
- v4.8 CSV 缺失时,对比子图 (1,2) 退化为 annotation 提示
- 若 lagged IC ≈ 0 → 行业层**纯描述性**,SI 用于报告 / 风险标签,不作选股信号(确认 v4.8 结论)
- 若 lagged IC > 0.05 显著 → 行业 SI(t) 是预测性指标,v4.12 行业轮动策略有基础
```

- [ ] **Step 14: 提交(2 commits,先代码后文档)**

Commit 1 (代码):
```bash
git add backtrace/dynamics/dynamics_si_lagged_ic.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v4.10 — 时序 SI 的 lagged IC 评估

新 CLI dynamics_si_lagged_ic.py(独立,不改 v4.8/v4.9 任何文件):
- compute_lagged_cross_sectional_ic 闭环 v4.8 — 真正预测性测试
- rolling_lagged_ic 60 日窗 / 20 步长(沿用 v4.8 框架)
- write_si_lagged_ic_summary + build_si_lagged_ic_html(3 子图)
- 4 个新单元测试(48 total)

约束兑现:_dynamics_core.py / 3 caller / dynamics_si_ic.py(v4.8) /
dynamics_si_timeseries.py(v4.9) / compute_sector_stability_timeseries 0 行修改。"
```

Commit 2 (文档):
```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README §3.10 v4.10 时序 SI 的 lagged IC"
```

---

## Self-Review(Plan v1)

1. **Spec 覆盖**:
   - [x] spec §3.1 lagged vs contemporaneous 定义 → Step 4 注释
   - [x] spec §3.2 forward return(中位数法) → Step 3
   - [x] spec §3.3 rolling window 60 日 / step 20 日 → Step 5
   - [x] spec §3.4 lagged 对齐(`asof_date <= t - horizon`) → Step 4
   - [x] spec §3.5 跨期汇总 5 数 → Step 6
   - [x] spec §4.2 输出 4 文件 → Step 10 main()
   - [x] spec §5 HTML 3 子图 → Step 8
   - [x] spec §6 测试 4 个 → Steps 7 / 9
   - [x] spec §7 0 行修改约束 → 严格遵守(新文件独立 CLI)

2. **Placeholder scan**: 无 TBD / TODO。

3. **类型一致性**:
   - `compute_lagged_cross_sectional_ic` 返回 5 列(`asof_date, horizon, ic, p_value, n_industries`)— verify against spec §4.2
   - `write_si_lagged_ic_summary` 返回 `(summary_df, text_str)` 元组 — 与 Step 6 签名一致
   - `rolling_lagged_ic` 返回 5 列(`window_end_date, horizon, ic, p_value, n_industries`)— 与 spec §4.2 一致

4. **潜在风险**:
   - `compute_industry_forward_returns` 数据量大,可能慢 — 单元测试只 5 行业,E2E 真实数据慢属正常
   - lagged 对齐用 `asof_date <= t - horizon` 而非精确 `==` — 文档化在 Step 4 注释
   - `n_industries >= 5` 阈值(避免 spearmanr 在 < 3 时退化)— Step 4

5. **修复记录**(inline):
   - Step 6 重命名 `write_si_lagged_ic_summary` 返回签名,分离 CSV 路径与文本路径
   - Step 8 子图 (1,2) graceful degradation(v4.8 CSV 不存在时 annotation,不报错)
   - Step 10 main() 用 `si_ts['industry_l1'].unique()` 过滤 members_by_ind,避免空 industry 算 forward

---

## 执行提示

**SDD 推荐**(沿用 v4.7/v4.8/v4.9 模式):
- 1 implementer subagent 跑 Task 1 → task review → fix rounds → push
- 1 final code reviewer 整 branch 扫一遍 → push

**关键文件**:
- `backtrace/dynamics/dynamics_si_lagged_ic.py` (Task 1 new)
- `tests/test_dynamics_eigen.py` (Task 1 末尾追加)
- `backtrace/dynamics/README.md` (Task 1 Step 13)

**预期产出**:
- 2 commits total
- 48 tests pass
- 0 行修改:_dynamics_core.py / 3 caller / dynamics_si_ic.py(v4.8) / dynamics_si_timeseries.py(v4.9) / compute_sector_stability_timeseries