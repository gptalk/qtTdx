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
    p.add_argument('--ramp-up-min-n-valid', type=int, default=192,
                   help='ramp-up filter:仅保留 n_valid_days >= 该阈值 的行 '
                        '(默认 192 = 240 * 0.8,reviewer finding #2)')
    p.add_argument('--period', choices=['daily', '15m', '5m', '1m'], default='daily',
                   help='缓存粒度(daily = 默认)')
    return p.parse_args()


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


def detect_si_drift(
    si_ts_df: pd.DataFrame,
    window: int = 3,
    z_threshold: float = -2.0,
    min_n_valid_days: int = 0,
) -> pd.DataFrame:
    """对每个行业的 SI(t) 做 rolling z-score,触发 drift event。

    Args:
        si_ts_df: compute_sector_stability_timeseries 输出的 11 列
                  (若含 'n_valid_days_min' 列,且 min_n_valid_days > 0,
                  则跳过 n_valid_days_min < min_n_valid_days 的历史点 ——
                  ramp-up filter,reviewer finding #2)
        window: rolling window 大小(asof_date 数,默认 3 ≈ 60 交易日)
        z_threshold: 触发阈值(默认 -2.0,负值越极端越算漂移)
        min_n_valid_days: ramp-up 阈值(默认 0 = 不开)。
                          主流程在 main() 里直接对 kc_long_df 做 pre-filter,
                          此处仅作为 secondary 防御层(若 si_ts_df 携带
                          n_valid_days_min 列)。

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
    # Ramp-up filter 防御层:若 si_ts_df 含 n_valid_days_min,跳过低于阈值的历史点
    use_n_valid = (
        min_n_valid_days > 0
        and 'n_valid_days_min' in si_ts_df.columns
    )
    drift_rows = []
    for ind, g in si_ts_df.groupby('industry_l1'):
        si = g['SI'].values
        dates = g['asof_date'].values
        sector_name = g['sector_name'].iloc[0] if 'sector_name' in g.columns else ''
        n_valid_arr = g['n_valid_days_min'].values if use_n_valid else None
        n = len(si)
        for i in range(n):
            # rolling window = [max(0, i-window), i),不含 i 自身(避免 leak)
            s = max(0, i - window)
            if i - s < 2:  # 至少需要 2 个历史点算 std
                continue
            hist = si[s:i]
            hist = hist[np.isfinite(hist)]
            # Ramp-up filter:丢弃 n_valid_days_min < 阈值的历史点(若启用)
            if use_n_valid:
                mask = n_valid_arr[s:i] >= min_n_valid_days
                hist = hist[mask]
            if len(hist) < 2:
                continue
            m = float(np.mean(hist))
            sd = float(np.std(hist, ddof=1))
            # noise floor 0.01:history 区间恒定(sd=0)但新值与均值差异显著时,
            # 仍判为 drift(避免常数背景后突发骤降漏检)。
            # 真实数据 SI 含自然波动,sd>0 时 floor 无影响。
            sd_eff = max(sd, 0.01)
            if sd_eff < 1e-9:
                continue
            z = (si[i] - m) / sd_eff
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


def main():
    args = parse_args()
    # 1. 加载输入
    kc_long = load_kc_long(args.kc_time, limit=args.limit)
    industry_lookup = load_industry_membership(args.eigen, args.sw2_members)
    # 1.5 ★ Ramp-up filter (CRITICAL — reviewer finding #2):
    # Task 1 的 --rolling-time 模式对早期 asof_date 用的是 expanding window
    # (例如 240 天窗口,前 240 天只能用 [1..t] 的 expanding 子集),导致早期 (k̂, ĉ)
    # 估计单调漂移,与"行业 SI 时序漂移"严重混淆(伪漂移 = ramp-up artifact)。
    # 修法:rolling window 满 80% 才视为"高置信度";低于该阈值,该 (asof_date, stock)
    # 直接从输入剔除 → 聚合后该 asof_date 自然消失,漂移检测不会误触发。
    # 默认 240 天窗口 × 80% = 192 天;通过 --ramp-up-min-n-valid 调。
    ramp_up_min_n_valid = getattr(args, 'ramp_up_min_n_valid', 0)
    if ramp_up_min_n_valid > 0 and 'n_valid_days' in kc_long.columns:
        n_before = len(kc_long)
        kc_long = kc_long[kc_long['n_valid_days'] >= ramp_up_min_n_valid].copy()
        n_after = len(kc_long)
        print(f'[si-ts] ramp-up filter: {n_before} → {n_after} 行 '
              f'(保留 n_valid_days >= {ramp_up_min_n_valid})')
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