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

用法:
  PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py
  PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_lagged_ic.py --window 30 --step 10
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

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
    df = pd.read_csv(path, encoding='utf-8', dtype={'industry_l1': str})
    df['asof_date'] = pd.to_datetime(df['asof_date'])
    return df.sort_values(['industry_l1', 'asof_date']).reset_index(drop=True)


def load_industry_membership(kc_path: str, sw2_path: str) -> dict:
    """回查 code → industry_l1(沿用 v4.8 实现风格)。

    kc_estimates.csv 不含 industry_l1 时,用 sw2 members.csv 反查兜底。
    """
    if os.path.exists(kc_path):
        kc = pd.read_csv(kc_path, dtype={'code': str})
        if 'industry_l1' in kc.columns:
            kc = kc[kc['industry_l1'].notna() & (kc['industry_l1'].astype(str) != '')]
            lookup = dict(zip(kc['code'].astype(str), kc['industry_l1'].astype(str)))
            if lookup:
                return lookup
        if os.path.exists(sw2_path):
            sw2 = pd.read_csv(sw2_path, dtype={'member_code': str, 'sector_code': str})
            if {'member_code', 'sector_code'} <= set(sw2.columns):
                sw2 = sw2[['member_code', 'sector_code']].drop_duplicates('member_code')
                lookup = dict(zip(sw2['member_code'].astype(str),
                                  sw2['sector_code'].astype(str)))
                codes = set(kc['code'].astype(str)) if 'code' in kc.columns else None
                if codes:
                    lookup = {c: i for c, i in lookup.items() if c in codes}
                if lookup:
                    return lookup
    raise FileNotFoundError(
        f'无法构建 code → industry_l1 映射。{kc_path} 不含 industry_l1 列,'
        f'且 {sw2_path} 无法兜底。'
    )


def compute_industry_forward_returns(
    members_by_industry: dict,    # {industry_l1: [code1, code2, ...]}
    daily_dir: str,
    eval_dates,
    horizon: int,
) -> pd.DataFrame:
    """对每个行业在每个 eval_date 算 forward horizon 日收益(中位数收盘价法)。

    Returns:
        DataFrame with columns: asof_date, industry_l1, forward_return
        缺失 / 数据不足的 (行业, 日期) 直接不出行(后续 dropna 语义等价)
    """
    eval_dates = pd.DatetimeIndex(pd.to_datetime(list(eval_dates)))
    rows = []
    for ind, codes in members_by_industry.items():
        valid_codes = []
        close_by_code = {}
        for code in codes:
            csv = os.path.join(daily_dir, f'{code}.csv')
            if not os.path.exists(csv):
                continue
            df = pd.read_csv(csv, encoding='utf-8')
            if 'close' not in df.columns or 'date' not in df.columns:
                continue
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            close_by_code[code] = df['close']
            valid_codes.append(code)
        if len(valid_codes) < 3:
            continue
        median_df = pd.DataFrame(close_by_code).sort_index()
        if median_df.empty:
            continue
        median_close = median_df.median(axis=1)
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
    return pd.DataFrame(rows, columns=['asof_date', 'industry_l1', 'forward_return'])


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
        不可计算的日期(< 5 行业 / std=0 / 无可用 SI)直接不出行
    """
    from scipy.stats import spearmanr
    cols = ['asof_date', 'horizon', 'ic', 'p_value', 'n_industries']
    if si_ts_df is None or si_ts_df.empty or forward_returns_df is None or forward_returns_df.empty:
        return pd.DataFrame(columns=cols)
    # forward return 按 industry × date pivot 成宽表
    pivot = forward_returns_df.pivot_table(
        index='asof_date', columns='industry_l1', values='forward_return',
    )
    # SI 时序按 industry × date pivot
    si_pivot = si_ts_df.pivot_table(
        index='asof_date', columns='industry_l1', values='SI',
    )
    rows = []
    for t in pivot.index:
        # lagged 对齐:SI 在 (t - horizon) 时刻的排名。
        # SI 的 asof_date 通常是月度采样,不会恰好等于 t - horizon,
        # 故取 asof_date <= t - horizon 的最新一期(严格只用过去信息)。
        si_target_date = t - pd.Timedelta(days=horizon)
        si_avail = si_pivot.index[si_pivot.index <= si_target_date]
        if len(si_avail) == 0:
            continue
        si_date = si_avail.max()
        si_row = si_pivot.loc[si_date].dropna()
        fwd_row = pivot.loc[t].dropna()
        common = si_row.index.intersection(fwd_row.index)
        if len(common) < 5:
            continue
        si_vals = si_row[common].values.astype(float)
        fwd_vals = fwd_row[common].values.astype(float)
        # 跨截面 Spearman(避免常数方差)
        if np.std(si_vals) < 1e-9 or np.std(fwd_vals) < 1e-9:
            continue
        corr, p = spearmanr(si_vals, fwd_vals)
        if not np.isfinite(corr):
            continue
        rows.append({
            'asof_date': t,
            'horizon': int(horizon),
            'ic': float(corr),
            'p_value': float(p),
            'n_industries': int(len(common)),
        })
    return pd.DataFrame(rows, columns=cols)


def rolling_lagged_ic(
    daily_ic_df: pd.DataFrame,
    window: int = 60,
    step: int = 20,
) -> pd.DataFrame:
    """对每日 lagged IC 做滚动窗口平均。

    Args:
        daily_ic_df: compute_lagged_cross_sectional_ic 输出
        window: rolling window(默认 60 日)
        step: rolling step(默认 20 日)

    Returns:
        DataFrame with columns: window_end_date, horizon, ic, p_value, n_industries
        排序: (window_end_date ASC, horizon ASC)
    """
    cols = ['window_end_date', 'horizon', 'ic', 'p_value', 'n_industries']
    if daily_ic_df is None or daily_ic_df.empty:
        return pd.DataFrame(columns=cols)
    daily_ic_df = daily_ic_df.sort_values('asof_date').reset_index(drop=True)
    rows = []
    for h, g in daily_ic_df.groupby('horizon'):
        g = g.sort_values('asof_date').reset_index(drop=True)
        dates = g['asof_date'].values
        ics = g['ic'].values.astype(float)
        ps = g['p_value'].values.astype(float)
        ns = g['n_industries'].values.astype(float)
        n = len(g)
        # 滚动窗口(闭区间 [i-window+1, i],步长 step)
        for i in range(window - 1, n, step):
            window_ics = ics[i - window + 1: i + 1]
            window_ps = ps[i - window + 1: i + 1]
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
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows, columns=cols)
            .sort_values(['window_end_date', 'horizon'])
            .reset_index(drop=True))


def write_si_lagged_ic_summary(
    timeseries_df: pd.DataFrame,
    output_csv_path: str,
) -> tuple:
    """写跨期汇总 CSV,返回 (summary_df, text_str)。

    summary_df: 每 horizon 一行 × 6 列
        horizon, ic_mean, ic_std, ic_ir, p_value_mean, n_windows
    text_str: UTF-8 中文文本汇总(由 caller 写盘,避免路径耦合)
    """
    summary_cols = ['horizon', 'ic_mean', 'ic_std', 'ic_ir', 'p_value_mean', 'n_windows']
    if timeseries_df is None or timeseries_df.empty:
        timeseries_df = pd.DataFrame(columns=[
            'window_end_date', 'horizon', 'ic', 'p_value', 'n_industries',
        ])
        summary = pd.DataFrame(columns=summary_cols)
    else:
        summary = timeseries_df.groupby('horizon').agg(
            ic_mean=('ic', 'mean'),
            ic_std=('ic', 'std'),
            ic_ir=('ic', lambda s: s.mean() / s.std() if s.std() > 0 else 0.0),
            p_value_mean=('p_value', 'mean'),
            n_windows=('ic', 'count'),
        ).reset_index()
        summary['ic_std'] = summary['ic_std'].fillna(0.0)
        summary['ic_mean'] = summary['ic_mean'].round(4)
        summary['ic_std'] = summary['ic_std'].round(4)
        summary['ic_ir'] = summary['ic_ir'].round(4)
        summary['p_value_mean'] = summary['p_value_mean'].round(4)
        summary = summary[summary_cols]
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or '.', exist_ok=True)
    summary.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    # 文本汇总
    horizons_txt = (sorted(int(h) for h in timeseries_df['horizon'].unique())
                    if not timeseries_df.empty else '无')
    lines = [
        '=' * 70,
        'v4.10 时序 SI 的 lagged IC 评估',
        '=' * 70,
        f'窗口数: {len(timeseries_df)}',
        f'Horizons: {horizons_txt}',
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
    else:
        lines.append('(无 IC 窗口)')
    text = '\n'.join(lines) + '\n'
    return summary, text


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
    if timeseries_df is None:
        timeseries_df = pd.DataFrame()
    if summary_df is None:
        summary_df = pd.DataFrame()
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
            ), row=1, col=1)
    fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)
    fig.update_yaxes(range=[-0.5, 0.5], row=1, col=1, title_text='IC')
    # (1,2) v4.10 vs v4.8 对比
    v8_loaded = False
    if os.path.exists(v8_summary_path):
        try:
            v8 = pd.read_csv(v8_summary_path, encoding='utf-8-sig')
            if not v8.empty and not summary_df.empty:
                merged = summary_df.merge(v8, on='horizon', suffixes=('_v10', '_v8'))
                if not merged.empty:
                    fig.add_trace(go.Scatter(
                        x=merged['ic_mean_v8'], y=merged['ic_mean_v10'],
                        mode='markers+text',
                        text=merged['horizon'].astype(int).astype(str) + 'd',
                        textposition='top center',
                        marker=dict(size=14, color='indianred'),
                        name='horizon',
                    ), row=1, col=2)
                    # y=x 参考线
                    lim = max(
                        abs(float(merged['ic_mean_v10'].abs().max())),
                        abs(float(merged['ic_mean_v8'].abs().max())),
                        0.1,
                    )
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
    else:
        table_text = '(无 IC 窗口)'
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


def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(',') if str(h).strip()]
    # 1. 加载 SI 时序
    si_ts = load_sector_si_timeseries(args.si_timeseries)
    print(f'[lagged-ic] SI 时序: {len(si_ts)} 行 / '
          f'{si_ts["industry_l1"].nunique()} 行业 / '
          f'{si_ts["asof_date"].nunique()} 个 asof_date')
    # 限制行业(冒烟)
    if args.limit > 0:
        top_ind = (si_ts.groupby('industry_l1')['SI'].last()
                   .sort_values(ascending=False).head(args.limit).index)
        si_ts = si_ts[si_ts['industry_l1'].isin(top_ind)]
        print(f'[lagged-ic] --limit {args.limit} → {si_ts["industry_l1"].nunique()} 行业')
    # 2. 回查 industry membership(code → industry_l1)
    industry_lookup = load_industry_membership(args.kc_estimates, args.sw2_members)
    members_by_ind = {}
    for code, ind in industry_lookup.items():
        members_by_ind.setdefault(ind, []).append(code)
    keep = set(si_ts['industry_l1'].unique())
    members_by_ind = {ind: codes for ind, codes in members_by_ind.items() if ind in keep}
    print(f'[lagged-ic] 行业成员表: {len(members_by_ind)} 行业')
    # 3. eval_dates:SI 时序中所有 asof_date
    eval_dates = pd.DatetimeIndex(sorted(pd.to_datetime(si_ts['asof_date'].unique())))
    # 4. 对每个 horizon 算 daily lagged IC + rolling
    all_ts = []
    for h in horizons:
        fwd = compute_industry_forward_returns(members_by_ind, args.daily_dir, eval_dates, h)
        daily_ic = compute_lagged_cross_sectional_ic(si_ts, fwd, horizon=h)
        rolled = rolling_lagged_ic(daily_ic, window=args.window, step=args.step)
        print(f'[lagged-ic] horizon={h}d: {len(daily_ic)} 日 IC → {len(rolled)} 窗口')
        all_ts.append(rolled)
    timeseries = (pd.concat(all_ts, ignore_index=True) if all_ts
                  else pd.DataFrame(columns=[
                      'window_end_date', 'horizon', 'ic', 'p_value', 'n_industries']))
    # 5. 写出 summary + timeseries CSV
    summary, text = write_si_lagged_ic_summary(timeseries, args.ic_summary_output)
    os.makedirs(os.path.dirname(os.path.abspath(args.ic_timeseries_output)) or '.',
                exist_ok=True)
    timeseries.to_csv(args.ic_timeseries_output, index=False, encoding='utf-8-sig')
    print(f'[lagged-ic] 💾 {args.ic_summary_output} ({len(summary)} horizons)')
    print(f'[lagged-ic] 💾 {args.ic_timeseries_output} ({len(timeseries)} 窗口)')
    # 6. 写出文本汇总
    os.makedirs(os.path.dirname(os.path.abspath(args.txt_output)) or '.', exist_ok=True)
    with open(args.txt_output, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'[lagged-ic] 📝 {args.txt_output}')
    # 7. HTML
    build_si_lagged_ic_html(timeseries, summary, args.v8_summary, args.html_output)
    print(f'[lagged-ic] 🌐 {args.html_output}')


if __name__ == '__main__':
    main()
