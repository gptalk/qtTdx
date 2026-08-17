# -*- coding: utf-8 -*-
# dynamics_si_ic.py — v4.8 SI 与 forward return 的 IC 评估
#
# 目标:
#   读 v4.7 sector_si.csv + kc_estimates.csv + data/daily/<code>.csv
#   算各行业 forward 20d/60d 收益(中位数收盘价法)
#   滚动 60 日窗口,步长 20 日,跨截面 Spearman(SI 排名 vs forward return 排名)
#   输出 si_ic_summary.csv / si_ic_timeseries.csv / dynsys_si_ic.html / 文本汇总
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_ic.py
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_si_ic.py --window 30 --step 10
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 让 import 找到 dynamics 兄弟模块
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

CSV_OUT_DIR = 'data/dynamics'
DEFAULT_SI_CSV = os.path.join(CSV_OUT_DIR, 'sector_si.csv')
DEFAULT_KC_CSV = 'data/projection/kc_estimates.csv'
DEFAULT_SW2_CSV = 'data/sw2/members.csv'
DEFAULT_DAILY_DIR = 'data/daily'
DEFAULT_OUTPUT_HTML = 'backtrace/outputs/dynsys_si_ic.html'
DEFAULT_TXT_OUTPUT = 'backtrace/outputs/dynsys_si_ic_summary.txt'
DEFAULT_WINDOW = 60
DEFAULT_STEP = 20
DEFAULT_HORIZONS = (20, 60)


def parse_args():
    p = argparse.ArgumentParser(description='SI × forward return 滚动 Spearman IC 评估 (v4.8)')
    p.add_argument('--si-csv', default=DEFAULT_SI_CSV)
    p.add_argument('--kc-csv', default=DEFAULT_KC_CSV)
    p.add_argument('--sw2-csv', default=DEFAULT_SW2_CSV)
    p.add_argument('--daily-dir', default=DEFAULT_DAILY_DIR)
    p.add_argument('--output', default=DEFAULT_OUTPUT_HTML)
    p.add_argument('--window', type=int, default=DEFAULT_WINDOW)
    p.add_argument('--step', type=int, default=DEFAULT_STEP)
    p.add_argument('--horizons', type=str, default='20,60',
                   help='forward horizon (days), comma-separated, e.g. "20,60"')
    return p.parse_args()


def load_sector_si(path: str) -> pd.DataFrame:
    """读 sector_si.csv,返回至少含 industry_l1, SI 两列。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'sector_si.csv 不存在: {path};先跑 v4.7 产 SI')
    df = pd.read_csv(path, dtype={'industry_l1': str})
    return df[['industry_l1', 'SI']].copy()


def load_industry_membership(kc_path: str, sw2_path: str) -> dict[str, list[str]]:
    """回查 industry_l1 → [code, ...] 成员表。

    用 kc_estimates.csv 取 code → industry_l1 映射,再用 sw2/members 兜底(只取 sector_code)。"""
    if not os.path.exists(kc_path):
        raise FileNotFoundError(f'kc_estimates.csv 不存在: {kc_path}')
    kc = pd.read_csv(kc_path, dtype={'code': str})
    if 'industry_l1' not in kc.columns:
        # 走与 dynamics_eigen_analysis 相同的反查路径
        sw2 = pd.read_csv(sw2_path, dtype={'member_code': str, 'sector_code': str})
        sw2 = sw2[['member_code', 'sector_code']].drop_duplicates('member_code')
        sw2 = sw2.rename(columns={'member_code': 'code', 'sector_code': 'industry_l1'})
        kc = kc.merge(sw2, on='code', how='left')
    kc = kc[kc['industry_l1'].notna() & (kc['industry_l1'] != '')]
    membership = kc.groupby('industry_l1')['code'].apply(list).to_dict()
    return membership


def compute_industry_forward_returns(
    membership: dict, daily_dir: str, dates: pd.DatetimeIndex, horizon: int,
) -> pd.DataFrame:
    """算各行业 forward horizon 日收益(成员中位数收盘价)。

    Returns:
        [date × industry_l1] 矩阵,值 = (median_close(t+h) - median_close(t)) / median_close(t)
    """
    # 收集所有需要的 code
    all_codes = set()
    for codes in membership.values():
        all_codes.update(codes)
    # 读 daily CSV, 拼成 [date × code] close 矩阵
    close_by_code = {}
    for code in all_codes:
        path = os.path.join(daily_dir, f'{code}.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={'code': str})
        if 'date' not in df.columns or 'close' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')[['close']].sort_index()
        close_by_code[code] = df['close']
    close_df = pd.DataFrame(close_by_code).sort_index()
    # 各行业:成员中位数 close 在 dates 当日
    rows = {}
    for ind, codes in membership.items():
        ind_codes = [c for c in codes if c in close_df.columns]
        if len(ind_codes) < 3:
            continue
        median_close = close_df[ind_codes].median(axis=1)
        # 限制到 dates
        median_close = median_close.reindex(dates)
        fwd = median_close.shift(-horizon) / median_close - 1.0
        rows[ind] = fwd
    fwd_df = pd.DataFrame(rows)
    return fwd_df.dropna(how='all')


def rolling_cross_sectional_ic(
    si: dict, fwd_ret: pd.DataFrame, window: int = 60, step: int = 20,
) -> pd.DataFrame:
    """滚动 cross-sectional Spearman IC。

    每窗口 IC = 窗口内逐日 IC 的算术平均(spec §3.3:不 pool 重算,避免重复计权重)。

    Returns:
        per-window DataFrame: window_end_date, horizon, ic, p_value, n_industries
        (注:horizon 列在这里默认 0;main 里循环 horizons 赋值)
    """
    si_vec = pd.Series(si)
    fwd_aligned = fwd_ret[si_vec.index]  # 行业顺序与 SI 一致
    rows = []
    n_days = len(fwd_aligned)
    if n_days < window:
        return pd.DataFrame(columns=['window_end_date', 'horizon', 'ic', 'p_value', 'n_industries'])
    for end in range(window - 1, n_days, step):
        start = end - window + 1
        win = fwd_aligned.iloc[start:end + 1]
        # 逐日 IC, 取有效行业数 ≥ 5
        daily_ics = []
        n_per_day = []
        for _, row in win.iterrows():
            valid = row.dropna()
            if len(valid) < 5:
                continue
            ic, p = spearmanr(si_vec.loc[valid.index], valid.values)
            if not np.isnan(ic):
                daily_ics.append((ic, p))
                n_per_day.append(len(valid))
        if not daily_ics:
            continue
        avg_ic = float(np.mean([x[0] for x in daily_ics]))
        avg_p = float(np.mean([x[1] for x in daily_ics]))
        n_industries = int(round(float(np.mean(n_per_day)))) if n_per_day else 0
        rows.append({
            'window_end_date': fwd_aligned.index[end],
            'horizon': 0,  # 由 main 后续填
            'ic': avg_ic, 'p_value': avg_p,
            'n_industries': n_industries,
        })
    return pd.DataFrame(rows)


def write_si_ic_summary(ts_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """写跨期汇总 2 行(20d / 60d),返回汇总 DataFrame。"""
    if ts_df.empty:
        Path(output_path).write_text('horizon,ic_mean,ic_std,ic_ir,p_value_mean,n_windows\n', encoding='utf-8')
        return pd.DataFrame()
    summary_rows = []
    for h, grp in ts_df.groupby('horizon'):
        ic_mean = float(grp['ic'].mean())
        ic_std = float(grp['ic'].std(ddof=1)) if len(grp) > 1 else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
        summary_rows.append({
            'horizon': h, 'ic_mean': ic_mean, 'ic_std': ic_std,
            'ic_ir': ic_ir, 'p_value_mean': float(grp['p_value'].mean()),
            'n_windows': len(grp),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    return summary


def build_si_ic_html(
    ts_df: pd.DataFrame, summary_df: pd.DataFrame, si_dict: dict,
    fwd_ret_60d: pd.DataFrame, output_path: str,
) -> None:
    """1 个 HTML,3 子图(spec §5 布局):
    (1,1) Rolling IC(20d / 60d)时序 + IC=0 红虚线
    (1,2) 行业 SI vs 累计 forward 60d 收益 散点
    (2,1, 全宽) IC 跨期统计表
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rolling IC(20d / 60d)',
            '行业 SI vs 累计 forward 60d 收益',
            'IC 跨期统计',
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'table', 'colspan': 2}, None],
        ],
        row_heights=[0.6, 0.4],
        column_widths=[0.5, 0.5],
    )
    # (1,1) rolling IC 时序
    if ts_df.empty:
        fig.add_annotation(text='无 IC 窗口', xref='paper', yref='paper', x=0.25, y=0.75, showarrow=False)
    else:
        for h, grp in ts_df.groupby('horizon'):
            fig.add_trace(
                go.Scatter(
                    x=grp['window_end_date'], y=grp['ic'],
                    mode='lines+markers', name=f'horizon={h}d',
                ),
                row=1, col=1,
            )
        fig.add_hline(y=0, line_dash='dash', line_color='red', row=1, col=1)
    fig.update_yaxes(title_text='IC', row=1, col=1)
    # (1,2) SI vs 累计 forward 60d 散点
    if si_dict and not fwd_ret_60d.empty:
        cum_ret = (1.0 + fwd_ret_60d).prod() - 1.0
        si_x, ret_y, names = [], [], []
        for ind, si_val in si_dict.items():
            if ind in cum_ret.index and not np.isnan(cum_ret[ind]):
                si_x.append(si_val)
                ret_y.append(cum_ret[ind])
                names.append(ind)
        if si_x:
            fig.add_trace(
                go.Scatter(
                    x=si_x, y=ret_y, mode='markers',
                    text=names, name='行业',
                    hovertemplate='行业=%{text}<br>SI=%{x:.3f}<br>累计60d=%{y:.2%}<extra></extra>',
                ),
                row=1, col=2,
            )
            fig.add_hline(y=0, line_dash='dash', line_color='grey', row=1, col=2)
    fig.update_xaxes(title_text='SI', row=1, col=2)
    fig.update_yaxes(title_text='累计 forward 60d 收益', row=1, col=2)
    # (2,1 全宽) 统计表
    if not summary_df.empty:
        fig.add_trace(
            go.Table(
                header=dict(values=list(summary_df.columns), fill_color='lightblue'),
                cells=dict(values=[summary_df[c] for c in summary_df.columns]),
            ),
            row=2, col=1,
        )
    fig.update_layout(height=900, width=1200, title_text='SI × Forward Return Rolling IC (v4.8)')
    fig.write_html(output_path, include_plotlyjs='cdn')


def main():
    args = parse_args()
    si_df = load_sector_si(args.si_csv)
    si_dict = dict(zip(si_df['industry_l1'], si_df['SI']))
    print(f'[si-ic] 加载 {len(si_dict)} 行业 SI')
    membership = load_industry_membership(args.kc_csv, args.sw2_csv)
    print(f'[si-ic] 行业成员表:{len(membership)} 行业')
    # 选 daily_dir 内的所有日期并集作为 dates
    sample_code = next(iter(next(iter(membership.values()))))
    sample_path = os.path.join(args.daily_dir, f'{sample_code}.csv')
    sample = pd.read_csv(sample_path)
    dates = pd.to_datetime(sample['date']).sort_values().reset_index(drop=True)
    print(f'[si-ic] daily 范围: {dates.min().date()} 至 {dates.max().date()},共 {len(dates)} 日')
    # 算各 horizon 滚动 IC
    horizons = tuple(int(x) for x in args.horizons.split(','))
    all_ts = []
    fwd_ret_60d = pd.DataFrame()  # 留作散点图用(spec §5 (1,2))
    for h in horizons:
        fwd = compute_industry_forward_returns(membership, args.daily_dir, dates, horizon=h)
        if h == 60:
            fwd_ret_60d = fwd
        ts = rolling_cross_sectional_ic(si_dict, fwd, window=args.window, step=args.step)
        ts['horizon'] = h
        all_ts.append(ts)
        print(f'[si-ic] horizon={h}d: {len(ts)} 窗口')
    ts_df = pd.concat(all_ts, ignore_index=True) if all_ts else pd.DataFrame()
    # 写 CSV
    ts_csv = os.path.join(CSV_OUT_DIR, 'si_ic_timeseries.csv')
    ts_df.to_csv(ts_csv, index=False, encoding='utf-8-sig')
    print(f'[si-ic] 💾 {ts_csv} ({len(ts_df)} 窗口)')
    # 写汇总(spec §4.2: data/dynamics/si_ic_summary.csv)
    summary_csv = os.path.join(CSV_OUT_DIR, 'si_ic_summary.csv')
    summary = write_si_ic_summary(ts_df, summary_csv)
    print(f'[si-ic] 💾 {summary_csv}')
    print(f'[si-ic] 汇总:{summary.to_dict("records")}')
    # HTML
    build_si_ic_html(ts_df, summary, si_dict, fwd_ret_60d, args.output)
    print(f'[si-ic] 🌐 {args.output}')
    # 文本汇总(spec §4.2: backtrace/outputs/dynsys_si_ic_summary.txt)
    txt_path = DEFAULT_TXT_OUTPUT
    lines = ['SI × Forward Return 滚动 IC 汇总 (v4.8)', '=' * 48, '']
    if not summary.empty:
        for _, r in summary.iterrows():
            lines.append(
                f"horizon={int(r['horizon'])}d  ic_mean={r['ic_mean']:+.4f}  "
                f"ic_std={r['ic_std']:.4f}  ic_ir={r['ic_ir']:+.4f}  "
                f"p_value_mean={r['p_value_mean']:.4f}  n_windows={int(r['n_windows'])}"
            )
    else:
        lines.append('(无 IC 窗口)')
    Path(txt_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[si-ic] 📝 {txt_path}')


if __name__ == '__main__':
    main()
