# -*- coding: utf-8 -*-
"""
04_ic_metrics.py — 因子评估指标

对长表 (date, code, factor, label) 计算:
  - RankIC 每日 Spearman 相关
  - RankICIR = mean / std
  - RankIC t-stat, p-value, 正占比
  - IC 衰减(多持有期平均 IC)
  - 分组年化收益(十分位 vs 多空)
  - 月度换手率(相邻月因子相关性)
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# A. 单因子 RankIC 时序
# ============================================================
def daily_rankic(df: pd.DataFrame, factor_col: str, label_col: str) -> pd.Series:
    """
    每日横截面 Spearman 相关 → 时序 Series
    """
    def _one(d):
        sub = d[[factor_col, label_col]].dropna()
        if len(sub) < 20:
            return np.nan
        rho, _ = spearmanr(sub[factor_col], sub[label_col])
        return rho
    return df.groupby('date').apply(_one)


def ic_summary(ic_ts: pd.Series) -> dict:
    """IC 时序 → 摘要字典"""
    ic_ts = ic_ts.dropna()
    if len(ic_ts) == 0:
        return {'n_days': 0}
    return {
        'n_days':   len(ic_ts),
        'ic_mean':  ic_ts.mean(),
        'ic_std':   ic_ts.std(),
        'icir':     ic_ts.mean() / ic_ts.std() if ic_ts.std() > 0 else np.nan,
        'ic_t':     ic_ts.mean() / (ic_ts.std() / np.sqrt(len(ic_ts))) if ic_ts.std() > 0 else np.nan,
        'ic_pos':   (ic_ts > 0).mean(),
        'ic_abs_mean': ic_ts.abs().mean(),
    }


# ============================================================
# B. IC 衰减(对不同持有期)
# ============================================================
def ic_decay(df: pd.DataFrame, factor_col: str,
             label_col: str, horizons=(1, 5, 10, 20, 40, 60)) -> pd.DataFrame:
    """
    对每个 horizon,计算 factor 对未来 N 日收益的 RankIC 均值。
    """
    rows = []
    g = df.sort_values(['code', 'date']).groupby('code', group_keys=False)
    for h in horizons:
        # 用 shift(-h) 构造未来收益
        fwd = g['Close'].shift(-h) / df['Close'] - 1
        sub = df[[factor_col]].assign(_fwd=fwd).dropna()
        if len(sub) == 0:
            continue
        ic = sub.groupby('date').apply(
            lambda d: spearmanr(d[factor_col], d['_fwd'])[0] if len(d) > 20 else np.nan
        ).mean()
        rows.append({'horizon': h, 'rank_ic': ic})
    return pd.DataFrame(rows)


# ============================================================
# C. 分组(十分位)收益
# ============================================================
def quantile_returns(df: pd.DataFrame, factor_col: str, label_col: str,
                     n_quantiles: int = 10) -> pd.DataFrame:
    """
    每日横截面分 n 组,算每组平均未来收益;输出:
        date, q10, q9, ..., q1, long_short(q1-q10)
    """
    def _one(d):
        sub = d[[factor_col, label_col]].dropna()
        if len(sub) < n_quantiles * 5:
            return None
        sub['q'] = pd.qcut(sub[factor_col], q=n_quantiles, labels=False, duplicates='drop') + 1
        return sub.groupby('q')[label_col].mean()

    daily = df.groupby('date').apply(_one).dropna()
    if daily.empty:
        return daily
    # 多空 = 第 1 组 - 第 10 组(因子越大越好)
    daily['long_short'] = daily[n_quantiles] - daily[1]
    return daily


def quantile_summary(qret: pd.DataFrame, periods_per_year: int = 250) -> pd.DataFrame:
    """把分组收益表转成年化摘要"""
    rows = []
    for col in qret.columns:
        v = qret[col].dropna()
        if len(v) == 0:
            continue
        rows.append({
            'group':      col,
            'mean':       v.mean(),
            'std':        v.std(),
            'ann_ret':    v.mean() * periods_per_year,
            'ann_vol':    v.std()  * np.sqrt(periods_per_year),
            'sharpe':     v.mean() / v.std() * np.sqrt(periods_per_year) if v.std() > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================================
# D. 月度换手率(用月度因子截面相关性)
# ============================================================
def monthly_turnover(df: pd.DataFrame, factor_col: str) -> float:
    """
    把每日 rank 化后的因子按月聚合(均值),算相邻月 Spearman。
    换手率 = 1 - 相邻月 Spearman。
    """
    sub = df[['date', 'code', factor_col]].dropna().copy()
    sub['rank'] = sub.groupby('date')[factor_col].rank(method='first')
    monthly = sub.set_index('date').groupby([pd.Grouper(freq='M'), 'code'])['rank'].mean().unstack()
    # 相邻月相关
    corrs = []
    for i in range(1, len(monthly)):
        common = monthly.iloc[i].dropna().index.intersection(monthly.iloc[i-1].dropna().index)
        if len(common) < 30:
            continue
        c, _ = spearmanr(monthly.iloc[i][common], monthly.iloc[i-1][common])
        corrs.append(c)
    if not corrs:
        return np.nan
    return float(1 - np.mean(corrs))


# ============================================================
# E. 一键评估:输入 (panel, factor_col, label_col) → 摘要 DataFrame
# ============================================================
def full_evaluate(panel: pd.DataFrame, factor_col: str,
                  label_col: str, name: str = 'factor') -> pd.DataFrame:
    """对一个因子跑全套评估,返回一行摘要"""
    sub = panel[[factor_col, label_col, 'date', 'code']].dropna(subset=[factor_col, label_col])
    if len(sub) < 1000:
        return pd.DataFrame([{'name': name, 'note': '样本不足'}])

    ic_ts = daily_rankic(sub, factor_col, label_col)
    summ  = ic_summary(ic_ts)
    decay = ic_decay(sub, factor_col, label_col)
    qret  = quantile_returns(sub, factor_col, label_col, n_quantiles=10)
    qsum  = quantile_summary(qret) if not qret.empty else pd.DataFrame()
    turnover = monthly_turnover(sub, factor_col)

    out = {
        'name':         name,
        'n_obs':        len(sub),
        'ic_mean':      summ.get('ic_mean'),
        'ic_std':       summ.get('ic_std'),
        'icir':         summ.get('icir'),
        'ic_t':         summ.get('ic_t'),
        'ic_pos':       summ.get('ic_pos'),
        'ic_abs_mean':  summ.get('ic_abs_mean'),
        'turnover':     turnover,
        'ic_decay_20d': float(decay[decay['horizon']==20]['rank_ic'].iloc[0]) if (decay['horizon']==20).any() else np.nan,
        'long_short_sharpe': float(qsum[qsum['group']=='long_short']['sharpe'].iloc[0]) if 'long_short' in qsum['group'].values else np.nan,
    }
    return pd.DataFrame([out])


# ============================================================
# F. 独立运行
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from importlib import import_module
    cfg = import_module('00_config')

    panel_path = cfg.DATA_DIR / "panel.parquet"
    if not panel_path.exists():
        sys.exit(f"[FATAL] 找不到 {panel_path}")
    panel = pd.read_parquet(panel_path)

    # 拿标签当"假因子"演示
    res = full_evaluate(panel, cfg.LABEL_NAME, cfg.LABEL_NAME, name='label_itself')
    print(res.T)