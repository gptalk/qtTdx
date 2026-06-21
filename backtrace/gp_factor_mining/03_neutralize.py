# -*- coding: utf-8 -*-
"""
03_neutralize.py — 截面中性化(行业 + 市值)

A 股 alpha 因子挖掘必须做的事:
  - 剔除规模因子(小盘股天然有溢价,容易被 GP 抓成虚假 alpha)
  - 剔除行业因子(同一行业内相关性高,容易被 GP 抓出来)

做法:每日横截面 OLS
    factor ~ 1 + log(size) + C(industry)
    residual = factor - predicted

输入 panel 必须包含:
    date, code, factor
    size_col  : 规模代理(如 Amount 或 log_market_cap)
    ind_col   : 行业标签(str 或 int);若没有,用 size 分组近似
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ============================================================
# A. 行业 / 市值代理(若 panel 里没有)
# ============================================================
def add_size_proxy(panel: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    用过去 20 日均成交额(Amount 的 rolling mean)做规模代理。
    比 log_market_cap 粗糙,但 TDX 数据里直接有 Amount,够用。
    """
    p = panel.sort_values(['code', 'date']).copy()
    g = p.groupby('code', group_keys=False)
    p['size_proxy'] = g['Amount'].transform(lambda s: s.rolling(lookback, min_periods=5).mean())
    p['log_size']   = np.log1p(p['size_proxy'].fillna(0))
    return p


def add_industry_proxy(panel: pd.DataFrame, n_clusters: int = 20) -> pd.DataFrame:
    """
    没有真实行业数据时,按"过去 20 日收益率分布"做简易聚类分组,作为行业代理。
    真有 SW/ZX 行业分类时,直接读 panel['industry'] 即可,绕过此函数。
    """
    p = panel.sort_values(['code', 'date']).copy()
    g = p.groupby('code', group_keys=False)

    # 用过去 20 日的 ret / volume 截面 rank 当聚类特征
    p['ret_20']   = g['Close'].transform(lambda s: s.pct_change(20))
    p['vol_20']   = g['Close'].transform(lambda s: s.pct_change().rolling(20).std())
    feat = p.groupby('code')[['ret_20', 'vol_20']].transform('mean')

    # 简单分箱(快,不依赖 sklearn.cluster)
    p['ind_proxy'] = (
        pd.qcut(feat['ret_20'].rank(method='first'), q=n_clusters, labels=False, duplicates='drop').fillna(0).astype(int)
    ).astype(str)
    return p


# ============================================================
# B. 单日截面中性化
# ============================================================
def _neutralize_one_day(df_day: pd.DataFrame, factor_col: str,
                        size_col: str, ind_col: str) -> pd.Series:
    """对一日横截面做 OLS 中性化,返回残差 Series(索引与 df_day 对齐)"""
    y = df_day[factor_col].values
    if np.isnan(y).all():
        return pd.Series(np.nan, index=df_day.index)

    # 构造设计矩阵:截距 + log(size) + 行业 one-hot
    X_parts = [np.ones((len(df_day), 1))]
    if size_col in df_day.columns:
        log_size = np.log1p(df_day[size_col].fillna(0).values).reshape(-1, 1)
        X_parts.append(log_size)
    if ind_col in df_day.columns:
        dummies = pd.get_dummies(df_day[ind_col].astype(str), drop_first=True, dummy_na=True)
        X_parts.append(dummies.values)

    X_mat = np.hstack(X_parts)

    # 丢掉 y 为 nan 的行,fit 后再回填
    valid = ~np.isnan(y)
    if valid.sum() < X_mat.shape[1] + 5:
        return pd.Series(np.nan, index=df_day.index)

    model = LinearRegression().fit(X_mat[valid], y[valid])
    pred  = model.predict(X_mat)
    resid = y - pred

    out = pd.Series(np.nan, index=df_day.index)
    out.loc[df_day.index[valid]] = resid[valid]
    return out


# ============================================================
# C. 批量:对整个 panel 做截面中性化
# ============================================================
def neutralize(panel: pd.DataFrame, factor_col: str,
               size_col: str = 'size_proxy',
               ind_col:  str = 'ind_proxy') -> pd.DataFrame:
    """
    给 panel 加一列 '<factor_col>_neu' = 因子残差(已中性化)。
    原 factor 列保留,方便对比。
    """
    print(f"[03] 中性化:{factor_col}  on  {size_col} + {ind_col}")
    p = panel.copy()

    # 确保 size / industry 列存在
    if size_col not in p.columns:
        p = add_size_proxy(p)
    if ind_col not in p.columns:
        p = add_industry_proxy(p)

    # 按日分组做 OLS,concat 残差
    parts = []
    for d, sub in p.groupby('date'):
        r = _neutralize_one_day(sub, factor_col, size_col, ind_col)
        parts.append(r)

    p[f'{factor_col}_neu'] = pd.concat(parts).reindex(p.index)
    print(f"  → 新增列:{factor_col}_neu  "
          f"(非空 {p[f'{factor_col}_neu'].notna().sum():,} / {len(p):,})")
    return p


# ============================================================
# D. 独立运行
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    cfg = import_module('00_config')

    panel_path = cfg.DATA_DIR / "panel.parquet"
    if not panel_path.exists():
        sys.exit(f"[FATAL] 找不到 {panel_path},先跑 01_data_prep.py")

    panel = pd.read_parquet(panel_path)
    panel = add_size_proxy(panel)
    panel = add_industry_proxy(panel)
    print(f"size_proxy 非空 {panel['size_proxy'].notna().sum():,}")
    print(f"ind_proxy 分布:\n{panel['ind_proxy'].value_counts().head()}")

    panel = neutralize(panel, factor_col=cfg.LABEL_NAME)
    print(panel[[cfg.LABEL_NAME, f'{cfg.LABEL_NAME}_neu']].describe())