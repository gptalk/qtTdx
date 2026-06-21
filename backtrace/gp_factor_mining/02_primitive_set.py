# -*- coding: utf-8 -*-
"""
02_primitive_set.py — GP 算子集(原始 + 时序 + 截面)

约定:
  - 终端(Terminal)= 已经预计算好的"标量特征列",每行(每只股票每天)一个数
  - 函数(Function)= 元素级算术算子,gplearn 在每行上逐元素应用
  - 时序特征(delay / ma / rsi)在 panel 上预计算,然后作为 Terminal 进入 GP

为什么时序特征不进 gplearn?
  gplearn 原生不支持 group/rolling 上下文;把时序特征预算好当 Terminal 最稳。
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


# ============================================================
# A. 时序原始特征(在长表上 groupby('code') 计算)
# ============================================================
def add_timeseries_primitives(panel: pd.DataFrame) -> pd.DataFrame:
    """
    给 panel 加时序原始特征:
      ma_N, std_N, min_N, max_N, delta_N, delay_N, rsi_N, macd, atr_N,
      amount, vwap_proxy(close*volume)
    """
    print("[02.A] 算时序原始特征...")
    p = panel.sort_values(['code', 'date']).reset_index(drop=True)
    g = p.groupby('code', group_keys=False)

    # ---- 均线/统计 ----
    for N in (5, 10, 20, 60):
        p[f'ma_{N}']      = g['Close'].transform(lambda s: s.rolling(N, min_periods=1).mean())
        p[f'std_{N}']     = g['Close'].transform(lambda s: s.rolling(N, min_periods=2).std())
        p[f'min_{N}']     = g['Low'].transform(  lambda s: s.rolling(N, min_periods=1).min())
        p[f'max_{N}']     = g['High'].transform( lambda s: s.rolling(N, min_periods=1).max())

    # ---- 价格位移 ----
    for N in (1, 5, 10, 20):
        p[f'delta_{N}']   = p['Close'] - g['Close'].shift(N)
        p[f'ret_{N}']     = p['Close'] / g['Close'].shift(N) - 1
        p[f'delay_{N}']   = g['Close'].shift(N)                # 滞后收盘价
        p[f'delay_v_{N}'] = g['Volume'].shift(N)              # 滞后成交量

    # ---- 均线比(经典 alpha)----
    p['ma_ratio_5_20']  = p['ma_5']  / (p['ma_20'] + 1e-12)
    p['ma_ratio_10_60'] = p['ma_10'] / (p['ma_60'] + 1e-12)

    # ---- RSI(14) ----
    p['rsi_14'] = _rsi(p['Close'], 14)

    # ---- MACD ----
    ema12 = g['Close'].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = g['Close'].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    p['macd_dif'] = ema12 - ema26
    p['macd_dea'] = g['macd_dif'].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    p['macd_bar'] = 2 * (p['macd_dif'] - p['macd_dea'])

    # ---- ATR(14) ----
    p['atr_14'] = _atr(p, 14)

    # ---- 布林带 ----
    p['bbi']     = (p['ma_5'] + p['ma_10'] + p['ma_20'] + p['ma_60']) / 4
    p['bb_upper']= p['ma_20'] + 2 * p['std_20']
    p['bb_lower']= p['ma_20'] - 2 * p['std_20']
    p['bb_width']= (p['bb_upper'] - p['bb_lower']) / (p['ma_20'] + 1e-12)
    p['bb_pos']  = (p['Close'] - p['bb_lower']) / (p['bb_upper'] - p['bb_lower'] + 1e-12)

    # ---- 量能 ----
    p['vwap_proxy']  = p['Close'] * p['Volume']
    for N in (5, 20):
        p[f'vol_ma_{N}']   = g['Volume'].transform(lambda s: s.rolling(N, min_periods=1).mean())
        p[f'vol_ratio_{N}']= p['Volume'] / (p[f'vol_ma_{N}'] + 1e-12)

    # ---- 动量 ----
    p['mom_20'] = p['Close'] / (g['Close'].shift(20) + 1e-12) - 1

    # ---- K线形态 ----
    p['hl_ratio'] = (p['High'] - p['Low']) / (p['Close'] + 1e-12)
    p['oc_ratio'] = (p['Close'] - p['Open']) / (p['Open'] + 1e-12)

    # ---- 替换 inf ----
    p = p.replace([np.inf, -np.inf], np.nan)
    print(f"  → 新增 {p.shape[1] - panel.shape[1]} 个原始特征列")
    return p


def _rsi(close: pd.Series, N: int = 14) -> pd.Series:
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    # 用 EMA 计算更稳
    roll_up   = up.ewm(span=N, adjust=False).mean()
    roll_down = down.ewm(span=N, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - 100 / (1 + rs)


def _atr(panel: pd.DataFrame, N: int = 14) -> pd.Series:
    high  = panel['High']
    low   = panel['Low']
    close = panel['Close']
    prev_close = panel.groupby('code')['Close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.groupby(panel['code']).transform(lambda s: s.ewm(span=N, adjust=False).mean())


# ============================================================
# B. 截面/全市场特征(对每日横截面计算)
# ============================================================
def add_crosssection_primitives(panel: pd.DataFrame) -> pd.DataFrame:
    """
    给 panel 加截面特征:
      - 行业内 rank(需要行业;若无就用全市场 rank)
      - 与大盘的相关系数(rolling)
      - 截面分位
    """
    print("[02.B] 算截面特征...")
    p = panel.sort_values(['date', 'code']).reset_index(drop=True)

    # ---- 全市场 rank(已在 01_data_prep 里算过 cs_*;这里补几个比值截面特征)----
    if 'cs_Close' not in p.columns:
        p['cs_Close'] = p.groupby('date')['Close'].rank(pct=True)
    if 'cs_Volume' not in p.columns:
        p['cs_Volume'] = p.groupby('date')['Volume'].rank(pct=True)

    # ---- 收益率截面分位 ----
    for c in ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'mom_20']:
        if c in p.columns:
            p[f'cs_{c}'] = p.groupby('date')[c].rank(pct=True)

    # ---- 与自身 ma 的偏离(截面去量纲版)----
    if 'ma_20' in p.columns:
        p['dev_ma20'] = (p['Close'] - p['ma_20']) / (p['ma_20'] + 1e-12)
        p['cs_dev_ma20'] = p.groupby('date')['dev_ma20'].rank(pct=True)

    # ---- 量比截面 ----
    if 'vol_ratio_5' in p.columns:
        p['cs_vol_ratio_5'] = p.groupby('date')['vol_ratio_5'].rank(pct=True)

    print(f"  → 截面特征就绪")
    return p


# ============================================================
# C. 整理:把 panel 拆成 (X, y) 给 GP 用
# ============================================================
def build_xy(panel: pd.DataFrame, label: str):
    """
    输入:长表 panel
    输出:
      X      : DataFrame(每行一个样本,每列一个 Terminal)
      y      : Series(对应行的标签)
      meta   : DataFrame(date, code)  方便回溯
      feat_cols: 实际用到的特征列名
    """
    print(f"\n[02.C] 切 X/y(标签={label})...")

    # ---- 用 cs_* 列作为输入(已截面标准化,无市值/价格水平污染)----
    base_cols = [c for c in panel.columns if c.startswith('cs_') and c != f'cs_{label}']
    # 也可加上原始 ret_/mom_ 的截面分位
    extra_cols = [c for c in panel.columns if c.startswith('cs_') and c not in base_cols]

    # 拼接所有可用的截面特征 + 几个稳健的原始 rank 化列
    feat_cols = sorted(set(base_cols))

    # 去掉标签列
    feat_cols = [c for c in feat_cols if c != f'cs_{label}']

    # 过滤全 NaN 列
    valid = [c for c in feat_cols if panel[c].notna().sum() > 100]
    feat_cols = valid

    print(f"  候选 Terminal 列:{len(feat_cols)} 个")
    print(f"  示例:{feat_cols[:8]} ...")

    # ---- 丢掉缺失标签的行 ----
    keep = panel[label].notna()
    sub  = panel.loc[keep].copy()
    print(f"  有效样本(标签非空):{len(sub):,} 行")

    # ---- 再丢掉特征全 NaN 的行(每行)----
    row_valid = sub[feat_cols].notna().any(axis=1)
    sub = sub.loc[row_valid]
    print(f"  特征非空样本:{len(sub):,} 行")

    X    = sub[feat_cols].astype(np.float32)
    y    = sub[label].astype(np.float32)
    meta = sub[['date', 'code']].reset_index(drop=True)
    return X, y, meta, feat_cols


# ============================================================
# D. gplearn Function Set(纯算术 + 元素级)
# ============================================================
def make_function_set():
    """
    gplearn 0.4.3 要求 function_set 是 Function 对象列表(不是元组)。
    用 gplearn.functions.make_function 包一层;提供受保护的算子(防 div-by-0 等)。
    """
    from gplearn.functions import make_function

    def _protected_div(x1, x2):
        return np.where(np.abs(x2) > 1e-6, x1 / x2, 1.0)

    def _protected_log(x):
        return np.log(np.abs(x) + 1e-6)

    def _protected_sqrt(x):
        return np.sqrt(np.abs(x))

    def _safe_max(x1, x2):
        return np.where(x1 >= x2, x1, x2)

    def _safe_min(x1, x2):
        return np.where(x1 <= x2, x1, x2)

    def _sign(x):
        return np.sign(x)

    funcs = [
        make_function(function=np.add,            name='add',  arity=2),
        make_function(function=np.subtract,       name='sub',  arity=2),
        make_function(function=np.multiply,       name='mul',  arity=2),
        make_function(function=_protected_div,    name='div',  arity=2),
        make_function(function=np.abs,            name='abs',  arity=1),
        make_function(function=np.negative,       name='neg',  arity=1),
        make_function(function=_protected_sqrt,   name='sqrt', arity=1),
        make_function(function=_protected_log,    name='log',  arity=1),
        make_function(function=_safe_max,         name='max',  arity=2),
        make_function(function=_safe_min,         name='min',  arity=2),
        make_function(function=_sign,             name='sign', arity=1),
    ]
    return funcs


# ============================================================
# E. 独立运行:试算一下
# ============================================================
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    cfg = import_module('00_config')

    panel_path = cfg.DATA_DIR / "panel.parquet"
    if not panel_path.exists():
        sys.exit(f"[FATAL] 找不到 {panel_path},请先跑 01_data_prep.py")

    panel = pd.read_parquet(panel_path)
    print(f"载入 panel: {panel.shape}, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}, "
          f"{panel['code'].nunique()} 只\n")

    panel = add_timeseries_primitives(panel)
    panel = add_crosssection_primitives(panel)

    X, y, meta, feat_cols = build_xy(panel, cfg.LABEL_NAME)
    print(f"\nX.shape={X.shape}, y.shape={y.shape}")
    print(f"y 分布:mean={y.mean():.4f}, std={y.std():.4f}, "
          f"min={y.min():.4f}, max={y.max():.4f}")