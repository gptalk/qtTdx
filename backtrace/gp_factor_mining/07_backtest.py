# -*- coding: utf-8 -*-
"""
07_backtest.py — vectorbt 多因子选股回测

流程:
  1. 从 factor_pool/pool/ 读入选因子(test 段)
  2. IC 加权合成综合因子
  3. 截面 rank → top N(默认 50)等权
  4. 月末调仓 → vectorbt 回测
  5. 输出年化、夏普、最大回撤、换手
  6. 出 HTML 图(若 vbt 可用)

依赖:
  - vectorbt(`pip install vectorbt`)
  - 没有 vbt 时,本模块直接给出等权组合的"近似净值曲线"作为兜底
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
cfg     = import_module('00_config')
metrics = import_module('04_ic_metrics')

# vectorbt 可选
try:
    import vectorbt as vbt
    HAS_VBT = True
except Exception:
    HAS_VBT = False


# ============================================================
# A. 读因子库
# ============================================================
def load_pool():
    pool_path = cfg.FACTOR_DIR / "factor_pool.csv"
    if not pool_path.exists():
        sys.exit(f"[FATAL] 找不到 {pool_path},先跑 06_factor_pool.py")
    pool = pd.read_csv(pool_path)
    pool_dir = cfg.FACTOR_DIR / "pool"
    if not pool_dir.exists():
        sys.exit(f"[FATAL] {pool_dir} 不存在")
    return pool, pool_dir


# ============================================================
# B. 合成综合因子
# ============================================================
def combine_factors(pool: pd.DataFrame, pool_dir: Path, method: str = "ic_weighted"):
    """
    method:
      ic_weighted : 按样本外 IC 加权
      equal       : 等权
      zscore_avg  : 各自截面 zscore 后等权
    """
    test_dfs = []
    weights  = []

    for _, row in pool.iterrows():
        name = row['name']
        f = pool_dir / f"{name}_test.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f).rename(columns={'factor': name})
        test_dfs.append(df)
        # 用样本外 IC 的符号 × |IC| 做权重
        w = abs(row.get('test_ic', 0.0)) if pd.notna(row.get('test_ic')) else 0.0
        sign = np.sign(row.get('test_ic', 0.0)) if pd.notna(row.get('test_ic')) else 0.0
        weights.append(w * sign)

    if not test_dfs:
        sys.exit("[FATAL] 没有任何入选因子的 test parquet")

    base = test_dfs[0][['date', 'code']].copy()
    for df in test_dfs:
        base = base.merge(df, on=['date', 'code'], how='outer')

    factor_cols = [c for c in base.columns if c not in ('date', 'code')]

    if method == "ic_weighted":
        weights = np.asarray(weights)
        if weights.sum() == 0:
            method = "equal"
        else:
            weights = weights / weights.sum()
            base['composite'] = sum(w * base[c].fillna(0) for w, c in zip(weights, factor_cols))

    if method == "equal":
        base['composite'] = base[factor_cols].mean(axis=1)

    if method == "zscore_avg":
        for c in factor_cols:
            base[c] = base.groupby('date')[c].transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9)
            )
        base['composite'] = base[factor_cols].mean(axis=1)

    print(f"\n  因子合成:{method}")
    print(f"  入选因子:{factor_cols}")
    print(f"  IC 权重:{dict(zip(factor_cols, np.round(weights, 3)))}")

    return base[['date', 'code', 'composite']]


# ============================================================
# C. 截面 rank → top N → 月末调仓
# ============================================================
def monthly_topn(composite: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    每个月末取截面 rank top N,标记 1(持仓)/ 0(空仓)。
    返回 long DataFrame: index=date, columns=code, values=0/1
    """
    df = composite.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    # 每月最后一天作为调仓日
    last_days = df.groupby('month')['date'].max().reset_index()
    rebal_dates = sorted(last_days['date'].unique())

    # 每个调仓日:取 top N
    records = []
    for d in rebal_dates:
        sub = df[df['date'] == d].dropna(subset=['composite'])
        if len(sub) < top_n:
            continue
        top = sub.nlargest(top_n, 'composite')[['code']].copy()
        top['date'] = d
        top['weight'] = 1.0 / top_n
        records.append(top)

    weights = pd.concat(records, ignore_index=True)
    # pivot 成 (date × code) 矩阵
    pivot = weights.pivot(index='date', columns='code', values='weight').fillna(0.0)
    pivot = pivot.sort_index()
    print(f"  调仓次数:{len(pivot)},持仓列数:{pivot.shape[1]},月频")
    return pivot


# ============================================================
# D. 取收盘价宽表(给 vbt)
# ============================================================
def get_price_panel(codes: list, start: str, end: str):
    """从 01_data_prep 落盘的 panel 取收盘价;没有的话用 TQ 现拉"""
    p = cfg.DATA_DIR / "panel.parquet"
    if not p.exists():
        sys.exit(f"[FATAL] 找不到 {p}")
    panel = pd.read_parquet(p)
    sub = panel[(panel['date'] >= start) & (panel['date'] <= end) & panel['code'].isin(codes)]
    wide = sub.pivot(index='date', columns='code', values='Close').sort_index().ffill()
    return wide


# ============================================================
# E. 主流程
# ============================================================
def main():
    print("=" * 70)
    print("[07] 多因子选股回测")
    print("=" * 70)

    pool, pool_dir = load_pool()
    print(f"  因子库 {len(pool)} 条")

    composite = combine_factors(pool, pool_dir, method="ic_weighted")

    # 截面 rank IC(综合因子 vs label)
    panel = pd.read_parquet(cfg.DATA_DIR / "panel.parquet")
    eval_df = panel[['date', 'code', cfg.LABEL_NAME]].merge(composite, on=['date', 'code'])
    summ = metrics.full_evaluate(eval_df, 'composite', cfg.LABEL_NAME, name='composite')
    print("\n  综合因子 IC:")
    print(summ.T)

    # 月末 top N 持仓
    top_n = cfg.BACKTEST_TOP_N
    weights = monthly_topn(composite, top_n=top_n)
    codes = weights.columns.tolist()

    # 取价格
    price = get_price_panel(codes, cfg.TEST_START, cfg.TEST_END)
    # 对齐
    common = price.columns.intersection(weights.columns)
    price = price[common]
    weights = weights[common]

    # 持仓权重 → entries/exits(weight 0→1 买入,weight 1→0 卖出)
    entries = (weights > 0) & (weights.shift(1).fillna(0) == 0)
    exits   = (weights == 0) & (weights.shift(1).fillna(0) > 0)

    print(f"\n  价格矩阵:{price.shape}")
    print(f"  调仓矩阵:{weights.shape}")
    print(f"  买入信号:{entries.values.sum()},卖出信号:{exits.values.sum()}")

    # ---- vectorbt 回测 ----
    if HAS_VBT:
        print("\n  [vectorbt] 跑回测...")
        pf = vbt.Portfolio.from_signals(
            close    = price,
            entries  = entries,
            exits    = exits,
            init_cash= cfg.INIT_CASH,
            fees     = cfg.FEE_RATE,
            freq     = 'D',
            size_granularity=cfg.SIZE_GRANULARITY,
        )
        stats = pf.stats()
        print("\n=== 回测绩效 ===")
        print(stats)

        # 画净值图
        try:
            html_path = cfg.GP_DIR / f"gp_backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"
            pf.plot().write_html(str(html_path))
            print(f"\n  净值图已保存:{html_path}")
        except Exception as e:
            print(f"  [图保存失败] {e}")
    else:
        # ---- 兜底:自己算等权组合净值曲线 ----
        print("\n  [vectorbt 未装] 用 numpy 算近似等权净值曲线")
        ret = price.pct_change().fillna(0)
        # 每月初按权重持有,期间不变
        aligned_w = weights.reindex(price.index, method='ffill').fillna(0)
        port_ret = (ret * aligned_w).sum(axis=1)
        nav = (1 + port_ret).cumprod() * cfg.INIT_CASH
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std()  * np.sqrt(252)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
        max_dd  = (nav / nav.cummax() - 1).min()
        print(f"\n  累计收益:{(nav.iloc[-1]/cfg.INIT_CASH - 1):.2%}")
        print(f"  年化收益:{ann_ret:.2%}")
        print(f"  年化波动:{ann_vol:.2%}")
        print(f"  夏普比率:{sharpe:.2f}")
        print(f"  最大回撤:{max_dd:.2%}")

        # 简单出 CSV
        out = pd.DataFrame({'nav': nav, 'ret': port_ret})
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        out_path = cfg.GP_DIR / f"nav_curve_{ts}.csv"
        out.to_csv(out_path, encoding='utf-8-sig')
        print(f"  净值曲线 → {out_path}")

    # ---- 绩效自检 ----
    print("\n[INFO] 合格线参考:")
    print("  年化超额 > 10% / 夏普 > 1.5 / 最大回撤 < 25%")


if __name__ == "__main__":
    main()