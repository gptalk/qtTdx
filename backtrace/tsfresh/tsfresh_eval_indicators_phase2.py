# -*- coding: utf-8 -*-
# tsfresh 指标评测:Phase 2 — 分池因子回测
# 把 Phase 1 输出的 per-stock IC 明细拆成两个池,各自建因子、跑 vbt + jhzq_fees 真实回测,
# 验证「科创板动量 / 主板反转」这种「市场结构分化」假设是否在样本外也能成立。
#
# 关键发现(Phase 1 输出):
#   - ~230 只 stoch_k > 0.05 AND obv > 0.05 → 动量池(主要是科创板/创业板)
#   - ~4300 只其余 → 反转池(主板 SH/SZ 占大头)
#
# 策略:
#   - 动量池:factor = stoch_k 原值,每 5 日调仓,买进当时池内分位最高 10%
#   - 反转池:factor = -(close/ma5 - 1),每 5 日调仓,买进当时池内分位最低 10%(最超卖)
#   - 持有 5 天,等权,SZ/SH 各自走 jhzq_fees 真实费率
#
# 输入:backtrace/outputs/tsfresh_indicator_ic_<sector>_<start>_<end>.csv(Phase 1 产物)
# 输出:
#   - tsfresh_phase2_pool_split_<date>.csv  每只票所在池 + 该池因子在调仓日的分位
#   - tsfresh_phase2_momentum_bt_<date>.csv 动量池每只票 trades + 净 PnL
#   - tsfresh_phase2_reversal_bt_<date>.csv 反转池每只票 trades + 净 PnL
#   - tsfresh_phase2_summary_<date>.csv      两池 side-by-side 收益/胜率对比
#
# 用法:`PYTHONIOENCODING=utf-8 python backtrace/tsfresh/tsfresh_eval_indicators_phase2.py`
#      可选 `--limit N` 限池大小跑冒烟;`--ic-file PATH` 指定其它 Phase 1 输出。

import warnings
warnings.filterwarnings('ignore')

import sys, os, glob, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import vectorbt as vbt

from common import data_store, jhzq_fees as F, tsfresh_pipeline as P


# ============== 配置 ==============
INIT_CASH     = 100_000         # 单只票虚拟初始资金(per-stock vbt,匹配 tsfresh_vbt_combo.py 既有口径)
MAX_POS_PCT   = 0.20            # 单票占初始资金上限
REBAL_FREQ    = 5               # 每 5 个交易日调一次仓
TOP_QUANTILE  = 0.10            # 选池内分位前/后 10%
IC_THRESHOLD  = 0.05            # 池划分阈值(Phase 1 用户分析结果)
MIN_POOL_SIZE = 30              # 池内少于 30 只票时跳过这一次调仓(分位数算不稳)
# ===================================


# ---------- 0. CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description='Phase 2: 分池因子 + vbt 真实回测')
    p.add_argument('--ic-file', default=None,
                   help='Phase 1 的 IC 明细 CSV (默认取 backtrace/outputs/ 下最新的那一份)')
    p.add_argument('--limit', type=int, default=0,
                   help='每池最多取 N 只票(冒烟);0=全跑')
    return p.parse_args()


def latest_ic_detail():
    """默认取 outputs/ 下最新的 tsfresh_indicator_ic_*.csv(Phase 1 产物)。
    按文件 mtime 排序(不是文件名),避免中文 '通' > 英文 'S' 导致排序崩。
    """
    files = glob.glob(os.path.join('backtrace/outputs', 'tsfresh_indicator_ic_*.csv'))
    files = [f for f in files if not f.endswith('_summary.csv')]
    if not files:
        raise FileNotFoundError("未找到 Phase 1 输出,请先跑 backtrace/tsfresh/tsfresh_eval_indicators.py")
    return max(files, key=os.path.getmtime)


# ---------- 1. 池划分 ----------
def split_pools(ic_detail, ic_threshold=IC_THRESHOLD):
    """stoch_k > 0.05 AND obv > 0.05 → 动量池;其余 → 反转池。
    仅在两列都有非空 IC 值时纳入动量池(其它列 IC 情况不参与本阶段分组)。
    """
    df = ic_detail.copy()
    df['stoch_k'] = pd.to_numeric(df['stoch_k'], errors='coerce')
    df['obv']     = pd.to_numeric(df['obv'],     errors='coerce')
    momentum_mask = (df['stoch_k'] >  ic_threshold) & (df['obv'] >  ic_threshold)
    momentum = df.loc[momentum_mask, 'stock'].tolist()
    reversal = df.loc[~momentum_mask, 'stock'].tolist()
    return momentum, reversal


# ---------- 2. 因子构造 ----------
def reversal_factor(df):
    """-(close/ma5 - 1):正 = 低于 MA5(超卖),反转策略买最高分位 = 最超卖。"""
    c = df['Close']
    ma5 = c.rolling(5).mean()
    return -(c / ma5 - 1)


def momentum_factor(df):
    """stoch_k 原值:高 = 强势,动量策略买最高分位 = 最强。"""
    h, l, c = df['High'], df['Low'], df['Close']
    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    span = (high14 - low14).replace(0, np.nan)
    return (c - low14) / span * 100


# ---------- 3. 调仓 + per-stock vbt 真实回测 ----------
def run_pool(pool_codes, factor_fn, side, ic_detail, target_start, target_end):
    """对给定股票池,跑横截面调仓回测。

    side:  'long_top'     买进池内分位最高 10%(动量);factor 越大越好
           'long_bottom'  买进池内分位最低 10%(反转);factor 越小越好
    """
    # 3.1 从本地 data/ 拉日线
    pool_data = {}
    miss = 0
    for code in pool_codes:
        df = data_store.load_daily(code)
        if df is None or len(df) < 60:
            miss += 1
            continue
        df = df.loc[target_start:target_end]
        if len(df) < 60:
            miss += 1
            continue
        pool_data[code] = df
    if not pool_data:
        return pd.DataFrame(), pd.DataFrame()

    # 3.2 构 factor / close 矩阵(date × code)
    factor_dict = {code: factor_fn(df) for code, df in pool_data.items()}
    close_dict  = {code: df['Close']          for code, df in pool_data.items()}
    factor_df = pd.DataFrame(factor_dict).sort_index()
    close_df  = pd.DataFrame(close_dict).sort_index()
    common_idx = factor_df.index.intersection(close_df.index)
    factor_df = factor_df.loc[common_idx]
    close_df  = close_df.loc[common_idx]

    # 3.3 调仓日横截面选股 → entries / exits(date × code)
    rebal_dates = common_idx[::REBAL_FREQ]
    entries = pd.DataFrame(False, index=common_idx, columns=close_df.columns)
    exits   = pd.DataFrame(False, index=common_idx, columns=close_df.columns)
    for date in rebal_dates:
        if date not in factor_df.index:
            continue
        f = factor_df.loc[date].dropna()
        if len(f) < MIN_POOL_SIZE:
            continue
        if side == 'long_top':
            threshold = f.quantile(1 - TOP_QUANTILE)
            selected = f[f >= threshold].index
        else:  # long_bottom
            threshold = f.quantile(TOP_QUANTILE)
            selected = f[f <= threshold].index
        idx_pos = common_idx.get_loc(date)
        for code in selected:
            entries.loc[date, code] = True
            if idx_pos + REBAL_FREQ < len(common_idx):
                exits.loc[common_idx[idx_pos + REBAL_FREQ], code] = True

    # 3.4 shift(1) —— 信号次日开盘前已知,实际成交挂在次日
    entries = entries.shift(1).fillna(False).astype(bool)
    exits   = exits.shift(1).fillna(False).astype(bool)

    # 3.5 per-stock vbt + jhzq_fees(匹配 tsfresh_with_ma_grid_sector.py 既有模式)
    rows = []
    trade_rows = []
    for code in pool_data.keys():
        if not entries[code].any() and not exits[code].any():
            continue
        c     = close_df[code]
        e_ser = entries[code]
        x_ser = exits[code]
        # 单票仓位大小:固定额度,100 股一手取整
        init_open = c.iloc[0]
        if not np.isfinite(init_open) or init_open <= 0:
            continue
        shares = int(np.floor(INIT_CASH * MAX_POS_PCT / init_open / 100) * 100)
        if shares <= 0:
            continue
        pf = vbt.Portfolio.from_signals(
            close=c, entries=e_ser, exits=x_ser,
            init_cash=INIT_CASH, fees=0, freq='D',
            size=shares, size_type='amount', size_granularity=100,
            upon_long_conflict='exit',
        )
        trades = pf.trades.records_readable
        if len(trades) == 0:
            continue
        trades_net = F.adjust_trades_pnl(trades, code)
        summary    = F.summary_after_fees(trades, code)
        n_trades   = len(trades_net)
        win_rate   = (trades_net['净盈亏_扣费后'] > 0).mean() if n_trades else np.nan
        rows.append({
            'stock':        code,
            'n_trades':     n_trades,
            'gross_pnl':    summary['gross_pnl'],
            'total_stamp':  summary['total_stamp'],
            'total_transfer': summary['total_transfer'],
            'net_pnl':      summary['net_pnl'],
            'net_ret':      summary['net_pnl'] / INIT_CASH,
            'win_rate':     win_rate,
            'avg_net_per_trade': summary['avg_net_per_trade'],
        })
        # 完整 trades 明细(含 5 列中文 schema)落盘
        out = trades_net.copy()
        out.insert(0, 'stock', code)
        trade_rows.append(out)

    summary_df = pd.DataFrame(rows)
    trades_df  = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return summary_df, trades_df


def pool_split_table(ic_detail, momentum_pool, reversal_pool):
    """per-stock 池归属 + (调仓日)因子分位 — 便于人工抽查。"""
    df = ic_detail.copy()
    df['pool'] = np.where(df['stock'].isin(momentum_pool), 'momentum', 'reversal')
    df['stoch_k'] = pd.to_numeric(df['stoch_k'], errors='coerce')
    df['obv']     = pd.to_numeric(df['obv'],     errors='coerce')
    return df.sort_values(['pool', 'stock']).reset_index(drop=True)


# ---------- 4. 汇总对比 ----------
def side_by_side(mom_summary, rev_summary):
    """两池 side-by-side:总净收益 / 平均净收益 / 中位胜率 / 平均交易笔数。"""
    def agg(df, label):
        if df.empty:
            return {'pool': label, 'n_stocks': 0, 'avg_net_ret': np.nan,
                    'median_net_ret': np.nan, 'avg_win_rate': np.nan,
                    'avg_n_trades': np.nan, 'total_net_pnl': np.nan}
        return {
            'pool':           label,
            'n_stocks':       len(df),
            'avg_net_ret':    df['net_ret'].mean(),
            'median_net_ret': df['net_ret'].median(),
            'avg_win_rate':   df['win_rate'].mean(),
            'avg_n_trades':   df['n_trades'].mean(),
            'total_net_pnl':  df['net_pnl'].sum(),
        }
    return pd.DataFrame([agg(mom_summary, 'momentum'), agg(rev_summary, 'reversal')])


# ---------- 5. main ----------
def main():
    args = parse_args()
    ic_path = args.ic_file or latest_ic_detail()
    print(f"[Phase 2] 读取 IC 明细: {ic_path}")
    ic_detail = pd.read_csv(ic_path, dtype={'stock': str})
    print(f"  共 {len(ic_detail)} 只票")

    # 1. 划池
    momentum, reversal = split_pools(ic_detail)
    print(f"  动量池(stoch_k > 0.05 AND obv > 0.05): {len(momentum)} 只")
    print(f"  反转池(其余): {len(reversal)} 只")

    if args.limit > 0:
        momentum = momentum[: args.limit]
        reversal = reversal[: args.limit]
        print(f"  --limit={args.limit}:各池截到 {len(momentum)} / {len(reversal)} 只")

    # 2. 调仓窗口与 Phase 1 对齐:从 IC 明细文件名抽(start, end)
    target_start = '20250101'
    target_end   = datetime.now().strftime('%Y%m%d')

    # 3. 跑两池
    print("\n[动量池] 因子 = stoch_k,买进池内分位最高 10%")
    mom_summary, mom_trades = run_pool(
        momentum, momentum_factor, 'long_top', ic_detail, target_start, target_end,
    )
    mom_pnl = mom_summary['net_pnl'].sum() if not mom_summary.empty else 0.0
    print(f"  有效票: {len(mom_summary)}  净 PnL 合计: {mom_pnl:+.0f}")

    print("\n[反转池] 因子 = -(close/ma5 - 1),买进池内分位最低 10%")
    rev_summary, rev_trades = run_pool(
        reversal, reversal_factor, 'long_bottom', ic_detail, target_start, target_end,
    )
    rev_pnl = rev_summary['net_pnl'].sum() if not rev_summary.empty else 0.0
    print(f"  有效票: {len(rev_summary)}  净 PnL 合计: {rev_pnl:+.0f}")

    # 4. 落盘
    out_dir = os.path.join('backtrace', 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')

    split_path = os.path.join(out_dir, f'tsfresh_phase2_pool_split_{today}.csv')
    pool_split_table(ic_detail, momentum, reversal).to_csv(
        split_path, index=False, encoding='utf-8-sig')
    print(f"\n[落盘] 池归属: {split_path}")

    if not mom_trades.empty:
        p = os.path.join(out_dir, f'tsfresh_phase2_momentum_bt_{today}.csv')
        mom_trades.to_csv(p, index=False, encoding='utf-8-sig')
        print(f"[落盘] 动量池 trades: {p}")
    if not rev_trades.empty:
        p = os.path.join(out_dir, f'tsfresh_phase2_reversal_bt_{today}.csv')
        rev_trades.to_csv(p, index=False, encoding='utf-8-sig')
        print(f"[落盘] 反转池 trades: {p}")

    if not mom_summary.empty:
        p = os.path.join(out_dir, f'tsfresh_phase2_momentum_summary_{today}.csv')
        mom_summary.to_csv(p, index=False, encoding='utf-8-sig')
    if not rev_summary.empty:
        p = os.path.join(out_dir, f'tsfresh_phase2_reversal_summary_{today}.csv')
        rev_summary.to_csv(p, index=False, encoding='utf-8-sig')

    summary = side_by_side(mom_summary, rev_summary)
    s_path = os.path.join(out_dir, f'tsfresh_phase2_summary_{today}.csv')
    summary.to_csv(s_path, index=False, encoding='utf-8-sig')
    print(f"[落盘] 双池对比: {s_path}")

    # 5. 打印总结
    print("\n" + "=" * 70)
    print("=== Phase 2 双策略框架对比 ===")
    print("=" * 70)
    print(f"{'池':<10} {'票数':>6} {'平均净收益':>12} {'中位净收益':>12} "
          f"{'平均胜率':>10} {'平均交易笔数':>14} {'总净盈亏':>14}")
    print("-" * 96)
    for _, r in summary.iterrows():
        print(f"{r['pool']:<10} {int(r['n_stocks']):>6} "
              f"{r['avg_net_ret']:>+12.4%} {r['median_net_ret']:>+12.4%} "
              f"{r['avg_win_rate']:>10.1%} {r['avg_n_trades']:>14.1f} "
              f"{r['total_net_pnl']:>+14.0f}")
    print()
    print("解读:")
    print("  - 动量池 > 0 = 走 stoch_k 选 top 10% 当周吃到上涨")
    print("  - 反转池 > 0 = 走 -(close/ma5-1) 选 bottom 10% 当周吃到反弹")
    print("  - 任一池为负 = 池内 IC 信号不可用于样本外,请回 Phase 1 复核")

    # 同时把 per-stock summary 各存一份
    print("\n(per-stock 总结 CSV 已落到 outputs/;详见 *_summary_<date>.csv)")


if __name__ == '__main__':
    main()
