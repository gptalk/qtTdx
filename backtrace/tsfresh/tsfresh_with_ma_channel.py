# tsfresh + MA5 作独立通道 → vbt + jhzq_fees 真实扣费
# 核心创新:把 ma5 / ma10 / ma20 + close/ma5 偏离度 也作为通道喂给 tsfresh
# 对比 3 个 tsfresh 通道方案 + 1 个 MA5 基线
# 输出:tsfresh_with_ma_<code>_<start>_<end>.csv
# 用法:验证把"均线 / 偏离度"当额外 tsfresh 通道,能否挑出 MA5 看不出的反转点
#
# ⚠️ 已知陷阱:
#  1. MA 通道与 Close 结构性共线 — ma5/ma10/ma20 是 Close 的确定性变换(低通滤波),
#     喂给 tsfresh 相当于让模型对同一段价格做了两次滤波,显著特征可能成对出现。
#     跑完后看 _report_channel_composition 输出,确认 ma5/10/20 通道是否真带来增量
#     (而不是 Close 信息的冗余复制 — 那样的话 with_ma 跟 basic 没本质区别)。
#  2. **阈值选择偏差** — 0.60 是从上一版脚本(同一段 5 年回测、同一只票)网格搜索出的
#     「最优」,本次对比再用同一段数据验证,本质上是同一份样本的二次拟合。
#     这次默认改用 0.55(非拟合中点)以避开这个偏差;想保留 0.60 对比,见 --entry-th 参数。
#  3. fillna(0.0) 会人为制造 0 → 实值的跳变,改用 bfill(第一个有效值往前填)。
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'
TARGET_START = '20250101'
TARGET_END   = '20251231'
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95

INIT_TRAIN_SIZE = 200   # walk-forward 初始训练窗口
STEP = 50               # 重训步长
EXIT_TSF = 0.50
# ⚠️ 0.60 是同段数据上 grid search 的「最优」,二次拟合风险大 → 默认 0.55(中点)
ENTRY_TSF = 0.55
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# 计算 MA 通道
for w in [5, 10, 20]:
    df[f'ma{w}'] = vbt.MA.run(df['Close'], window=w).ma.ffill()
# close 相对 ma5 的偏离度(标准化)
df['rel_ma5'] = (df['Close'] - df['ma5']) / df['ma5'].replace(0, np.nan)

close_full = df['Close']
open_full  = df['Open']

# ---------- 2. 跑两组 tsfresh 通道方案 ----------
PROBA_CACHE = {}   # key=channels -> pd.Series(proba, index=date)


def _report_channel_composition(X_sel, label):
    """统计 X_sel 里各通道贡献的显著特征数 — 用来看 MA 通道是否带来真增量。

    ma5/ma10/ma20 是 Close 的低通滤波,与 Close 通道提取的特征高度相关。
    如果 MA 通道入选数极少 → with_ma 跟 basic 没本质区别,「加均线」无实质 alpha。
    """
    if X_sel.shape[1] == 0:
        return
    # 特征名形如 '{kind}__{feature_func}...',kind 在第一个 '__' 前
    channels = pd.Series([col.split('__', 1)[0] for col in X_sel.columns])
    counts = channels.value_counts()
    print(f"   [{label}] 各通道入选特征数:")
    for ch, n in counts.items():
        pct = n / len(channels) * 100
        print(f"     {ch:<10} {n:>4}  ({pct:5.1f}%)")
    # 如果 MA 通道占比超过 1/3,提醒是冗余复制风险
    ma_total = sum(counts.get(c, 0) for c in counts.index if c.startswith('ma') or c == 'rel_ma5')
    if ma_total > 0 and ma_total / len(channels) > 0.33:
        print(f"   [WARN] MA 相关通道占 {ma_total/len(channels):.0%},可能与 Close 通道冗余")


def build_tsfresh_proba(channels, label):
    """
    给定通道列表,跑全量 tsfresh + walk-forward。
    返回 proba Series(每个窗口结束日 1 个值)。
    """
    print(f"\n--- {label} ({len(channels)} 通道: {channels}) ---")
    df_fill = df[channels].copy().bfill()   # 用第一个有效值往前填(替代 fillna(0.0))
    # 避免 0 → 实值的跳变被 tsfresh 当成「突变」类特征 — 改 bfill 后头 19 天 ma20 为常数,
    # tsfresh 提不出有效信号(方差=0、自相关=1),比 0 填充更干净
    long_df = P.to_long_format(df_fill, channels=channels, id_value=STOCK_CODE)
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, df['Close'].values, verbose=False)
    print(f"   样本 {len(y_all)} 个  |  正样本 {y_all.mean():.1%}  |  特征 {X_all.shape[1]} 列")

    # FDR 特征筛选 — **只在前 INIT_TRAIN_SIZE 段做**(防未来信息泄漏)
    X_train0 = X_all.iloc[:INIT_TRAIN_SIZE]
    y_train0 = y_all.iloc[:INIT_TRAIN_SIZE]
    X_sel_initial = P.select_relevant(X_train0, y_train0, verbose=False)
    selected_cols = X_sel_initial.columns.tolist()
    if len(selected_cols) == 0:
        print(f"   [WARN] FDR=0,用全量 {X_all.shape[1]} 特征")
        X_sel = X_all
    else:
        X_sel = X_all[selected_cols]   # 筛出的列名 + 全期索引 → walk-forward 可按 pos 切片
    print(f"   FDR 显著 {X_sel.shape[1]} 列 (前 {INIT_TRAIN_SIZE} 个样本筛)")
    _report_channel_composition(X_sel, label=label)

    # walk-forward
    print(f"   walk-forward (initial={INIT_TRAIN_SIZE}, retrain every {STEP})...")
    date_index = pd.DatetimeIndex(df.index)
    proba_records = []
    scaler_w = clf_w = None
    for pos, idx in enumerate(X_sel.index):
        end_t = idx[1]
        if end_t >= len(date_index):
            continue
        if pos < INIT_TRAIN_SIZE:
            proba_records.append((date_index[end_t], np.nan))
            continue
        if pos == INIT_TRAIN_SIZE or (pos - INIT_TRAIN_SIZE) % STEP == 0:
            scaler_w, clf_w = P.fit_logreg(X_sel.iloc[:pos], y_all.iloc[:pos], verbose=False)
        p = float(clf_w.predict_proba(scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
        proba_records.append((date_index[end_t], p))

    proba = pd.Series([v for _, v in proba_records],
                      index=pd.DatetimeIndex([d for d, _ in proba_records]),
                      name='proba').sort_index()
    proba = proba[~proba.index.duplicated(keep='last')].dropna()
    print(f"   proba {len(proba)} 个  |  mean={proba.mean():.3f}")
    return proba


# 5 通道原始
PROBA_CACHE['basic'] = build_tsfresh_proba(
    channels=['Open', 'High', 'Low', 'Close', 'Volume'],
    label='basic(5 通道 OHLCV)',
)

# 8 通道(原始 + MA5/10/20 + 偏离度)
PROBA_CACHE['with_ma'] = build_tsfresh_proba(
    channels=['Open', 'High', 'Low', 'Close', 'Volume',
              'ma5', 'ma10', 'ma20', 'rel_ma5'],
    label='with_ma(8 通道 = 5 OHLCV + MA5/10/20 + rel_ma5)',
)


# ---------- 3. 4 组对比策略 ----------
shares_per_trade = int(np.floor(INIT_CASH * MAX_POS_PCT / float(open_full.iloc[0]) / 100) * 100)
print(f"\n每笔 {shares_per_trade} 股  (≈ {shares_per_trade * float(open_full.iloc[0]):,.0f} 元)\n")


def build_signals(strategy):
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(close_full, window=5).ma.ffill()
        entries = close_full.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits   = close_full.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    if strategy['kind'] == 'tsfresh':
        proba = PROBA_CACHE[strategy['proba_key']]
        aligned = proba.reindex(close_full.index)
        if strategy['mode'] == 'pure':
            entries = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
            exits   = (aligned < EXIT_TSF).shift(1).fillna(False).astype(bool)
        elif strategy['mode'] == 'confirmed':
            # MA5 触发 + tsfresh 确认(双条件 AND)
            ma = vbt.MA.run(close_full, window=5).ma.ffill()
            ma_entry = close_full.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
            tsf_entry = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
            entries = ma_entry & tsf_entry
            exits   = close_full.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits
    raise ValueError(strategy['kind'])


def run_one(strategy):
    name, entries, exits = build_signals(strategy)
    if entries.sum() == 0:
        return {'strategy': name, 'trades': 0,
                'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
                'net_pnl': 0.0, 'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'zero_friction_ret': 0.0}

    pf_zero = vbt.Portfolio.from_signals(
        close=close_full, entries=entries, exits=exits, price=open_full,
        init_cash=INIT_CASH, fees=0, slippage=0, freq='D',
        size=shares_per_trade, size_type='amount', size_granularity=100,
        upon_long_conflict='exit',
    )
    trades = pf_zero.trades.records_readable
    zero_ret = pf_zero.total_return()
    if len(trades) == 0:
        return {'strategy': name, 'trades': 0,
                'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
                'net_pnl': 0.0, 'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'zero_friction_ret': zero_ret}

    summary = F.summary_after_fees(trades, STOCK_CODE)
    summary['strategy'] = name
    summary['zero_friction_ret'] = zero_ret
    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades)
    summary['profit_factor'] = (
        trades[pnl_col][trades[pnl_col] > 0].sum() /
        abs(trades[pnl_col][trades[pnl_col] < 0].sum())
        if (trades[pnl_col] < 0).sum() > 0 else float('inf')
    )
    summary['net_ret'] = summary['net_pnl'] / INIT_CASH
    return summary


STRATEGIES = [
    {'name': 'MA5_baseline',          'kind': 'ma'},
    {'name': 'tsfresh_basic_p60',     'kind': 'tsfresh', 'proba_key': 'basic',   'mode': 'pure'},
    {'name': 'tsfresh_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'pure'},
    {'name': 'MA5_AND_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'confirmed'},
]

print("=" * 70)
print("=== 4 策略对比 ===")
print("=" * 70)
for strat in STRATEGIES:
    name, entries, _ = build_signals(strat)
    print(f"  [{strat['name']}] entry 信号数: {int(entries.sum())}")

results = [run_one(s) for s in STRATEGIES]
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100


def fmt_pct(x):
    return f"{x:>7.2%}" if pd.notna(x) and x != float('inf') else "     inf"
def fmt_money(x):
    return f"{x:>12,.2f}" if pd.notna(x) else "          N/A"
def fmt_pp(x):
    return f"{x:>6.1f}pp" if pd.notna(x) else "  N/A"

print("\n" + "=" * 110)
print("=== 4 策略对比表(basic vs with_ma vs MA5 复合) ===")
print("=" * 110)
print(f"{'策略':<22} {'笔数':>5} {'零摩擦':>8} "
      f"{'净收益':>12} {'净收益率':>8} {'胜率':>6} {'盈亏比':>8} {'摩擦吃损':>10}")
print("-" * 110)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<22} {int(r['trades']):>5} "
          f"{fmt_pct(r['zero_friction_ret'])} "
          f"{fmt_money(r['net_pnl'])} {fmt_pct(r['net_ret'])} "
          f"{fmt_pct(r['win_rate'])} {fmt_pct(r['profit_factor'])} "
          f"{fmt_pp(r['friction_loss_pp'])}")

# 保存
out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
                       f'tsfresh_with_ma_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv')
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
