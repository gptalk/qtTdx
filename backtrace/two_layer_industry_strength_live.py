# 真策略版:分别用 10 / 5 / 1 日窗口跑 α 选股真策略,对比 Sharpe / 最大回撤 / 与大盘对比
# 评估 Sharpe / 最大回撤 / 与大盘对比
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

import tsfresh_config as C
import tsfresh_pipeline as P
import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
INDEX_CODE      = '000001.SH'           # 大盘基准
REBAL_FREQ      = 5                     # 调仓周期(交易日,5 = 周)
TOP_INDUSTRY    = 5                     # 选 Top K 行业
TOP_PER_INDUSTRY = 5                    # 每行业 Top M → 总持仓 25 只
TARGET_START    = '20210101'            # 5 年回测
TARGET_END      = '20251231'
INIT_CASH       = 1_000_000

# ===== 三个窗口分别独立测试 =====
WINDOWS_TO_TEST = [10, 5, 1]
# ===================================


# ---------- 1. 拉大盘 + 缺省行业板块(list_type=11, 128 个) + 全部成分股 5 年 close ----------
print("=" * 70)
print("[1/3] 拉数据(2021-2025)...")
df_idx = P.load_ohlcva(INDEX_CODE, verbose=False)
industry_list = tq.get_stock_list('11', list_type=1)
industry_codes = [it['Code'] for it in industry_list if it and 'Code' in it]
print(f"  大盘: {df_idx.shape[0]} 行")
print(f"  缺省行业板块(list_type=11): {len(industry_codes)} 个")

df_ind = tq.get_market_data(
    field_list=['Close'],
    stock_list=industry_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
industry_close_df = df_ind['Close']
print(f"  行业指数矩阵: {industry_close_df.shape}")

print("\n[2/3] 拉全部 128 行业成分股 5 年 close...")
members_cache = {}
def get_members(code):
    if code not in members_cache:
        members_cache[code] = tq.get_stock_list_in_sector(code)
    return members_cache[code]

all_member_codes = set()
for code in industry_codes:
    members = get_members(code)
    if members:
        all_member_codes.update(members)
all_member_codes = sorted(all_member_codes)
print(f"  128 行业去重成分股: {len(all_member_codes)} 只")

df_stocks = tq.get_market_data(
    field_list=['Close'],
    stock_list=all_member_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
stocks_close_df = df_stocks['Close']
print(f"  个股 close 矩阵: {stocks_close_df.shape}")
tq.close()

# 对齐索引
common_idx = industry_close_df.index.intersection(stocks_close_df.index).intersection(df_idx.index)
industry_close_df = industry_close_df.loc[common_idx]
stocks_close_df = stocks_close_df.loc[common_idx]
df_idx = df_idx.loc[common_idx]
print(f"\n[对齐后] 共同交易日: {len(common_idx)}  ({common_idx[0].date()} ~ {common_idx[-1].date()})")


# ---------- 2. 三窗口独立回测 ----------
print("\n[3/3] 三窗口分别独立回测(10/5/1 日)...")
print("=" * 70)


def run_window(window):
    """用指定窗口跑一次完整 5 年回测,返回周收益 DataFrame"""
    rebal_dates = common_idx[window :: REBAL_FREQ]
    weekly_returns = []
    prev_holdings = {}

    for reb_date in rebal_dates:
        reb_pos = common_idx.get_loc(reb_date)

        # 行业 α
        ind_ret = industry_close_df.iloc[reb_pos] / industry_close_df.iloc[reb_pos - window] - 1
        top_industries = ind_ret.nlargest(TOP_INDUSTRY).index.tolist()

        # 每行业 Top M
        new_holdings = {}
        for ind_code in top_industries:
            members = get_members(ind_code)
            if not members:
                continue
            stock_ret = {}
            for s in members:
                if s not in stocks_close_df.columns:
                    continue
                if reb_pos < window:
                    continue
                sr = stocks_close_df[s].iloc[reb_pos] / stocks_close_df[s].iloc[reb_pos - window] - 1
                if not np.isnan(sr):
                    stock_ret[s] = sr
            if not stock_ret:
                continue
            top_stocks = sorted(stock_ret.items(), key=lambda x: x[1], reverse=True)[:TOP_PER_INDUSTRY]
            for s, _ in top_stocks:
                new_holdings[s] = 1.0 / (TOP_INDUSTRY * TOP_PER_INDUSTRY)

        # 持有期收益
        next_pos = reb_pos + REBAL_FREQ
        if next_pos >= len(common_idx):
            next_pos = len(common_idx)
        next_date = common_idx[next_pos - 1]

        if prev_holdings:
            port_ret = 0.0
            for s, w in prev_holdings.items():
                if s not in stocks_close_df.columns:
                    continue
                s_close = stocks_close_df[s].loc[reb_date:next_date].dropna()
                if len(s_close) < 2:
                    continue
                sr = s_close.iloc[-1] / s_close.iloc[0] - 1
                port_ret += w * sr
            weekly_returns.append({
                'date': reb_date,
                'holdings_count': len(prev_holdings),
                'weekly_return': port_ret,
            })
        else:
            weekly_returns.append({
                'date': reb_date,
                'holdings_count': 0,
                'weekly_return': 0.0,
            })

        prev_holdings = new_holdings

    return pd.DataFrame(weekly_returns).set_index('date')


def eval_window(df_rets, label):
    total_weeks = len(df_rets)
    if total_weeks == 0:
        return None
    total_ret = (1 + df_rets['weekly_return']).prod() - 1
    ann_ret = (1 + total_ret) ** (52 / total_weeks) - 1
    ann_vol = df_rets['weekly_return'].std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + df_rets['weekly_return']).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    yearly = (df_rets.assign(year=df_rets.index.year)
                       .groupby('year')['weekly_return']
                       .apply(lambda x: (1 + x).prod() - 1))
    return {
        'window':       label,
        'total_weeks':  total_weeks,
        'total_ret':    total_ret,
        'ann_ret':      ann_ret,
        'ann_vol':      ann_vol,
        'sharpe':       sharpe,
        'max_dd':       dd.min(),
        'yearly':       yearly,
    }


results = []
for w in WINDOWS_TO_TEST:
    print(f"\n--- WINDOW = {w} 日 ---")
    df_rets = run_window(w)
    stat = eval_window(df_rets, f'{w}d')
    if stat:
        results.append(stat)
        print(f"  5 年累计:{stat['total_ret']:+.2%}  年化:{stat['ann_ret']:+.2%}  "
              f"Sharpe:{stat['sharpe']:.2f}  最大回撤:{stat['max_dd']:.2%}")

idx_close_aligned = df_idx['Close'].loc[common_idx]
idx_total = float(idx_close_aligned.iloc[-1] / idx_close_aligned.iloc[0] - 1)


# ---------- 3. 对比输出 ----------
print("\n" + "=" * 70)
print(f"=== 三窗口对比(128 缺省行业 × 5×5 = 25 只,每周调仓)===")
print("=" * 70)
print(f"{'窗口':<10} {'5 年累计':>12} {'年化':>10} {'波动':>10} {'Sharpe':>8} {'最大回撤':>12}")
print("-" * 70)
for r in results:
    print(f"{r['window']:>8}   {r['total_ret']:>+12.2%} {r['ann_ret']:>+10.2%} "
          f"{r['ann_vol']:>10.2%} {r['sharpe']:>8.2f} {r['max_dd']:>12.2%}")
print(f"{'大盘':>8}   {idx_total:>+12.2%}")
print()

# 按年分解
years = sorted(results[0]['yearly'].index)
print("按年收益分解:")
print(f"{'窗口':<10}", end='')
for y in years:
    print(f"  {y:>6}", end='')
print(f"  {'总':>8}")
for r in results:
    print(f"{r['window']:>8} ", end='')
    for y in years:
        v = r['yearly'].get(y, 0)
        print(f"  {v:>+6.1%}", end='')
    print(f"  {r['total_ret']:>+8.1%}")

print(f"{'大盘':>8} ", end='')
for y in years:
    yr_idx = df_idx['Close'].loc[common_idx]
    yr_first = yr_idx[yr_idx.index.year == y].iloc[0]
    yr_last = yr_idx[yr_idx.index.year == y].iloc[-1]
    yr_ret = float(yr_last / yr_first - 1)
    print(f"  {yr_ret:>+6.1%}", end='')
yr_first_all = idx_close_aligned.iloc[0]
yr_last_all = idx_close_aligned.iloc[-1]
print(f"  {idx_total:>+8.1%}")

# 排序找最佳窗口
best = max(results, key=lambda r: r['sharpe'])
worst = min(results, key=lambda r: r['sharpe'])
print(f"\n[最佳窗口] WINDOW = {best['window']}(Sharpe = {best['sharpe']:.2f})")
print(f"[最差窗口] WINDOW = {worst['window']}(Sharpe = {worst['sharpe']:.2f})")

# 保存最佳窗口的明细
out_csv = os.path.join(
    os.path.dirname(__file__),
    f'two_layer_industry_live_{best["window"]}_{INDEX_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
run_window(int(best['window'].replace('d', ''))).to_csv(out_csv, encoding='utf-8-sig')
print(f"\n[最佳窗口] 周收益已保存到 {out_csv}")