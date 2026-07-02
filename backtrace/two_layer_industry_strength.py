# 双层横截面:行业板块(880xxx)+ 通达信88 强势股
# 第 1 层:588 个行业指数 vs 上证大盘,找 Top K 强势行业
# 第 2 层:通达信88 个股,先按"近 20 日收益"排名,再过滤"属于 Top 行业"的股
#
# 注意:TQ 的 880xxx 行业指数本身有 K 线但拿不到成分股,所以用"行业代码前缀"匹配做粗过滤
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

import tsfresh_pipeline as P

tq.initialize(__file__)

# ============== 配置 ==============
SECTOR_CODE    = '通达信88'             # 仅作为后备(实际不用)
INDEX_CODE     = '000001.SH'            # 大盘基准
WINDOW         = 20                     # 近 N 日累计收益
TOP_INDUSTRY_N = 10                     # 第 1 层后取 Top N 强势行业
TOP_STOCK_N    = 20                     # 第 2 层后取 Top M 强势股
TARGET_START   = '20250101'
TARGET_END     = '20251231'
# ===================================

# ---------- 1. 拉大盘 + 缺省行业板块(list_type=11, 128 个真行业)----------
print("=" * 70)
print("[1/3] 拉大盘 + 缺省行业板块...")
df_idx = P.load_ohlcva(INDEX_CODE, verbose=False)
# list_type=11 是缺省行业板块(申万二级),比 list_type=1 的 588 个干净(无事件型)
industry_list = tq.get_stock_list('11', list_type=1)
print(f"大盘: {df_idx.shape[0]} 行")
print(f"缺省行业板块: {len(industry_list)} 个 (list_type=11)")


# ---------- 2. 拉行业板块日线(只拉 daily close)----------
print("\n[1/3] 拉所有行业指数 2025 年 close...")
industry_codes = [it['Code'] for it in industry_list if it and 'Code' in it]
# 一次批量拉全部 588 个
end = '20251231'
start = '20250101'
industry_close_df = pd.DataFrame()
try:
    df_ind = tq.get_market_data(
        field_list=['Close'],
        stock_list=industry_codes,
        start_time=start, end_time=end,
        dividend_type='front', period='1d', fill_data=True,
    )
    industry_close_df = df_ind['Close']    # columns=industry_codes
    print(f"  -> 行业指数矩阵:{industry_close_df.shape} (NaN 个数: {industry_close_df.isna().sum().sum()})")
except Exception as e:
    print(f"[ERR] 批量拉行业指数失败:{e}")


# ---------- 3. 第 1 层:行业 vs 大盘,Top K 强势行业 ----------
print("\n" + "=" * 70)
print(f"[第 1 层] 行业指数 vs 大盘(WINDOW={WINDOW} 日)")
print("=" * 70)

idx_close = df_idx['Close'].reindex(industry_close_df.index).ffill()
idx_ret = float(idx_close.iloc[-1] / idx_close.iloc[-WINDOW - 1] - 1)
print(f"近 {WINDOW} 日大盘累计收益:{idx_ret:+.2%}")

layer1_rows = []
for code in industry_close_df.columns:
    s = industry_close_df[code].dropna()
    if len(s) < WINDOW + 1:
        continue
    r = float(s.iloc[-1] / s.iloc[-WINDOW - 1] - 1)
    industry_name = next((it['Name'] for it in industry_list if it.get('Code') == code), code)
    layer1_rows.append({
        'industry_code': code,
        'industry_name': industry_name,
        'ret': r,
        'alpha_vs_idx': r - idx_ret,
    })

df_layer1 = pd.DataFrame(layer1_rows).sort_values('alpha_vs_idx', ascending=False).reset_index(drop=True)
print(f"有效行业指数 {len(df_layer1)} 个")
print(f"\nTop {TOP_INDUSTRY_N} 强势行业:")
print(df_layer1.head(TOP_INDUSTRY_N).to_string(index=False))

top_industries = df_layer1.head(TOP_INDUSTRY_N)
top_industry_codes = set(top_industries['industry_code'])


# ---------- 4. 第 2 层:Top 10 行业的成分股 + 个股 α ----------
print("\n" + "=" * 70)
print(f"[第 2 层] Top {TOP_INDUSTRY_N} 行业成分股(用 get_stock_list_in_sector)vs 板块指数")
print("=" * 70)

tq.initialize(__file__)

# ---------- 4. 拉 Top 10 行业的成分股(去重)----------
print("\n" + "=" * 70)
print(f"[第 2 层] 拉 Top {TOP_INDUSTRY_N} 行业全部成分股...")
print("=" * 70)

all_member_codes = set()
for _, ind_row in top_industries.iterrows():
    code = ind_row['industry_code']
    members = tq.get_stock_list_in_sector(code)   # 必须带 .SH,默认 block_type=0
    if members:
        all_member_codes.update(members)
all_member_codes = sorted(all_member_codes)
print(f"Top {TOP_INDUSTRY_N} 行业去重成分股:{len(all_member_codes)} 只")

# 批量拉这些成分股日线
print(f"\n批量拉 {len(all_member_codes)} 只成分股日线(2025)...")
df_stocks = tq.get_market_data(
    field_list=['Close'],
    stock_list=all_member_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
stocks_close_df = df_stocks['Close']   # columns = all_member_codes
print(f"  -> 个股 close 矩阵:{stocks_close_df.shape}")
tq.close()

# ---------- 5. 第 2 层:每板块成分股 vs 板块指数 α ----------
print("\n" + "=" * 70)
print(f"[第 2 层] 每板块 Top 5 个股 vs 板块指数 α")
print("=" * 70)

all_top_stocks = []
for _, ind_row in top_industries.iterrows():
    code, name, ind_ret, ind_alpha = ind_row['industry_code'], ind_row['industry_name'], ind_row['ret'], ind_row['alpha_vs_idx']
    members = tq.get_stock_list_in_sector(code)
    if not members or members[0] not in stocks_close_df.columns:
        print(f"  [{name} ({code})] 无成分股或拉不到日线,跳过")
        continue

    # 算每只成分股近 20 日 α vs 板块指数
    stock_rows = []
    for s in members:
        if s not in stocks_close_df.columns:
            continue
        s_close = stocks_close_df[s].dropna()
        if len(s_close) < WINDOW + 1:
            continue
        s_ret = float(s_close.iloc[-1] / s_close.iloc[-WINDOW - 1] - 1)
        stock_rows.append({
            'industry_code': code,
            'industry_name': name,
            'industry_ret': ind_ret,
            'stock': s,
            'stock_ret': s_ret,
            'alpha_vs_idx': s_ret - idx_ret,
            'alpha_vs_industry': s_ret - ind_ret,
        })

    if not stock_rows:
        print(f"  [{name} ({code})] 全部成分股日线不足,跳过")
        continue
    df_members = pd.DataFrame(stock_rows).sort_values('alpha_vs_industry', ascending=False)
    print(f"\n  [{name} ({code})] α={ind_alpha*100:+.1f}pp, 成分股 {len(members)} 只 → 有日线 {len(df_members)} 只,Top 5 vs 板块:")
    print(df_members.head(5)[['stock', 'stock_ret', 'alpha_vs_industry']].to_string(index=False))
    all_top_stocks.extend(stock_rows)

# 综合 Top 成分股排名(按 vs 大盘 α)
df_all_top = pd.DataFrame(all_top_stocks)
if not df_all_top.empty:
    df_all_top = df_all_top.sort_values('alpha_vs_idx', ascending=False).reset_index(drop=True)
    print(f"\n[综合] Top 10 行业全部成分股 {len(df_all_top)} 只,按 α vs 大盘 排名 Top 20:")
    print(df_all_top.head(20)[['stock', 'industry_name', 'stock_ret',
                                  'alpha_vs_idx', 'alpha_vs_industry']].to_string(index=False))
else:
    df_all_top = pd.DataFrame()
    print("\n[WARN] 行业成分股拉数失败")


# ---------- 6. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(__file__),
    f'two_layer_industry_strong_{INDEX_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
# 保存:Top 行业 + Top 行业成分股排名
with open(out_csv, 'w', encoding='utf-8-sig') as f:
    f.write("# === Top 10 强势行业 (588 行业指数 vs 大盘) ===\n")
    df_layer1.head(TOP_INDUSTRY_N).to_csv(f, index=False)
    f.write("\n# === Top 10 行业 全部成分股 (α vs 大盘) ===\n")
    df_all_top.to_csv(f, index=False)

print(f"\n结果已保存到 {out_csv}")


# ---------- 7. 对账 ----------
top_ind_avg_ret = df_layer1.head(TOP_INDUSTRY_N)['ret'].mean()
if not df_all_top.empty:
    top_comp_avg_ret = df_all_top.head(20)['stock_ret'].mean()
    print(f"\n[对账]")
    print(f"  大盘 近 {WINDOW} 日收益:                {idx_ret:+.2%}")
    print(f"  Top 10 行业平均近 {WINDOW} 日收益:      {top_ind_avg_ret:+.2%}")
    print(f"  Top 行业成分股 20 平均近 {WINDOW} 日收益: {top_comp_avg_ret:+.2%}")

tq.close()