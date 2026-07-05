# 双层横截面相对强度选股
# 第 1 层:大盘 vs 通达信88 板块指数,找"跑赢大盘"的强势股
# 第 2 层:强势股内,个股 vs 板块,找"跑赢板块"的强势个股
# 最后对 Top N 跑 walk-forward tsfresh + vbt + jhzq_fees 验证
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
SECTOR_NAME   = '通达信88'
INDEX_CODE    = '000001.SH'             # 上证综指(大盘)
WINDOW        = 20                      # 近 N 日累计收益
TOP_STRONG_N  = 20                      # 第 1 层后取 Top N 强势股
TARGET_START  = '20250101'
TARGET_END    = '20251231'
INIT_CASH     = 100_000
MAX_POS_PCT   = 0.95

# walk-forward 参数
INIT_TRAIN_SIZE = 200
STEP = 50
ENTRY_TSF = 0.60
EXIT_TSF = 0.50
# ===================================


# ---------- 1. 拉数据(板块 + 大盘) ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 拉板块全部成员(带 Amount)...")
stock_data = P.load_sector(sector_name=SECTOR_NAME, verbose=True)
print(f"[{INDEX_CODE}] 拉大盘...")
df_idx = P.load_ohlcva(INDEX_CODE, verbose=True)
if not stock_data or df_idx is None:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)

print("=" * 70)


# ---------- 2. 计算"近 WINDOW 日累计收益" ----------
def recent_return(close_series, window=WINDOW):
    """近 WINDOW 日累计收益率(对最后 window 个 bar)"""
    s = close_series.dropna()
    if len(s) < window + 1:
        return np.nan
    return float(s.iloc[-1] / s.iloc[-window - 1] - 1)


df_demo_idx = df_idx.loc[TARGET_START:TARGET_END]
idx_ret = recent_return(df_demo_idx['Close'])
print(f"近 {WINDOW} 日大盘({INDEX_CODE})累计收益:{idx_ret:+.2%}")

# ---------- 第 1 层:个股 vs 大盘,找"跑赢大盘"的强势股 ----------
print("\n" + "=" * 70)
print(f"[第 1 层] 个股 vs 大盘(WINDOW={WINDOW} 日,找跑赢大盘的强势股)")
print("=" * 70)
layer1_rows = []
for code, raw in stock_data.items():
    df = raw.loc[TARGET_START:TARGET_END]
    if len(df) < WINDOW + 5:
        continue
    r = recent_return(df['Close'])
    if np.isnan(r):
        continue
    alpha_vs_idx = r - idx_ret
    layer1_rows.append({'stock': code, 'ret': r, 'alpha_vs_idx': alpha_vs_idx})

df_layer1 = pd.DataFrame(layer1_rows).sort_values('alpha_vs_idx', ascending=False).reset_index(drop=True)
print(f"有效股票 {len(df_layer1)} / {len(stock_data)} 只")
print(f"\nTop {TOP_STRONG_N} 强势股(跑赢大盘):")
print(df_layer1.head(TOP_STRONG_N).to_string(index=False))

top_strong_codes = df_layer1.head(TOP_STRONG_N)['stock'].tolist()
print(f"\n[第 1 层筛选] {len(top_strong_codes)} 只强势股进入第 2 层")


# ---------- 第 2 层:强势股 vs 板块(通达信88 整体)指数 ----------
print("\n" + "=" * 70)
print(f"[第 2 层] 强势股 vs 板块(通达信88 整体等权指数)")
print("=" * 70)

# 计算"板块指数"= 88 只股票的等权日收益累计
df_layer1_close = pd.DataFrame({c: raw['Close'].reindex(df_idx.index)
                                 for c, raw in stock_data.items()
                                 if c in df_layer1['stock'].values})
df_layer1_close = df_layer1_close.dropna(how='all').ffill().bfill()
# 等权板块指数(每天所有 88 只股票的 close 中位数,抗个股缺失)
sector_index = df_layer1_close.median(axis=1)
sector_ret = recent_return(sector_index)
print(f"近 {WINDOW} 日通达信88 板块指数累计收益:{sector_ret:+.2%}")

layer2_rows = []
for code in top_strong_codes:
    raw = stock_data[code]
    df = raw.loc[TARGET_START:TARGET_END]
    r = recent_return(df['Close'])
    if np.isnan(r):
        continue
    alpha_vs_sector = r - sector_ret
    layer2_rows.append({
        'stock': code,
        'ret': r,
        'alpha_vs_idx': float(df_layer1.loc[df_layer1['stock'] == code, 'alpha_vs_idx'].iloc[0]),
        'alpha_vs_sector': alpha_vs_sector,
    })

df_layer2 = pd.DataFrame(layer2_rows).sort_values('alpha_vs_sector', ascending=False).reset_index(drop=True)
print(f"\nTop {TOP_STRONG_N} 强势股 × 相对板块排名(跑赢板块的强势个股):")
print(df_layer2.head(TOP_STRONG_N).to_string(index=False))


# ---------- 3. 验证:Top N 强势股 vs 全 88 板块(简单持有收益对比) ----------
print("\n" + "=" * 70)
print(f"[验证] Top 强势股 vs 88 整体(纯持有 {WINDOW} 日,不分时点)对比")
print("=" * 70)

# 整段持有收益(从 TARGET_START 到 TARGET_END)
def full_period_return(raw, code):
    df = raw.loc[TARGET_START:TARGET_END]
    if len(df) < 10:
        return None
    return float(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1)

# Top 强势股平均持有收益
top_avg = df_layer2['ret'].mean()
print(f"Top {len(df_layer2)} 强势股平均近 {WINDOW} 日收益:{top_avg:+.2%}")

# 全 88 股票平均持有收益
all_avg = df_layer1['ret'].mean()
print(f"全 88 股票平均近 {WINDOW} 日收益:{all_avg:+.2%}")
print(f"差值(Top 强 - 88 均):{(top_avg - all_avg)*100:+.2f} pp")

# 大盘对照
print(f"大盘 近 {WINDOW} 日收益:{idx_ret:+.2%}")


# ---------- 4. 保存结果 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'two_layer_strong_stocks_{SECTOR_NAME}_{TARGET_START}_{TARGET_END}.csv',
)
df_layer2.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\nTop 强势股排名已保存到 {out_csv}")

# 后续:可以用 tsfresh_vbt_combo.py 单独对每只强势股跑 walk-forward
print(f"\n[下一步] 对每只 Top 强势股,可以复用 tsfresh_vbt_combo.py 跑 walk-forward:")
print(f"  python backtrace/tsfresh_vbt_combo.py  (改 STOCK_CODE = '{df_layer2.iloc[0]['stock']}')")

tq.close()