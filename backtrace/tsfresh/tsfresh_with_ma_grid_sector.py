# tsfresh_with_ma 跨通达信88 板块验证
# 对每只股票跑 basic vs with_ma 两套方案,看 with_ma 是否稳定胜出
# 输出:tsfresh_with_ma_grid_<code>_<start>_<end>.csv
# 用法:板块层面跑一遍,验证 with_ma 方案在多数票上是否真的更稳(不是过拟合单只票)
import warnings
warnings.filterwarnings('ignore')

import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
SECTOR_NAME   = '通达信88'
TARGET_START  = '20250101'
TARGET_END    = '20251231'
INIT_CASH     = 100_000
MAX_POS_PCT   = 0.95

INIT_TRAIN_SIZE = 200
STEP = 50
ENTRY_TSF = 0.60
EXIT_TSF  = 0.50

CHANNELS_BASIC = ['Open', 'High', 'Low', 'Close', 'Volume']
CHANNELS_MA    = CHANNELS_BASIC + ['ma5', 'ma10', 'ma20', 'rel_ma5']
# ===================================


# ---------- 1. 拉全板块 ----------
print("=" * 70)
print(f"[{SECTOR_NAME}] 拉板块全部成员...")
stock_data = P.load_sector(sector_name=SECTOR_NAME, verbose=True)
print("=" * 70)
if not stock_data:
    print("❌ 板块数据为空"); tq.close(); raise SystemExit(1)

# ---------- 2. 对每只股票跑 2 方案 ----------
results = []
total = len(stock_data)
t_start = time.time()

for i, (code, raw) in enumerate(stock_data.items(), 1):
    # Walkforward 需要足够数据(>250天),先跑在完整 500 天上
    df_full = W.add_ma_channels(raw.copy())

    # 2025 切片用于回测(243天),需确保足够长
    df_demo = df_full.loc[TARGET_START:TARGET_END].copy()
    if len(df_demo) < 100:
        print(f"  [{i:2d}/{total}] {code} 回测区间不足 ({len(df_demo)} 天),跳过")
        continue

    print(f"  [{i:2d}/{total}] {code} (全 {len(df_full)} 天 / 回测 {len(df_demo)} 天) ...")
    rec = {'stock': code, 'n_days': len(df_demo)}

    # basic
    try:
        proba_basic, _ = W.tsfresh_walkforward_proba(
            df_full, CHANNELS_BASIC,
            init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=False)
        entries, exits = B.build_proba_signals(
            proba_basic, df_demo.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF)
        r_basic = B.run_vbt_backtest(
            df_demo, entries, exits, code,
            init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
            print_rejection_warning=False)
        rec['basic_net_ret'] = r_basic['net_ret']
        rec['basic_trades']  = r_basic['trades']
        rec['basic_winrate'] = r_basic['win_rate']
    except Exception as e:
        print(f"     basic 失败:{e}")
        rec['basic_net_ret'] = None

    # with_ma
    try:
        proba_ma, _ = W.tsfresh_walkforward_proba(
            df_full, CHANNELS_MA,
            init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=False)
        entries, exits = B.build_proba_signals(
            proba_ma, df_demo.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF)
        r_ma = B.run_vbt_backtest(
            df_demo, entries, exits, code,
            init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
            print_rejection_warning=False)
        rec['with_ma_net_ret'] = r_ma['net_ret']
        rec['with_ma_trades']  = r_ma['trades']
        rec['with_ma_winrate'] = r_ma['win_rate']
    except Exception as e:
        print(f"     with_ma 失败:{e}")
        rec['with_ma_net_ret'] = None

    rec['with_ma_wins'] = (
        (rec.get('with_ma_net_ret') or -1) > (rec.get('basic_net_ret') or -1)
        if rec.get('with_ma_net_ret') is not None and rec.get('basic_net_ret') is not None
        else None
    )
    results.append(rec)

    elapsed = time.time() - t_start
    eta = elapsed / i * (total - i)
    print(f"     basic={rec.get('basic_net_ret', 'NA')}, with_ma={rec.get('with_ma_net_ret', 'NA')},"
          f" 胜={rec.get('with_ma_wins', 'NA')} | 累计 {elapsed:.0f}s,ETA {eta:.0f}s")

# ---------- 3. 汇总 ----------
df_res = pd.DataFrame(results)
df_res['alpha_diff_pp'] = (df_res['with_ma_net_ret'] - df_res['basic_net_ret']) * 100

print("\n" + "=" * 90)
print("=== 88 板块 basic vs with_ma 汇总 ===")
print("=" * 90)
both_valid = df_res.dropna(subset=['basic_net_ret', 'with_ma_net_ret'])
print(f"有效股票数:{len(both_valid)} / {len(df_res)}")
print(f"with_ma 胜出(>basic):{(both_valid['with_ma_wins'] == True).sum()} 只"
      f"  ({(both_valid['with_ma_wins'] == True).mean():.1%})")
print(f"with_ma 跑赢平均提升:{both_valid['alpha_diff_pp'].mean():+.2f} pp(中位数 {both_valid['alpha_diff_pp'].median():+.2f} pp)")
print(f"\n平均净收益:basic={both_valid['basic_net_ret'].mean():.2%}  with_ma={both_valid['with_ma_net_ret'].mean():.2%}")
print(f"中位净收益:basic={both_valid['basic_net_ret'].median():.2%}  with_ma={both_valid['with_ma_net_ret'].median():.2%}")

# Top 10 with_ma 表现最好
top = both_valid.sort_values('with_ma_net_ret', ascending=False).head(10)
print("\n=== Top 10 with_ma 净收益最高 ===")
print(top[['stock', 'basic_net_ret', 'with_ma_net_ret', 'alpha_diff_pp']].to_string(index=False))

# 保存
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_with_ma_sector_{SECTOR_NAME}_{TARGET_START}_{TARGET_END}.csv',
)
df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
