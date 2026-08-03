# vbt 回测,按江海证券真实手续费率扣费(佣金 + 印花税 + 沪市过户费)
# 资金账号 / 密码 不在本脚本
# 输出:vbt_jhzq_<code>_<start>_<end>_trades.csv(trades 表 + 5 列扣费字段)
# 用法:与 vbt_simple_backtest.py 对照 — 验证"真实扣费 vs 零摩擦"差距有多大
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import vectorbt as vbt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
from tqcenter import tq
from common import tsfresh_pipeline as P
from common import jhzq_fees as F

tq.initialize(__file__)

# =============== 配置 ===============
STOCK_CODE   = '688318.SH'       # 单只沪市股(有过户费)
TARGET_START = '20250701'
TARGET_END   = '20251231'
WINDOW       = 5                 # MA5
INIT_CASH    = 100000
MAX_POS_PCT  = 0.95              # 单次最大占用 95% 资金,留 buffer 给手续费
# =====================================

# 1. 加载 K 线
print(f"[{STOCK_CODE}] 加载数据(TQ → CSV 回退)...")
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print(f"❌ {STOCK_CODE} 数据缺失")
    tq.close(); raise SystemExit(1)
df = df.loc[TARGET_START:TARGET_END].copy()
if len(df) == 0:
    print(f"❌ 区间 {TARGET_START}~{TARGET_END} 无行情")
    tq.close(); raise SystemExit(1)

print(f"回测区间:{df.index[0].date()} → {df.index[-1].date()}  ({len(df)} 交易日)\n")
close, openp = df['Close'], df['Open']

# 2. MA5 信号(shift(1) 规避未来函数;VBT upon_long_conflict 自动防同 K 线连开)
ma = vbt.MA.run(close, window=WINDOW).ma.ffill()
entries = close.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
exits   = close.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
print(f"信号:买入 {entries.sum()} 次,卖出 {exits.sum()} 次")

# 3. vbt 回测(fees=0,所有手续费后置统一扣)
#    按初始开盘价 × 95% / 100 取整,固定手数,后续信号都用同一手数
init_open   = float(openp.iloc[0])
shares_per_trade = int(np.floor(INIT_CASH * MAX_POS_PCT / init_open / 100) * 100)
print(f"\n[vbt] fees=0,每笔 {shares_per_trade} 股(= {shares_per_trade*init_open:.0f} 元,"
      f"占初始资金 {shares_per_trade*init_open/INIT_CASH:.1%})...")

portfolio = vbt.Portfolio.from_signals(
    close=close, entries=entries, exits=exits,
    price=openp,
    init_cash=INIT_CASH,
    fees=0,
    freq='D',
    size=shares_per_trade,
    size_type='amount',
    size_granularity=100,
    upon_long_conflict='exit',
)
print(f"  -> vbt 未扣费总收益 {portfolio.total_return():.2%}")

# 4. 后置扣费
trades_gross = portfolio.trades.records_readable
if len(trades_gross) == 0:
    print("\n[WARN] 无成交,退出"); tq.close(); raise SystemExit(0)

print(f"\n[费后调整] 佣金 万 0.85(免5) + 印花税 卖出 万 5 + 沪市过户费 万 0.1...")
trades_net = F.adjust_trades_pnl(trades_gross, STOCK_CODE)

# 5. 汇总
summary = F.summary_after_fees(trades_gross, STOCK_CODE)
print("\n" + "=" * 70)
print("=== 江海证券真实费率回测结果 ===")
print("=" * 70)
mkt = "沪市,双向过户费" if F.is_sh(STOCK_CODE) else "深市,无过户费"
print(f"标的:{STOCK_CODE}  ({mkt})")
print(f"区间:{TARGET_START} → {TARGET_END}  {len(df)} 交易日")
print(f"策略:MA{WINDOW}  次日开盘成交  单次仓位上限 {MAX_POS_PCT:.0%}\n")

print(f"成交笔数:          {summary['trades']}")
print(f"未扣费毛收益:      {summary['gross_pnl']:>12,.2f} 元")
print(f"合计印花税:        -{summary['total_stamp']:>10,.2f} 元")
print(f"合计过户费:        -{summary['total_transfer']:>10,.2f} 元")
print(f"全费用后净收益:    {summary['net_pnl']:>12,.2f} 元")
print(f"单笔平均净盈亏:    {summary['avg_net_per_trade']:>12,.2f} 元")

# 净收益率(分母 = INIT_CASH,与 vbt total_return() 同口径)
real_net_ret = summary['net_pnl'] / INIT_CASH
print(f"\n净收益率(分母 INIT_CASH={INIT_CASH:,}): {real_net_ret:.2%}")

# 6. 对比 vbt 默认(双边 万 10)
print("\n=== 对比:vbt 默认双边 万 10 vs 江海真实费率 ===")
portfolio_default = vbt.Portfolio.from_signals(
    close=close, entries=entries, exits=exits,
    price=openp, init_cash=INIT_CASH, fees=0.001,
    freq='D',
    size=shares_per_trade,
    size_type='amount',
    size_granularity=100,
    upon_long_conflict='exit',
)
default_ret = portfolio_default.total_return()
diff_pp = (real_net_ret - default_ret) * 100
print(f"vbt 默认 万 10 收益率: {default_ret:.2%}")
print(f"江海真实扣费收益率:    {real_net_ret:.2%}")
print(f"差异(江海 - 默认):    {diff_pp:+.2f} 个百分点")

# 7. 保存扣费明细
out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
                       f'vbt_jhzq_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}_trades.csv')
trades_net.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n扣费明细已保存到 {out_csv}")

# 8. vbt 自带统计(参考)
print("\n=== vbt 自带统计(只扣了双边佣金 0) ===")
print(portfolio.stats())

tq.close()
