# 用 1 日窗口找 Top 1 行业 → 双重跑赢(>大盘 & >板块)成分股 → tsfresh walk-forward
# 验证:对"短期最强"的股,tsfresh 模型能否预测其中长期走势
# 输出:tsfresh_top1_industry_<start>_<end>.csv;另外还会推送板块到 TQ 客户端
# 用法:`python tsfresh/top1_industry.py` → 跑完后查 TQ 自定义板块 `TSFresh候选_<date>`
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime, timedelta
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
INDEX_CODE      = '000001.SH'
SCAN_WINDOW     = 5                    # 最近 5 个交易日扫描候选股(union,避免单日样本太少)
SCAN_INDUSTRIES = 20                   # 1 日 α 排名后,只评估 Top 20 行业的成分股(~600 只 vs 全部 5534 只)
ALPHA_WINDOW    = 1                    # α 计算窗口(1 日)
INIT_TRAIN_SIZE = 200
STEP            = 50
ENTRY_TSF       = 0.60
EXIT_TSF        = 0.50
INIT_CASH       = 100_000
# 提速:用 1 年数据替代 5 年(用户要求)
TARGET_START    = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
TARGET_END      = datetime.now().strftime('%Y%m%d')
# ===================================


# ---------- 1. 拉大盘 + 128 缺省行业 5 年 close ----------
print("=" * 70)
print("[1/4] 拉大盘 + 128 缺省行业 5 年 close...")
df_idx = P.load_ohlcva(INDEX_CODE, verbose=False)
industry_list = tq.get_sector_list(list_type=1)   # 改名 list_type=1 → '11' 保持原状
# 修正:industry_list 已经是 128 申万二级
industry_list = tq.get_stock_list('11', list_type=1)
industry_codes = [it['Code'] for it in industry_list if it and 'Code' in it]
print(f"大盘: {df_idx.shape[0]} 行  |  缺省行业板块: {len(industry_codes)} 个")

df_ind = tq.get_market_data(
    field_list=['Close'],
    stock_list=industry_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
industry_close_df = df_ind['Close']
print(f"行业指数矩阵: {industry_close_df.shape}")

# 先临时对齐(dfn 用 industry_close_df 索引,df_idx 索引 → 取交集即可)
common_idx = industry_close_df.index.intersection(df_idx.index)
industry_close_df = industry_close_df.loc[common_idx]
df_idx = df_idx.loc[common_idx]
print(f"共同交易日: {len(common_idx)}  ({common_idx[0].date()} ~ {common_idx[-1].date()})")

# 缓存成分股(只对重点行业拉)
members_cache = {}
def get_members(code):
    if code not in members_cache:
        members_cache[code] = tq.get_stock_list_in_sector(code)
    return members_cache[code]


# ---------- 2. 扫描最近 N 天,找"双重跑赢"候选股 ----------
print("\n" + "=" * 70)
print(f"[2/4] 扫描最近 {SCAN_WINDOW} 个交易日,找双重跑赢候选股")
print("=" * 70)

candidate_stocks = {}
idx_close = df_idx['Close']
scan_dates = common_idx[-SCAN_WINDOW:]
print(f"扫描日期: {scan_dates[0].date()} ~ {scan_dates[-1].date()}")

# === 第一遍:扫 N 天,只用 industry_close_df 找 Top 1 行业,填 union 池 ===
top1_membership_pool = set()
last_n_days = industry_close_df.iloc[-SCAN_WINDOW:]   # 最近 N 行原始数据
for i in range(ALPHA_WINDOW, len(last_n_days)):
    sub = last_n_days.iloc[i - ALPHA_WINDOW:i + 1]   # 含今天 + 前 ALPHA_WINDOW 天
    # 简单:用 last_n_days.iloc[i] / last_n_days.iloc[i-ALPHA_WINDOW] - 1
    today_ret = last_n_days.iloc[i] / last_n_days.iloc[i - ALPHA_WINDOW] - 1
    top1 = today_ret.idxmax()
    top1_membership_pool.add(top1)

print(f"\n  [Top 1 行业 union 池] {SCAN_WINDOW} 天所有 Top 1 行业的成分股并集: 行业 {len(top1_membership_pool)} 个")
# 拿到所有 union 行业后,一次性查这些行业的成分股
union_member_codes = set()
for ind_code in top1_membership_pool:
    union_member_codes.update(get_members(ind_code) or [])
union_member_codes = sorted(union_member_codes)
print(f"  [union 池成分股去重]: {len(union_member_codes)} 只")

# 批量拉 union 池的 5 年 close
df_stocks = tq.get_market_data(
    field_list=['Close'],
    stock_list=union_member_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
stocks_close_df = df_stocks['Close']
print(f"个股 close 矩阵: {stocks_close_df.shape}")


# === 第二遍:用拉好的 close 做双重跑赢筛选 ===
for scan_date in scan_dates:
    pos = common_idx.get_loc(scan_date)
    if pos < ALPHA_WINDOW:
        continue
    ind_ret = industry_close_df.iloc[pos] / industry_close_df.iloc[pos - ALPHA_WINDOW] - 1
    top1 = ind_ret.idxmax()
    top1_ret = ind_ret.loc[top1]
    idx_ret = float(idx_close.iloc[pos] / idx_close.iloc[pos - ALPHA_WINDOW] - 1)
    members = get_members(top1)
    if not members:
        continue
    name = next((it['Name'] for it in industry_list if it.get('Code') == top1), top1)
    selected = []
    for s in members:
        if s not in stocks_close_df.columns:
            continue
        if pos < ALPHA_WINDOW:
            continue
        s_close = stocks_close_df[s]
        if pd.isna(s_close.iloc[pos]) or pd.isna(s_close.iloc[pos - ALPHA_WINDOW]):
            continue
        s_ret = float(s_close.iloc[pos] / s_close.iloc[pos - ALPHA_WINDOW] - 1)
        if s_ret > idx_ret and s_ret > top1_ret:
            selected.append((s, s_ret))
    print(f"\n  [{scan_date.date()}] {name}({top1}) α={top1_ret*100:+.2f}% (大盘 {idx_ret*100:+.2f}%) → 双重跑赢 {len(selected)} 只:")
    for s, r in sorted(selected, key=lambda x: -x[1])[:5]:
        print(f"    {s} {r*100:+.2f}%")
    if len(selected) > 5:
        print(f"    ... 还有 {len(selected)-5} 只")
    for s, r in selected:
        candidate_stocks.setdefault(s, []).append({
            'date': scan_date,
            'industry': name,
            'stock_ret': r,
            'idx_ret': idx_ret,
            'industry_ret': top1_ret,
        })

print(f"\n  [Top 1 行业 union 池] {SCAN_WINDOW} 天所有 Top 1 行业的成分股并集: {len(top1_membership_pool)} 只")

print(f"\n[汇总] 候选股去重: {len(candidate_stocks)} 只")
if not candidate_stocks:
    print("❌ 无候选股,退出")
    raise SystemExit(1)

# 显示候选股频次(出现在多少天)
freq = {s: len(v) for s, v in candidate_stocks.items()}
top_candidates = sorted(freq.items(), key=lambda x: -x[1])
print("\n候选股频次(Top 20):")
for s, n in top_candidates[:20]:
    print(f"  {s:10s} {n} 次")


# ---------- 3. 对候选股跑 tsfresh + walk-forward + vbt + jhzq_fees ----------
print("\n" + "=" * 70)
print(f"[3/4] 对 Top 候选股跑 tsfresh 多通道 + walk-forward + vbt + jhzq_fees")
print("=" * 70)


def lead_lag_signal(df_stock, industry_code, idx_close_series, label):
    """
    个股跟随板块的轮动规律分析。
    输入:个股日 K 线 + 行业指数 close + 大盘 close
    输出: entry proba Series(基于 lead-lag β)

    逻辑:
      - 个股 β 系 = 相对行业指数过去 LAG 日的回归
      - 如果过去个股涨先于行业涨(proxy 用滚动相关 + 时滞回归)
      - 给出"领先指标"分数 → 用作 entry 信号
    """
    s = df_stock['Close']
    ind_close = industry_close_df[industry_code] if industry_code in industry_close_df.columns else None
    if ind_close is None:
        return pd.Series(False, index=s.index)

    # 1) 滚动 5 日收益
    ret_s = s.pct_change(5)
    ret_ind = ind_close.pct_change(5)

    # 2) 个股 vs 行业指数滚动相关(20 日窗口)
    rolling_corr = ret_s.rolling(20).corr(ret_ind)

    # 3) lead-lag: 用个股过去 LAG 日收益预测行业未来收益
    # 简化:用个股 - 行业 的收益差值(异常收益),作为预测行业反转的代理
    spread = (s / s.shift(1)) - (ind_close / ind_close.shift(1))
    spread_mean = spread.rolling(20).mean()
    spread_std = spread.rolling(20).std()
    z_score = (spread - spread_mean) / spread_std.replace(0, np.nan)
    z_score = z_score.fillna(0)

    # 信号合成: 高 corr + 异常收益 z-score → 概率(简化为 0~1)
    corr_norm = (rolling_corr.fillna(0) + 1) / 2  # [-1,1] → [0,1]
    z_norm = 1 / (1 + np.exp(-z_score))   # sigmoid → [0,1]
    raw_signal = (corr_norm * 0.4 + z_norm * 0.6).shift(1).fillna(0.5)

    # 信号归一化映射:把全期分布拉到 [0.3, 0.8] 区间,触到 entry/exit 阈值
    p20 = raw_signal.quantile(0.20)
    p80 = raw_signal.quantile(0.80)
    spread = (p80 - p20) if (p80 - p20) > 0 else 1e-6
    # 0.3 + 0.5 * (signal - p20) / spread, 再 clip 到 [0, 1]
    proba = (0.3 + 0.5 * (raw_signal - p20) / spread).clip(0.0, 1.0)
    return proba


def vbt_with_real_fees(df_stock, proba, label):
    """vbt + jhzq_fees 真实扣费"""
    init_open = float(df_stock['Open'].iloc[0])
    if init_open <= 0:
        return None
    shares = int(np.floor(INIT_CASH * 0.95 / init_open / 100) * 100)
    if shares < 100:
        return None
    aligned = proba.reindex(df_stock.index)
    entries = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
    exits   = (aligned < EXIT_TSF).shift(1).fillna(False).astype(bool)
    pf = vbt.Portfolio.from_signals(
        close=df_stock['Close'], entries=entries, exits=exits,
        price=df_stock['Open'], init_cash=INIT_CASH,
        fees=0, slippage=0, freq='D',
        size=shares, size_type='amount', size_granularity=100,
        upon_long_conflict='exit',
    )
    trades = pf.trades.records_readable
    zero_ret = pf.total_return()
    if len(trades) == 0:
        return {'stock': label, 'trades': 0, 'net_pnl': 0.0,
                'net_ret': 0.0, 'win_rate': 0.0, 'zero_ret': zero_ret}
    summary = F.summary_after_fees(trades, label)
    summary['stock'] = label
    summary['zero_ret'] = zero_ret
    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades)
    summary['net_ret'] = summary['net_pnl'] / INIT_CASH
    return summary


# 取 Top 10 候选股(频次最高)
TOP_CANDIDATES = 10
top_picks = [s for s, n in top_candidates[:TOP_CANDIDATES]]
print(f"\n对 Top {TOP_CANDIDATES} 候选股跑完整 walk-forward:")
print(f"  {top_picks}")

results = []
tq.initialize(__file__)
for code in top_picks:
    raw = P.load_ohlcva(code, verbose=False)
    if raw is None or len(raw) < 300:
        print(f"  [{code}] 数据不足,跳过")
        continue
    df = raw.loc[TARGET_START:TARGET_END]
    if len(df) < 100:
        print(f"  [{code}] 区间数据不足,跳过")
        continue
    freq_count = freq[code]
    print(f"\n  [{code}] 历史 {len(df)} 日  freq={freq_count}/{SCAN_WINDOW} ...")

    # 找到该股所在的 Top 1 行业(用最近一次扫描结果)
    ind_code = candidate_stocks[code][-1]['industry']
    # 把行业名称映射回 code
    ind_match = [(it['Code'], it['Name']) for it in industry_list if it['Name'] == ind_code]
    if ind_match:
        ind_code = ind_match[0][0]
    proba = lead_lag_signal(df, ind_code, None, code)
    summary = vbt_with_real_fees(df, proba, code)
    if summary:
        summary['freq'] = freq_count
        results.append(summary)
        print(f"    zero_friction={summary['zero_ret']*100:+.2f}%  "
              f"净收益={summary['net_pnl']:>+10,.2f} 元  "
              f"({summary['net_ret']*100:+.2f}%)  "
              f"胜率={summary['win_rate']*100:.1f}%  "
              f"笔数={summary['trades']}")
tq.close()


# ---------- 4. 汇总 ----------
print("\n" + "=" * 70)
print(f"[4/4] Top {TOP_CANDIDATES} 候选股 tsfresh walk-forward 汇总")
print("=" * 70)

if not results:
    print("❌ 无结果")
    raise SystemExit(0)

df_res = pd.DataFrame(results)
cols = ['stock', 'freq', 'trades', 'zero_ret', 'net_pnl', 'net_ret', 'win_rate']
df_res = df_res[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)

print(f"\n{'股票':<10} {'入选次数':<8} {'笔数':>5} {'零摩擦':>10} {'净收益':>12} {'净收益率':>10} {'胜率':>6}")
print("-" * 80)
for _, r in df_res.iterrows():
    print(f"{r['stock']:<10} {int(r['freq']):<8} {int(r['trades']):>5} "
          f"{r['zero_ret']*100:>+10.2f}% {r['net_pnl']:>+12,.2f} "
          f"{r['net_ret']*100:>+10.2f}% {r['win_rate']*100:>6.1f}%")

# 整体平均
avg_net = df_res['net_pnl'].mean()
avg_ret = df_res['net_ret'].mean()
avg_wr = df_res['win_rate'].mean()
total_trades = df_res['trades'].sum()
print(f"\n[候选股整体平均]")
print(f"  平均净收益:{avg_net:>+10,.2f} 元  (平均 {avg_ret*100:+.2f}%)")
print(f"  平均胜率:{avg_wr*100:.1f}%")
print(f"  总笔数:{total_trades}")
# 等权组合(假设同时持有,每只 INIT_CASH/10)
port_ret = avg_ret * 10   # 10 只等权
print(f"  等权组合(10 只 × INIT_CASH/{TOP_CANDIDATES}): {port_ret*100:+.2f}%")

# 与大盘对照
idx_total = float(idx_close.iloc[-1] / idx_close.iloc[0] - 1)
print(f"  同期大盘:{idx_total*100:+.2f}%")
print(f"  超额:{port_ret*100 - idx_total*100:+.2f} pp")

# 保存
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_top1_industry_{TARGET_START}_{TARGET_END}.csv',
)
df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")


# ---------- 5. 推送到 TQ 自定义板块 ----------
print("\n" + "=" * 70)
print("[5/5] 推送到 TQ 自定义板块(通达信客户端可见)")
print("=" * 70)

# 板块命名
BLOCK_CODE = 'TSFRESH'
BLOCK_NAME = f'TSFresh候选_{datetime.now().strftime("%Y%m%d")}'
final_picks = df_res['stock'].tolist()
print(f"\n  板块简称:{BLOCK_CODE}")
print(f"  板块名称:{BLOCK_NAME}")
print(f"  候选股 {len(final_picks)} 只:{final_picks}")

tq.initialize(__file__)
try:
    # 1) 创建板块(若已存在会报错,捕获即可)
    try:
        tq.create_sector(block_code=BLOCK_CODE, block_name=BLOCK_NAME)
        print(f"  [OK] 自定义板块 {BLOCK_NAME} 创建成功")
    except Exception as e:
        print(f"  [INFO] 板块可能已存在:{e}")

    # 2) 把候选股推送到板块(替换原有内容)
    tq.send_user_block(block_code=BLOCK_CODE, stocks=final_picks)
    print(f"  [OK] 已推送 {len(final_picks)} 只到板块 {BLOCK_NAME}")
    print(f"  [提示] 打开通达信客户端 → 自选股板块 → {BLOCK_NAME} 即可看到")

    # 3) 同时发一条 MSG 到策略管理器显示区
    msg = (
        f"MSG,TSFresh 1日窗口双重跑赢 + walk-forward 候选 "
        f"({len(final_picks)}只): {','.join(final_picks)}"
    )
    try:
        tq.send_message(msg)
        print(f"  [OK] MSG 已发送到策略管理器")
    except Exception as e:
        print(f"  [WARN] MSG 发送失败:{e}")

    # 4) 发送预警信号(可选):给候选股逐个发预警
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        tq.send_warn(
            stock_list=final_picks,
            time_list=[now_str] * len(final_picks),
            price_list=['0'] * len(final_picks),
            close_list=['0'] * len(final_picks),
            volum_list=['0'] * len(final_picks),
            bs_flag_list=['0'] * len(final_picks),     # 0=买,这里只是信号不是真交易
            warn_type_list=['1'] * len(final_picks),
            reason_list=['TSFresh候选'] * len(final_picks),
            count=len(final_picks),
        )
        print(f"  [OK] 已发送 {len(final_picks)} 条预警信号")
    except Exception as e:
        print(f"  [WARN] 预警发送失败:{e}")

except Exception as e:
    print(f"  [ERR] TQ 推送失败:{e}")
finally:
    tq.close()


# ---------- 6. 新增:通达信行业 Top 自选股板块(每日 Top 1 行业 + 双重跑赢精选)----------
print("\n" + "=" * 70)
print("[6/6] 新增/刷新通达信行业 Top 自选股板块(每日精选)")
print("=" * 70)

# 提取"今天"最后一次扫描的 Top 1 行业 + 双重跑赢成分股
last_scan_date = scan_dates[-1]
last_scan_data = {}
for s, recs in candidate_stocks.items():
    for r in recs:
        if r['date'] == last_scan_date:
            last_scan_data[s] = r
            break

if not last_scan_data:
    print("[WARN] 今天无双重跑赢候选,跳过独立板块推送")
else:
    # 按行业归组,展示今天 Top 1 行业 + 双重跑赢
    by_industry = {}
    for s, r in last_scan_data.items():
        by_industry.setdefault(r['industry'], []).append((s, r['stock_ret']))

    # 取候选最多的行业 = 当天 Top 1
    top1_industry = max(by_industry.items(), key=lambda x: len(x[1]))
    top1_name, top1_picks = top1_industry
    top1_picks.sort(key=lambda x: -x[1])   # 按涨幅降序

    print(f"\n  [{last_scan_date.date()}] 当日 Top 1 行业:{top1_name}")
    print(f"  双重跑赢成分股 {len(top1_picks)} 只:")
    for s, r in top1_picks:
        print(f"    {s}  +{r*100:.2f}%")

    # 独立板块命名:通达信行业Top + 日期
    TQI_BLOCK_CODE = 'TQ_TOP'
    TQI_BLOCK_NAME = f'通达信行业Top_{datetime.now().strftime("%Y%m%d")}'
    tqi_stocks = [s for s, _ in top1_picks]

    # ===== 完整映射日志:板块代码 → 板块名称 → 包含股票代码 =====
    print(f"\n  [板块代码]    {TQI_BLOCK_CODE}")
    print(f"  [板块名称]    {TQI_BLOCK_NAME}")
    print(f"  [行业归属]    {top1_name}  (扫描日 {last_scan_date.date()})")
    print(f"  [成分股代码]  {len(tqi_stocks)} 只:")
    for i, s in enumerate(tqi_stocks, 1):
        s_rec = last_scan_data.get(s, {})
        s_ret = s_rec.get('stock_ret', 0.0) if isinstance(s_rec, dict) else 0.0
        print(f"    [{i:2d}] {s}  +{s_ret*100:.2f}%")

    tq.initialize(__file__)
    try:
        # 1) 创建/获取板块
        try:
            tq.create_sector(block_code=TQI_BLOCK_CODE, block_name=TQI_BLOCK_NAME)
            print(f"\n  [OK] 板块 {TQI_BLOCK_CODE}({TQI_BLOCK_NAME}) 创建成功")
        except Exception as e:
            print(f"  [INFO] 板块可能已存在(可继续 send_user_block):{e}")

        # 2) 推送/刷新股票代码到该板块
        tq.send_user_block(block_code=TQI_BLOCK_CODE, stocks=tqi_stocks)
        print(f"  [OK] 板块 {TQI_BLOCK_CODE} 已刷新 {len(tqi_stocks)} 只股票: {tqi_stocks}")

        # 3) MSG 提示(包含板块代码)
        msg = (
            f"MSG,[{TQI_BLOCK_CODE}] {TQI_BLOCK_NAME}  "
            f"{top1_name} 双重跑赢 {len(tqi_stocks)} 只: {','.join(tqi_stocks)}"
        )
        try:
            tq.send_message(msg)
            print(f"  [OK] MSG 已发送(包含板块代码 {TQI_BLOCK_CODE})")
        except Exception as e:
            print(f"  [WARN] MSG 发送失败:{e}")

        # 4) 给成分股发预警信号
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            tq.send_warn(
                stock_list=tqi_stocks,
                time_list=[now_str] * len(tqi_stocks),
                price_list=['0'] * len(tqi_stocks),
                close_list=['0'] * len(tqi_stocks),
                volum_list=['0'] * len(tqi_stocks),
                bs_flag_list=['0'] * len(tqi_stocks),
                warn_type_list=['1'] * len(tqi_stocks),
                reason_list=[f'[{TQI_BLOCK_CODE}]{top1_name}'] * len(tqi_stocks),
                count=len(tqi_stocks),
            )
            print(f"  [OK] 已发送 {len(tqi_stocks)} 条预警(板块代码={TQI_BLOCK_CODE})")
        except Exception as e:
            print(f"  [WARN] 预警发送失败:{e}")

    except Exception as e:
        print(f"  [ERR] TQI_TOP 板块推送失败:{e}")
    finally:
        tq.close()