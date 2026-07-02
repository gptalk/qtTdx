# 用 1 日窗口找 Top 1 行业 → 双重跑赢(>大盘 & >板块)成分股 → tsfresh walk-forward
# 验证:对"短期最强"的股,tsfresh 模型能否预测其中长期走势
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from datetime import datetime
from tqcenter import tq

import tsfresh_config as C
import tsfresh_pipeline as P
import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
INDEX_CODE      = '000001.SH'
SCAN_WINDOW     = 5                    # 最近 5 个交易日扫描候选股(union,避免单日样本太少)
ALPHA_WINDOW    = 1                    # α 计算窗口(1 日)
INIT_TRAIN_SIZE = 200
STEP            = 50
ENTRY_TSF       = 0.60
EXIT_TSF        = 0.50
INIT_CASH       = 100_000
TARGET_START    = '20210101'           # walk-forward 回测区间
TARGET_END      = '20251231'
# ===================================


# ---------- 1. 拉大盘 + 128 缺省行业 + 全成分股 5 年 close ----------
print("=" * 70)
print("[1/4] 拉数据...")
df_idx = P.load_ohlcva(INDEX_CODE, verbose=False)
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

# 缓存成分股
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
print(f"全部去重成分股: {len(all_member_codes)} 只")

df_stocks = tq.get_market_data(
    field_list=['Close'],
    stock_list=all_member_codes,
    start_time=TARGET_START, end_time=TARGET_END,
    dividend_type='front', period='1d', fill_data=True,
)
stocks_close_df = df_stocks['Close']
print(f"个股 close 矩阵: {stocks_close_df.shape}")

# 对齐索引
common_idx = industry_close_df.index.intersection(stocks_close_df.index).intersection(df_idx.index)
industry_close_df = industry_close_df.loc[common_idx]
stocks_close_df = stocks_close_df.loc[common_idx]
df_idx = df_idx.loc[common_idx]
print(f"共同交易日: {len(common_idx)}  ({common_idx[0].date()} ~ {common_idx[-1].date()})")
tq.close()


# ---------- 2. 扫描最近 N 天,找"双重跑赢"候选股 ----------
print("\n" + "=" * 70)
print(f"[2/4] 扫描最近 {SCAN_WINDOW} 个交易日,找双重跑赢候选股")
print("=" * 70)

candidate_stocks = {}   # {stock: [data_points]} 用于后续分析
idx_close = df_idx['Close']

scan_dates = common_idx[-SCAN_WINDOW:]   # 最近 N 个交易日
print(f"扫描日期: {scan_dates[0].date()} ~ {scan_dates[-1].date()}")

for scan_date in scan_dates:
    pos = common_idx.get_loc(scan_date)
    if pos < ALPHA_WINDOW:
        continue
    # 1) 找 1 日 α 最大的行业
    ind_ret = industry_close_df.iloc[pos] / industry_close_df.iloc[pos - ALPHA_WINDOW] - 1
    top1 = ind_ret.idxmax()
    top1_ret = ind_ret.loc[top1]
    # 2) 大盘 1 日收益
    idx_ret = float(idx_close.iloc[pos] / idx_close.iloc[pos - ALPHA_WINDOW] - 1)
    # 3) 该行业成分股,筛"双重跑赢"
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
        # 双重跑赢:股票 > 大盘 AND 股票 > 行业
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


def tsfresh_walkforward(df_stock, label):
    """单只股:tsfresh + walk-forward predict → proba Series"""
    long_df = P.to_long_format(df_stock, channels=['Open', 'High', 'Low', 'Close', 'Volume'],
                               id_value=label)
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, df_stock['Close'].values, verbose=False)

    X_sel = P.select_relevant(X_all, y_all, verbose=False)
    if X_sel.shape[1] == 0:
        X_sel = X_all   # 兜底

    date_index = pd.DatetimeIndex(df_stock.index)
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
    return proba[~proba.index.duplicated(keep='last')].dropna()


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

    proba = tsfresh_walkforward(df, code)
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
    os.path.dirname(__file__),
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