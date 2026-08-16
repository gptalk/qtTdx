# -*- coding: utf-8 -*-
# dynamics_state_backtest.py — 按状态分布筛选股票 + vbt 回测 + IC 评估
#
# 目标(用户 Task 3):
#   把 stock 列表按"主导状态"(resonance / against / independent 等)分组,
#   每组跑 vbt 等权 basket 回测,看哪组 IC/收益更好。
#
# 流程:
#   1. 读 --input(股票列表 CSV,data/projection/stocks.csv)
#   2. 每只票:
#        load_pair → compute_movement_projection → compute_dynamics → classify_states
#        → 统计 state 分布(7 个状态 + none)
#   3. 每只票:state_prop = {state_name: 频率};dominant_state = argmax
#   4. 按 dominant_state 分组(可选:只在 3 大类 resonance / against / independent 中筛)
#   5. 每组:close_df 等权 basket → vbt.Portfolio.from_holding(buy-and-hold) → 收益/Sharpe
#   6. IC:每只票的 state_prop(向量)vs forward return(向量)→ 跨截面 Spearman
#
# 输出:
#   - data/dynamics/state_distribution.csv — 每只票的状态分布(每行一只票,每列一状态)
#   - data/dynamics/backtest_per_state.csv — 每组 basket 的总收益/Sharpe/MaxDD
#   - data/dynamics/state_ic.csv — 每个状态的 IC(Spearman + p-value)
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --limit 50
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_state_backtest.py --horizon 5 --target-states resonance,against
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
from collections import Counter
import numpy as np
import pandas as pd

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P
from projection._projection_core import (
    load_pair, compute_movement_projection, compute_dynamics, classify_states,
)
from dynamics import STATE_LABELS

CSV_OUT_DIR = 'data/dynamics'
DEFAULT_INPUT = 'data/projection/stocks.csv'

# 关注的 3 大状态(用户 Task 3 明确点名)
FOCUS_STATES = ('resonance', 'against', 'independent')


def parse_args():
    p = argparse.ArgumentParser(description='按状态分布筛选股票 + vbt basket 回测 + IC 评估')
    p.add_argument('--input', default=DEFAULT_INPUT,
                   help=f'股票列表 CSV。默认 {DEFAULT_INPUT}')
    p.add_argument('--days', type=int, default=240, help='回看天数。默认 240')
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0=全部')
    p.add_argument('--market-baseline', action='store_true',
                   help='回退大盘基线(默认走行业基线)')
    p.add_argument('--index', default=None,
                   help='强制指定基线指数')
    p.add_argument('--classify-thresholds', default='0.10,0.50,30,90',
                   help='状态分类阈值 R_low,R_high,theta_following_deg,theta_against_deg')
    p.add_argument('--lambda-q', type=float, default=-1.0,
                   help='λ_q;-1 走 median 自适应')
    p.add_argument('--target-states', default=','.join(FOCUS_STATES),
                   help=f'要纳入 IC/basket 回测的状态(逗号分隔)。默认 {",".join(FOCUS_STATES)}')
    p.add_argument('--min-prop', type=float, default=0.05,
                   help='纳入某状态组的最小 state 占比(dominant_state 必须 ≥ 此)。默认 0.05')
    p.add_argument('--init-cash', type=float, default=1e6, help='vbt 初始资金。默认 100 万')
    p.add_argument('--use-vbt', action='store_true',
                   help='跑 vbt 回测(默认先跑以验证);不传则跳过 vbt 只算 IC')
    return p.parse_args()


def load_stock_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'股票列表文件不存在: {path}\n请新建(最少一列 code),例:\n'
            f'  code,name\n  002475.SZ,立讯精密'
        )
    df = pd.read_csv(path, dtype={'code': str})
    if 'code' not in df.columns:
        raise ValueError(f'输入文件 {path} 必须有 code 列')
    names = df['name'] if 'name' in df.columns else [None] * len(df)
    return [
        (str(c).strip(), str(n).strip() if isinstance(n, str) else None)
        for c, n in zip(df['code'], names)
    ]


def process_one(stock_code, days, prefer_industry, index_code,
                lambda_q, classify_thresholds):
    """跑一只票的状态分类,返回 (state_props dict, dominant_state, total_days)。

    state_props: {state_name: 频率 ∈ [0, 1]},7 个 STATE_LABELS + 'none'
    dominant_state: argmax(state_props);空数据时为 None
    total_days: 状态序列长度
    """
    try:
        loaded = load_pair(stock_code, days, P,
                           prefer_industry=prefer_industry,
                           index_code=index_code, lag=0)
        mv = compute_movement_projection(loaded['stock_df'], loaded['index_df'])
        dyn = compute_dynamics(mv, lambda_q=lambda_q)
        r_low, r_high, theta_f, theta_a = classify_thresholds
        states = classify_states(dyn['R'], dyn['theta'], dyn['E_self'],
                                (r_low, r_high, theta_f, theta_a))
        if len(states) == 0:
            return None, None, 0
        counter = Counter(states)
        total = len(states)
        state_props = {s: counter.get(s, 0) / total for s in list(STATE_LABELS) + ['none']}
        dominant_state = max(state_props, key=state_props.get)
        return state_props, dominant_state, total
    except Exception:
        return None, None, 0


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    # 阈值
    try:
        R_LOW, R_HIGH, THETA_FOLLOWING_DEG, THETA_AGAINST_DEG = (
            float(x) for x in args.classify_thresholds.split(',')
        )
    except Exception as e:
        raise SystemExit(f'--classify-thresholds 解析失败: {args.classify_thresholds!r}\n{e}')
    classify_thresholds = (
        R_LOW, R_HIGH, np.deg2rad(THETA_FOLLOWING_DEG), np.deg2rad(THETA_AGAINST_DEG),
    )
    lambda_q = None if args.lambda_q < 0 else args.lambda_q
    prefer_industry = not args.market_baseline
    target_states = tuple(s.strip() for s in args.target_states.split(',') if s.strip())

    stock_list = load_stock_list(args.input)
    if args.limit > 0:
        stock_list = stock_list[:args.limit]

    print(f'输入: {args.input} ({len(stock_list)} 只)')
    print(f'回看天数: {args.days} | 目标状态: {target_states}')
    print(f'分类阈值: R<{R_LOW}/{R_HIGH}, θ<{THETA_FOLLOWING_DEG}°/>{THETA_AGAINST_DEG}°')
    print(f'λ_q={"median 自适应" if lambda_q is None else f"{lambda_q:.4e}"}')
    print(f'输出: {CSV_OUT_DIR}/\n')

    # 1. 每只票的状态分布
    rows = []
    fail = 0
    for i, (code, name) in enumerate(stock_list, 1):
        props, dom, total = process_one(
            code, args.days, prefer_industry, args.index,
            lambda_q, classify_thresholds,
        )
        if props is None:
            fail += 1
            print(f'[{i}/{len(stock_list)}] {code} ✗ 状态分类失败')
            continue
        row = {'code': code, 'name': name or '', 'dominant_state': dom, 'total_days': total}
        row.update(props)
        rows.append(row)
        print(
            f'[{i}/{len(stock_list)}] {code} dom={dom:11s} '
            f'resonance={props["resonance"]:.0%} '
            f'against={props["against"]:.0%} '
            f'independent={props["independent"]:.0%} '
            f'({total}d)'
        )

    if not rows:
        print('没有任何股票跑出状态分类,终止')
        return
    state_df = pd.DataFrame(rows)
    state_path = os.path.join(CSV_OUT_DIR, 'state_distribution.csv')
    state_df.to_csv(state_path, index=False, encoding='utf-8')
    print(f'\n=== 状态分布 ===')
    print(f'  总计: {len(state_df)} 只(失败 {fail})')
    print(f'  dominant_state 计数:')
    print(state_df['dominant_state'].value_counts().to_string(header=False))
    print(f'  清单: {state_path}')

    # 2. 按 dominant_state 分组 + 每组 vbt basket
    if args.use_vbt:
        try:
            import vectorbt as vbt
        except ImportError:
            print('⚠ vbt 未安装,跳过 basket 回测')
            vbt = None
    else:
        vbt = None

    # 每组 stock 列表
    groups = {s: [] for s in target_states}
    for _, r in state_df.iterrows():
        if r['dominant_state'] in target_states and r[r['dominant_state']] >= args.min_prop:
            groups[r['dominant_state']].append(r['code'])
    print(f'\n=== 分组 ===')
    for s in target_states:
        print(f'  {s}: {len(groups[s])} 只(占比 ≥ {args.min_prop:.0%})')

    # 拉收盘价(用于 basket 回测 + IC 的 forward return)— 无论 vbt 是否可用都跑
    all_codes = sorted({c for codes in groups.values() for c in codes})
    close_dict = {}
    if not all_codes:
        print('没有任何股票分组,跳过回测 + IC')
    else:
        for c in all_codes:
            try:
                df = P.load_ohlcva(c, lookback_years=args.days / 240)
                if df is None or df.empty:
                    continue
                # 大小写容错:Close/close 都收
                close_col = next((col for col in df.columns if col.lower() == 'close'), None)
                if close_col is None:
                    print(f'  {c} 无 close 列(列: {df.columns.tolist()[:5]}...)')
                    continue
                close_dict[c] = df[close_col]
            except Exception as e:
                print(f'  {c} 拉 close 失败: {e}')

    # basket 回测(vbt 不可用时跳过,close_dict 仍可用于 IC)
    if not all_codes:
        pass
    elif vbt is None:
        print('vbt 不可用,跳过 basket 回测')
    else:
        if not close_dict:
            print('没有拉取到任何 close,跳过 basket 回测')
        else:
            close_df = pd.DataFrame(close_dict).dropna(how='all')
            # 对齐共同日期
            close_df = close_df.ffill().dropna()
            print(f'\n=== basket 回测(等权 buy-and-hold,初始资金 {args.init_cash:.0f}) ===')
            print(f'  close_df: {close_df.shape[0]} 日 × {close_df.shape[1]} 只')
            bt_rows = []
            for s in target_states:
                codes_in_group = [c for c in groups[s] if c in close_df.columns]
                if not codes_in_group:
                    print(f'  [{s}] 0 只 in close_df,跳过')
                    bt_rows.append({'state': s, 'n_stocks': 0})
                    continue
                sub_close = close_df[codes_in_group]
                pf = vbt.Portfolio.from_holding(sub_close, init_cash=args.init_cash, freq='D')
                # vbt 1.0.0 stats 多 portfolio 时给 warning,聚合后取首行
                with np.errstate(all='ignore'):
                    stats = pf.stats()
                # 安全取值(stats 是 Series,index 是 stat 名,值是聚合值)
                def _g(k, default=np.nan):
                    if hasattr(stats, '__contains__') and k in stats.index:
                        return float(stats[k])
                    return default
                # vbt 1.0 列名:'Total Return [%]' 是百分比形式,要 /100
                total_ret_pct = _g('Total Return [%]')
                max_dd_pct = _g('Max Drawdown [%]')
                bt_rows.append({
                    'state': s,
                    'n_stocks': len(codes_in_group),
                    'total_return': (total_ret_pct / 100.0) if np.isfinite(total_ret_pct) else np.nan,
                    'sharpe_ratio': _g('Sharpe Ratio'),
                    'max_drawdown': (max_dd_pct / 100.0) if np.isfinite(max_dd_pct) else np.nan,
                })
                print(
                    f'  [{s}] {len(codes_in_group)} 只 '
                    f'总收益={bt_rows[-1]["total_return"]:.2%} '
                    f'Sharpe={bt_rows[-1]["sharpe_ratio"]:+.2f} '
                    f'MaxDD={bt_rows[-1]["max_drawdown"]:.2%}'
                )
            bt_df = pd.DataFrame(bt_rows)
            bt_path = os.path.join(CSV_OUT_DIR, 'backtest_per_state.csv')
            bt_df.to_csv(bt_path, index=False, encoding='utf-8')
            print(f'  清单: {bt_path}')

    # 3. IC:每只票的 state_prop vs forward return
    # forward_return(股票 i) = (close[-1] - close[0]) / close[0] 在回看期内
    # state_prop 是 scalar(0..1)。rank 相关(Spearman) → IC
    # 对每个状态算一条 IC
    from scipy.stats import spearmanr
    print(f'\n=== IC(每只票 state_prop vs forward return; 跨截面 Spearman) ===')
    ic_rows = []
    # 先算 forward_return
    fwd_returns = {}
    for c in all_codes:
        if c in close_dict:
            s = close_dict[c].dropna()
            if len(s) >= 2:
                fwd_returns[c] = float((s.iloc[-1] - s.iloc[0]) / s.iloc[0])
    for s in target_states:
        # 筛选:有 forward_return + 有 state_prop 的票
        x_prop, y_ret = [], []
        for _, r in state_df.iterrows():
            c = r['code']
            if c in fwd_returns and r[s] >= args.min_prop:
                x_prop.append(r[s])
                y_ret.append(fwd_returns[c])
        if len(x_prop) < 3:
            ic_rows.append({'state': s, 'n': len(x_prop), 'ic': np.nan, 'p_value': np.nan})
            print(f'  [{s}] n={len(x_prop)} (样本太少,跳过)')
            continue
        # n=2 时 Spearman 会退化为 ±1 但仍算;大于 3 才有意义
        ic, p = spearmanr(x_prop, y_ret)
        ic_rows.append({'state': s, 'n': len(x_prop), 'ic': ic, 'p_value': p})
        print(f'  [{s}] n={len(x_prop)} IC={ic:+.3f} p={p:.3f}')
    ic_df = pd.DataFrame(ic_rows)
    ic_path = os.path.join(CSV_OUT_DIR, 'state_ic.csv')
    ic_df.to_csv(ic_path, index=False, encoding='utf-8')
    print(f'  清单: {ic_path}')


if __name__ == '__main__':
    main()