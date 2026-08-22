# -*- coding: utf-8 -*-
# state_kc_analysis.py — 状态分类 × (k̂, ĉ) 关联分析
#
# 假设:不同动力学状态(跟随 / 弱偏离 / 加速偏离 / 独立 / 逆势 / 回归 / 共振)的
# 个股,其 (k̂, ĉ) 应该呈现可区分的分布特征。
#   - 共振/跟随 个股 → 可能 c 高(波动耗散快)
#   - 加速偏离 / 独立个股 → 可能 c 低或负(波动放大)
#   - 逆势个股 → 可能 k 低或负(位置偏离不被拉回)
#
# 实现:
#   1. 读 kc_estimates.csv → 每只票 (k̂, ĉ)
#   2. 读 dynamics_*.csv → 每只票每天状态(label)
#   3. 计算每只票「状态分布」(各状态占比)
#   4. 按状态分组,看 k̂/ĉ 在该状态出现日的分布
#
# 输出:
#   - per_stock_state_dist.csv — 每只票一行,8 列(各状态占比 + (k̂, ĉ))
#   - per_state_kc_stats.csv   — 每状态一行,6 列(count, k̂ med/mean, ĉ med/mean)
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/state_kc_analysis.py
#   PYTHONIOENCODING=utf-8 python backtrace/projection/state_kc_analysis.py --limit 10
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import pandas as pd

CSV_OUT_DIR = 'data/projection'
KC_FILE = 'kc_estimates.csv'
PER_STOCK_OUT = 'per_stock_state_dist.csv'
PER_STATE_OUT = 'per_state_kc_stats.csv'

STATE_ORDER = ['follow', 'weak_div', 'accelerating', 'independent',
               'against', 'returning', 'resonance', 'none']
STATE_LABELS_CN = {
    'follow': '跟随', 'weak_div': '弱偏离', 'accelerating': '加速偏离',
    'independent': '独立', 'against': '逆势', 'returning': '回归',
    'resonance': '共振', 'none': '无',
}


def parse_args():
    p = argparse.ArgumentParser(
        description='状态分类 × (k̂, ĉ) 关联分析',
    )
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0 = 全部')
    p.add_argument(
        '--status-filter', default='ok',
        help='只对 kc_estimates status 以指定前缀开头的票分析(默认 "ok")',
    )
    p.add_argument(
        '--drop-none', action='store_true',
        help='在 per_state 统计中丢掉 "none" 状态(数据缺失日)',
    )
    p.add_argument(
        '--period', choices=['daily', '15m', '5m', '1m'], default='daily',
        help='缓存粒度(仅作审计/记录;不影响分析,读现有 CSV)',
    )
    return p.parse_args()


def load_kc_estimates(status_filter: str = 'ok') -> dict:
    """读 kc_estimates.csv → {(index_tag, stock_tag): (k_hat, c_hat, code, name)}"""
    path = os.path.join(CSV_OUT_DIR, KC_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} 不存在;先跑 parameter_fit.py')
    df = pd.read_csv(path, dtype={
        'code': str, 'index_code': str, 'index_tag': str, 'stock_tag': str,
    })
    if status_filter:
        df = df[df['status'].str.startswith(status_filter, na=False)]
    out = {}
    for _, row in df.iterrows():
        out[(row['index_tag'], row['stock_tag'])] = {
            'k_hat': float(row['k_hat']),
            'c_hat': float(row['c_hat']),
            'code': row['code'],
            'name': row['name'] if isinstance(row['name'], str) else '',
        }
    return out


def state_distribution(dynamics_csv: str, stock_tag: str) -> dict:
    """读 dynamics_*.csv,返回 {state: count, ...}。空返回 {state: 0, ...}"""
    counts = {s: 0 for s in STATE_ORDER}
    try:
        df = pd.read_csv(dynamics_csv)
    except Exception:
        return counts
    col = f'Dyn_State_{stock_tag}'
    if col not in df.columns:
        return counts
    vc = df[col].value_counts().to_dict()
    for s, c in vc.items():
        if s in counts:
            counts[s] = int(c)
    return counts


def per_state_kc_table(per_stock_rows: list, drop_none: bool = False) -> pd.DataFrame:
    """聚合:每个状态的 k̂/ĉ 分布(count / median / mean)。

    输入 per_stock_rows 是 dict 列表,每 dict 含:
      'index_tag', 'stock_tag', 'state' (单一日的状态), 'k_hat', 'c_hat'
    """
    df = pd.DataFrame(per_stock_rows)
    if drop_none and 'state' in df.columns:
        df = df[df['state'] != 'none']
    if len(df) == 0:
        return pd.DataFrame(columns=[
            'state', 'state_cn', 'count',
            'k_hat_median', 'k_hat_mean', 'c_hat_median', 'c_hat_mean',
        ])
    grouped = df.groupby('state').agg(
        count=('k_hat', 'size'),
        k_hat_median=('k_hat', 'median'),
        k_hat_mean=('k_hat', 'mean'),
        c_hat_median=('c_hat', 'median'),
        c_hat_mean=('c_hat', 'mean'),
    ).reset_index()
    # 加中文标签 + 固定状态顺序
    grouped['state_cn'] = grouped['state'].map(STATE_LABELS_CN)
    grouped['state'] = pd.Categorical(grouped['state'], categories=STATE_ORDER, ordered=True)
    grouped = grouped.sort_values('state').reset_index(drop=True)
    return grouped


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    kc_map = load_kc_estimates(args.status_filter)
    print(f'已加载 {len(kc_map)} 条 (k̂, ĉ) (status_filter={args.status_filter!r})')

    per_stock_rows = []    # 输出 1
    per_state_long = []    # 输出 2(用于聚合)

    targets = list(kc_map.items())
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f'目标: {len(targets)} 只 (limit={args.limit})')

    for i, ((index_tag, stock_tag), kc) in enumerate(targets, 1):
        dyn_csv = os.path.join(CSV_OUT_DIR, f'dynamics_{index_tag}_{stock_tag}.csv')
        if not os.path.exists(dyn_csv):
            print(f'[{i}/{len(targets)}] {kc["code"]} ✗ dynamics CSV 不存在')
            continue
        counts = state_distribution(dyn_csv, stock_tag)
        total = sum(counts.values())
        if total == 0:
            continue

        # 输出 1:per_stock_state_dist(占比 + (k̂, ĉ))
        row = {
            'code': kc['code'],
            'name': kc['name'],
            'index_code': f'{index_tag}.SH',
            'index_tag': index_tag,
            'stock_tag': stock_tag,
            'k_hat': kc['k_hat'],
            'c_hat': kc['c_hat'],
            'n_days': total,
            **{f'pct_{s}': counts[s] / total for s in STATE_ORDER},
        }
        per_stock_rows.append(row)

        # 输出 2:per-state 长表(每只票每天一行,带 (k̂, ĉ))
        try:
            df = pd.read_csv(dyn_csv)
            states = df[f'Dyn_State_{stock_tag}'].tolist()
        except Exception:
            continue
        for s in states:
            if not isinstance(s, str) or s not in STATE_ORDER:
                continue
            per_state_long.append({
                'code': kc['code'],
                'state': s,
                'k_hat': kc['k_hat'],
                'c_hat': kc['c_hat'],
            })

        # 进度:打印 top-3 状态
        top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top3_str = ', '.join(f'{STATE_LABELS_CN.get(s, s)}={c}({c/total:.0%})' for s, c in top3)
        print(f'[{i}/{len(targets)}] {kc["code"]} k={kc["k_hat"]:+.4f} '
              f'c={kc["c_hat"]:+.4f} top: {top3_str}')

    # 输出 1:per_stock_state_dist
    if per_stock_rows:
        ps_df = pd.DataFrame(per_stock_rows)
        ps_path = os.path.join(CSV_OUT_DIR, PER_STOCK_OUT)
        ps_df.to_csv(ps_path, index=False, encoding='utf-8')
        print(f'\n  → per_stock_state_dist: {ps_path} ({len(ps_df)} 只)')

    # 输出 2:per_state_kc_stats
    if per_state_long:
        state_df = per_state_kc_table(per_state_long, drop_none=args.drop_none)
        st_path = os.path.join(CSV_OUT_DIR, PER_STATE_OUT)
        state_df.to_csv(st_path, index=False, encoding='utf-8')
        print(f'  → per_state_kc_stats:  {st_path} ({len(state_df)} 个状态)')

        print(f'\n=== 状态 × (k̂, ĉ) 分布 ===')
        print(state_df.to_string(index=False))

        # 简单 sanity check:是否某状态 ĉ 显著偏高?(高 ĉ = 强阻尼 = 系统耗散快)
        if 'c_hat_median' in state_df.columns and len(state_df) >= 3:
            c_meds = state_df['c_hat_median'].dropna()
            if len(c_meds) >= 2:
                print(f'\n  ĉ 中位数极差: max={c_meds.max():.3f} min={c_meds.min():.3f} '
                      f'(差距 {c_meds.max() - c_meds.min():.3f})')


if __name__ == '__main__':
    main()