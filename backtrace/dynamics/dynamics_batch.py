# -*- coding: utf-8 -*-
# dynamics_batch.py — 批量跑离散动力系统(load → describe → simulate → CSV + manifest)
#
# 与 projection_batch.py --dynamics 的关系:
#   - 那个只跑描述层(q/θ/R/E/state/forces),产 dynamics_*.csv + forces_*.csv + manifest
#   - 这个在描述层之上 N 步前向模拟,产 simulation_*.csv + 更详细的 manifest
#
# 默认输入与 projection_batch 一致(data/projection/stocks.csv),不冲突。
# 默认输出 data/dynamics/ — 与 data/projection/ 隔离避免覆盖。
#
# 参数:
#   --input              path  股票列表 CSV(列:code, 可选 name)。默认 data/projection/stocks.csv
#   --days               int   回看天数。默认 240
#   --horizon            int   模拟步数 N。默认 5
#   --limit              int   最多处理多少只;0 = 全部
#   --market-baseline    flag  全部回退大盘基线(覆盖默认行业基线)
#   --index              str   强制指定基线指数(覆盖个股自动解析)
#   --lambda-q           float 锚定强度系数 λ_q。-1 走 median 自适应
#   --classify-thresholds str  4 浮点:默认 0.10,0.50,30,90
#   --k-restore / --c-damp float 力模型系数;默认 0
#   --k-from-fit / --c-from-fit flag 从 data/projection/kc_estimates.csv 加载
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --limit 50
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --horizon 10
#   PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_batch.py --input stocks.csv
#
# 输出:
#   - 每只票:data/dynamics/{dynamics,forces,simulation}_<idx>_<stk>.csv
#   - 清单:data/dynamics/batch_manifest.csv,列:
#     code, name, index_code, index_name, rows, horizon, k_restore, c_damp,
#     desc_csv_path, frc_csv_path, sim_csv_path,
#     sim_mean_R, sim_max_E_self, sim_state_dist, status
#     sim_state_dist 用 "{follow:3, accelerating:1, ...}" 字符串表示
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
    load_pair, compute_movement_projection, compute_dynamics,
    classify_states, build_dynamics_df,
    compute_forces, build_forces_df,
)
from dynamics import (
    build_simulation_df, simulate_trajectory,
    make_rolling_mean_f_self_predictor, make_constant_f_self_predictor,
    make_ar1_f_self_predictor,
)

CSV_OUT_DIR = 'data/dynamics'
CSV_FALLBACK_INPUT = 'data/projection/stocks.csv'


def load_kc_map(status_filter='ok'):
    """从 data/projection/kc_estimates.csv 读 (k̂, ĉ) by (index_tag, stock_tag)。"""
    path = os.path.join('data/projection', 'kc_estimates.csv')
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={
        'code': str, 'index_code': str, 'index_tag': str, 'stock_tag': str,
    })
    if status_filter:
        df = df[df['status'].str.startswith(status_filter, na=False)]
    out = {}
    for _, row in df.iterrows():
        try:
            out[(row['index_tag'], row['stock_tag'])] = (
                float(row['k_hat']), float(row['c_hat']),
            )
        except (KeyError, ValueError):
            continue
    return out


def parse_args():
    p = argparse.ArgumentParser(description='批量离散动力系统(产 description + force + simulation CSVs + manifest)')
    p.add_argument('--input', default=CSV_FALLBACK_INPUT,
                   help=f'股票列表 CSV(列:code, 可选 name)。默认 {CSV_FALLBACK_INPUT}')
    p.add_argument('--days', type=int, default=240, help='回看天数。默认 240')
    p.add_argument('--horizon', type=int, default=5, help='模拟步数 N。默认 5')
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0=全部')
    p.add_argument('--market-baseline', action='store_true',
                   help='回退大盘基线(SZ→深证成指/SH→上证综指);默认走行业基线')
    p.add_argument('--index', default=None,
                   help='强制指定基线指数(覆盖个股自动解析)')
    p.add_argument('--lambda-q', type=float, default=-1.0, help='锚定强度 λ_q;-1 走 median 自适应')
    p.add_argument('--classify-thresholds', default='0.10,0.50,30,90',
                   help='状态分类阈值 R_low,R_high,theta_following_deg,theta_against_deg。默认 0.10,0.50,30,90')
    p.add_argument('--k-restore', type=float, default=0.0, help='恢复系数 k。默认 0.0')
    p.add_argument('--c-damp', type=float, default=0.0, help='阻尼系数 c。默认 0.0')
    p.add_argument('--k-from-fit', action='store_true', help='从 kc_estimates.csv 加载 k̂')
    p.add_argument('--c-from-fit', action='store_true', help='从 kc_estimates.csv 加载 ĉ')
    p.add_argument('--f-self-mode', default='rolling',
                   choices=['rolling', 'constant', 'oracle', 'ar1'],
                   help='F_self 预测模式:rolling=末日滚动均值(default)/constant=末日瞬时值/'
                        'oracle=末日观测残差恒定外推/ar1=AR(1) 自回归(per-dim 估 ρ/μ)')
    p.add_argument('--f-self-window', type=int, default=10,
                   help='F_self 窗口(rolling=滚动均值天数;ar1=最少有效样本数)。默认 10')
    return p.parse_args()


def load_stock_list(path):
    """读 CSV:必须有 code 列,name 可选。返回 [(code, name|None), ...]。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'股票列表文件不存在: {path}\n请新建(最少一列 code),例:\n'
            f'  code,name\n  002475.SZ,立讯精密\n  600519.SH,贵州茅台'
        )
    df = pd.read_csv(path, dtype={'code': str})
    if 'code' not in df.columns:
        raise ValueError(f'输入文件 {path} 必须有 code 列')
    names = df['name'] if 'name' in df.columns else [None] * len(df)
    return [
        (str(c).strip(), str(n).strip() if isinstance(n, str) else None)
        for c, n in zip(df['code'], names)
    ]


def process_one(stock_code, stock_name, days, prefer_industry, index_code,
                horizon, lambda_q, classify_thresholds,
                k_restore, c_damp, kc_overrides=None,
                f_self_mode='rolling', f_self_window=10):
    """处理一只股票。返回 manifest 行 dict(失败也返回,status 字段说明原因)。"""
    try:
        # 1. 加载
        loaded = load_pair(stock_code, days, P, prefer_industry=prefer_industry,
                           index_code=index_code, lag=0)
        data_stock = loaded['stock_df']
        data_index = loaded['index_df']
        common_idx = loaded['common_idx']
        index_code = loaded['index_code']
        index_name = loaded['index_name']
        index_tag = loaded['index_tag']
        stock_tag = loaded['stock_tag']

        if len(common_idx) < horizon + 2:
            return {
                'code': stock_code, 'name': stock_name or '', 'index_code': index_code,
                'index_name': index_name, 'rows': len(common_idx), 'horizon': horizon,
                'k_restore': k_restore, 'c_damp': c_damp,
                'desc_csv_path': '', 'frc_csv_path': '', 'sim_csv_path': '',
                'sim_mean_R': np.nan, 'sim_max_E_self': np.nan, 'sim_state_dist': '',
                'status': f'failed: 数据 {len(common_idx)} 行 < horizon+2={horizon + 2}',
            }

        # k / c 覆盖
        eff_k, eff_c = k_restore, c_damp
        if kc_overrides is not None:
            kc = kc_overrides.get((index_tag, stock_tag))
            if kc is not None:
                eff_k, eff_c = kc

        # 2. 描述层
        mv = compute_movement_projection(data_stock, data_index)
        dyn = compute_dynamics(mv, lambda_q=lambda_q)
        r_low, r_high, theta_following_rad, theta_against_rad = classify_thresholds
        states_desc = classify_states(
            dyn['R'], dyn['theta'], dyn['E_self'],
            (r_low, r_high, theta_following_rad, theta_against_rad),
        )
        desc_df = build_dynamics_df(common_idx[1:], dyn, states_desc, index_tag, stock_tag)
        desc_csv = os.path.join(CSV_OUT_DIR, f'dynamics_{index_tag}_{stock_tag}.csv')
        desc_df.to_csv(desc_csv, index=False, encoding='utf-8')

        # 3. 力分解
        frc = compute_forces(dyn, mv, k_restore=eff_k, c_damp=eff_c)
        frc_df = build_forces_df(common_idx[1:], frc, index_tag, stock_tag)
        frc_csv = os.path.join(CSV_OUT_DIR, f'forces_{index_tag}_{stock_tag}.csv')
        frc_df.to_csv(frc_csv, index=False, encoding='utf-8')

        # 4. 模拟
        v_S_init = mv['stock_move'][-1]
        # NEW(2026-08-17 时间轴重构 v2):v_M_seq / β_seq 改用 N+1 个状态量,
        # 让所有 N 个 a_M(t) = v_M(t+1) - v_M(t) 都有效(无 NaN 跳步)。
        beta_seq = mv['proj_coeff'][-(horizon + 1):]   # (N+1,)
        v_M_seq = mv['index_move'][-(horizon + 1):]    # (N+1, 2)
        u_full = mv['stock_move'] - mv['proj_coeff'][:, None] * mv['index_move']
        d_full = np.zeros_like(mv['stock_move'])
        if len(u_full) >= 2:
            d_full[1:] = np.cumsum(u_full[:-1], axis=0)
        # NEW(2026-08-17 v2):d[t+1]=d[t]+u[t] 递推。d(0)=d_full[-1] 为自然初始条件,
        # d(1) = d_full[-1] + u_full[-1](最直观的物理定义,无需 u 补偿)
        d_init = d_full[-1]
        u_init = u_full[-1]
        a_u_vec = np.full_like(mv['stock_move'], np.nan)
        a_v_vec = np.full_like(mv['index_move'], np.nan)
        if len(mv['stock_move']) >= 2:
            a_u_vec[:-1] = np.diff(mv['stock_move'], axis=0)
            a_v_vec[:-1] = np.diff(mv['index_move'], axis=0)
        # a_u_vec / a_v_vec 末行是 NaN(np.diff 丢一行);取倒数第二行
        # (T-2) 作为"最近一个有效加速度"用于 F_self 分解
        a_S_recent = a_u_vec[-2] if np.isfinite(a_u_vec[-2]).all() else np.array([0.0, 0.0])
        a_M_recent = a_v_vec[-2] if np.isfinite(a_v_vec[-2]).all() else np.array([0.0, 0.0])
        F_self_last = (
            a_S_recent - beta_seq[-1] * a_M_recent
            + eff_k * d_init + eff_c * u_init
        )
        # F_self 历史(末日之前,长度 T-1)— 用于 rolling 预测器
        # F_self_full[τ] = a_S(τ) - β(τ)·a_M(τ) + k·d(τ) + c·u(τ),只在有效段填
        F_self_full = np.full_like(mv['stock_move'], np.nan)
        valid = np.isfinite(a_u_vec).all(axis=1) & np.isfinite(a_v_vec).all(axis=1)
        F_self_full[valid] = (
            a_u_vec[valid] - mv['proj_coeff'][valid, None] * a_v_vec[valid]
            + eff_k * d_full[valid] + eff_c * u_full[valid]
        )

        # F_self 预测器选择
        if f_self_mode == 'rolling':
            F_self_predictor = make_rolling_mean_f_self_predictor(
                F_self_full, window=f_self_window,
            )
            F_self_seq_for_manifest = None
        elif f_self_mode == 'constant':
            F_self_predictor = make_constant_f_self_predictor(F_self_last)
            F_self_seq_for_manifest = None
        elif f_self_mode == 'ar1':
            # AR(1) 自回归:per-dim 估 ρ/μ;数据不足自动退化到常数(用均值)
            F_self_predictor = make_ar1_f_self_predictor(
                F_self_full, min_history=f_self_window,
            )
            F_self_seq_for_manifest = None
        else:  # oracle — 末日观测残差恒定外推(旧默认行为)
            F_self_predictor = None
            F_self_seq_for_manifest = np.tile(F_self_last, (horizon, 1))

        q_t_seq = dyn['q_t'][-horizon:]

        sim = simulate_trajectory(
            v_S_init=v_S_init, v_M_seq=v_M_seq, beta_seq=beta_seq,
            F_self_seq=F_self_seq_for_manifest,
            F_self_predictor=F_self_predictor,
            d_init=d_init,
            # v3:u_init 删除(派生量,simulate_trajectory 在 t=0 自动派生 u[0])
            k=eff_k, c=eff_c, q_t_seq=q_t_seq,
            classify_thresholds=(r_low, r_high, theta_following_rad, theta_against_rad),
        )
        sim_df = build_simulation_df(sim, list(range(horizon + 1)), index_tag, stock_tag)
        sim_csv = os.path.join(CSV_OUT_DIR, f'simulation_{index_tag}_{stock_tag}.csv')
        sim_df.to_csv(sim_csv, index=False, encoding='utf-8')

        # 模拟摘要
        sim_R = sim['R'][:horizon]
        sim_E_self = sim['E_self'][:horizon]
        sim_states = sim['state'][:horizon]
        state_dist = Counter(sim_states)
        sim_state_dist = '{' + ', '.join(
            f'{s}:{c}' for s, c in state_dist.most_common()
        ) + '}' if state_dist else ''

        return {
            'code': stock_code, 'name': stock_name or '', 'index_code': index_code,
            'index_name': index_name, 'rows': len(common_idx), 'horizon': horizon,
            'k_restore': eff_k, 'c_damp': eff_c,
            'desc_csv_path': desc_csv, 'frc_csv_path': frc_csv, 'sim_csv_path': sim_csv,
            'sim_mean_R': float(np.nanmean(sim_R)),
            'sim_max_E_self': float(np.nanmax(sim_E_self)),
            'sim_state_dist': sim_state_dist,
            'status': 'ok',
        }
    except Exception as e:
        return {
            'code': stock_code, 'name': stock_name or '', 'index_code': '',
            'index_name': '', 'rows': 0, 'horizon': horizon,
            'k_restore': k_restore, 'c_damp': c_damp,
            'desc_csv_path': '', 'frc_csv_path': '', 'sim_csv_path': '',
            'sim_mean_R': np.nan, 'sim_max_E_self': np.nan, 'sim_state_dist': '',
            'status': f'failed: {type(e).__name__}: {e}',
        }


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    # 阈值
    try:
        R_LOW, R_HIGH, THETA_FOLLOWING_DEG, THETA_AGAINST_DEG = (
            float(x) for x in args.classify_thresholds.split(',')
        )
    except Exception as e:
        raise SystemExit(
            f'--classify-thresholds 解析失败: {args.classify_thresholds!r}\n{e}'
        )
    if not (0 < R_LOW < R_HIGH < 1):
        raise SystemExit(f'R_low={R_LOW} / R_high={R_HIGH} 必须 0 < R_low < R_high < 1')
    if not (0 < THETA_FOLLOWING_DEG < THETA_AGAINST_DEG < 180):
        raise SystemExit('theta_following / theta_against 必须 0 < following < against < 180')
    classify_thresholds = (
        R_LOW, R_HIGH, np.deg2rad(THETA_FOLLOWING_DEG), np.deg2rad(THETA_AGAINST_DEG),
    )

    lambda_q = None if args.lambda_q < 0 else args.lambda_q

    kc_overrides = None
    if args.k_from_fit or args.c_from_fit:
        kc_overrides = load_kc_map(status_filter='ok')
        if not kc_overrides:
            print('[--k-from-fit/--c-from-fit] ⚠ kc_estimates.csv 没有 ok 记录,使用默认值')
            print('  请先跑:python backtrace/projection/parameter_fit.py')
        else:
            print(f'[--k-from-fit/--c-from-fit] 已加载 {len(kc_overrides)} 条拟合值')

    stock_list = load_stock_list(args.input)
    if args.limit > 0:
        stock_list = stock_list[:args.limit]

    prefer_industry = not args.market_baseline
    if args.index:
        baseline = f'显式指定基线 {args.index}(所有股票共用)'
    elif prefer_industry:
        baseline = '申万二级行业(按个股解析;新股/非 A 股自动回退大盘)'
    else:
        baseline = '大盘指数(深证成指/上证综指)'

    lq_str = 'median 自适应' if lambda_q is None else f'{lambda_q:.4e}'
    f_self_str = {
        'rolling': f'末日滚动均值 W={args.f_self_window}',
        'constant': '末日瞬时值(常数)',
        'oracle': '末日观测残差恒定外推',
        'ar1': f'AR(1) 自回归(最少有效样本={args.f_self_window})',
    }[args.f_self_mode]
    print(f'输入: {args.input} ({len(stock_list)} 只)')
    print(f'回看天数: {args.days} | 模拟步数: N={args.horizon}')
    print(f'基线: {baseline}')
    print(f'分类阈值: R<{R_LOW}/{R_HIGH}, θ<{THETA_FOLLOWING_DEG}°/>{THETA_AGAINST_DEG}°')
    print(f'λ_q={lq_str} | k={args.k_restore} c={args.c_damp}')
    print(f'F_self 模式: {f_self_str}')
    print(f'输出: {CSV_OUT_DIR}/\n')

    manifest = []
    for i, (code, name) in enumerate(stock_list, 1):
        label = f'{code} ({name})' if name else code
        print(f'[{i}/{len(stock_list)}] {label}...', end=' ', flush=True)
        row = process_one(
            code, name, args.days, prefer_industry, args.index,
            args.horizon, lambda_q, classify_thresholds,
            args.k_restore, args.c_damp, kc_overrides,
            args.f_self_mode, args.f_self_window,
        )
        manifest.append(row)
        if row['status'] == 'ok':
            print(
                f'✓ R̄={row["sim_mean_R"]:.3f} 状态={row["sim_state_dist"] or "(空)"} → '
                f'{os.path.basename(row["sim_csv_path"])}'
            )
        else:
            print(f'✗ {row["status"]}')

    manifest_df = pd.DataFrame(manifest, columns=[
        'code', 'name', 'index_code', 'index_name', 'rows', 'horizon',
        'k_restore', 'c_damp',
        'desc_csv_path', 'frc_csv_path', 'sim_csv_path',
        'sim_mean_R', 'sim_max_E_self', 'sim_state_dist', 'status',
    ])
    manifest_path = os.path.join(CSV_OUT_DIR, 'batch_manifest.csv')
    manifest_df.to_csv(manifest_path, index=False, encoding='utf-8')

    ok = sum(1 for r in manifest if r['status'] == 'ok')
    fail = len(manifest) - ok
    print(f'\n=== 汇总 ===')
    print(f'  全部成功: {ok}/{len(manifest)} | 失败: {fail}')
    print(f'  清单: {manifest_path}')


if __name__ == '__main__':
    main()
