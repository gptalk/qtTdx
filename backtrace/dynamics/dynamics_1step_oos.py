# -*- coding: utf-8 -*-
# dynamics_1step_oos.py — 用 predict_next_state 跑 OOS 1 步预测(纯动力学基线)
#
# 与 prediction_ode.py 的关系:
#   - prediction_ode.py 用 OLS 拟合的 (k̂, ĉ) 跑 OOS 1 步预测(给模型加拟合自由度)
#   - 本脚本用 k=c=0,F_self 用 --f-self-window 滚动均值(纯动力学,无拟合)
#   - 两个脚本的输出 schema 兼容(都是 prediction_<idx>_<stk>.csv + prediction_summary.csv)
#
# 重要坑(已修):
#   - 若直接传 a_S_now 给 predict_next_state,在 k=c=0 下 F_self ≡ a_S_now,
#     导致 v_pred = v_S + a_S = v_S_next(数学恒等式,无预测意义,100% 命中率是假的)
#   - 因此本脚本默认 F_self_pred = rolling-mean(F_self(t-W:t)),W=--f-self-window
#
# 模型(用户 prompt §14-19):
#   a_pred(t) = β(t)·a_M(t) - k·d(t) - c·u(t) + F_self_pred(t)
#   v_pred(t+1) = v_S(t) + a_pred(t)
#
# 评估指标:
#   - direction_hit_rate: v_pred 与 v_actual 同号比例(2-D 用 cos > 0)
#   - rmse: ‖v_pred - v_actual‖ 的 RMS
#   - median_cos: cos(夹角) 的中位数
#
# 输入:stock list CSV(data/projection/stocks.csv 或 --input 指定)
# 输出:
#   - 每只票:data/dynamics/prediction_<idx>_<stk>.csv (T-2 行,11 列)
#   - 汇总:data/dynamics/prediction_summary.csv (每只票一行)
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import pandas as pd

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from common import tsfresh_pipeline as P
from projection._projection_core import (
    load_pair, compute_movement_projection, compute_dynamics,
)
from dynamics import predict_next_state

CSV_OUT_DIR = 'data/dynamics'
DEFAULT_INPUT = 'data/projection/stocks.csv'


def parse_args():
    p = argparse.ArgumentParser(description='OOS 1 步预测(纯动力学基线,F_self 用滚动均值,无拟合)')
    p.add_argument('--input', default=DEFAULT_INPUT,
                   help=f'股票列表 CSV(列:code, 可选 name)。默认 {DEFAULT_INPUT}')
    p.add_argument('--days', type=int, default=240, help='回看天数。默认 240')
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0=全部')
    p.add_argument('--market-baseline', action='store_true',
                   help='回退大盘基线(SZ→深证成指/SH→上证综指);默认走行业基线')
    p.add_argument('--index', default=None,
                   help='强制指定基线指数(覆盖个股自动解析)')
    p.add_argument('--k', type=float, default=0.0, help='恢复系数 k。默认 0')
    p.add_argument('--c', type=float, default=0.0, help='阻尼系数 c。默认 0')
    p.add_argument('--f-self-window', type=int, default=10,
                   help='F_self 滚动均值窗口(天);0 = 用末日瞬时值(会触发恒等式陷阱)')
    p.add_argument('--min-valid-days', type=int, default=20,
                   help='最少有效预测天数,少于此跳过。默认 20')
    return p.parse_args()


def load_stock_list(path):
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


def predict_one(stock_code, days, prefer_industry, index_code,
                k=0.0, c=0.0, f_self_window=10):
    """对一只股票跑 1 步 OOS 预测。

    F_self 预测器(避免恒等式陷阱):
        F_self_pred(t) = mean( F_self(τ) for τ in [t-W, t-1] )
        其中 F_self(τ) = a_S(τ) - β(τ)·a_M(τ) + k·d(τ) + c·u(τ)
        若 W=0 → F_self_pred(t) = F_self(t-1) 末日瞬时值(会严重过拟合)
        若 W≥T → 用全历史均值

    Returns:
        pred_df, summary, index_code, index_tag, stock_tag
    """
    loaded = load_pair(stock_code, days, P, prefer_industry=prefer_industry,
                       index_code=index_code, lag=0)
    data_stock = loaded['stock_df']
    data_index = loaded['index_df']
    common_idx = loaded['common_idx']
    index_code = loaded['index_code']
    index_tag = loaded['index_tag']
    stock_tag = loaded['stock_tag']

    mv = compute_movement_projection(data_stock, data_index)
    delta_u = mv['stock_move']              # (T-1, 2) — v_S
    delta_v = mv['index_move']              # (T-1, 2) — v_M
    beta = mv['proj_coeff']                 # (T-1,)
    T = len(delta_u)
    # 锚定强度 q_t 序列(与 simulate_trajectory 共享):从 description 层拿
    dyn = compute_dynamics(mv, lambda_q=None)
    q_t_seq = dyn['q_t']                    # (T-1,)

    # 加速度:末行 NaN(np.diff 丢一行)
    a_u = np.full_like(delta_u, np.nan)     # a_S
    a_v = np.full_like(delta_v, np.nan)     # a_M
    if T >= 2:
        a_u[:-1] = np.diff(delta_u, axis=0)
        a_v[:-1] = np.diff(delta_v, axis=0)

    # 速度 / 位置偏离
    u_vec = delta_u - beta[:, None] * delta_v
    d_vec = np.zeros_like(delta_u)
    if T >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    # 残差序列:F_self(τ) = a_S(τ) - β(τ)·a_M(τ) + k·d(τ) + c·u(τ)
    # 注意 a_S / a_M 在 τ=0..T-3 有效,τ=T-2 是 NaN;这里只填有效段
    F_self_full = np.full_like(delta_u, np.nan)
    valid = np.isfinite(a_u).all(axis=1) & np.isfinite(a_v).all(axis=1)
    F_self_full[valid] = (
        a_u[valid] - beta[valid, None] * a_v[valid]
        + k * d_vec[valid] + c * u_vec[valid]
    )

    # 主循环:t ∈ [0, T-2] — 预测 Δu(t+1)
    rows = []
    for t in range(T - 1):
        # 当前步必须有限
        if not np.isfinite(a_v[t]).all():
            continue
        # F_self 预测:F_self_pred(t) = mean( F_self[t-W : t] )
        # τ ∈ [max(0, t-W), t-1] 必须至少 1 个有效值
        lo = max(0, t - f_self_window)
        hi = t
        f_self_hist = F_self_full[lo:hi]
        valid_hist = np.isfinite(f_self_hist).all(axis=1)
        if not valid_hist.any():
            continue
        F_self_pred = np.nanmean(f_self_hist, axis=0)
        actual = delta_u[t + 1]
        if not np.isfinite(actual).all():
            continue
        a_pred, v_pred = predict_next_state(
            v_S_now=delta_u[t],
            a_M_now=a_v[t],
            beta_now=beta[t],
            d_now=d_vec[t],
            u_now=u_vec[t],
            F_self_now=F_self_pred,
            k=k, c=c,
            q_now=float(q_t_seq[t]),
        )
        if not np.isfinite(v_pred).all():
            continue
        cos = float(np.dot(v_pred, actual) / (
            np.linalg.norm(v_pred) * np.linalg.norm(actual) + 1e-30
        ))
        dir_match = int(cos > 0)
        err_mag = float(np.linalg.norm(v_pred - actual))
        rows.append({
            'Date': common_idx[t + 1],
            'Delta_Vol_actual': actual[0],
            'Delta_Amt_actual': actual[1],
            'Delta_Vol_pred': v_pred[0],
            'Delta_Amt_pred': v_pred[1],
            'a_pred_Vol': a_pred[0],
            'a_pred_Amt': a_pred[1],
            'F_self_Vol': F_self_pred[0],
            'F_self_Amt': F_self_pred[1],
            'CosAngle': cos,
            'DirMatch': dir_match,
            'ErrMag': err_mag,
        })

    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        return pred_df, {
            'n_valid_days': 0, 'dir_hit_rate': np.nan,
            'rmse': np.nan, 'mean_err_mag': np.nan, 'median_cos': np.nan,
        }, index_code, index_tag, stock_tag

    summary = {
        'n_valid_days': len(pred_df),
        'dir_hit_rate': float(pred_df['DirMatch'].mean()),
        'rmse': float(np.sqrt(np.mean(pred_df['ErrMag'] ** 2))),
        'mean_err_mag': float(pred_df['ErrMag'].mean()),
        'median_cos': float(pred_df['CosAngle'].median()),
    }
    return pred_df, summary, index_code, index_tag, stock_tag


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    stock_list = load_stock_list(args.input)
    if args.limit > 0:
        stock_list = stock_list[:args.limit]
    prefer_industry = not args.market_baseline

    print(f'输入: {args.input} ({len(stock_list)} 只)')
    print(f'回看天数: {args.days} | 力模型: k={args.k}, c={args.c}(无拟合)')
    if args.f_self_window == 0:
        print(f'⚠ --f-self-window=0 会触发恒等式陷阱(F_self_pred ≡ a_S_obs → 命中率虚高)')
        print(f'  建议 ≥ 5(默认 10)')
    print(f'F_self 预测: 滚动均值窗口 W={args.f_self_window}')
    print(f'基线: {"显式 " + args.index if args.index else "申万二级行业(自动)" if prefer_industry else "大盘指数"}')
    print(f'输出: {CSV_OUT_DIR}/')

    summary_rows = []
    skipped = {'too_few': 0, 'failed': 0}
    for i, (code, name) in enumerate(stock_list, 1):
        try:
            pred_df, summary, index_code, index_tag, stock_tag = predict_one(
                code, args.days, prefer_industry, args.index,
                k=args.k, c=args.c, f_self_window=args.f_self_window,
            )
        except Exception as e:
            skipped['failed'] += 1
            print(f'[{i}/{len(stock_list)}] {code} ✗ {type(e).__name__}: {e}')
            continue
        if summary['n_valid_days'] < args.min_valid_days:
            skipped['too_few'] += 1
            print(f'[{i}/{len(stock_list)}] {code} - 有效天数 {summary["n_valid_days"]} < {args.min_valid_days},跳过')
            continue
        pred_csv = os.path.join(CSV_OUT_DIR, f'prediction_{index_tag}_{stock_tag}.csv')
        pred_df.to_csv(pred_csv, index=False, encoding='utf-8')
        summary_rows.append({
            'code': code,
            'name': name or '',
            'index_code': index_code,
            'index_tag': index_tag,
            'stock_tag': stock_tag,
            'k': args.k,
            'c': args.c,
            'f_self_window': args.f_self_window,
            'n_valid_days': summary['n_valid_days'],
            'dir_hit_rate': summary['dir_hit_rate'],
            'rmse': summary['rmse'],
            'mean_err_mag': summary['mean_err_mag'],
            'median_cos': summary['median_cos'],
        })
        print(
            f'[{i}/{len(stock_list)}] {code} W={args.f_self_window} '
            f'命中率={summary["dir_hit_rate"]:.1%} '
            f'RMSE={summary["rmse"]:.2e} '
            f'cos={summary["median_cos"]:+.3f} '
            f'({summary["n_valid_days"]}d)'
        )

    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_path = os.path.join(CSV_OUT_DIR, 'prediction_summary.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        hit = out_df['dir_hit_rate'].to_numpy()
        rmse = out_df['rmse'].to_numpy()
        cos = out_df['median_cos'].to_numpy()
        print(f'\n=== 汇总 ===')
        print(f'  预测完成: {len(summary_rows)} 只')
        print(f'  有效天数不足: {skipped["too_few"]} | 失败: {skipped["failed"]}')
        print(f'  方向命中率: median={np.median(hit):.1%} '
              f'p25={np.percentile(hit, 25):.1%} '
              f'p75={np.percentile(hit, 75):.1%}')
        print(f'  RMSE:        median={np.median(rmse):.2e}')
        print(f'  median_cos:  median={np.median(cos):+.3f}')
        print(f'  清单: {out_path}')


if __name__ == '__main__':
    main()