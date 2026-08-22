# -*- coding: utf-8 -*-
# prediction_ode.py — 用拟合的 (k̂, ĉ) 滚动预测下日个股速度 v_S
#
# 模型(用户 prompt §14-17):
#   a_S = β·a_M - k·d - c·u + F_self
#   v_{t+1} = v_t + a_t · Δt        (Δt = 1 交易日)
#
# 给定历史拟合的 (k̂, ĉ),逐日预测:
#   a_pred(t) = β(t)·a_M(t) - k̂·d(t) - ĉ·u(t)
#   v_pred(t+1) = v_S(t) + a_pred(t)
#
# 与实际 v_S(t+1) 对比,产两个核心指标:
#   - direction_hit_rate: 方向预测命中率(Δv_S_pred 与 Δv_S_actual 同号比例)
#   - rmse: 速度幅值预测误差(2-D ‖v_pred − v_actual‖)
#
# 输入:data/projection/movement_*.csv + kc_estimates.csv(由 parameter_fit.py 产出)
# 输出:data/projection/prediction_<idx>_<stk>.csv(每只票一日一行,17 列)
#       data/projection/prediction_summary.csv(每只票一行汇总)
#
# 用法:
#   PYTHONIOENCODING=utf-8 python backtrace/projection/prediction_ode.py
#   PYTHONIOENCODING=utf-8 python backtrace/projection/prediction_ode.py --limit 10
#   PYTHONIOENCODING=utf-8 python backtrace/projection/prediction_ode.py --input stocks.csv
#
# 注:本脚本是「动力学模型的 OOS 评估」,不是交易信号。命中率高 ≠ 能赚钱(还要看
# magnitude + 摩擦);只是验证 (k̂, ĉ) + 残差基线 F_self = a_S - β·a_M + k·d + c·u
# 的「拟合优度」和「方向预测力」。
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import pandas as pd

CSV_OUT_DIR = 'data/projection'
PRED_OUT_NAME = 'prediction_summary.csv'


def parse_args():
    p = argparse.ArgumentParser(
        description='用拟合 (k̂, ĉ) 滚动预测下日个股速度 v_S,产方向命中率 + RMSE',
    )
    p.add_argument(
        '--input', default=None,
        help=(
            '可选股票列表 CSV(列:code;可选 name/index_code)。'
            '不传则扫描 data/projection/movement_*.csv 全部处理。'
        ),
    )
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0 = 全部')
    p.add_argument(
        '--use-status-filter', default='ok',
        help=(
            '只对 kc_estimates.csv 里 status 以指定前缀开头的票做预测;默认 "ok"。'
            '空字符串 = 全部。例:"ok" / "ok (restoring" / ""'
        ),
    )
    p.add_argument(
        '--min-valid-days', type=int, default=20,
        help='最少有效预测天数(默认 20)。少于则跳过。',
    )
    p.add_argument(
        '--period', choices=['daily', '15m', '5m', '1m'], default='daily',
        help='缓存粒度(仅作审计/记录;不影响预测,读现有 movement CSV)',
    )
    return p.parse_args()


def load_kc_estimates(status_filter: str = 'ok') -> pd.DataFrame | None:
    """读 kc_estimates.csv;返回 (index_tag, stock_tag) → k_hat/c_hat 的 dict。"""
    path = os.path.join(CSV_OUT_DIR, 'kc_estimates.csv')
    if not os.path.exists(path):
        return None
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


def predict_one(movement_csv: str, stock_tag: str, index_tag: str,
                k_hat: float, c_hat: float):
    """对一只股票跑滚动预测,返回 (pred_df, summary_dict)。

    pred_df:每日一行,T-2 行(末行 NaN drop),17 列
      Date, ΔS_actual_x/y, ΔS_pred_x/y, a_pred_x/y,
      DirMatch(0/1), ErrMag(‖·‖)
    summary_dict:dict with code/stock_tag/index_tag/k_hat/c_hat/n_valid/dir_hit_rate/rmse
    """
    df = pd.read_csv(movement_csv)
    delta_u = df[[f'Move_Delta_Vol_{stock_tag}',
                  f'Move_Delta_Amt_{stock_tag}']].to_numpy()
    delta_v = df[[f'Move_Delta_Vol_{index_tag}',
                  f'Move_Delta_Amt_{index_tag}']].to_numpy()
    beta = df['Move_Proj_Coeff'].to_numpy()

    # 重建 2-D u_vec / d_vec / a_u_vec / a_v_vec(同 parameter_fit)
    u_vec = delta_u - beta[:, None] * delta_v
    d_vec = np.zeros_like(delta_u)
    if len(u_vec) >= 2:
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    a_u_vec = np.full_like(delta_u, np.nan)
    a_v_vec = np.full_like(delta_v, np.nan)
    if len(delta_u) >= 2:
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)

    # 预测:a_pred(t) = β(t)·a_M(t) - k̂·d(t) - ĉ·u(t)
    # 有效行:t ≥ 0 AND t ≤ T-2(末行 NaN 跳过)
    T = len(delta_u)
    a_pred = np.full((T, 2), np.nan)
    for t in range(T - 1):                                # 末行 a_v/t 衍生 NaN 跳过
        if not (np.isfinite(a_v_vec[t]).all() and np.isfinite(d_vec[t]).all()
                and np.isfinite(u_vec[t]).all()):
            continue
        a_pred[t] = (beta[t] * a_v_vec[t]
                     - k_hat * d_vec[t]
                     - c_hat * u_vec[t])

    # 预测速度变化:v_pred(t+1) = v_S(t) + a_pred(t) = Δu(t) + a_pred(t)
    # 实际速度变化:v_actual(t+1) = Δu(t+1) - Δu(t)
    # 等等 — Δu(t) 就是 v_S(t)。v_pred(t+1) 应该是下日的 Δu(t+1) 预测:
    #   Δu_pred(t+1) = v_S(t) + a_pred(t) = Δu(t) + a_pred(t)
    delta_u_pred = np.full((T, 2), np.nan)
    for t in range(T - 1):
        if np.isfinite(a_pred[t]).all():
            delta_u_pred[t] = delta_u[t] + a_pred[t]      # 预测下日的 Δu

    # 实际下日 Δu(t+1):t ∈ [0, T-2],shift -1
    # 命中条件:t 的预测 Δu_pred(t) vs t+1 的实际 Δu(t+1)
    rows = []
    for t in range(T - 1):
        if not (np.isfinite(delta_u_pred[t]).all()
                and np.isfinite(delta_u[t + 1]).all()):
            continue
        actual = delta_u[t + 1]
        pred = delta_u_pred[t]
        # 方向匹配:2-D 投到一个主轴(V + A 平均方向),避免维度抵消
        # 简化:用 L2 norm 的方向差异
        # 真实方向:sign(corr(pred, actual)) > 0
        cos = np.dot(pred, actual) / (
            np.linalg.norm(pred) * np.linalg.norm(actual) + 1e-30
        )
        dir_match = int(cos > 0)                          # 夹角 < 90° = 同方向
        err_mag = float(np.linalg.norm(pred - actual))
        rows.append({
            'Date': df['Date'].iloc[t],
            'Delta_Vol_actual': actual[0],
            'Delta_Amt_actual': actual[1],
            'Delta_Vol_pred': pred[0],
            'Delta_Amt_pred': pred[1],
            'a_pred_Vol': a_pred[t, 0],
            'a_pred_Amt': a_pred[t, 1],
            'CosAngle': cos,
            'DirMatch': dir_match,
            'ErrMag': err_mag,
        })
    pred_df = pd.DataFrame(rows)
    if len(pred_df) == 0:
        return pred_df, {
            'n_valid_days': 0, 'dir_hit_rate': np.nan, 'rmse': np.nan,
        }
    summary = {
        'n_valid_days': len(pred_df),
        'dir_hit_rate': float(pred_df['DirMatch'].mean()),
        'rmse': float(np.sqrt(np.mean(pred_df['ErrMag'] ** 2))),
        'mean_err_mag': float(pred_df['ErrMag'].mean()),
        'median_cos': float(pred_df['CosAngle'].median()),
    }
    return pred_df, summary


def list_movement_csvs(input_csv):
    """同 parameter_fit.list_movement_csvs(签名相同,逻辑一致)"""
    if input_csv:
        df = pd.read_csv(input_csv, dtype={'code': str, 'index_code': str})
        out = []
        for _, row in df.iterrows():
            code = str(row['code']).strip()
            name = str(row['name']).strip() if 'name' in df.columns and pd.notna(row['name']) else None
            if 'index_code' in df.columns and pd.notna(row['index_code']):
                index_code = str(row['index_code']).strip()
            else:
                suf = code.split('.')[-1]
                index_code = '000001.SH' if suf == 'SH' else '399001.SZ'
            stock_tag = code.split('.')[0]
            index_tag = index_code.split('.')[0]
            mv_csv = os.path.join(CSV_OUT_DIR, f'movement_{index_tag}_{stock_tag}.csv')
            out.append((code, name, mv_csv, index_tag, stock_tag, index_code))
        return out
    if not os.path.isdir(CSV_OUT_DIR):
        raise FileNotFoundError(f'{CSV_OUT_DIR} 不存在;先跑 batch --movement')
    out = []
    for fn in sorted(os.listdir(CSV_OUT_DIR)):
        if not fn.startswith('movement_') or not fn.endswith('.csv'):
            continue
        stem = fn[len('movement_'):-len('.csv')]
        parts = stem.split('_')
        if len(parts) < 2:
            continue
        index_tag = parts[0]
        stock_tag = '_'.join(parts[1:])
        suf = stock_tag[:6]
        code_guess = stock_tag + ('.SH' if suf.startswith(('6', '9', '5')) else '.SZ')
        mv_csv = os.path.join(CSV_OUT_DIR, fn)
        out.append((code_guess, None, mv_csv, index_tag, stock_tag, index_tag + '.SH'))
    return out


def main():
    args = parse_args()
    os.makedirs(CSV_OUT_DIR, exist_ok=True)

    # 1. 加载拟合结果
    kc_map = load_kc_estimates(status_filter=args.use_status_filter)
    if kc_map is None:
        raise SystemExit(
            f'{CSV_OUT_DIR}/kc_estimates.csv 不存在;先跑 parameter_fit.py'
        )
    print(f'已加载 {len(kc_map)} 条 (k̂, ĉ) (status_filter={args.use_status_filter!r})')

    # 2. 扫描目标
    targets = list_movement_csvs(args.input)
    if args.limit > 0:
        targets = targets[:args.limit]
    print(f'目标: {len(targets)} 只 (limit={args.limit})')

    # 3. 逐只预测
    summary_rows = []
    skipped = {'no_kc': 0, 'too_few': 0, 'failed': 0}
    for i, (code, name, mv_csv, index_tag, stock_tag, index_code) in enumerate(targets, 1):
        if (index_tag, stock_tag) not in kc_map:
            skipped['no_kc'] += 1
            continue
        kc = kc_map[(index_tag, stock_tag)]
        k_hat, c_hat = kc['k_hat'], kc['c_hat']
        try:
            pred_df, summary = predict_one(mv_csv, stock_tag, index_tag, k_hat, c_hat)
        except Exception as e:
            skipped['failed'] += 1
            print(f'[{i}/{len(targets)}] {code} ✗ {type(e).__name__}: {e}')
            continue
        if summary['n_valid_days'] < args.min_valid_days:
            skipped['too_few'] += 1
            continue
        # 落 per-stock CSV
        pred_csv = os.path.join(
            CSV_OUT_DIR, f'prediction_{index_tag}_{stock_tag}.csv'
        )
        pred_df.to_csv(pred_csv, index=False, encoding='utf-8')

        summary_rows.append({
            'code': code,
            'name': name or kc.get('name', ''),
            'index_code': index_code,
            'index_tag': index_tag,
            'stock_tag': stock_tag,
            'k_hat': k_hat,
            'c_hat': c_hat,
            'n_valid_days': summary['n_valid_days'],
            'dir_hit_rate': summary['dir_hit_rate'],
            'rmse': summary['rmse'],
            'mean_err_mag': summary['mean_err_mag'],
            'median_cos': summary['median_cos'],
        })
        print(
            f'[{i}/{len(targets)}] {code} '
            f'k={k_hat:+.4f} c={c_hat:+.4f} '
            f'命中率={summary["dir_hit_rate"]:.1%} '
            f'RMSE={summary["rmse"]:.2e} '
            f'cos={summary["median_cos"]:+.3f} '
            f'({summary["n_valid_days"]}d)'
        )

    # 4. 汇总
    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_path = os.path.join(CSV_OUT_DIR, PRED_OUT_NAME)
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f'\n=== 汇总 ===')
        print(f'  预测完成: {len(summary_rows)} 只')
        print(f'  无 kc_estimates: {skipped["no_kc"]} | '
              f'有效天数不足: {skipped["too_few"]} | '
              f'失败: {skipped["failed"]}')
        hit = out_df['dir_hit_rate'].to_numpy()
        rmse = out_df['rmse'].to_numpy()
        cos = out_df['median_cos'].to_numpy()
        print(f'  方向命中率: median={np.median(hit):.1%} '
              f'p25={np.percentile(hit, 25):.1%} '
              f'p75={np.percentile(hit, 75):.1%}')
        print(f'  RMSE:        median={np.median(rmse):.2e}')
        print(f'  median_cos:  median={np.median(cos):+.3f}')
        print(f'  清单: {out_path}')


if __name__ == '__main__':
    main()