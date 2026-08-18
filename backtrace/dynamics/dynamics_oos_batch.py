# -*- coding: utf-8 -*-
# dynamics_oos_batch.py — v5.10 Task 1: scaffold + per-stock metrics + aggregator
#
# 本模块给 v5.10 「全市场 OOS 分布」分析打地基:
#   1. compute_oos_metrics — 单股调用 load_oos_predictions(v5.9) → 算 hit/rmse/mae/dir_acc
#   2. aggregate_oos_metrics — 批量汇总 → median/quantiles/ranked
#
# 设计要点:
#   - 0 重写 projection / dynamics 数学(全部 import)
#   - 0 新依赖(numpy/pandas 已存在)
#   - 0 plotly(可视化留给 v5.10 后置 Task)
#   - REPO_ROOT sys.path 沿用 v5.9.1 修复模式
#
# 已知坑:
#   - load_oos_predictions 不传 lambda_q 时由 compute_dynamics 自适应(None = 自适应)
#   - M1 tsfresh shadow tolerated(详见 dynamics_1step_oos README)
#   - n_oos == 0 时全部返回 NaN(命中"零样本"边界)
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
REPO_ROOT = os.path.dirname(BACKTRACE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import argparse
import logging
import numpy as np
import pandas as pd

from backtrace.dynamics.dynamics_oos_viz import load_oos_predictions

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DEFAULTS = dict(
    days=250,
    limit=0,
    prefer_industry=True,
    top_n=5,
)
DEFAULT_OUTPUT = 'backtrace/outputs/dynsys_oos_full_market.html'


# === Per-stock OOS metrics =================================================
def compute_oos_metrics(
    stock_code: str,
    days: int = 250,
    *,
    prefer_industry: bool = True,
    k: float | None = None,
    c: float | None = None,
    f_self_window: int = 10,
) -> dict:
    """Per-stock OOS prediction quality metrics.

    Returns dict with keys:
        code: str
        n_oos: int
        hit_rate: float
        rmse: float
        mae: float
        direction_accuracy: float
        k_used: float
        c_used: float
    """
    # 1) 跑 v5.9 的对齐预测
    oos = load_oos_predictions(
        stock_code=stock_code,
        days=days,
        prefer_industry=prefer_industry,
        k=k,
        c=c,
        f_self_window=f_self_window,
    )

    common_idx = oos['common_idx']
    a_pred = oos['a_pred']
    a_actual = oos['a_actual']
    state_pred = oos['state_pred']
    state_actual = oos['state_actual']
    k_used = oos['k_used']
    c_used = oos['c_used']

    n_oos = len(common_idx)

    # 2) 零样本兜底(避免 NaN propagation)
    if n_oos == 0:
        return {
            'code': stock_code,
            'n_oos': 0,
            'hit_rate': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'direction_accuracy': float('nan'),
            'k_used': float(k_used),
            'c_used': float(c_used),
        }

    # 3) 幅度误差
    a_pred_mag = np.linalg.norm(a_pred, axis=1)
    a_actual_mag = np.linalg.norm(a_actual, axis=1)
    error = a_pred_mag - a_actual_mag

    # 4) rmse / mae(nan 防护,虽然本函数上游不应产生 NaN)
    rmse = float(np.sqrt(np.nanmean(np.square(error))))
    mae = float(np.nanmean(np.abs(error)))

    # 5) 状态命中率
    n_hit = sum(1 for sp, sa in zip(state_pred, state_actual) if sp == sa)
    hit_rate = float(n_hit / n_oos)

    # 6) 方向一致率(按加速度幅度符号)
    n_same_dir = sum(
        1
        for p, a in zip(a_pred_mag, a_actual_mag)
        if np.sign(p) == np.sign(a)
    )
    direction_accuracy = float(n_same_dir / n_oos)

    return {
        'code': stock_code,
        'n_oos': int(n_oos),
        'hit_rate': hit_rate,
        'rmse': rmse,
        'mae': mae,
        'direction_accuracy': direction_accuracy,
        'k_used': float(k_used),
        'c_used': float(c_used),
    }


# === Cross-stock aggregator =================================================
def aggregate_oos_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-stock metrics into population summary.

    Returns dict with:
        n_stocks: int
        median_hit_rate, p25_hit_rate, p75_hit_rate: float
        median_rmse, median_mae: float
        median_direction_acc: float
        ranked: list[dict]  # sorted by hit_rate desc
    """
    # 1) 空列表兜底
    if not metrics_list:
        return {
            'n_stocks': 0,
            'median_hit_rate': float('nan'),
            'p25_hit_rate': float('nan'),
            'p75_hit_rate': float('nan'),
            'median_rmse': float('nan'),
            'median_mae': float('nan'),
            'median_direction_acc': float('nan'),
            'ranked': [],
        }

    # 2) DataFrame + 中位数 / 四分位
    df = pd.DataFrame(metrics_list)
    ranked = df.sort_values('hit_rate', ascending=False).to_dict('records')

    return {
        'n_stocks': int(len(df)),
        'median_hit_rate': float(np.nanmedian(df['hit_rate'])),
        'p25_hit_rate': float(np.nanpercentile(df['hit_rate'], 25)),
        'p75_hit_rate': float(np.nanpercentile(df['hit_rate'], 75)),
        'median_rmse': float(np.nanmedian(df['rmse'])),
        'median_mae': float(np.nanmedian(df['mae'])),
        'median_direction_acc': float(np.nanmedian(df['direction_accuracy'])),
        'ranked': ranked,
    }


__all__ = [
    'compute_oos_metrics',
    'aggregate_oos_metrics',
    'DEFAULTS',
    'DEFAULT_OUTPUT',
]
