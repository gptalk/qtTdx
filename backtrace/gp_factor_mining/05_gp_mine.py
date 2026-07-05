# -*- coding: utf-8 -*-
"""
05_gp_mine.py — GP 多轮残差因子挖掘(核心)

流程:
  Round 1: y0 = fwd_ret_20d → 训练 SymbolicRegressor → best_factor_1
  Round 2: y1 = y0 - best_factor_1  → 训练 → best_factor_2
  ...
  Round K: y_K = y_(K-1) - best_factor_K  → ...

每轮产出:
  - 因子公式(program)
  - 训练期/测试期 IC 摘要
  - 因子值长表(parquet)
  - 边际 IC 增益

依赖:gplearn(`pip install gplearn`)
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# gplearn 可选:装了才能跑,没装给出清晰报错
try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import _Fitness
    HAS_GPLEARN = True
except Exception:
    HAS_GPLEARN = False

# sklearn fallback:没 gplearn 也能用 ElasticNet 顶一下(线性基线)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
cfg        = import_module('00_config')
prim       = import_module('02_primitive_set')
metrics    = import_module('04_ic_metrics')
neutralize = import_module('03_neutralize')


# ============================================================
# A. 数据准备
# ============================================================
def load_panel():
    """载入 01_data_prep 落盘的 panel"""
    p = cfg.DATA_DIR / "panel.parquet"
    if not p.exists():
        sys.exit(f"[FATAL] 找不到 {p},先跑 01_data_prep.py")
    return pd.read_parquet(p)


def make_xy(panel: pd.DataFrame):
    """
    切训练 / 测试,构造 (X_train, y_train) / (X_test, y_test)
    + meta(date, code)
    """
    panel = prim.add_timeseries_primitives(panel)
    panel = prim.add_crosssection_primitives(panel)
    panel = neutralize.add_size_proxy(panel)
    panel = neutralize.add_industry_proxy(panel)

    X, y, meta, feat_cols = prim.build_xy(panel, cfg.LABEL_NAME)

    train_mask = (meta['date'] >= cfg.TRAIN_START) & (meta['date'] <= cfg.TRAIN_END)
    test_mask  = (meta['date'] >= cfg.TEST_START)  & (meta['date'] <= cfg.TEST_START)
    # 注:TEST_START 含以后;如果用户不想这样,把 TEST_END 也加进条件

    # 安全:训练 / 测试 内部还可以再细分一份 hold-out
    X_tr, y_tr, m_tr = X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True), meta[train_mask].reset_index(drop=True)
    X_te, y_te, m_te = X[test_mask].reset_index(drop=True),  y[test_mask].reset_index(drop=True),  meta[test_mask].reset_index(drop=True)

    print(f"\n  训练集: {len(X_tr):,} 行  "
          f"({m_tr['date'].min().date() if len(m_tr) else 'N/A'} ~ "
          f"{m_tr['date'].max().date() if len(m_tr) else 'N/A'})")
    print(f"  测试集: {len(X_te):,} 行  "
          f"({m_te['date'].min().date() if len(m_te) else 'N/A'} ~ "
          f"{m_te['date'].max().date() if len(m_te) else 'N/A'})")
    print(f"  Terminal 数: {len(feat_cols)}\n")

    return X_tr, y_tr, m_tr, X_te, y_te, m_te, feat_cols


# ============================================================
# B. 单轮 GP 训练
# ============================================================
def train_one_round(X: pd.DataFrame, y: pd.Series, feat_cols, round_idx: int,
                    sample_frac: float = 0.30):
    """
    用 gplearn SymbolicRegressor 跑一代进化。
    训练数据太大时,随机抽 sample_frac 提速(默认 30%)。
    """
    if not HAS_GPLEARN:
        raise RuntimeError("gplearn 未安装,先 `pip install gplearn`")

    funcs = prim.make_function_set()

    # 抽样(防内存 / 提速)
    if sample_frac < 1.0 and len(X) > 50000:
        n = int(len(X) * sample_frac)
        idx = np.random.RandomState(cfg.RANDOM_STATE + round_idx).choice(
            len(X), size=n, replace=False
        )
        Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    else:
        Xs, ys = X, y

    print(f"  round {round_idx}: 抽样 {len(Xs):,} / {len(X):,} 行  "
          f"种群 {cfg.POP_SIZE}  代数 {cfg.N_GENERATIONS}")

    # 用 |Spearman| 当 fitness(gplearn 默认 MSE;这里关心秩相关)
    from scipy.stats import spearmanr
    def _abs_spearman_fitness(y_true, y_pred, sample_weight=None):
        if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
            return 0.0
        rho, _ = spearmanr(y_true, y_pred)
        return float(abs(rho)) if rho == rho else 0.0  # NaN guard
    rankic_fit = _Fitness(function=_abs_spearman_fitness, greater_is_better=True)

    reg = SymbolicRegressor(
        population_size       = cfg.POP_SIZE,
        generations           = cfg.N_GENERATIONS,
        tournament_size       = cfg.TOURNAMENT_SIZE,
        function_set          = funcs,
        metric                = rankic_fit,
        stopping_criteria     = 0.005,
        const_range           = (-2.0, 2.0),
        parsimony_coefficient = cfg.PARSIMONY_COEFFICIENT,
        p_crossover           = cfg.P_CROSSOVER,
        p_subtree_mutation    = cfg.P_SUBTREE_MUTATION,
        p_hoist_mutation      = cfg.P_HOIST_MUTATION,
        p_point_mutation      = cfg.P_POINT_MUTATION,
        p_point_replace       = cfg.P_POINT_REPLACE,
        init_depth            = (2, cfg.MAX_DEPTH_INIT),
        init_method           = 'half and half',
        feature_names         = list(X.columns),
        random_state          = cfg.RANDOM_STATE + round_idx,
        n_jobs                = -1,
        verbose               = 0,
        low_memory            = True,
    )

    t0 = time.time()
    reg.fit(Xs.values, ys.values)
    print(f"  round {round_idx}: 训练 {time.time()-t0:.1f}s, "
          f"最佳程序长度={reg._program.length_}, depth={reg._program.depth_}")

    return reg


# ============================================================
# C. 多轮残差挖掘主循环
# ============================================================
def multi_round_residual_mine(X_tr, y_tr, m_tr, X_te, y_te, m_te, feat_cols):
    """
    多轮残差 GP 挖掘主流程。
    返回:list of dict,每个 dict 是一轮的产出。
    """
    if not HAS_GPLEARN:
        sys.exit("[FATAL] 本模块依赖 gplearn,请先 `pip install gplearn`")

    np.random.seed(cfg.RANDOM_STATE)

    y_train_cur = y_tr.copy().values
    y_test_cur  = y_te.copy().values

    # 累计预测(测试集用)
    cum_pred_te = np.zeros_like(y_test_cur, dtype=np.float64)

    rounds_out = []
    prev_icir  = 0.0

    for r in range(1, cfg.N_RESIDUAL_ROUNDS + 1):
        print(f"\n{'='*70}\n[Round {r}/{cfg.N_RESIDUAL_ROUNDS}]\n{'='*70}")

        # 训练
        reg = train_one_round(X_tr, pd.Series(y_train_cur), feat_cols, round_idx=r)

        # 预测(训练集残差 + 测试集)
        pred_tr = reg.predict(X_tr.values).astype(np.float64)
        pred_te = reg.predict(X_te.values).astype(np.float64)

        # 累计预测(测试集)
        cum_pred_te += pred_te

        # ---- 评估这一轮 ----
        # 训练:本轮预测 vs 当前残差
        ic_tr_ts = metrics.daily_rankic(
            pd.DataFrame({'date': m_tr['date'], 'code': m_tr['code'],
                          'factor': pred_tr, 'label': y_train_cur}),
            'factor', 'label'
        )
        summ_tr = metrics.ic_summary(ic_tr_ts)

        # 测试:累计预测 vs 真实标签
        ic_te_ts = metrics.daily_rankic(
            pd.DataFrame({'date': m_te['date'], 'code': m_te['code'],
                          'factor': cum_pred_te, 'label': y_te.values}),
            'factor', 'label'
        )
        summ_te = metrics.ic_summary(ic_te_ts)

        marginal_icir = summ_tr['icir'] - prev_icir
        print(f"  训练期 |IC|={summ_tr['ic_mean']:.4f}  ICIR={summ_tr['icir']:.3f}  "
              f"边际={marginal_icir:.3f}")
        print(f"  测试期 |IC|={summ_te['ic_mean']:.4f}  ICIR={summ_te['icir']:.3f}")

        # ---- 公式字符串 ----
        formula = str(reg._program)
        print(f"\n  Round {r} 公式: {formula[:120]}{'...' if len(formula)>120 else ''}")

        rounds_out.append({
            'round':        r,
            'formula':      formula,
            'program_length': reg._program.length_,
            'program_depth':  reg._program.depth_,
            'train':        summ_tr,
            'test':         summ_te,
            'marginal_icir': marginal_icir,
            'pred_tr':      pred_tr,
            'pred_te':      pred_te,
            'ic_tr_ts':     ic_tr_ts,
            'ic_te_ts':     ic_te_ts,
        })

        # ---- 残差更新:训练集 ----
        y_train_cur = y_train_cur - pred_tr
        prev_icir   = summ_tr['icir']

        # 早停:边际 ICIR 不再增长
        if r > 1 and marginal_icir < cfg.MIN_IMPROVE_IC:
            print(f"\n  [早停] 边际 ICIR {marginal_icir:.4f} < {cfg.MIN_IMPROVE_IC},停止")
            break

    return rounds_out


# ============================================================
# D. 结果保存
# ============================================================
def save_results(rounds_out, m_tr, m_te, X_tr, X_te, feat_cols):
    """把每轮产物落盘到 cfg.FACTOR_DIR"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary_rows = []
    for r in rounds_out:
        s_tr, s_te = r['train'], r['test']
        summary_rows.append({
            'round':          r['round'],
            'formula':        r['formula'],
            'prog_length':    r['program_length'],
            'prog_depth':     r['program_depth'],
            'train_ic':       s_tr.get('ic_mean'),
            'train_icir':     s_tr.get('icir'),
            'train_n_days':   s_tr.get('n_days'),
            'test_ic':        s_te.get('ic_mean'),
            'test_icir':      s_te.get('icir'),
            'test_n_days':    s_te.get('n_days'),
            'marginal_icir':  r['marginal_icir'],
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = cfg.FACTOR_DIR / f"factor_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 因子汇总 → {summary_path}")

    # 每轮的因子值(长表)
    for r in rounds_out:
        rid = r['round']
        # 训练
        tr_df = pd.DataFrame({
            'date': m_tr['date'].values,
            'code': m_tr['code'].values,
            'factor': r['pred_tr'],
        })
        tr_df.to_parquet(cfg.FACTOR_DIR / f"factor_r{rid}_train_{timestamp}.parquet", index=False)
        # 测试
        te_df = pd.DataFrame({
            'date': m_te['date'].values,
            'code': m_te['code'].values,
            'factor': r['pred_te'],
        })
        te_df.to_parquet(cfg.FACTOR_DIR / f"factor_r{rid}_test_{timestamp}.parquet", index=False)

    # 公式 dump(JSON)
    formula_path = cfg.FACTOR_DIR / f"factor_formulas_{timestamp}.json"
    formulas = {f"r{r['round']}": r['formula'] for r in rounds_out}
    with open(formula_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'train_window': [cfg.TRAIN_START, cfg.TRAIN_END],
                'test_window':  [cfg.TEST_START,  cfg.TEST_END],
                'hold_period':  cfg.HOLD_PERIOD,
                'pop_size':     cfg.POP_SIZE,
                'generations':  cfg.N_GENERATIONS,
                'n_rounds':     len(rounds_out),
            },
            'formulas': formulas,
            'feat_cols': list(feat_cols),
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] 公式 dump → {formula_path}")
    return timestamp


# ============================================================
# E. 入口
# ============================================================
def main():
    print("=" * 70)
    print("[05] GP 多轮残差因子挖掘")
    print(f"     训练: {cfg.TRAIN_START} ~ {cfg.TRAIN_END}")
    print(f"     测试: {cfg.TEST_START} ~ {cfg.TEST_END}")
    print(f"     标签: {cfg.LABEL_NAME}  种群: {cfg.POP_SIZE}  "
          f"轮数: {cfg.N_RESIDUAL_ROUNDS}")
    print("=" * 70)

    if not HAS_GPLEARN:
        sys.exit("[FATAL] gplearn 未安装:\n"
                 "    pip install gplearn\n"
                 "装完再跑本模块。")

    panel = load_panel()
    X_tr, y_tr, m_tr, X_te, y_te, m_te, feat_cols = make_xy(panel)

    if len(X_tr) == 0 or len(X_te) == 0:
        sys.exit("[FATAL] 训练集或测试集为空,检查时间窗")

    rounds_out = multi_round_residual_mine(X_tr, y_tr, m_tr, X_te, y_te, m_te, feat_cols)

    ts = save_results(rounds_out, m_tr, m_te, X_tr, X_te, feat_cols)

    # 摘要打印
    print("\n" + "=" * 70)
    print(f"[05] 完成!timestamp={ts}")
    print("=" * 70)
    print("\n各轮 |IC| / ICIR:")
    for r in rounds_out:
        s_tr, s_te = r['train'], r['test']
        print(f"  r{r['round']}:  训练 IC={s_tr['ic_mean']:+.4f}  "
              f"ICIR={s_tr['icir']:+.3f}  |  "
              f"测试 IC={s_te['ic_mean']:+.4f}  "
              f"ICIR={s_te['icir']:+.3f}  |  "
              f"边际={r['marginal_icir']:+.3f}")

    print(f"\n下一步:跑 06_factor_pool.py 入库;07_backtest.py 出回测。")


if __name__ == "__main__":
    main()