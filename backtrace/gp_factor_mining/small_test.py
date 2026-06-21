# -*- coding: utf-8 -*-
"""
small_test.py — 端到端小规模冒烟测试

目的:
  1. 验证整个 pipeline (01→02→03→05) 跑得通
  2. 不依赖 TQ(本环境 TDX server 未启动),用合成数据
  3. 用极小参数 (200/5/2) 几分钟内出结果

合成数据设计:
  - 30 只"股票",每只 3 年 (≈ 750 个交易日)
  - 注入一个已知 alpha 信号:
        未来 20 日收益 ≈ 0.5 * ret_5 + 噪声
    让 GP 有机会把它挖出来
  - 不注入行业 / 市值 alpha,只用截面时序信号

运行:
    cd backtrace/gp_factor_mining
    python small_test.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys, os, time, json
from pathlib import Path

import numpy as np
import pandas as pd

# ========================= 配置 =========================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
cfg = import_module('00_config')

# 强制覆盖成"小规模"
cfg.USE_TQ                 = False
cfg.TRAIN_START            = "2018-01-01"
cfg.TRAIN_END              = "2020-06-30"
cfg.TEST_START             = "2020-07-01"
cfg.TEST_END               = "2021-06-30"
cfg.DATA_FETCH_START       = "2017-10-01"
cfg.HOLD_PERIOD            = 20
cfg.MIN_PRICE              = 1.0
cfg.POP_SIZE               = 200
cfg.N_GENERATIONS          = 5
cfg.PARSIMONY_COEFFICIENT  = 0.005
cfg.N_RESIDUAL_ROUNDS      = 2
cfg.MIN_IMPROVE_IC         = 0.001
cfg.RANDOM_STATE           = 42

prim       = import_module('02_primitive_set')
neutralize = import_module('03_neutralize')
metrics    = import_module('04_ic_metrics')


# ========================= A. 合成数据 =========================
def gen_synthetic_panel(n_stocks=30, n_days=1000, seed=42) -> pd.DataFrame:
    """
    合成多只股票日线 panel。注入已知 alpha:
        fwd_ret_20 ≈ 0.5 * ret_5 + 噪声
    """
    print(f"\n[GEN] 合成 {n_stocks} 只 × {n_days} 个交易日 ...")
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start='2017-10-01', periods=n_days)  # 工作日
    codes = [f"TEST{str(i).zfill(4)}.SZ" for i in range(n_stocks)]

    rows = []
    for c in codes:
        # 每只股票一个随机 drift / vol
        drift = rng.normal(0.0005, 0.002)
        vol   = abs(rng.normal(0.02, 0.005))
        s0    = rng.uniform(10, 50)
        rets  = rng.normal(drift, vol, n_days)
        close = s0 * (1 + pd.Series(rets)).cumprod()
        o = close * (1 + rng.normal(0, 0.003, n_days))
        h = np.maximum(o, close) * (1 + abs(rng.normal(0, 0.003, n_days)))
        l = np.minimum(o, close) * (1 - abs(rng.normal(0, 0.003, n_days)))
        v = rng.lognormal(15, 0.5, n_days)
        a = close * v
        df = pd.DataFrame({
            'date':   dates,
            'code':   c,
            'Open':   o,
            'High':   h,
            'Low':    l,
            'Close':  close,
            'Volume': v,
            'Amount': a,
        })
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(['code', 'date']).reset_index(drop=True)

    # 注入真实可挖的 alpha:让 fwd_ret_20 与 ret_5 强相关
    # 但要把 ret_5 当成"未来信息优势"被 GP 学到比较难,
    # 所以直接把 ret_5 的符号注入到未来收益里
    print("[GEN] 注入 alpha: fwd_ret_20 ← 0.5 * ret_5 + noise")
    g = panel.groupby('code', group_keys=False)
    panel['ret_5'] = panel['Close'] / g['Close'].shift(5) - 1
    panel['fwd_ret_20d'] = g['Close'].shift(-20) / panel['Close'] - 1
    # 重写:让 fwd_ret ≈ 0.5 * ret_5 + 0.01 * 随机噪声 (留些样本外不可预测的)
    noise = rng.normal(0, 0.04, len(panel))
    panel['fwd_ret_20d'] = 0.5 * panel['ret_5'].fillna(0) + noise
    # 清掉 NaN (头尾各 20 行)
    panel.loc[panel['ret_5'].isna(), 'fwd_ret_20d'] = np.nan
    panel.loc[panel.groupby('code').tail(20).index, 'fwd_ret_20d'] = np.nan

    print(f"[GEN] panel: {panel.shape}, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}, "
          f"{panel['code'].nunique()} 只")

    # 截面标准化 cs_*
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'fwd_ret_20d']
    for c in num_cols:
        col = panel[c]
        r = col.rank(method='first')
        mu, sd = r.mean(), r.std()
        panel[f'cs_{c}'] = (col.groupby(panel['date']).rank(method='first')
                              .groupby(panel['date']).transform(lambda s: (s - s.mean()) / s.std()))

    return panel


# ========================= B. 跑 GP 训练 =========================
def run_small_gp(panel: pd.DataFrame):
    """复用 05_gp_mine.py 的核心,但在小参数上跑"""
    from gplearn.genetic import SymbolicRegressor
    from gplearn.fitness import _Fitness
    from scipy.stats import spearmanr

    print("\n" + "="*70)
    print("[GP] 特征工程 → 切训练/测试")
    print("="*70)

    # 时序 + 截面原始特征
    panel = prim.add_timeseries_primitives(panel)
    panel = prim.add_crosssection_primitives(panel)

    # size / industry 代理
    panel = neutralize.add_size_proxy(panel)
    panel = neutralize.add_industry_proxy(panel)

    # 切 X/y
    X, y, meta, feat_cols = prim.build_xy(panel, cfg.LABEL_NAME)
    # build_xy 返回的 X/y 继承 panel 原索引,meta 已 reset;统一对齐
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    train_mask = (meta['date'] >= cfg.TRAIN_START) & (meta['date'] <= cfg.TRAIN_END)
    test_mask  = (meta['date'] >= cfg.TEST_START)  & (meta['date'] <= cfg.TEST_END)

    X_tr, y_tr, m_tr = X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True), meta[train_mask].reset_index(drop=True)
    X_te, y_te, m_te = X[test_mask].reset_index(drop=True),  y[test_mask].reset_index(drop=True),  meta[test_mask].reset_index(drop=True)

    print(f"\n  训练: {len(X_tr):,} 行 ({m_tr['date'].min().date()} ~ {m_tr['date'].max().date()})")
    print(f"  测试: {len(X_te):,} 行 ({m_te['date'].min().date()} ~ {m_te['date'].max().date()})")
    print(f"  Terminal: {len(feat_cols)} 个")
    if len(X_tr) < 100:
        print("[FATAL] 训练集太小,无法训练")
        return

    # ---- 单轮 GP ----
    funcs = prim.make_function_set()

    # gplearn 0.4.3: 自定义 metric 必须包成 _Fitness(给函数 + 方向)
    def _abs_spearman(y_true, y_pred, sample_weight=None):
        if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
            return 0.0
        rho, _ = spearmanr(y_true, y_pred)
        return abs(rho) if rho == rho else 0.0  # NaN guard

    rankic_fitness = _Fitness(function=_abs_spearman, greater_is_better=True)

    print(f"\n[GP] 训练 (pop={cfg.POP_SIZE}, gen={cfg.N_GENERATIONS}) ...")
    t0 = time.time()
    reg = SymbolicRegressor(
        population_size       = cfg.POP_SIZE,
        generations           = cfg.N_GENERATIONS,
        tournament_size       = cfg.TOURNAMENT_SIZE,
        function_set          = funcs,
        metric                = rankic_fitness,
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
        random_state          = cfg.RANDOM_STATE,
        n_jobs                = -1,
        verbose               = 0,
        low_memory            = True,
    )
    reg.fit(X_tr.values, y_tr.values)
    train_sec = time.time() - t0
    print(f"[GP] 训练耗时 {train_sec:.1f}s, "
          f"最佳程序长度={reg._program.length_}, depth={reg._program.depth_}")

    # ---- 预测 + 评估 ----
    pred_tr = reg.predict(X_tr.values)
    pred_te = reg.predict(X_te.values)

    def _eval(y_pred, y_true, m, tag):
        sub = pd.DataFrame({'date': m['date'].values, 'code': m['code'].values,
                            'factor': y_pred, 'label': y_true.values})
        ic_ts = metrics.daily_rankic(sub, 'factor', 'label')
        s = metrics.ic_summary(ic_ts)
        print(f"  {tag:<8} |IC|={s['ic_mean']:+.4f}  ICIR={s['icir']:+.3f}  "
              f"|IC|>0 占比={(ic_ts>0).mean():.2%}  n_days={s['n_days']}")
        return s

    print("\n[GP] 评估:")
    s_tr = _eval(pred_tr, y_tr, m_tr, "训练")
    s_te = _eval(pred_te, y_te, m_te, "测试")

    formula = str(reg._program)
    print(f"\n[GP] 最佳公式: {formula}")
    print(f"     (长度 {reg._program.length_}, 深度 {reg._program.depth_})")

    # ---- 落盘 ----
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    summary_path = cfg.FACTOR_DIR / f"small_test_summary_{timestamp}.csv"
    pd.DataFrame([{
        'stage':  'train', 'ic_mean': s_tr.get('ic_mean'),
        'icir':   s_tr.get('icir'), 'n_days': s_tr.get('n_days'),
    }, {
        'stage':  'test',  'ic_mean': s_te.get('ic_mean'),
        'icir':   s_te.get('icir'), 'n_days': s_te.get('n_days'),
    }]).to_csv(summary_path, index=False, encoding='utf-8-sig')

    formula_path = cfg.FACTOR_DIR / f"small_test_formula_{timestamp}.txt"
    with open(formula_path, 'w', encoding='utf-8') as f:
        f.write(f"# small_test GP best formula\n")
        f.write(f"# train: {cfg.TRAIN_START} ~ {cfg.TRAIN_END}\n")
        f.write(f"# test:  {cfg.TEST_START} ~ {cfg.TEST_END}\n")
        f.write(f"# train_ic={s_tr.get('ic_mean'):.4f} icir={s_tr.get('icir'):.3f}\n")
        f.write(f"# test_ic ={s_te.get('ic_mean'):.4f} icir={s_te.get('icir'):.3f}\n")
        f.write(f"# length={reg._program.length_}, depth={reg._program.depth_}\n\n")
        f.write(formula + "\n")

    # 因子值也落盘
    pd.DataFrame({
        'date': m_tr['date'].values, 'code': m_tr['code'].values,
        'factor': pred_tr, 'split': 'train',
    }).to_parquet(cfg.FACTOR_DIR / f"small_test_factor_train_{timestamp}.parquet", index=False)
    pd.DataFrame({
        'date': m_te['date'].values, 'code': m_te['code'].values,
        'factor': pred_te, 'split': 'test',
    }).to_parquet(cfg.FACTOR_DIR / f"small_test_factor_test_{timestamp}.parquet", index=False)

    print(f"\n[OK] 公式 → {formula_path}")
    print(f"[OK] 摘要 → {summary_path}")
    print(f"[OK] 因子值 → small_test_factor_{{train,test}}_{timestamp}.parquet")

    return {
        'train_ic':  s_tr.get('ic_mean'),
        'train_icir': s_tr.get('icir'),
        'test_ic':   s_te.get('ic_mean'),
        'test_icir': s_te.get('icir'),
        'formula':   formula,
        'train_sec': train_sec,
        'timestamp': timestamp,
    }


# ========================= C. 入口 =========================
def main():
    print("="*70)
    print("[SMALL TEST] GP 因子挖掘 — 冒烟测试")
    print(f"  训练:{cfg.TRAIN_START} ~ {cfg.TRAIN_END}")
    print(f"  测试:{cfg.TEST_START}  ~ {cfg.TEST_END}")
    print(f"  pop={cfg.POP_SIZE}, gen={cfg.N_GENERATIONS}, rounds={cfg.N_RESIDUAL_ROUNDS}")
    print("="*70)

    # 1) 合成数据并落盘
    panel = gen_synthetic_panel(n_stocks=30, n_days=1000)
    panel_path = cfg.DATA_DIR / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    print(f"[OK] 合成 panel 已落盘 → {panel_path}  (shape={panel.shape})")

    # 2) 跑 GP
    res = run_small_gp(panel)

    print("\n" + "="*70)
    print("[DONE] 小规模测试完成")
    print("="*70)
    if res:
        print(f"  训练 |IC| = {res['train_ic']:+.4f}  ICIR = {res['train_icir']:+.3f}")
        print(f"  测试 |IC| = {res['test_ic']:+.4f}  ICIR = {res['test_icir']:+.3f}")
        print(f"  训练耗时 {res['train_sec']:.1f}s")
        print(f"  公式: {res['formula'][:200]}")
        # 合理性判断
        train_ok = abs(res['train_ic']) > 0.03 and abs(res['train_icir']) > 0.5
        if train_ok:
            print("\n  [OK] 训练期 IC / ICIR 看起来合理")
        else:
            print("\n  [WARN] 训练期 IC / ICIR 偏低(合成数据应该能挖出来,可能参数要调)")

if __name__ == "__main__":
    main()