# -*- coding: utf-8 -*-
"""
06_factor_pool.py — 因子库管理:入库 / 去重 / 体检

读 05 产出的 factor_summary_*.csv + factor_r*_*.parquet,
对每个因子做:
  1. 重新计算样本内 / 外 IC / ICIR / 换手
  2. 入库门槛(cfg.IN_SAMPLE_ICIR_MIN 等)
  3. 与已入库因子做相关性去重(> MAX_CORR_WITH_POOL 剔除)
  4. 写入 factor_pool.csv(主表)
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
cfg     = import_module('00_config')
metrics = import_module('04_ic_metrics')


# ============================================================
# A. 找最新一次的挖掘产物
# ============================================================
def latest_run():
    """返回最新的 (summary_df, formulas_json_path, timestamp)"""
    summaries = sorted(cfg.FACTOR_DIR.glob('factor_summary_*.csv'))
    if not summaries:
        sys.exit(f"[FATAL] {cfg.FACTOR_DIR} 下找不到 factor_summary_*.csv,"
                 f"先跑 05_gp_mine.py")
    latest = summaries[-1]
    ts     = latest.stem.replace('factor_summary_', '')
    formulas_path = cfg.FACTOR_DIR / f"factor_formulas_{ts}.json"
    return pd.read_csv(latest), formulas_path, ts


def load_panel():
    """载入 01_data_prep 落盘的 panel;缺失时 sys.exit"""
    p = cfg.DATA_DIR / "panel.parquet"
    if not p.exists():
        sys.exit(f"[FATAL] 找不到 {p}")
    return pd.read_parquet(p)


# ============================================================
# B. 体检每个因子
# ============================================================
def evaluate_one(factor_df: pd.DataFrame, panel: pd.DataFrame, label: str, name: str):
    """合并因子值到 panel,跑全套指标"""
    df = panel[['date', 'code', label]].merge(factor_df, on=['date', 'code'], how='inner')
    df = df.rename(columns={'factor': name})
    return metrics.full_evaluate(df, name, label, name=name)


# ============================================================
# C. 入库决策
# ============================================================
def passes_gate(row: pd.Series) -> bool:
    """入库门槛"""
    if row['train_icir'] < cfg.IN_SAMPLE_ICIR_MIN:
        return False
    if abs(row['test_ic']) < cfg.OUT_SAMPLE_IC_MIN:
        return False
    if row['test_icir'] < cfg.OUT_SAMPLE_ICIR_MIN:
        return False
    if pd.notna(row.get('turnover')) and row['turnover'] > cfg.MAX_TURNOVER:
        return False
    return True


# ============================================================
# D. 相关性去重
# ============================================================
def decorrelate(accepted: list, pool_df: pd.DataFrame) -> list:
    """
    pool_df: 已经合并好的 (date, code, factor_i) 长表,各列是候选因子值
    accepted: 按 round 顺序已经接受的因子名 list
    返回:再筛一遍,剔除与已入库因子 corr > 阈值的
    """
    if not accepted:
        return accepted

    # 计算每个候选因子与已入库的截面平均相关系数
    final = []
    for cand in accepted:
        ok = True
        for exist in final:
            # 月度相关(快,稳)
            sub = pool_df[['date', cand, exist]].dropna()
            if len(sub) < 1000:
                continue
            sub['m'] = sub['date'].dt.to_period('M')
            monthly = sub.groupby('m').apply(
                lambda d: spearmanr(d[cand], d[exist])[0] if len(d) > 30 else np.nan
            )
            avg_corr = monthly.mean()
            if pd.notna(avg_corr) and abs(avg_corr) > cfg.MAX_CORR_WITH_POOL:
                print(f"  [去重] {cand} 与 {exist} 月度相关 {avg_corr:.3f},剔除")
                ok = False
                break
        if ok:
            final.append(cand)
    return final


# ============================================================
# E. 主流程
# ============================================================
def main():
    print("=" * 70)
    print("[06] 因子入库")
    print("=" * 70)

    summary, formulas_path, ts = latest_run()
    print(f"  最新一轮挖掘:{ts},共 {len(summary)} 个因子\n")

    panel   = load_panel()
    label   = cfg.LABEL_NAME
    pool_df = panel[['date', 'code', label]].copy()

    rows = []
    factor_data = {}    # name -> (train_df, test_df)

    for _, r in summary.iterrows():
        rid   = int(r['round'])
        name  = f"gp_r{rid}"
        tr_p  = cfg.FACTOR_DIR / f"factor_r{rid}_train_{ts}.parquet"
        te_p  = cfg.FACTOR_DIR / f"factor_r{rid}_test_{ts}.parquet"
        if not tr_p.exists() or not te_p.exists():
            print(f"  [跳过] r{rid}:parquet 缺失")
            continue

        tr_df = pd.read_parquet(tr_p)
        te_df = pd.read_parquet(te_p)

        # 训练 / 测试 评估
        tr_eval = evaluate_one(tr_df, panel, label, name)
        te_eval = evaluate_one(te_df, panel, label, name)
        row = {
            'name':       name,
            'round':      rid,
            'formula':    r['formula'],
            'prog_length':int(r['prog_length']),
            'train_ic':   float(tr_eval['ic_mean'].iloc[0])  if tr_eval['ic_mean'].notna().any() else None,
            'train_icir': float(tr_eval['icir'].iloc[0])    if tr_eval['icir'].notna().any()   else None,
            'test_ic':    float(te_eval['ic_mean'].iloc[0])  if te_eval['ic_mean'].notna().any() else None,
            'test_icir':  float(te_eval['icir'].iloc[0])    if te_eval['icir'].notna().any()   else None,
            'turnover':   float(te_eval['turnover'].iloc[0]) if pd.notna(te_eval['turnover'].iloc[0]) else None,
        }
        rows.append(row)
        factor_data[name] = (tr_df, te_df)

        # 加入 pool_df 给后续去重用
        tr_df2 = tr_df.rename(columns={'factor': name})
        pool_df = pool_df.merge(tr_df2, on=['date', 'code'], how='left')

    eval_df = pd.DataFrame(rows)
    print("\n=== 因子体检 ===")
    print(eval_df.to_string(index=False))

    # ---- 入库门槛 ----
    accepted = []
    print("\n=== 入库决策 ===")
    for _, r in eval_df.iterrows():
        if passes_gate(r):
            accepted.append(r['name'])
            print(f"  [✓] {r['name']}: 训练 ICIR={r['train_icir']:.2f}  "
                  f"测试 IC={r['test_ic']:.3f}  ICIR={r['test_icir']:.2f}")
        else:
            print(f"  [✗] {r['name']}: 不达标 (训练 ICIR={r['train_icir']:.2f}, "
                  f"测试 IC={r['test_ic']:.3f}, 测试 ICIR={r['test_icir']:.2f})")

    # ---- 相关性去重 ----
    if len(accepted) > 1:
        print("\n=== 相关性去重 ===")
        accepted = decorrelate(accepted, pool_df)

    print(f"\n最终入库:{len(accepted)} 个因子 → {accepted}")

    # ---- 落盘 ----
    pool_path = cfg.FACTOR_DIR / "factor_pool.csv"
    if pool_path.exists():
        old = pd.read_csv(pool_path)
        # 去掉同 ts 的旧条目
        old = old[old['timestamp'] != ts] if 'timestamp' in old.columns else old
        new_rows = eval_df[eval_df['name'].isin(accepted)].copy()
        new_rows['timestamp'] = ts
        final = pd.concat([old, new_rows], ignore_index=True)
    else:
        new_rows = eval_df[eval_df['name'].isin(accepted)].copy()
        new_rows['timestamp'] = ts
        final = new_rows

    final.to_csv(pool_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 因子库 → {pool_path}")

    # 复制入选因子的 parquet 到 pool 子目录,方便 07 回测直接读取
    pool_dir = cfg.FACTOR_DIR / "pool"
    pool_dir.mkdir(exist_ok=True)
    for name in accepted:
        tr_df, te_df = factor_data[name]
        tr_df.to_parquet(pool_dir / f"{name}_train.parquet", index=False)
        te_df.to_parquet(pool_dir / f"{name}_test.parquet", index=False)
    print(f"[OK] 入选因子 parquet → {pool_dir}")


if __name__ == "__main__":
    main()