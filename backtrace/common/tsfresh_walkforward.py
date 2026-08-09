# -*- coding: utf-8 -*-
"""tsfresh walk-forward proba 生成 + MA 通道工具。

约定(写在本模块 docstring 顶部,3 个原脚本迁移后共享):
  1. **bfill 而非 fillna(0.0)** — 避免序列开头 0 → 实值跳变被 tsfresh 当成「突变」特征。
     理由见原 with_ma_channel.py:96-98 注释。
  2. **FDR 严格在 init_train_size 段筛** — 防止特征选择时用上 walk-forward 期内的未来样本
     (原 grid_sector 在 X_all 上筛有泄漏风险,迁移后自动修好)。
  3. **X_sel 列对齐** — 在全期 X_all[selected_cols] 上做,索引仍覆盖全期。
"""
import numpy as np
import pandas as pd
import vectorbt as vbt

from common import tsfresh_pipeline as P


def add_ma_channels(ohlcv_df, windows=(5, 10, 20), add_rel=True):
    """原地加 ma5/ma10/ma20 + rel_ma5(Close 相对 ma5 的偏离度)。
    复制 df 后修改,避免污染上游调用方。
    使用 bfill(替代原 grid_sector 的 fillna(0.0))——见 with_ma_channel.py:96-98 注释。
    """
    out = ohlcv_df.copy()
    for w in windows:
        out[f'ma{w}'] = vbt.MA.run(out['Close'], window=w).ma.bfill()
    if add_rel:
        out['rel_ma5'] = ((out['Close'] - out['ma5']) / out['ma5'].replace(0, np.nan)).bfill()
    return out


def report_channel_composition(X_sel, label=''):
    """统计 X_sel 各通道入选特征数。若 ma*/rel_ma5 占比 > 33% 打印 [WARN] 冗余风险。
    从原 with_ma_channel.py:_report_channel_composition 搬过来。
    """
    if X_sel.shape[1] == 0:
        return
    channels = pd.Series([col.split('__', 1)[0] for col in X_sel.columns])
    counts = channels.value_counts()
    prefix = f'[{label}] ' if label else ''
    print(f'   {prefix}各通道入选特征数:')
    for ch, n in counts.items():
        pct = n / len(channels) * 100
        print(f'     {ch:<10} {n:>4}  ({pct:5.1f}%)')
    ma_total = sum(counts.get(c, 0) for c in counts.index
                   if c.startswith('ma') or c == 'rel_ma5')
    if ma_total > 0 and ma_total / len(channels) > 0.33:
        print(f'   [WARN] MA 相关通道占 {ma_total/len(channels):.0%},'
              f'可能与 Close 通道冗余')


def tsfresh_walkforward_proba(ohlcv_df, channels, *,
                              init_train_size=200, step=50,
                              fillna='bfill', id_value=None, verbose=True):
    """跑 tsfresh 全流程 → 返回 (proba, X_sel)。

    行为约定:
      - **FDR 在 X.iloc[:init_train_size] 段筛**(防泄漏)
      - **bfill 替代 fillna(0.0)**(防早期跳变特征)
      - **walk-forward**:初始 init_train_size,之后每 step 重训一次
      - **proba 在 date_index[end_t] 当日计算**,下游 shift(1) 视作次日开盘成交

    Raises:
      ValueError: 样本数 < init_train_size(无法满足 FDR 限制 + 留 walk-forward 起点)
    """
    if len(ohlcv_df) < init_train_size + step:
        raise ValueError(
            f'样本数 {len(ohlcv_df)} < init_train_size({init_train_size}) + step({step}),'
            f'无法跑 walk-forward'
        )

    df_fill = ohlcv_df[channels].copy()
    if fillna == 'bfill':
        df_fill = df_fill.bfill()
    elif fillna == 'zero':
        df_fill = df_fill.fillna(0.0)
    else:
        raise ValueError(f"fillna 必须是 'bfill' 或 'zero',收到 {fillna!r}")

    if verbose:
        print(f'   通道 {len(channels)} 个: {channels}')

    id_val = id_value if id_value is not None else (ohlcv_df.name or 'X')
    long_df = P.to_long_format(df_fill, channels=channels, id_value=id_val)
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, ohlcv_df['Close'].values, verbose=False)
    if verbose:
        print(f'   样本 {len(y_all)} 个  |  正样本 {y_all.mean():.1%}  |  '
              f'特征 {X_all.shape[1]} 列')

    # FDR 严格限制在前 init_train_size 段(防未来信息泄漏)
    X_train0 = X_all.iloc[:init_train_size]
    y_train0 = y_all.iloc[:init_train_size]
    X_sel_initial = P.select_relevant(X_train0, y_train0, verbose=False)
    selected_cols = X_sel_initial.columns.tolist()
    if len(selected_cols) == 0:
        if verbose:
            print(f'   [WARN] FDR=0,用全量 {X_all.shape[1]} 特征')
        X_sel = X_all
    else:
        X_sel = X_all[selected_cols]
    if verbose:
        print(f'   FDR 显著 {X_sel.shape[1]} 列 (前 {init_train_size} 个样本筛)')

    date_index = pd.DatetimeIndex(ohlcv_df.index)
    proba_records = []
    scaler_w = clf_w = None
    for pos, idx in enumerate(X_sel.index):
        end_t = idx[1]
        if end_t >= len(date_index):
            continue
        if pos < init_train_size:
            proba_records.append((date_index[end_t], np.nan))
            continue
        if pos == init_train_size or (pos - init_train_size) % step == 0:
            scaler_w, clf_w = P.fit_logreg(X_sel.iloc[:pos], y_all.iloc[:pos],
                                            verbose=False)
        p = float(clf_w.predict_proba(
            scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
        proba_records.append((date_index[end_t], p))

    proba = pd.Series([v for _, v in proba_records],
                      index=pd.DatetimeIndex([d for d, _ in proba_records]),
                      name='proba').sort_index()
    proba = proba[~proba.index.duplicated(keep='last')].dropna()
    if verbose:
        print(f'   proba {len(proba)} 个  |  mean={proba.mean():.3f}')
    return proba, X_sel
