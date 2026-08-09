# -*- coding: utf-8 -*-
"""Shared math + I/O for projection_2d.py and projection_batch.py.

Single source of truth for: market→index map, local-cache loading,
2-D vector projection math, and 19-column result DataFrame assembly.

No plotly / HTML / file writes — those are the calling scripts' jobs.
"""
import numpy as np
import pandas as pd


# 市场 → 大盘指数(Code, Name)。改个股交易所后缀即自动切换大盘。
MARKET_TO_INDEX = {
    'SZ': ('399001.SZ', '深证成指'),
    'SH': ('000001.SH', '上证综指'),
}


def resolve_index(stock_code):
    """由 STOCK_CODE 后缀派生 (INDEX_CODE, INDEX_NAME);未知后缀抛 ValueError。"""
    suffix = stock_code.split('.')[-1]
    if suffix not in MARKET_TO_INDEX:
        raise ValueError(
            f"未识别 STOCK_CODE 后缀: {stock_code!r}\n"
            f"支持: {sorted(MARKET_TO_INDEX)} (对应 深证成指 / 上证综指)"
        )
    return MARKET_TO_INDEX[suffix]


def project_u_onto_v(u, v):
    """2-D 向量 u 投影到 v;v 零向量返回零向量。"""
    v_norm_sq = np.dot(v, v)
    if v_norm_sq == 0:
        return np.zeros_like(u)
    return (np.dot(u, v) / v_norm_sq) * v


def _safe_ratio(num, den, default=np.nan):
    """标量除法,安全处理 0/NaN/Inf — 不会触发 numpy RuntimeWarning。

    无效输入 → 返回 default(np.nan),由调用方决定过滤或填值。
    batch 跑 5500+ 只票时,NaN 安全能避免日志被 RuntimeWarning 刷屏。
    """
    if not np.isfinite(den) or den == 0:
        return default
    res = num / den
    if not np.isfinite(res):
        return default
    return res


def load_pair(stock_code, days, pipeline):
    """从本地 data/ 缓存加载 (stock_df, index_df) 共同交易日的最近 `days` 行。

    Returns dict: stock_df, index_df, common_idx, index_code, index_name,
                  index_tag, stock_tag。
    Raises RuntimeError if either cache entry is missing.
    """
    index_code, index_name = resolve_index(stock_code)
    index_tag = index_code.split('.')[0]
    stock_tag = stock_code.split('.')[0]

    data_index_full = pipeline.load_ohlcva(index_code, use_tq=False, verbose=True)
    data_stock_full = pipeline.load_ohlcva(stock_code, use_tq=False, verbose=True)
    if data_index_full is None:
        raise RuntimeError(
            f"本地缓存缺失 {index_code}。请先跑:\n"
            f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
        )
    if data_stock_full is None:
        raise RuntimeError(
            f"本地缓存缺失 {stock_code}。请先跑:\n"
            f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py"
        )

    data_index = data_index_full[['Volume', 'Amount', 'Close']].tail(days).dropna()
    data_stock = data_stock_full[['Volume', 'Amount', 'Close']].tail(days).dropna()
    common_idx = data_index.index.intersection(data_stock.index)
    return {
        'stock_df': data_stock.loc[common_idx],
        'index_df': data_index.loc[common_idx],
        'common_idx': common_idx,
        'index_code': index_code,
        'index_name': index_name,
        'index_tag': index_tag,
        'stock_tag': stock_tag,
    }


def _safe_minmax(values, v_min, v_range):
    """Min-Max 归一化:max == min(常数列,如停牌 / 无成交)→ 整列归零。
    否则 0/0 = NaN 污染后续所有投影系数和残差。
    """
    if not np.isfinite(v_range) or v_range == 0:
        return np.zeros_like(values)
    return (values - v_min) / v_range


def compute_vectors(stock_df, index_df, index_tag, stock_tag):
    """Min-Max 归一化 Vol/Amt。返回 (vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params_str)。"""
    vec_index = index_df[['Volume', 'Amount']].values
    vec_stock = stock_df[['Volume', 'Amount']].values

    vol_min_ix, vol_max_ix = vec_index[:, 0].min(), vec_index[:, 0].max()
    amt_min_ix, amt_max_ix = vec_index[:, 1].min(), vec_index[:, 1].max()
    vol_min_st, vol_max_st = vec_stock[:, 0].min(), vec_stock[:, 0].max()
    amt_min_st, amt_max_st = vec_stock[:, 1].min(), vec_stock[:, 1].max()

    vec_index_norm = np.column_stack([
        _safe_minmax(vec_index[:, 0], vol_min_ix, vol_max_ix - vol_min_ix),
        _safe_minmax(vec_index[:, 1], amt_min_ix, amt_max_ix - amt_min_ix),
    ])
    vec_stock_norm = np.column_stack([
        _safe_minmax(vec_stock[:, 0], vol_min_st, vol_max_st - vol_min_st),
        _safe_minmax(vec_stock[:, 1], amt_min_st, amt_max_st - amt_min_st),
    ])

    norm_params = (
        f"vol_{index_tag}:[{vol_min_ix:.2e},{vol_max_ix:.2e}] "
        f"amt_{index_tag}:[{amt_min_ix:.2e},{amt_max_ix:.2e}] "
        f"vol_{stock_tag}:[{vol_min_st:.2e},{vol_max_st:.2e}] "
        f"amt_{stock_tag}:[{amt_min_st:.2e},{amt_max_st:.2e}]"
    )
    return vec_index, vec_stock, vec_index_norm, vec_stock_norm, norm_params


def compute_projections(vec_stock_norm, vec_index_norm):
    """对每行跑 project_u_onto_v,返回 7 个 np.array。"""
    projections, residuals, dot_after = [], [], []
    proj_coeffs, proj_mags, proj_prices, resi_prices = [], [], [], []

    for i in range(len(vec_stock_norm)):
        u = vec_stock_norm[i]
        v = vec_index_norm[i]
        proj = project_u_onto_v(u, v)
        residual = u - proj
        projections.append(proj)
        residuals.append(residual)
        dot_after.append(np.dot(residual, v))
        proj_coeffs.append(_safe_ratio(np.dot(u, v), np.dot(v, v), default=0.0))
        proj_mags.append(np.linalg.norm(proj))
        proj_prices.append(_safe_ratio(proj[1], proj[0], default=np.sign(proj[1]) if np.isfinite(proj[1]) else 0.0))
        resi_price = _safe_ratio(residual[1], residual[0])
        # NaN 或 | > 3 视为 Volume≈0 导致的异常比值,限幅到已算值
        if not np.isfinite(resi_price) or abs(resi_price) > 3:
            past = np.abs(resi_prices[:-2]) if len(resi_prices) > 2 else None
            cap = float(np.nanmax(past)) if past is not None and len(past) > 0 and np.any(np.isfinite(past)) else 0.0
            sign = np.sign(residual[1]) if np.isfinite(residual[1]) else 0.0
            resi_price = sign * cap
        resi_prices.append(resi_price)

    return {
        'projections': np.array(projections),
        'residuals': np.array(residuals),
        'dot_after': np.array(dot_after),
        'proj_coeffs': np.array(proj_coeffs),
        'proj_mags': np.array(proj_mags),
        'proj_prices': np.array(proj_prices),
        'resi_prices': np.array(resi_prices),
    }


def build_result_df(common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
                    projections, residuals, dot_after, proj_coeffs, proj_mags,
                    proj_prices, resi_prices, norm_params, index_tag, stock_tag):
    """组装 19 列结果 DataFrame(raw + norm + 投影 + 残差 + 4 个汇总 + 正交验证 + 归一化参数)。"""
    return pd.DataFrame({
        'Date': common_idx,
        f'Vol_{index_tag}_raw': vec_index[:, 0],
        f'Amt_{index_tag}_raw': vec_index[:, 1],
        f'Vol_{stock_tag}_raw': vec_stock[:, 0],
        f'Amt_{stock_tag}_raw': vec_stock[:, 1],
        f'Vol_{index_tag}_norm': vec_index_norm[:, 0],
        f'Amt_{index_tag}_norm': vec_index_norm[:, 1],
        f'Vol_{stock_tag}_norm': vec_stock_norm[:, 0],
        f'Amt_{stock_tag}_norm': vec_stock_norm[:, 1],
        'Proj_Vol': projections[:, 0],
        'Proj_Amt': projections[:, 1],
        'Residual_Vol': residuals[:, 0],
        'Residual_Amt': residuals[:, 1],
        'Proj_Coeff': proj_coeffs,
        'Proj_Magnitude': proj_mags,
        'Proj_Price': proj_prices,
        'Resi_Price': resi_prices,
        'Dot_After_Proj': dot_after,
        'Norm_Params': [norm_params] * len(common_idx),
    })