# -*- coding: utf-8 -*-
"""Shared math + I/O for projection_2d.py and projection_batch.py.

Single source of truth for: market→index map, stock→industry map,
local-cache loading, 2-D vector projection math, and 19-column result
DataFrame assembly.

No plotly / HTML / file writes — those are the calling scripts' jobs.
"""
import os

import numpy as np
import pandas as pd


# === 数据根目录(沿用 common.data_store 的 DATA_DIR) ===
# 这里避免直接 import common.data_store 以保持 _projection_core.py 的最小依赖。
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(_PROJECT_DIR, 'data')
SW2_MEMBERS_CSV = os.path.join(DATA_DIR, 'sw2', 'members.csv')   # member_code -> (sector_code, sector_name)


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


# === 个股 → 申万二级行业(平行于 MARKET_TO_INDEX) ===
# 申万二级行业用通达信行业代码 881xxx.SH;日线在 data/sectors/ 下,直接由
# tsfresh_pipeline.load_ohlcva(code) 走 sectors kind 拉取。
#
# 数据源:data/sw2/members.csv (sector_code, sector_name, member_code)
# 一只票出现在多个行业 → 保留首个(申万二级互有重叠,首条一般是主行业)。
def _build_industry_map(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"行业成分股表缺失: {csv_path}\n"
            f"需先准备 data/sw2/members.csv(列:sector_code, sector_name, member_code)"
        )
    df = pd.read_csv(csv_path, dtype={'sector_code': str, 'member_code': str})
    seen_member = set()
    member_map = {}
    sector_map = {}
    for _, row in df.iterrows():
        sc = row['sector_code']
        sn = row['sector_name']
        mc = row['member_code']
        # 行业代码 → 行业名(同名行业重复出现取首条即可)
        if sc not in sector_map:
            sector_map[sc] = sn
        # 个股 → 所属行业
        if mc not in seen_member:
            seen_member.add(mc)
            member_map[mc] = (sc, sn)
    return member_map, sector_map


INDUSTRY_MAP, SECTOR_NAME_MAP = _build_industry_map(SW2_MEMBERS_CSV)


def resolve_industry(stock_code):
    """由 STOCK_CODE 查 申万二级行业:(industry_code, industry_name)。
    未在 members.csv 中(新股/退市/非沪深)→ 抛 ValueError,提示更新 sw2/members.csv。
    """
    if stock_code not in INDUSTRY_MAP:
        raise ValueError(
            f"{stock_code} 不在 {SW2_MEMBERS_CSV} 中\n"
            f"(新股/退市/非沪深 A 股都会落空)\n"
            f"请更新 {SW2_MEMBERS_CSV} 或回退到 resolve_index() 用大盘。"
        )
    return INDUSTRY_MAP[stock_code]


def resolve_index_name(index_code):
    """由 index_code 反查人类可读名称。

    支持:
    - 大盘指数 (000001.SH / 399001.SZ) → MARKET_TO_INDEX
    - 申万二级行业 (881xxx.SH) → SECTOR_NAME_MAP(来 sw2/members.csv)
    - 其它 → 返回 '自定义基线'
    """
    if index_code in dict(MARKET_TO_INDEX.values()) or index_code in MARKET_TO_INDEX.values():
        # 反向查
        for k, v in MARKET_TO_INDEX.items():
            if v[0] == index_code:
                return v[1]
    if index_code in SECTOR_NAME_MAP:
        return SECTOR_NAME_MAP[index_code]
    return '自定义基线'


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


def load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag: int = 0):
    """从本地 data/ 缓存加载 (stock_df, index_df) 共同交易日的最近 `days` 行。

    Args:
        stock_code:        个股代码,带 .SH / .SZ 后缀(如 002475.SZ)
        days:              回看交易日数(取最近 N 行共同交易日)
        pipeline:          tsfresh_pipeline 实例(load_ohlcva 用)
        prefer_industry:   bool。True 时基线 = 个股所在申万二级行业(缺失回退大盘)
        index_code:        str 或 None。显式基线代码,优先级最高。
                           None 时按 prefer_industry / 默认大盘自动解析。
                           示例:'881427.SH'(申万体育)/ '000001.SH'(上证综指)/ '399001.SZ'(深证成指)
        lag: 0 = 当前(Volume, Amount)(默认,行为与改动前一致);
             >=1 时还附带 Volume.shift(1) / Amount.shift(1),首行 prev=NaN 被 dropna。
             本次仅实现 lag=0 / lag=1。

    基线选择优先级:
      1. index_code(显式传入,最高优先级)→ 强制用它,可传大盘/任意行业指数/自定义代码
      2. prefer_industry=True → 个股所在申万二级行业,缺失回退大盘
      3. 默认 → 大盘(SZ→深证成指 / SH→上证综指)

    Returns:
        dict: stock_df, index_df, common_idx, index_code, index_name,
              index_tag, stock_tag。

    Raises:
        RuntimeError: 本地缓存缺失(需先跑 backtrace/data_fetch/fetch_daily.py)
        ValueError: lag >= 1 但 stock/index 数据 < 2 行。
    """
    if index_code:
        # 显式传入基线:手动指定大盘或行业指数(如 881427.SH 半导体)。
        # 这里不做严格校验;pipeline.load_ohlcva 会在缓存缺失时报错。
        index_name = resolve_index_name(index_code)
    elif prefer_industry:
        try:
            index_code, index_name = resolve_industry(stock_code)
        except ValueError:
            index_code, index_name = resolve_index(stock_code)
    else:
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

    # lag >= 1 时附加 prev 列;前置 < 2 行检查(避免 shift 全 NaN 后静默丢所有行)
    if lag >= 1:
        if len(data_index_full) < 2 or len(data_stock_full) < 2:
            raise ValueError(
                f"--two-day-vec 需要 ≥2 行数据,"
                f"实际 {index_code}={len(data_index_full)} 行, {stock_code}={len(data_stock_full)} 行"
            )
        data_index_full = data_index_full.assign(
            Volume_prev=data_index_full['Volume'].shift(1),
            Amount_prev=data_index_full['Amount'].shift(1),
        )
        data_stock_full = data_stock_full.assign(
            Volume_prev=data_stock_full['Volume'].shift(1),
            Amount_prev=data_stock_full['Amount'].shift(1),
        )

    cols = ['Volume', 'Amount', 'Close']
    if lag >= 1:
        cols = ['Volume', 'Amount', 'Volume_prev', 'Amount_prev', 'Close']
    data_index = data_index_full[cols].tail(days).dropna()
    data_stock = data_stock_full[cols].tail(days).dropna()
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


def compute_vectors(stock_df, index_df, index_tag, stock_tag, lag: int = 0):
    """Min-Max 归一化 Vol/Amt(及可选的 Vol_prev/Amt_prev)。

    Args:
        lag: 0 = 当前 (Volume, Amount) 2-D(默认,与改动前一致);
             >=1 时还取 Volume.shift(1) / Amount.shift(1), 输出向量维度 = 2 * (lag + 1)。
             本次仅实现 lag=0 / lag=1。
    """
    assert lag <= 1, f"compute_vectors: lag={lag} not implemented (only lag=0 or lag=1)"
    cols = ['Volume', 'Amount']
    if lag >= 1:
        cols += ['Volume_prev', 'Amount_prev']
    # 防呆: DataFrame 缺列时直接报错(比 KeyError 友好)
    for c in cols:
        if c not in stock_df.columns:
            raise KeyError(f"compute_vectors(lag={lag}) 需要 stock_df 含列 {c!r}")
        if c not in index_df.columns:
            raise KeyError(f"compute_vectors(lag={lag}) 需要 index_df 含列 {c!r}")

    vec_index = index_df[cols].values
    vec_stock = stock_df[cols].values

    # 每个维度独立 Min-Max(向量化)
    norms_ix = np.zeros_like(vec_index)
    norms_st = np.zeros_like(vec_stock)
    params_parts = []
    for j, c in enumerate(cols):
        v_min_ix, v_max_ix = vec_index[:, j].min(), vec_index[:, j].max()
        v_min_st, v_max_st = vec_stock[:, j].min(), vec_stock[:, j].max()
        norms_ix[:, j] = _safe_minmax(vec_index[:, j], v_min_ix, v_max_ix - v_min_ix)
        norms_st[:, j] = _safe_minmax(vec_stock[:, j], v_min_st, v_max_st - v_min_st)
        params_parts.append(f"{c}_{index_tag}:[{v_min_ix:.2e},{v_max_ix:.2e}]")
        params_parts.append(f"{c}_{stock_tag}:[{v_min_st:.2e},{v_max_st:.2e}]")
    norm_params = " ".join(params_parts)

    return vec_index, vec_stock, norms_ix, norms_st, norm_params


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


def compute_movement_projection(stock_df, index_df):
    """运动向量投影 — 把 stock 当日 vs 前一日的 (ΔV, ΔA) 投到 index 同维运动方向上。

    与 compute_projections 的区别:
      - 那个算"当前成交状态" (Vt, At) 投影到 (Vi, Ai);
      - 这个算"成交状态变化" (Vt - Vt-1, At - At-1) 投影到 (Vi - Vi-1, Ai - Ai-1)。

    Args:
        stock_df, index_df: 必须含 'Volume' / 'Amount' 列,且按日期对齐。
                          首行因 .diff() 缺前一日数据,返回的所有数组都丢首日
                          (caller 需自行用 common_idx[1:] 切片)。

    Returns:
        dict with keys:
            stock_move:   ndarray (T-1, 2)  — 个股运动向量 (ΔV_s, ΔA_s)
            index_move:   ndarray (T-1, 2)  — 指数运动向量 (ΔV_i, ΔA_i)
            v_dot_v:      ndarray (T-1,)   — v·v (= ΔV_i² + ΔA_i²,β 分母)
            u_dot_v:      ndarray (T-1,)   — u·v (= ΔV_s·ΔV_i + ΔA_s·ΔA_i,β 分子)
            proj_coeff:   ndarray (T-1,)   — 映射系数 β_t = (u·v) / (v·v)
            proj:         ndarray (T-1, 2)  — 投影向量 β_t * v
            residual:     ndarray (T-1, 2)  — 正交残差 u - proj
            proj_mag:     ndarray (T-1,)   — ‖proj‖
            resi_mag:     ndarray (T-1,)   — ‖residual‖
            dot_after:    ndarray (T-1,)   — residual · v,理想正交 = 0
            proj_prices:  ndarray (T-1,)   — β·ΔA / β·ΔV (= ΔA_i / ΔV_i,大盘边际成交均价)
            resi_prices:  ndarray (T-1,)   — residual_ΔA / residual_ΔV(分母 0 → 0)
    """
    for c in ('Volume', 'Amount'):
        if c not in stock_df.columns:
            raise KeyError(f"compute_movement_projection 需要 stock_df 含列 {c!r}")
        if c not in index_df.columns:
            raise KeyError(f"compute_movement_projection 需要 index_df 含列 {c!r}")

    stock_dv = stock_df['Volume'].diff().to_numpy()
    stock_da = stock_df['Amount'].diff().to_numpy()
    index_dv = index_df['Volume'].diff().to_numpy()
    index_da = index_df['Amount'].diff().to_numpy()

    u = np.column_stack([stock_dv, stock_da])  # (T, 2)
    v = np.column_stack([index_dv, index_da])  # (T, 2)

    # 丢首行 (diff NaN) — 与 caller 后续切片 common_idx[1:] 对齐
    u = u[1:]
    v = v[1:]

    v_dot_v = np.sum(v * v, axis=1)                          # (T-1,)
    u_dot_v = np.sum(u * v, axis=1)                          # (T-1,)
    proj_coeff = np.divide(u_dot_v, v_dot_v,
                           out=np.zeros_like(u_dot_v),
                           where=v_dot_v > 1e-12)             # 防 /0
    proj = proj_coeff[:, None] * v                            # (T-1, 2)
    residual = u - proj                                        # (T-1, 2)
    proj_mag = np.linalg.norm(proj, axis=1)
    resi_mag = np.linalg.norm(residual, axis=1)
    dot_after = np.sum(residual * v, axis=1)                  # 理想正交 → 0

    # proj_price / resi_price:投影向量/残差向量的 Amount/Volume 比。
    # 几何含义:
    #   - proj_price = β·ΔA / (β·ΔV) = ΔA_i / ΔV_i
    #     即大盘的边际成交均价(β 抵消);刻画「市场状态切换时成交均价如何变化」。
    #   - resi_price = residual_dA / residual_dV
    #     个股运动偏离大盘方向的程度,用每手价格度量 — 识别个股是否在大盘放量
    #     之外额外放量(>大盘均价)/缩量(<大盘均价)。
    # 沿用 compute_projections 的安全除法 + cap 限幅逻辑(见 277-284)。
    proj_prices = _movement_safe_ratios(proj, axis=1)
    resi_prices = _movement_safe_ratios(residual, axis=1, cap_to='self')
    # for i, r in enumerate(resi_prices):|
    #     if not np.isfinite(r) or abs(r) > 3:
    #         past = np.abs(resi_prices[:i])
    #         cap = float(np.nanmax(past)) if len(past) > 0 and np.any(np.isfinite(past)) else 0.0
    #         sign = np.sign(residual[i, 1]) if np.isfinite(residual[i, 1]) else 0.0
    #         resi_prices[i] = sign * cap

    return {
        'stock_move': u,
        'index_move': v,
        'v_dot_v': v_dot_v,
        'u_dot_v': u_dot_v,
        'proj_coeff': proj_coeff,
        'proj': proj,
        'residual': residual,
        'proj_mag': proj_mag,
        'resi_mag': resi_mag,
        'dot_after': dot_after,
        'proj_prices': proj_prices,
        'resi_prices': resi_prices,
    }


def _movement_safe_ratios(arr2d, axis=1, cap_to=None):
    """对 2-D 数组每行算 arr[:,1] / arr[:,0],0/NaN/Inf 走 _safe_ratio 容错。

    与 compute_projections 中 proj_prices / resi_prices 计算同款,只是输入是
    2-D 数组(批处理)而非单向量 — 状态投影走 per-row 循环,运动向量可以一次性
    numpy 算,效率更好。
    """
    num = arr2d[:, 1]
    den = arr2d[:, 0]
    out = np.divide(num, den,
                    out=np.zeros_like(num, dtype=float),
                    where=(den != 0) & np.isfinite(den) & np.isfinite(num))
    return out


def build_movement_result_df(common_idx, mv, index_tag, stock_tag):
    """组装运动投影结果 DataFrame — 15 列。

    Columns:
        Date, ΔV_{index_tag}, ΔA_{index_tag}, ΔV_{stock_tag}, ΔA_{stock_tag},
        Proj_Coeff, Proj_Delta_Vol, Proj_Delta_Amt,
        Resi_Delta_Vol, Resi_Delta_Amt,
        Proj_Magnitude, Resi_Magnitude, Dot_After_Proj,
        Proj_Price, Resi_Price

    caller 负责传 common_idx[1:] (丢首行与 diff 对齐)。
    """
    return pd.DataFrame({
        'Date': common_idx,
        f'ΔV_{index_tag}': mv['index_move'][:, 0],
        f'ΔA_{index_tag}': mv['index_move'][:, 1],
        f'ΔV_{stock_tag}': mv['stock_move'][:, 0],
        f'ΔA_{stock_tag}': mv['stock_move'][:, 1],
        'Proj_Coeff': mv['proj_coeff'],
        'Proj_Delta_Vol': mv['proj'][:, 0],
        'Proj_Delta_Amt': mv['proj'][:, 1],
        'Resi_Delta_Vol': mv['residual'][:, 0],
        'Resi_Delta_Amt': mv['residual'][:, 1],
        'Proj_Magnitude': mv['proj_mag'],
        'Resi_Magnitude': mv['resi_mag'],
        'Dot_After_Proj': mv['dot_after'],
        'Proj_Price': mv['proj_prices'],
        'Resi_Price': mv['resi_prices'],
    })


def build_movement_intermediate_df(common_idx, mv, stock_df, index_df,
                                   index_tag, stock_tag):
    """组装运动投影的「逐日复核」DataFrame — 22 列,覆盖每一步的中间值。

    用途:`projection_2d.py --movement` 顺手落一份 CSV 到 data/projection/intermediate/,
    方便人工逐日核对公式:`Δ` / `β = u·v/v·v` / `proj = β·v` / `resi = u - proj` /
    `resi·v ≈ 0` / `proj_price = ΔA_i/ΔV_i` / `resi_price = resi_ΔA/resi_ΔV`。

    与 `build_movement_result_df` 的差别:
      - 多 4 列原始值(Vol/Ama 当日,非 Δ)— 验证 diff 正确性
      - 多 2 列中间点积(V_dot_V / U_dot_V)— 验证 β 分子分母
      - 多 1 列 Resi_Price_Raw — 与 Resi_Price 一并保留便于发现 > 3 异常

    Args:
        common_idx: caller 传 common_idx[1:] (与 diff 丢首行对齐)。
        mv:         `compute_movement_projection` 返回的 dict(必须含 v_dot_v/u_dot_v)。
        stock_df / index_df: 原始 (Vol/Ama) 序列,caller 同样切片丢首行。

    Columns:
        Date, Vol_{idx}, Amt_{idx}, Vol_{stk}, Amt_{stk},
        ΔV_{idx}, ΔA_{idx}, ΔV_{stk}, ΔA_{stk},
        V_dot_V, U_dot_V, Proj_Coeff,
        Proj_ΔV, Proj_ΔA, Resi_ΔV, Resi_ΔA,
        Resi_dot_V, Proj_Magnitude, Resi_Magnitude,
        Proj_Price, Resi_Price_Raw, Resi_Price
    """
    # raw Vol/Ama 当日 — caller 已丢首行 (stock_df.iloc[1:], index_df.iloc[1:])
    v_idx_raw = index_df['Volume'].to_numpy()[1:]
    a_idx_raw = index_df['Amount'].to_numpy()[1:]
    v_stk_raw = stock_df['Volume'].to_numpy()[1:]
    a_stk_raw = stock_df['Amount'].to_numpy()[1:]

    # Resi_Price_Raw:不经 _movement_safe_ratios 的原始比值,residual_ΔV=0 时显示 inf/NaN
    # 而不是被替换为 0 — 复核时更易一眼看出除零异常行
    resi_raw_num = mv['residual'][:, 1]
    resi_raw_den = mv['residual'][:, 0]
    resi_price_raw = np.divide(
        resi_raw_num, resi_raw_den,
        out=np.full_like(resi_raw_num, fill_value=np.nan, dtype=float),
        where=(resi_raw_den != 0) & np.isfinite(resi_raw_den) & np.isfinite(resi_raw_num),
    )

    return pd.DataFrame({
        'Date': common_idx,
        f'Vol_{index_tag}': v_idx_raw,
        f'Amt_{index_tag}': a_idx_raw,
        f'Vol_{stock_tag}': v_stk_raw,
        f'Amt_{stock_tag}': a_stk_raw,
        f'ΔV_{index_tag}': mv['index_move'][:, 0],
        f'ΔA_{index_tag}': mv['index_move'][:, 1],
        f'ΔV_{stock_tag}': mv['stock_move'][:, 0],
        f'ΔA_{stock_tag}': mv['stock_move'][:, 1],
        'V_dot_V': mv['v_dot_v'],
        'U_dot_V': mv['u_dot_v'],
        'Proj_Coeff': mv['proj_coeff'],
        'Proj_ΔV': mv['proj'][:, 0],
        'Proj_ΔA': mv['proj'][:, 1],
        'Resi_ΔV': mv['residual'][:, 0],
        'Resi_ΔA': mv['residual'][:, 1],
        'Resi_dot_V': mv['dot_after'],
        'Proj_Magnitude': mv['proj_mag'],
        'Resi_Magnitude': mv['resi_mag'],
        'Proj_Price': mv['proj_prices'],
        'Resi_Price_Raw': resi_price_raw,
        'Resi_Price': mv['resi_prices'],
    })


def build_result_df(common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
                    projections, residuals, dot_after, proj_coeffs, proj_mags,
                    proj_prices, resi_prices, norm_params, index_tag, stock_tag, lag: int = 0):
    """组装 19/27 列结果 DataFrame(raw + norm + 投影 + 残差 + 4 个汇总 + 正交验证 + 归一化参数)。"""
    prev_cols_raw = {}
    prev_cols_norm = {}
    if lag >= 1:
        # 假设 vec_* 已是 4 列 (Vol_t, Amt_t, Vol_prev, Amt_prev),取 [2:4]
        assert vec_index.shape[1] >= 4 and vec_stock.shape[1] >= 4, (
            f"build_result_df(lag={lag}) 需要 vec_index/vec_stock 有 4 列,"
            f" 实际 shape={vec_index.shape}, {vec_stock.shape}"
        )
        prev_cols_raw = {
            f'Vol_{index_tag}_prev_raw': vec_index[:, 2],
            f'Amt_{index_tag}_prev_raw': vec_index[:, 3],
            f'Vol_{stock_tag}_prev_raw': vec_stock[:, 2],
            f'Amt_{stock_tag}_prev_raw': vec_stock[:, 3],
        }
        prev_cols_norm = {
            f'Vol_{index_tag}_prev_norm': vec_index_norm[:, 2],
            f'Amt_{index_tag}_prev_norm': vec_index_norm[:, 3],
            f'Vol_{stock_tag}_prev_norm': vec_stock_norm[:, 2],
            f'Amt_{stock_tag}_prev_norm': vec_stock_norm[:, 3],
        }

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
        **prev_cols_raw,    # lag=0 时为空 dict,不引入新列
        **prev_cols_norm,
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