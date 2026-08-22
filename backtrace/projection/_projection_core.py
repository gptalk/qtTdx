# -*- coding: utf-8 -*-
"""Shared math + I/O for projection_2d.py and projection_batch.py.

Single source of truth for: market→index map, stock→industry map,
local-cache loading, 2-D vector projection math (原始量纲), and 21-column
result DataFrame assembly.

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


def load_pair(stock_code, days, pipeline, prefer_industry=False, index_code=None, lag: int = 0, *, period: str = 'daily'):
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

    if period not in ('daily', '15m', '5m', '1m'):
        raise ValueError(f"period 必须是 (daily, 15m, 5m, 1m) 之一,收到 {period!r}")

    data_index_full = pipeline.load_ohlcva(index_code, use_tq=False, verbose=True, period=period)
    data_stock_full = pipeline.load_ohlcva(stock_code, use_tq=False, verbose=True, period=period)
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


def compute_projections(vec_stock, vec_index):
    """对每行跑 project_u_onto_v,返回 9 个 np.array(原 6 + 新增 3 个幅度量)。

    Args:
        vec_stock:  ndarray (T, k) — 个股原始向量(Volume, Amount, ...)
                   切到原始量纲后 `proj_prices` 是真实边际成交均价(元/手),
                   `magnitudes` 是原始 ‖u‖ (1e7+ 量纲),`relative_move` 是
                   「个股成交规模 / 大盘成交规模」的真实倍数。
        vec_index:  ndarray (T, k) — 大盘原始向量,同 k 列结构。

    Returns dict keys:
        projections / residuals / dot_after / proj_coeffs / proj_mags /
        proj_prices: 6 个核心投影输出(原始量纲)。
        state_stock_mag:    ndarray (T,) — |u| 原始向量模长(元/手量纲)
        state_index_mag:    ndarray (T,) — |v| 原始向量模长
        state_relative_move: ndarray (T,) — |u| / |v|,|v|≈0 时 → 0(沿用 β 默认容错)

    设计变更(2026-08-16):state 投影不再产出 `resi_prices`。原因 — 2-D 投影几何
    上 residual ⊥ v 强制 `resi_price = -v[0]/v[1] = -1/proj_price`,对所有共享
    同一大盘基线的个股取值完全一样,与个股信息无关,选股无效。
    下游选股请用 `state_*_mag`(个股相关)|或 lag=1 (4-D) — 退化自动消失。
    注:motion 投影的 `Move_Resi_Price` 保留(基于 Δ 向量,非退化)。
    """
    projections, residuals, dot_after = [], [], []
    proj_coeffs, proj_mags, proj_prices = [], [], []
    state_stock_mag, state_index_mag = [], []

    for i in range(len(vec_stock)):
        u = vec_stock[i]
        v = vec_index[i]
        proj = project_u_onto_v(u, v)
        residual = u - proj
        projections.append(proj)
        residuals.append(residual)
        dot_after.append(np.dot(residual, v))
        proj_coeffs.append(_safe_ratio(np.dot(u, v), np.dot(v, v), default=0.0))
        proj_mags.append(np.linalg.norm(proj))
        proj_prices.append(_safe_ratio(proj[1], proj[0], default=np.sign(proj[1]) if np.isfinite(proj[1]) else 0.0))
        state_stock_mag.append(np.linalg.norm(u))
        state_index_mag.append(np.linalg.norm(v))

    state_stock_mag = np.array(state_stock_mag)
    state_index_mag = np.array(state_index_mag)
    state_relative_move = np.divide(
        state_stock_mag, state_index_mag,
        out=np.zeros_like(state_stock_mag),
        where=state_index_mag > 1e-12,
    )

    return {
        'projections': np.array(projections),
        'residuals': np.array(residuals),
        'dot_after': np.array(dot_after),
        'proj_coeffs': np.array(proj_coeffs),
        'proj_mags': np.array(proj_mags),
        'proj_prices': np.array(proj_prices),
        'state_stock_mag': state_stock_mag,
        'state_index_mag': state_index_mag,
        'state_relative_move': state_relative_move,
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
            stock_move:    ndarray (T-1, 2)  — 个股运动向量 (ΔV_s, ΔA_s)
            index_move:    ndarray (T-1, 2)  — 指数运动向量 (ΔV_i, ΔA_i)
            stock_move_mag: ndarray (T-1,)  — |u| = √(ΔV_s² + ΔA_s²)【运动幅度】
            index_move_mag: ndarray (T-1,)  — |v| = √(ΔV_i² + ΔA_i²)【大盘运动幅度】
            relative_move:  ndarray (T-1,)  — |u| / |v|【个股相对大盘运动倍数;|v|≈0 时 → 0】
            v_dot_v:       ndarray (T-1,)   — v·v (= ΔV_i² + ΔA_i²,β 分母)
            u_dot_v:       ndarray (T-1,)   — u·v (= ΔV_s·ΔV_i + ΔA_s·ΔA_i,β 分子)
            proj_coeff:    ndarray (T-1,)   — 映射系数 β_t = (u·v) / (v·v)
            proj:          ndarray (T-1, 2)  — 投影向量 β_t * v
            residual:      ndarray (T-1, 2)  — 正交残差 u - proj
            proj_mag:      ndarray (T-1,)   — ‖proj‖
            resi_mag:      ndarray (T-1,)   — ‖residual‖
            dot_after:     ndarray (T-1,)   — residual · v,理想正交 = 0
            proj_prices:   ndarray (T-1,)   — β·ΔA / β·ΔV (= ΔA_i / ΔV_i,大盘边际成交均价)
            resi_prices:   ndarray (T-1,)   — residual_ΔA / residual_ΔV(分母 0 → 0)
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

    # 8 维度框架的「幅度」量 — 与 Price (方向斜率) 区分:
    #   stock_move_mag / index_move_mag 描述运动大小(单位:手 / 元,与 Price 同空间)
    #   relative_move 描述「个股比大盘大几倍」,大盘没动个股暴动时 → 大值
    #   β 阈值同款保护:v 太小时 relative_move → 0,避免除零
    stock_move_mag = np.linalg.norm(u, axis=1)
    index_move_mag = np.linalg.norm(v, axis=1)
    relative_move = np.divide(stock_move_mag, index_move_mag,
                              out=np.zeros_like(stock_move_mag),
                              where=index_move_mag > 1e-12)

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
        'stock_move_mag': stock_move_mag,
        'index_move_mag': index_move_mag,
        'relative_move': relative_move,
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
    """组装运动投影结果 DataFrame — 18 列(8 维度框架)。

    所有运动向量相关列加 Move_ 前缀,与 state projection 的 State_ 前缀列明确区分。
    列分组:
      1. 运动向量本身:Move_Delta_Vol/Amt_{idx/stk} (4 列)
      2. 幅度量:Move_Stock_Magnitude / Move_Index_Magnitude / Move_Relative_Move (3 列)
      3. 投影侧:Move_Proj_Coeff / Vol / Amt / Magnitude / Price (5 列)
      4. 残差侧:Move_Resi_Vol / Amt / Magnitude / Price (4 列)
      5. 正交验证:Move_Dot_After (1 列)

    设计要点:
      - Magnitude (‖proj‖ / ‖resi‖) 描述运动「大小」,Price (slope) 描述「方向斜率」
      - 这两个维度在用户分析中应分别看 — 用户曾贴分析指出 Price 数值小不一定说明
        「运动弱」,而是「方向斜率浅」;真正识别「大盘没动 / 个股暴动」靠 Magnitude +
        Relative_Move
      - caller 负责传 common_idx[1:] (丢首行与 diff 对齐)
    """
    return pd.DataFrame({
        'Date': common_idx,
        f'Move_Delta_Vol_{index_tag}': mv['index_move'][:, 0],
        f'Move_Delta_Amt_{index_tag}': mv['index_move'][:, 1],
        f'Move_Delta_Vol_{stock_tag}': mv['stock_move'][:, 0],
        f'Move_Delta_Amt_{stock_tag}': mv['stock_move'][:, 1],
        'Move_Stock_Magnitude': mv['stock_move_mag'],
        'Move_Index_Magnitude': mv['index_move_mag'],
        'Move_Relative_Move': mv['relative_move'],
        'Move_Proj_Coeff': mv['proj_coeff'],
        'Move_Proj_Vol': mv['proj'][:, 0],
        'Move_Proj_Amt': mv['proj'][:, 1],
        'Move_Proj_Magnitude': mv['proj_mag'],
        'Move_Proj_Price': mv['proj_prices'],
        'Move_Resi_Vol': mv['residual'][:, 0],
        'Move_Resi_Amt': mv['residual'][:, 1],
        'Move_Resi_Magnitude': mv['resi_mag'],
        'Move_Resi_Price': mv['resi_prices'],
        'Move_Dot_After': mv['dot_after'],
    })


def build_movement_intermediate_df(common_idx, mv, stock_df, index_df,
                                   index_tag, stock_tag):
    """组装运动投影的「逐日复核」DataFrame — 25 列,覆盖每一步的中间值。

    用途:`projection_2d.py --movement` 顺手落一份 CSV 到 data/projection/intermediate/,
    方便人工逐日核对公式:`Δ` / `β = u·v/v·v` / `proj = β·v` / `resi = u - proj` /
    `resi·v ≈ 0` / `proj_price = ΔA_i/ΔV_i` / `resi_price = resi_ΔA/resi_ΔV` /
    `|u| / |v| / R = |u|/|v|`。

    与 `build_movement_result_df`(18 列)的差别:
      - 多 4 列原始值(Move_Vol/Amt_{idx/stk},非 Δ)— 验证 diff 正确性
      - 多 2 列中间点积(Move_V_dot_V / Move_U_dot_V)— 验证 β 分子分母
      - 多 1 列 Move_Resi_Price_Raw — 裸除法,residual_ΔV=0 时显示 NaN(暴露异常)
      - 多 0 列 magnitude — build_result_df 已含

    Args:
        common_idx: caller 传 common_idx[1:] (与 diff 丢首行对齐)。
        mv:         `compute_movement_projection` 返回的 dict。
        stock_df / index_df: 原始 (Vol/Ama) 序列,caller 同样切片丢首行。
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
        f'Move_Vol_{index_tag}': v_idx_raw,
        f'Move_Amt_{index_tag}': a_idx_raw,
        f'Move_Vol_{stock_tag}': v_stk_raw,
        f'Move_Amt_{stock_tag}': a_stk_raw,
        f'Move_Delta_Vol_{index_tag}': mv['index_move'][:, 0],
        f'Move_Delta_Amt_{index_tag}': mv['index_move'][:, 1],
        f'Move_Delta_Vol_{stock_tag}': mv['stock_move'][:, 0],
        f'Move_Delta_Amt_{stock_tag}': mv['stock_move'][:, 1],
        'Move_V_dot_V': mv['v_dot_v'],
        'Move_U_dot_V': mv['u_dot_v'],
        'Move_Stock_Magnitude': mv['stock_move_mag'],
        'Move_Index_Magnitude': mv['index_move_mag'],
        'Move_Relative_Move': mv['relative_move'],
        'Move_Proj_Coeff': mv['proj_coeff'],
        'Move_Proj_Vol': mv['proj'][:, 0],
        'Move_Proj_Amt': mv['proj'][:, 1],
        'Move_Proj_Magnitude': mv['proj_mag'],
        'Move_Proj_Price': mv['proj_prices'],
        'Move_Resi_Vol': mv['residual'][:, 0],
        'Move_Resi_Amt': mv['residual'][:, 1],
        'Move_Resi_Magnitude': mv['resi_mag'],
        'Move_Resi_Price_Raw': resi_price_raw,
        'Move_Resi_Price': mv['resi_prices'],
        'Move_Dot_After': mv['dot_after'],
    })


def build_result_df(common_idx, vec_index, vec_stock, vec_index_norm, vec_stock_norm,
                    projections, residuals, dot_after, proj_coeffs, proj_mags,
                    proj_prices, state_stock_mag, state_index_mag,
                    state_relative_move, norm_params, index_tag, stock_tag, lag: int = 0):
    """组装 21/29 列 state 投影结果 DataFrame(raw + norm + 幅度量 + 投影 + 残差 + 4 个汇总 + 归一化参数)。

    lag=0 → 21 列;lag=1 → 29 列(追加 4 prev_raw + 4 prev_norm)。

    所有 projection / magnitude / price 相关列加 State_ 前缀,与 movement 的 Move_ 列
    明确区分。列分组:
      1. 原始值(4 列)+ 归一化值(4 列)+ prev 列(lag=1 时 +8)
      2. 幅度量:State_Stock_Magnitude / State_Index_Magnitude / State_Relative_Move (3 列)
      3. 投影侧:State_Proj_Vol / Amt / Coeff / Magnitude / Price (5 列)
      4. 残差侧:State_Resi_Vol / Amt (2 列)
      5. 正交验证:State_Dot_After (1 列)
      6. 归一化参数:Norm_Params (1 列)

    设计变更(2026-08-16):删除 `State_Resi_Price` 列 — 2-D 投影几何上
    `resi_price = -v[0]/v[1] = -1/State_Proj_Price`,对所有共享同一大盘的个股
    取值完全一样(2-D 退化),与个股信息无关,选股无效。
    残差方向斜率仍可通过 lag=1 (4-D) 退化自然消失后产出,或由下游另行计算
    个股相关的 `State_Resi_Magnitude`(本批次不实现,见后续规划)。
    """
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
        'State_Stock_Magnitude': state_stock_mag,
        'State_Index_Magnitude': state_index_mag,
        'State_Relative_Move': state_relative_move,
        'State_Proj_Vol': projections[:, 0],
        'State_Proj_Amt': projections[:, 1],
        'State_Resi_Vol': residuals[:, 0],
        'State_Resi_Amt': residuals[:, 1],
        'State_Proj_Coeff': proj_coeffs,
        'State_Proj_Magnitude': proj_mags,
        'State_Proj_Price': proj_prices,
        'State_Dot_After': dot_after,
        'Norm_Params': [norm_params] * len(common_idx),
    })


# === 动力学层(2026-08-16 新增)=================================================
# 叠在 compute_movement_projection 之上,产出 9 个指标 + 7 状态分类。
# 公式与设计见 docs/superpowers/specs/2026-08-16-market-stock-dynamics-design.md。
#
# 关键约定:
#   - 时间步 Δt = 1 个交易日 → 速度 v ≡ Δu / 1 = Δu(直接复用 mv['stock_move'] 等)
#   - 加速度是速度的二阶差分 → 长度比 mv 再短 1 行(末行 NaN)
#   - 锚定强度 q_t = ‖ΔM‖ / (‖ΔM‖ + λ_q),λ_q 默认 median(‖ΔM‖) 自适应窗口
#   - 不引入质量 m → 耦合度 R = ‖v_resi‖² / ‖v_S‖² 与 m 无关
#   - 不输出 price-based 残差(2-D 退化,见 compute_projections 注释)

# 7 个状态标签 + 配色,供 classify_states 与 HTML 共用
STATE_LABELS = ['follow', 'weak_div', 'accelerating', 'independent',
                'against', 'returning', 'resonance', 'none']
STATE_COLORS = {
    'follow': '#2ecc71',        # 绿
    'weak_div': '#f1c40f',      # 黄
    'accelerating': '#e74c3c',  # 红
    'independent': '#9b59b6',   # 紫
    'against': '#8b4513',       # 棕
    'returning': '#1abc9c',     # 青
    'resonance': '#ff69b4',     # 粉
    'none': '#7f8c8d',          # 灰
}
STATE_LABELS_CN = {
    'follow': '跟随',
    'weak_div': '弱偏离',
    'accelerating': '加速偏离',
    'independent': '独立',
    'against': '逆势',
    'returning': '回归',
    'resonance': '共振',
    'none': '无',
}


def compute_dynamics(mv: dict, lambda_q):
    """基于运动投影输出,计算 9 个动力学指标。

    Args:
        mv: compute_movement_projection() 的返回值。必用键:
              'stock_move' (T-1, 2) / 'index_move' (T-1, 2) /
              'proj' (T-1, 2) / 'residual' (T-1, 2) /
              'proj_mag' / 'resi_mag' / 'index_move_mag' / 'stock_move_mag' /
              'proj_coeff' / 'dot_after'(可选,目前未用)
        lambda_q: 锚定强度系数。None / NaN → 自适应 = median(‖ΔM‖) of window
                  (退化情况下 fallback 到 1e-12 避免除零)。

    Returns:
        dict with keys:
            q_t:           ndarray (T-1,) ∈ [0, 1)
            theta:         ndarray (T-1,) 弧度;退化(‖Δu‖·‖ΔM‖=0)→ NaN
            R:             ndarray (T-1,) ∈ [0, 1];退化(‖v_S‖=0)→ 0
            v_S_mag:       ndarray (T-1,)
            v_M_mag:       ndarray (T-1,)
            v_proj_mag:    ndarray (T-1,)
            v_resi_mag:    ndarray (T-1,)
            E_market:      ndarray (T-1,)
            E_self:        ndarray (T-1,)
            E_total:       ndarray (T-1,)
            a_S_mag:       ndarray (T-1,) 首末 NaN,中间 T-2 有效
            a_M_mag:       ndarray (T-1,) 同上
            lambda_q_used: float,实际使用的 λ_q(便于 caller 报告)
    """
    delta_u = mv['stock_move']            # (T-1, 2) 个股 Δu,Δt=1 → 速度 v_S
    delta_v = mv['index_move']            # (T-1, 2) 大盘 Δv → v_M
    proj = mv['proj']                     # (T-1, 2) β·Δv,无 q 阻尼
    T_minus_1 = delta_u.shape[0]

    # ---- λ_q 自适应 ----
    delta_v_mag = np.linalg.norm(delta_v, axis=1)
    if lambda_q is None or not np.isfinite(lambda_q):
        lambda_q_used = float(np.median(delta_v_mag)) if delta_v_mag.size else 0.0
        if not np.isfinite(lambda_q_used) or lambda_q_used <= 0:
            lambda_q_used = 1e-12
    else:
        lambda_q_used = float(lambda_q)
        if lambda_q_used <= 0:
            lambda_q_used = 1e-12

    # ---- 锚定强度 q_t ∈ [0, 1) ----
    q_t = delta_v_mag / (delta_v_mag + lambda_q_used)

    # ---- 偏离角 θ ∈ [0, π] ----
    delta_u_mag = np.linalg.norm(delta_u, axis=1)
    denom = delta_u_mag * delta_v_mag
    cos_theta = np.divide(
        np.einsum('ij,ij->i', delta_u, delta_v),
        denom,
        out=np.full(T_minus_1, np.nan, dtype=float),
        where=(denom > 1e-12) & np.isfinite(denom),
    )
    cos_theta_clipped = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta_clipped)

    # ---- 速度分解 ----
    v_S = delta_u
    v_M = delta_v
    v_proj = q_t[:, None] * proj                   # 沿大盘方向 + q 阻尼
    v_resi = v_S - v_proj                          # 正交分量(带阻尼后)

    v_S_mag = np.linalg.norm(v_S, axis=1)
    v_M_mag = delta_v_mag
    v_proj_mag = np.linalg.norm(v_proj, axis=1)
    v_resi_mag = np.linalg.norm(v_resi, axis=1)

    # ---- 耦合度 R = ‖v_resi‖² / ‖v_S‖² ∈ [0, 1] ----
    sq_resi = v_resi_mag ** 2
    sq_S = v_S_mag ** 2
    R = np.divide(
        sq_resi, sq_S,
        out=np.zeros_like(sq_resi),
        where=(sq_S > 1e-12) & np.isfinite(sq_S),
    )

    # ---- 动能 = ½·‖v‖² ----
    E_market = 0.5 * v_proj_mag ** 2
    E_self = 0.5 * v_resi_mag ** 2
    E_total = 0.5 * v_S_mag ** 2                 # = E_market + E_self (正交保证)

    # ---- 加速度(末行 NaN,右补 NaN,与速度时序对齐) ----
    # 约定:a_S[i] = v_S[i+1] - v_S[i](前向差),代表「第 i 天发生的速度变化」。
    # 末行 i = L-1 没有 v_S[L] 观测 → NaN。有效 T-2 值在 indices 0..L-2。
    def _accel_right_pad_nan(arr):
        """np.diff 后右补 NaN,保持长度 L。
        输入 arr 长度 L → 输出长度 L,前 L-1 个为有效差分,末行 NaN。"""
        if len(arr) < 2:
            return np.full_like(arr, np.nan, dtype=float)
        diff = np.diff(arr)
        out = np.empty(len(arr), dtype=float)
        out[:-1] = diff
        out[-1] = np.nan
        return out

    a_S_mag = _accel_right_pad_nan(v_S_mag)
    a_M_mag = _accel_right_pad_nan(v_M_mag)

    return {
        'q_t': q_t,
        'theta': theta,
        'R': R,
        'v_S_mag': v_S_mag,
        'v_M_mag': v_M_mag,
        'v_proj_mag': v_proj_mag,
        'v_resi_mag': v_resi_mag,
        'E_market': E_market,
        'E_self': E_self,
        'E_total': E_total,
        'a_S_mag': a_S_mag,
        'a_M_mag': a_M_mag,
        'lambda_q_used': lambda_q_used,
    }


def classify_states(R, theta, E_self, thresholds):
    """按 §4 优先级规则逐日打标签。

    Args:
        R:        ndarray (T-1,) 耦合度
        theta:    ndarray (T-1,) 偏离角(弧度)
        E_self:   ndarray (T-1,) 特异动能(用于斜率)
        thresholds: 4 元组 (R_low, R_high, theta_following_rad, theta_against_rad)

    Returns:
        list[str],长度 T-1,值 ∈ STATE_LABELS。

    优先级:against > resonance > accelerating > returning >
            independent > weak_div > follow > none。
    前 2 天(i=0,1)跳过 accelerating / returning(斜率不够窗口)。
    """
    R_low, R_high, theta_following_rad, theta_against_rad = thresholds
    n = len(R)
    states = ['none'] * n

    # 3 日斜率(对 R 与 E_self 各做线性回归)
    R_slope = np.full(n, np.nan, dtype=float)
    E_slope = np.full(n, np.nan, dtype=float)
    if n >= 3:
        x_axis = np.arange(3, dtype=float)
        for i in range(2, n):
            if np.any(~np.isfinite(R[i - 2:i + 1])) or np.any(~np.isfinite(E_self[i - 2:i + 1])):
                continue
            R_slope[i] = np.polyfit(x_axis, R[i - 2:i + 1], 1)[0]
            E_slope[i] = np.polyfit(x_axis, E_self[i - 2:i + 1], 1)[0]

    def _f(x):
        return bool(np.isfinite(x))

    for i in range(n):
        r_i, th_i = R[i], theta[i]
        if not _f(r_i) or not _f(th_i):
            states[i] = 'none'
            continue
        # 1) 逆势
        if th_i >= theta_against_rad:
            states[i] = 'against'
            continue
        # 2) 共振
        if r_i >= R_high and th_i < theta_following_rad:
            states[i] = 'resonance'
            continue
        # 3) 加速偏离(需 i>=2)
        if i >= 2 and _f(R_slope[i]) and _f(E_slope[i]):
            if R_slope[i] > 0 and E_slope[i] > 0:
                states[i] = 'accelerating'
                continue
            if R_slope[i] < 0 and E_slope[i] < 0:
                states[i] = 'returning'
                continue
        # 5) 独立
        if r_i >= R_high and th_i < theta_against_rad:
            states[i] = 'independent'
            continue
        # 6) 弱偏离
        if R_low <= r_i < R_high and th_i < theta_against_rad:
            states[i] = 'weak_div'
            continue
        # 7) 跟随
        if r_i < R_low and th_i < theta_following_rad:
            states[i] = 'follow'
            continue
        # 兜底
        states[i] = 'none'

    return states


def build_dynamics_df(common_idx, dyn, states, index_tag, stock_tag):
    """组装 14 列动力学结果 DataFrame。

    Args:
        common_idx: 调用方传 common_idx[1:],长度 T-1(与运动投影对齐)
        dyn: compute_dynamics() 返回的 dict
        states: classify_states() 返回的 list[str]
        index_tag / stock_tag: 用于列名(<INDEX_TAG> / <STOCK_TAG> 后缀)

    Returns:
        pd.DataFrame,长度 T-1,14 列。
        加速度列首末 NaN,与速度时序对齐。
    """
    return pd.DataFrame({
        'Date': common_idx,
        f'Dyn_q_{index_tag}': dyn['q_t'],
        f'Dyn_Theta_{stock_tag}': dyn['theta'],                # 弧度
        f'Dyn_Coupling_{stock_tag}': dyn['R'],
        'Dyn_E_Market': dyn['E_market'],
        'Dyn_E_Self': dyn['E_self'],
        'Dyn_E_Total': dyn['E_total'],
        f'Dyn_V_Mag_{stock_tag}': dyn['v_S_mag'],
        f'Dyn_V_Mag_{index_tag}': dyn['v_M_mag'],
        'Dyn_V_Proj_Mag': dyn['v_proj_mag'],
        'Dyn_V_Resi_Mag': dyn['v_resi_mag'],
        f'Dyn_A_Mag_{stock_tag}': dyn['a_S_mag'],              # 首末 NaN
        f'Dyn_A_Mag_{index_tag}': dyn['a_M_mag'],              # 首末 NaN
        f'Dyn_State_{stock_tag}': states,
    })


# === 力模型层(2026-08-16 动力学 §14-17 实现)====================================
# 沿用用户原始 prompt §14-22 的离散动力学方程:
#   a_i = β·a_M  −  k·d  −  c·u  +  F_self
# 默认 k = c = 0(无阻尼基线),F_self 由残差定义,等于个股不能被市场 / 恢复 / 阻尼
# 三项解释的那部分加速度。
#
# 输入:compute_dynamics() 返回的 dyn(用 v_S_mag / v_M_mag / a_S_mag / a_M_mag /
#       实际 2-D 向量从 compute_movement_projection() 的 mv[' stock_move '] /
#       mv[' index_move '] / mv[' proj_coeff '] 取)
# 输出:4 个力的标量模长(2-D 向量已内部计算后取 ‖·‖),便于画 stacked / line。

def compute_forces(dyn, mv, k_restore=0.0, c_damp=0.0):
    """把个股加速度分解成 4 个力:F_market / F_restore / F_damp / F_self。

    Args:
        dyn: compute_dynamics() 的返回值。本函数用 keys:
              'v_S_mag' (T-1,), 'v_M_mag' (T-1,),
              'a_S_mag' (T-1, 末行 NaN), 'a_M_mag' (T-1, 末行 NaN)。
        mv:  compute_movement_projection() 的返回值。本函数用 keys:
              'stock_move' (T-1, 2)、'index_move' (T-1, 2)、
              'proj_coeff' (T-1,) — β。
        k_restore: float,恢复系数 k。默认 0.0 = 无均值回复力。
        c_damp:    float,阻尼系数 c。默认 0.0 = 无阻尼。

    Returns:
        dict with keys (均为 T-1 长):
            F_market:   ndarray (T-1,) 末行 NaN(= β·a_M 沿 a_M 方向的标量投影)
            F_restore:  ndarray (T-1,) ‖−k·d‖(标量幅值)
            F_damp:     ndarray (T-1,) ‖−c·u‖(标量幅值)
            F_self:     ndarray (T-1,) 末行 NaN(由方程残差定义:残差模长)
            d_mag:      ndarray (T-1,) ‖d‖ 位置偏离
            u_mag:      ndarray (T-1,) ‖u‖ 速度偏离

        物理含义:
            ‖a_S‖ ≈ √(‖F_market‖² + ‖F_self‖²  + cross terms)
            设 k = c = 0 时,F_self ≡ a_S - F_market(残差就是它)。

    注:严格按方程 a_S = β·a_M - k·d - c·u + F_self,这里输出的是每个力的
       2-D 向量模长。cross-term 信息在模长运算中丢失(‖a + b‖² ≠ ‖a‖² + ‖b‖²),
       但 4 个 ‖·‖ 的相对大小仍可比,适合画 stacked-area 看力分配比例。
    """
    delta_u = mv['stock_move']              # (T-1, 2) v_S 向量
    delta_v = mv['index_move']              # (T-1, 2) v_M 向量
    beta = mv['proj_coeff']                 # (T-1,)   标量映射系数
    T_minus_1 = delta_u.shape[0]

    # ---- 重建 2-D 加速度(末行 NaN,前 T-2 行用前向 np.diff) ----
    a_u_vec = np.full_like(delta_u, np.nan)     # (T-1, 2)
    a_v_vec = np.full_like(delta_v, np.nan)     # (T-1, 2)
    if T_minus_1 >= 2:
        a_u_vec[:-1] = np.diff(delta_u, axis=0)
        a_v_vec[:-1] = np.diff(delta_v, axis=0)

    # ---- 速度偏离 u = v_S - β·v_M(2-D 向量) ----
    u_vec = delta_u - beta[:, None] * delta_v

    # ---- 位置偏离 d = ∫u dt = Σ_{j<t} u[j],2-D 累积 ----
    d_vec = np.zeros_like(delta_u)
    if T_minus_1 >= 2:
        # cumsum[i] = sum_{j<i} u[j],所以 d[1:] = cumsum(u[:-1]),d[0] = 0
        d_vec[1:] = np.cumsum(u_vec[:-1], axis=0)

    # ---- 4 个力的 2-D 向量 ----
    F_market_vec = beta[:, None] * a_v_vec                  # β·a_M
    F_restore_vec = -k_restore * d_vec                      # -k·d
    F_damp_vec = -c_damp * u_vec                            # -c·u
    F_self_vec = a_u_vec - F_market_vec - F_restore_vec - F_damp_vec

    # ---- 取模长 ----
    F_market = np.linalg.norm(F_market_vec, axis=1)         # 末行 NaN
    F_restore = np.linalg.norm(F_restore_vec, axis=1)
    F_damp = np.linalg.norm(F_damp_vec, axis=1)
    F_self = np.linalg.norm(F_self_vec, axis=1)             # 末行 NaN
    d_mag = np.linalg.norm(d_vec, axis=1)
    u_mag = np.linalg.norm(u_vec, axis=1)

    return {
        'F_market': F_market,
        'F_restore': F_restore,
        'F_damp': F_damp,
        'F_self': F_self,
        'd_mag': d_mag,
        'u_mag': u_mag,
        'k_restore': k_restore,
        'c_damp': c_damp,
    }


def build_forces_df(common_idx, frc, index_tag, stock_tag):
    """组装 8 列力分解结果 DataFrame。

    Args:
        common_idx: 调用方传 common_idx[1:],长度 T-1。
        frc:        compute_forces() 返回的 dict。
        index_tag / stock_tag: 列名后缀。

    Returns:
        pd.DataFrame,长度 T-1,8 列:Date + 4 力模长 + d + u。
        F_market / F_self 末行 NaN(因加速度末行 NaN)。
    """
    return pd.DataFrame({
        'Date': common_idx,
        f'Frc_Market_{index_tag}': frc['F_market'],     # β·‖a_M‖,末行 NaN
        f'Frc_Restore_{stock_tag}': frc['F_restore'],   # ‖k·d‖
        f'Frc_Damp_{stock_tag}': frc['F_damp'],         # ‖c·u‖
        f'Frc_Self_{stock_tag}': frc['F_self'],         # 残差,末行 NaN
        f'Frc_Deviation_{stock_tag}': frc['d_mag'],     # 位置偏离 ‖d‖
        f'Frc_VelDev_{stock_tag}': frc['u_mag'],        # 速度偏离 ‖u‖
        f'Frc_Sum_{stock_tag}': (
            np.sqrt(frc['F_market']**2 + frc['F_self']**2)  # √(‖F_M‖² + ‖F_self‖²)
        ),                                               # 与 ‖a_S‖ 接近(2-D 不严格守恒)
    })