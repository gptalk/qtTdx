# -*- coding: utf-8 -*-
# _dynamics_core.py — 离散动力系统入口(描述层 + 力模型 + N 步轨迹模拟)
#
# 数学源头:backtrace/projection/_projection_core.py
#   - compute_dynamics / classify_states / build_dynamics_df   → 描述层
#   - compute_forces    / build_forces_df                       → 力模型
#   - STATE_LABELS / STATE_COLORS / STATE_LABELS_CN            → 状态配色
#
# 本模块新增 3 个真正属于"动力系统"的功能:
#   1. predict_next_state  — 1 步预测(扩展 prediction_ode.py 的散装函数为可调用 API)
#   2. simulate_trajectory — N 步前向模拟(已知未来大盘 + β + 残差,个股怎么演化)
#   3. build_simulation_df — 模拟结果 DataFrame
#
# 不重写任何 projection 层的数学 — 单一来源真理。
import os
import sys
import numpy as np
import pandas as pd

# 把 projection/ 加进 path,导入其数学
_PROJECTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'projection')
if _PROJECTION_DIR not in sys.path:
    sys.path.insert(0, _PROJECTION_DIR)

# 复用 projection 层的所有动力相关函数
from _projection_core import (
    # 描述层
    compute_dynamics,
    classify_states,
    build_dynamics_df,
    # 力模型
    compute_forces,
    build_forces_df,
    # 状态分类配色
    STATE_LABELS,
    STATE_COLORS,
    STATE_LABELS_CN,
)

# === 1 步预测(用户 prompt §19) ==============================================
def predict_next_state(
    v_S_now: np.ndarray,        # (2,) 当前个股速度
    a_M_now: np.ndarray,        # (2,) 当前大盘加速度
    beta_now: float,            # 当前 β
    d_now: np.ndarray,          # (2,) 当前位置偏离
    u_now: np.ndarray,          # (2,) 当前速度偏离
    F_self_now: np.ndarray | None = None,  # (2,) 当前残差;None 时按 F_self = a_S - β·a_M 推
    a_S_now: np.ndarray | None = None,     # (2,) 当前个股加速度(F_self=None 时必传)
    k: float = 0.0,
    c: float = 0.0,
    q_now: float = 1.0,         # 锚定强度 q_t;默认 1.0 = 无阻尼(向后兼容旧 caller)
) -> tuple:
    """1 步预测下一个交易日的个股加速度 / 速度。

    模型(用户 prompt §14-19,**统一版**;与 simulate_trajectory 共享同一方程):
        a_pred = q_now * β·a_M - k·d - c·u + F_self
        v_pred = v_S + a_pred          (Δt = 1)

    旧版本仅返 (a_pred, v_pred, delta_S_pred) — 但 delta_S_pred ≡ v_pred(纯冗余),
    现已删除第 3 个返回值。旧 caller 若解构 3 元组会报 too many values to unpack。

    时间轴约定(全篇):见 simulate_trajectory 顶部 docstring。

    Args:
        v_S_now:  当前 v_S(2-D 向量,ΔVol/ΔAmt 量纲)
        a_M_now:  当前 a_M(2-D 向量,代表"从 step t 到 step t+1 的市场速度变化")
        beta_now: 当前 β
        d_now:    当前 d(2-D 位置偏离累积)
        u_now:    当前 u(2-D 速度偏离)
        F_self_now: (可选)外部给定的残差;若 None 则由 a_S_now 推 F_self = a_S - q·β·a_M + k·d + c·u
        a_S_now:   (F_self_now=None 时必传)当前 a_S
        k: 恢复系数
        c: 阻尼系数
        q_now: 锚定强度(与 simulate_trajectory 的 q_t_seq[t] 同语义);
               默认 1.0 表示无阻尼。description 层用 q = ‖ΔM‖/(‖ΔM‖+λ_q) ∈ [0,1]。

    Returns:
        (a_pred, v_pred),都是 (2,) ndarray
    """
    if F_self_now is None:
        if a_S_now is None:
            raise ValueError("F_self_now=None 时必须传 a_S_now 才能推残差")
        F_self_now = a_S_now - q_now * beta_now * a_M_now + k * d_now + c * u_now
    a_pred = q_now * beta_now * a_M_now - k * d_now - c * u_now + F_self_now
    v_pred = v_S_now + a_pred                  # Δt = 1
    return a_pred, v_pred


# === F_self 预测器(用户 prompt §14-19 中"残差外推"的扩展) ===================
def make_rolling_mean_f_self_predictor(
    F_self_history: np.ndarray,        # (T_hist, 2) 已知残差历史(可含 NaN)
    window: int = 10,                  # 滚动窗口长度(天)
) -> 'callable':
    """构造"滚动均值 F_self 预测器"。

    模拟过程中 F_self_pred 不随 sim step 变化 — 用末日之前 [end-W, end) 的均值
    作为整段模拟的常数残差。比 dynamics_batch.py 默认的"末日瞬时值复制 N 次"更稳:
    末日瞬时值含噪声,滚动均值能平滑掉单日抖动。

    Args:
        F_self_history: (T_hist, 2) — 已知残差(模拟起点之前)
        window: 滚动窗口(天)。默认 10。设 0 = 全历史均值;设大 = 长期均值

    Returns:
        predictor: callable(t: int, hist: dict | None) -> ndarray (2,)
            - t: 模拟步号(0..N-1),本预测器忽略 t(常数预测)
            - hist: 预留接口,本预测器忽略
    """
    # 计算"末日滚动均值",作为常数预测
    end = len(F_self_history)
    if window <= 0 or end == 0:
        # 全历史均值
        valid = np.isfinite(F_self_history).all(axis=1) if end > 0 else np.array([])
    else:
        start = max(0, end - window)
        seg = F_self_history[start:end]
        valid = np.isfinite(seg).all(axis=1) if len(seg) > 0 else np.array([])

    if len(valid) == 0 or not valid.any():
        F_self_const = np.zeros(2)
    elif window <= 0 or end == 0:
        F_self_const = np.nanmean(F_self_history[valid], axis=0)
    else:
        seg = F_self_history[max(0, end - window):end]
        F_self_const = np.nanmean(seg[valid], axis=0)

    def predictor(t, hist=None):
        return F_self_const.copy()
    predictor.__doc__ = (
        f'Rolling-mean F_self predictor (window={window}, '
        f'F_self_const=({F_self_const[0]:+.3e}, {F_self_const[1]:+.3e}))'
    )
    return predictor


def make_constant_f_self_predictor(F_self_const: np.ndarray) -> 'callable':
    """构造"常数 F_self 预测器" — 给一个固定值,模拟过程中不变。

    等价于 dynamics_batch.py 默认的"末日瞬时值复制 N 次"行为。
    """
    arr = np.asarray(F_self_const, dtype=float)

    def predictor(t, hist=None):
        return arr.copy()
    predictor.__doc__ = f'Constant F_self predictor ({arr[0]:+.3e}, {arr[1]:+.3e})'
    return predictor


# === Forecast 模式:预生成 v_M_seq / beta_seq / q_t_seq(用户 Task 4) ===========
def forecast_v_M_random_walk(
    v_M_init: np.ndarray,         # (2,) 起点 v_M(取末日真实 v_M)
    n_steps: int,
    sigma_per_step: np.ndarray | float,   # 标量或 (2,) 每步噪声 std
    random_state: int | np.random.Generator = 42,
) -> np.ndarray:
    """随机游走生成 v_M_seq = (n_steps, 2)。

    v_M(t=0) = v_M_init
    v_M(t)   = v_M(t-1) + noise[t-1],noise[t-1] ~ N(0, sigma²)
    (noise 是 (n_steps, 2),索引 0..N-1 对应 step t=1..N)
    即 v_M(t) = v_M_init + Σ_{τ=0..t-1} noise[τ]

    用于 simulate_trajectory 的 forecast 模式:模拟起点之后没有真实大盘,
    用历史观测到的 sigma 来外推未来 N 步。σ 推荐 = 历史 diff 的 std。

    Args:
        v_M_init:        (2,) 起点(末日观测 v_M)
        n_steps:         模拟步数 N
        sigma_per_step:  每步 std;标量 → 各维度同;tuple/ndarray (2,) → 各维度独立
        random_state:    int 或 Generator;默认 42(可复现)
    """
    rng = np.random.default_rng(random_state)
    if np.isscalar(sigma_per_step):
        noise = rng.normal(0, float(sigma_per_step), size=(n_steps, 2))
    else:
        # 对角噪声(各维度独立)
        sig = np.asarray(sigma_per_step, dtype=float)
        if sig.shape != (2,):
            raise ValueError(f"sigma_per_step 应为标量或 (2,),收到 {sig.shape}")
        noise = rng.normal(0, 1.0, size=(n_steps, 2)) * sig
    v_M_seq = np.zeros((n_steps, 2), dtype=float)
    v_M_seq[0] = v_M_init
    for t in range(1, n_steps):
        v_M_seq[t] = v_M_seq[t - 1] + noise[t - 1]
    return v_M_seq


def forecast_v_M_last_value(v_M_last: np.ndarray, n_steps: int) -> np.ndarray:
    """末值恒定:v_M(t) = v_M_last for t = 0..N-1。

    最朴素的 forecast 假设:未来大盘速度 = 末日观测。适合大盘震荡行情。
    """
    return np.tile(np.asarray(v_M_last, dtype=float), (n_steps, 1))


def forecast_beta_last_value(beta_last: float, n_steps: int) -> np.ndarray:
    """β(t) = beta_last 恒定。适合 β 短期平稳的股票(行业基线稳定)。"""
    return np.full(n_steps, float(beta_last), dtype=float)


def forecast_beta_rolling_mean(beta_history: np.ndarray, n_steps: int,
                                window: int = 10) -> np.ndarray:
    """β(t) = mean(beta_history[-W:]) 恒定。比末值更平滑,降低单日 β 噪声。"""
    if window <= 0 or len(beta_history) == 0:
        b_mean = float(np.nanmean(beta_history)) if len(beta_history) > 0 else 1.0
    else:
        seg = beta_history[-window:]
        b_mean = float(np.nanmean(seg))
    return np.full(n_steps, b_mean, dtype=float)


def forecast_q_t_constant(q_t_last: float, n_steps: int) -> np.ndarray:
    """q_t 恒定。用于 forecast 模式 — 没有未来 ‖ΔM‖,沿用末日观测。"""
    return np.full(n_steps, float(q_t_last), dtype=float)


# === N 步前向模拟(用户 prompt §19 + §14-18) =================================
def simulate_trajectory(
    v_S_init: np.ndarray,           # (2,) 起点速度(取末日真实 v_S)
    v_M_seq: np.ndarray,            # (N, 2) 未来 N 天大盘速度
    beta_seq: np.ndarray,           # (N,)   未来 N 天 β
    d_init: np.ndarray,             # (2,)   起点位置偏离
    u_init: np.ndarray,             # (2,)   起点速度偏离
    k: float = 0.0,
    c: float = 0.0,
    q_t_seq: np.ndarray | None = None,   # (N,) 锚定强度;None 时默认 1
    classify_thresholds: tuple = (0.10, 0.50, np.deg2rad(30), np.deg2rad(90)),
    F_self_seq: np.ndarray | None = None,         # (N, 2) 残差序列;若 predictor=None 则必传
    F_self_predictor: 'callable | None' = None,   # 残差预测器: t -> (2,);优先级高于 F_self_seq
) -> dict:
    """N 步前向模拟。

    时间轴约定(全篇):
        v_M_seq[t]   第 t 步的大盘速度(2-D 向量,ΔVol/ΔAmt 量纲)
        a_M_seq[t]   = v_M_seq[t+1] - v_M_seq[t],代表"从 step t 到 step t+1 发生的市场速度变化"
                      (前向差;末步 t=N-1 留 NaN,因无 v_M_seq[N] 可作下一差)
        v_S_seq[t]   个股速度;t=0 = v_S_init(末日观测);t=1..N 由递推产生
        a_S_seq[t]   = v_S_seq[t+1] - v_S_seq[t],递推产出(末行 NaN)
        u_seq[t]     = v_S_seq[t] - β_seq[t] * v_M_seq[t],t 时刻速度偏离
        d_seq[t+1]   = d_seq[t] + u_seq[t](递推:在 step t 内加 step t 的 u,与 spec 写法一致)
        F_self(t)    = F_self_seq[t] 或 F_self_predictor(t),step t 的特异力

    动力学方程(用户 prompt §14-19,**统一版**;与 predict_next_state 共享):
        a_t = q_seq[t] * β_seq[t] * a_M_seq[t] - k * d_seq[t] - c * u_seq[t] + F_self(t)
        v_seq[t+1] = v_seq[t] + a_t

    注:本函数采用「**沿 v_M(t) 方向的真正正交分解**」(Gram-Schmidt)计算 E_market /
    E_self / R,与 description 层 `compute_dynamics` 的 `v_proj = q·β·v_M` 不同——
    description 层用「β 回归投影」(设计选择:β 是回归斜率,语义独立),simulation 层
    用「严格正交」(与 v_M 严格正交,R ∈ [0, 1] 不需 clip)。

    模式 1 — Oracle(已知未来大盘/β/残差):
        传 F_self_seq = (N, 2),走"末日观测残差恒定外推"。适合调试 / 描述层验证。
    模式 2 — Forecast(残差用预测器):
        传 F_self_predictor = callable(t) -> (2,)。推荐 make_rolling_mean_f_self_predictor
        或 make_constant_f_self_predictor。

    Args:
        v_S_init: (2,) 起点速度(末日真实 v_S)
        v_M_seq:  (N, 2) 未来 N 天大盘速度
        beta_seq: (N,)   未来 N 天 β
        d_init / u_init: (2,) 起点状态。**注意**:若希望 d_seq[t+1] = d_seq[t] + u_seq[t] 时
                       模拟起点处 d_seq[1] 与 description 层 d_full[-1] 一致,需要
                       d_init = d_full[-1] - u_full[-1]("前推一格"以消去累积 u)。
        k / c: 力模型系数
        q_t_seq: (N,) 锚定强度,None 时默认全 1(无阻尼);与 description 层 λ_q 同语义
        classify_thresholds: 4 元组,与 classify_states 同
        F_self_seq: (N, 2) 残差序列。predictor=None 时必传;predictor≠None 时被忽略。
        F_self_predictor: callable(t, hist=None) -> (2,) 残差预测器。优先级最高。

    Returns:
        dict with keys(均为长度 N+1,index 0=起点):
            v_seq / d_seq / u_seq: ndarray (N+1, 2)
            a_seq:                 ndarray (N+1, 2) 末行 NaN
            E_total / E_market / E_self: ndarray (N+1,)
            R:                     ndarray (N+1,)
            theta:                 ndarray (N+1,) 弧度
            state:                 list[str] (N+1,)
            v_M_seq_used:          ndarray (N, 2) 回放(便于 caller 画图)
            beta_seq_used:         ndarray (N,)
            F_market / F_restore / F_damp / F_self: ndarray (N+1,) 各力模长(末行 NaN for F_market/Self)
            F_self_predictor_used: callable | None — 回放 caller 传进来的预测器
    """
    N = v_M_seq.shape[0]
    if beta_seq.shape[0] != N:
        raise ValueError(f"beta_seq 长度 {beta_seq.shape[0]} != v_M_seq {N}")
    if F_self_predictor is None and F_self_seq is None:
        raise ValueError("F_self_predictor 和 F_self_seq 必须传一个")
    if F_self_seq is not None and F_self_seq.shape != (N, 2):
        raise ValueError(f"F_self_seq 形状 {F_self_seq.shape} != ({N}, 2)")
    if q_t_seq is None:
        q_t_seq = np.ones(N)
    elif q_t_seq.shape[0] != N:
        raise ValueError(f"q_t_seq 长度 {q_t_seq.shape[0]} != v_M_seq {N}")

    v_seq = np.zeros((N + 1, 2))
    d_seq = np.zeros((N + 1, 2))
    u_seq = np.zeros((N + 1, 2))
    a_seq = np.full((N + 1, 2), np.nan)
    F_market = np.full(N + 1, np.nan)
    F_restore = np.full(N + 1, np.nan)
    F_damp = np.full(N + 1, np.nan)
    F_self = np.full(N + 1, np.nan)

    # t=0 = 起点
    v_seq[0] = v_S_init
    d_seq[0] = d_init
    u_seq[0] = u_init
    # t=0 的力:用 v_S_init 等于刚走完的上一步,初值力 = (q*β*a_M, -k*d, -c*u, F_self)
    # 末步观察的外推 — 上一步 a_M 用 v_M_seq[0] - 0(假设 0 之前的市场速度)
    # 这里我们只对 t=0 设近似力:
    if N >= 1:
        # t=0 的 F_self:用 predictor(t=0) 或 F_self_seq[0]
        if F_self_predictor is not None:
            F_self_0 = np.asarray(F_self_predictor(0, None), dtype=float)
        else:
            F_self_0 = F_self_seq[0]
        F_market[0] = abs(q_t_seq[0] * beta_seq[0] * 0.0)  # 无前一步 → 0
        F_restore[0] = float(np.linalg.norm(k * d_init))
        F_damp[0] = abs(c * np.linalg.norm(u_init))
        F_self[0] = float(np.linalg.norm(F_self_0))

    # 大盘加速度 a_M(t) = v_M_seq[t+1] - v_M_seq[t],长度 N-1;末步 a_M(N-1) = NaN
    a_M_seq = np.full((N, 2), np.nan)
    if N >= 2:
        a_M_seq[:-1] = np.diff(v_M_seq, axis=0)

    # 主体循环
    for t in range(N):
        d_t = d_seq[t]
        u_t = u_seq[t]
        v_t = v_seq[t]
        # a_M(t) 末步 NaN — 此时模型也只产出 NaN 加速,后续 v_{N} 标 NaN
        a_M_t = a_M_seq[t]
        # 残差:F_self_predictor 优先;否则 F_self_seq[t]
        if F_self_predictor is not None:
            F_self_t = np.asarray(F_self_predictor(t, None), dtype=float)
        else:
            F_self_t = F_self_seq[t]
        if not np.isfinite(a_M_t).all():
            # 末步:无 a_M 输入 → 市场驱动力 = 0
            a_pred = -k * d_t - c * u_t + F_self_t
            F_market_t = 0.0
        else:
            # q_t 阻尼 a_M(与 description 层 compute_dynamics 一致)
            a_pred = (
                q_t_seq[t] * beta_seq[t] * a_M_t
                - k * d_t - c * u_t
                + F_self_t
            )
            F_market_t = float(np.linalg.norm(q_t_seq[t] * beta_seq[t] * a_M_t))
        a_seq[t] = a_pred
        # 力分解(取模长)— F_self 来自 F_self_t(预给序列 or 预测器)
        F_market[t] = F_market_t
        F_restore[t] = float(np.linalg.norm(k * d_t))
        F_damp[t] = float(np.linalg.norm(c * u_t))
        F_self[t] = float(np.linalg.norm(F_self_t))

        # 步 t → t+1
        v_seq[t + 1] = v_t + a_pred
        # 下日 β;末步用 β_t 兜底(避免越界)
        beta_next = beta_seq[t + 1] if t + 1 < N else beta_seq[t]
        v_M_next = v_M_seq[t + 1] if t + 1 < N else v_M_seq[t]
        u_seq[t + 1] = v_seq[t + 1] - beta_next * v_M_next
        # 位置偏离递推(spec 写法):d[t+1] = d[t] + u[t]
        # 注:此式要求 caller 传 d_init = d_full[-1] - u_full[-1],
        # 使 d_seq[1] 与 description 层 d_full[-1] 一致(详见 docstring)
        d_seq[t + 1] = d_t + u_seq[t]

    # 末步 (t=N) 力的"回声":用末态估算
    F_restore[N] = float(np.linalg.norm(k * d_seq[N]))
    F_damp[N] = float(np.linalg.norm(c * u_seq[N]))

    # 派生量:R / θ / E / state
    # 真正正交分解(沿 v_M(t) 方向 Gram-Schmidt 投影):
    #   v_proj(t) = (v_S(t) · v_M(t) / |v_M(t)|²) · v_M(t)
    #   v_res(t)  = v_S(t) - v_proj(t)        ← 严格 ⊥ v_M(t)
    # 这样 R = |v_res|² / |v_S|² ∈ [0, 1] 严格成立(无需 clip)。
    # 与 description 层 v_proj = q·β·v_M 不同(description 用 β 回归投影,语义独立)。
    v_S_mag = np.linalg.norm(v_seq, axis=1)
    v_proj_mag_sq = np.full(N + 1, np.nan, dtype=float)
    v_resi_mag = np.full(N + 1, np.nan, dtype=float)
    for t in range(N):
        v_M_t = v_M_seq[t]
        v_M_mag_sq_t = float(np.dot(v_M_t, v_M_t))
        if v_M_mag_sq_t > 1e-12:
            coeff = float(np.dot(v_seq[t], v_M_t)) / v_M_mag_sq_t
            v_proj = coeff * v_M_t
            v_res = v_seq[t] - v_proj
            v_proj_mag_sq[t] = float(np.dot(v_proj, v_proj))
            v_resi_mag[t] = float(np.linalg.norm(v_res))
    E_market = 0.5 * v_proj_mag_sq
    E_self = 0.5 * v_resi_mag ** 2
    # E_total = 0.5 * |v_S|²(只与 v_S 自身有关,与正交分解无关)
    E_total = 0.5 * v_S_mag ** 2
    # R = ‖v_res‖² / ‖v_S‖²(真正交保证 ≤ 1,不需 clip)
    R = np.divide(
        v_resi_mag ** 2, v_S_mag ** 2,
        out=np.zeros_like(v_resi_mag),
        where=(v_S_mag ** 2 > 1e-12) & np.isfinite(v_resi_mag ** 2),
    )
    # θ — t=0..N-2 用 (v_t, v_M_t),t=N-1 无 v_M_N → NaN,t=N 不算(末行)
    v_M_mag = np.linalg.norm(v_M_seq, axis=1)
    cos_theta = np.full(N + 1, np.nan, dtype=float)
    for t in range(N):
        denom = v_S_mag[t] * v_M_mag[t]
        if denom > 1e-12 and np.isfinite(denom):
            cos = np.dot(v_seq[t], v_M_seq[t]) / denom
            cos_theta[t] = float(np.clip(cos, -1.0, 1.0))
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # 状态分类 — R/theta 在 t=0..N-1 都有限,直接 classify
    R_low, R_high, theta_following, theta_against = classify_thresholds
    state = classify_states(
        R[:N], theta[:N], E_self[:N],
        (R_low, R_high, theta_following, theta_against),
    )
    # t=N 的 state = 'none'(无对应分类输入)
    state = state + ['none']

    return {
        'v_seq': v_seq,
        'a_seq': a_seq,
        'd_seq': d_seq,
        'u_seq': u_seq,
        'v_S_mag': v_S_mag,
        'v_M_mag': np.concatenate([v_M_mag, [v_M_mag[-1]]]),  # 长度对齐 N+1
        'E_total': E_total,
        'E_market': E_market,
        'E_self': E_self,
        'R': R,
        'theta': theta,
        'state': state,
        'v_M_seq_used': v_M_seq,
        'beta_seq_used': beta_seq,
        'q_t_seq_used': q_t_seq,
        'F_market': F_market,
        'F_restore': F_restore,
        'F_damp': F_damp,
        'F_self': F_self,
        'k_restore': k,
        'c_damp': c,
        'F_self_predictor_used': F_self_predictor,
    }


# === 模拟结果 DataFrame 组装 =================================================
def build_simulation_df(sim: dict, dates, index_tag: str, stock_tag: str) -> pd.DataFrame:
    """把 simulate_trajectory() 的输出组装成 CSV-ready DataFrame。

    Args:
        sim:  simulate_trajectory() 返回的 dict
        dates: (N+1,) 日期序列;None 时用 0..N 整数占位
        index_tag / stock_tag: 列名后缀

    Returns:
        pd.DataFrame,长度 N+1,14 列:
            Date, Sim_a_{idx}, Sim_a_{st}, Sim_v_{st}_Vol, Sim_v_{st}_Amt,
            Sim_d_{st}, Sim_u_{st}, Sim_R, Sim_Theta, Sim_E_Total,
            Sim_E_Market, Sim_E_Self, Sim_State
    """
    N_plus_1 = sim['v_seq'].shape[0]
    if dates is None:
        dates = list(range(N_plus_1))
    elif len(dates) != N_plus_1:
        raise ValueError(
            f"dates 长度 {len(dates)} != N+1 = {N_plus_1}"
        )

    return pd.DataFrame({
        'Date': dates,
        f'Sim_v_M_{index_tag}': np.concatenate([
            np.linalg.norm(sim['v_M_seq_used'], axis=1),  # (N,)
            [np.nan],                                     # pad 到 N+1
        ]),
        f'Sim_a_{stock_tag}': np.linalg.norm(sim['a_seq'], axis=1),
        f'Sim_v_{stock_tag}_Vol': sim['v_seq'][:, 0],
        f'Sim_v_{stock_tag}_Amt': sim['v_seq'][:, 1],
        f'Sim_d_{stock_tag}': np.linalg.norm(sim['d_seq'], axis=1),
        f'Sim_u_{stock_tag}': np.linalg.norm(sim['u_seq'], axis=1),
        'Sim_R': sim['R'],
        'Sim_Theta': sim['theta'],
        'Sim_E_Total': sim['E_total'],
        'Sim_E_Market': sim['E_market'],
        'Sim_E_Self': sim['E_self'],
        f'Sim_F_Market_{index_tag}': sim['F_market'],
        f'Sim_F_Restore_{stock_tag}': sim['F_restore'],
        f'Sim_F_Damp_{stock_tag}': sim['F_damp'],
        f'Sim_F_Self_{stock_tag}': sim['F_self'],
        f'Sim_State_{stock_tag}': sim['state'],
        # 调试用 — k / c 末行对齐(标量)
        'Sim_k_restore': [sim['k_restore']] * N_plus_1,
        'Sim_c_damp': [sim['c_damp']] * N_plus_1,
    })


# === 公开 re-export ==========================================================
__all__ = [
    # 复用 projection 层
    'compute_dynamics',
    'classify_states',
    'build_dynamics_df',
    'compute_forces',
    'build_forces_df',
    'STATE_LABELS',
    'STATE_COLORS',
    'STATE_LABELS_CN',
    # 新增
    'predict_next_state',
    'simulate_trajectory',
    'build_simulation_df',
    # F_self 预测器
    'make_rolling_mean_f_self_predictor',
    'make_constant_f_self_predictor',
    # Forecast 模式
    'forecast_v_M_random_walk',
    'forecast_v_M_last_value',
    'forecast_beta_last_value',
    'forecast_beta_rolling_mean',
    'forecast_q_t_constant',
]
