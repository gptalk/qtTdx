# Spec v5.12 — Dashboard regime color-coding (k̂ vs ĉ dominance)

> **Date:** 2026-08-19
> **Base:** v5.11 `kc_estimates` integration (`378d448`)
> **Branch:** new (from `main` HEAD = `378d448`)

## 1. 问题

v5.11 让 `k_used` / `c_used` 从 placeholder (0.0) 变成真实拟合值,业务方终于能在 dashboard 上看到每只股票的 (k̂, ĉ) 标注。但 dashboard **视觉上没体现**这层信息 —— 散点仍然按 hit-rate Viridis 着色,业务方必须 hover 进去才能看到 k̂/ĉ,根本看不出"共振风险 / 过阻尼 / 平衡"的批量分布。

**业务后果**:
- 100+ 只股票批量 OOS 跑完后,业务方要逐个 hover 才能识别"哪些是高 k̂ 共振风险 / 哪些是高 ĉ 过阻尼稳定 / 哪些是平衡"
- v5.10 dashboard 的价值 = 命中质量分布;v5.11 + v5.12 应该叠加**动力学结构分布**
- 没有 regime 颜色,`parameter_fit.py` 算出来的 (k̂, ĉ) 价值被埋没在 hover 里

## 2. 目标

**核心**:在 v5.10 dashboard 的 (2,1) hit-rate × RMSE 散点上,按 (k̂, ĉ) dominance 分 3 种 regime 配色 + legend,让业务方一眼看出"批量里有多少共振股 / 过阻尼股 / 平衡股"。

**业务价值**:批量 OOS 跑完后,业务方先看 legend 比例(共振风险股占比),再下钻看具体哪只。

**非目标 (YAGNI)**:
- ❌ 不动 protected files (11 个)
- ❌ 不动 `build_top5_small_multiples` (v5.10 top-N 详细图,scope 外)
- ❌ 不做 dominance 时序漂移 (那是 v5.13+ 范畴)
- ❌ 不改 hover 字段以外的其他 panel (1,1 / 1,2 / 2,2 不动)
- ❌ 不在 `compute_oos_metrics` 加新输出列 (regime 在 dashboard 里现算)
- ❌ 不做绝对值 / 相对值的混合阈值 (一个简单 ratio 就够)

## 3. 设计

### 3.1 Regime 分类规则

```python
def classify_regime(k: float, c: float, threshold: float = 0.1) -> str:
    """按 |k| / |c| 比例分 3 个 regime。

    threshold = 相对差异容忍度:|k| / |c| 在 [1/(1+threshold), 1+threshold] 内视为平衡。
    防止 1.05/0.95 这种微小差异被分成两类的"伪边界"问题。

    Returns:
        'k_dominant' — |k| > |c| * (1 + threshold)
        'c_dominant' — |c| > |k| * (1 + threshold)
        'balanced'   — 上述之外(等价: ratio 在 [1/(1+threshold), 1+threshold])
    """
```

| 输入 | regime | 颜色 | 业务解读 |
|---|---|---|---|
| `k=0.5, c=0.1, threshold=0.1` | `k_dominant` | 红 `#e74c3c` | 共振风险 |
| `k=0.1, c=0.5, threshold=0.1` | `c_dominant` | 蓝 `#3498db` | 过阻尼稳定 |
| `k=0.5, c=0.49, threshold=0.1` | `balanced` | 绿 `#2ecc71` | 平衡 |
| `k=0.0, c=0.0` | `balanced` (ratio=1) | 绿 | v5.11 placeholder 状态 — 视觉上"无信息" |
| `k=-0.5, c=0.1` | `k_dominant` (|k|=0.5 > |c|*1.1) | 红 | 负 k 也是共振(anti-restoring) |

**边界处理**:
- `k=0` 且 `c=0` → balanced (视为"无信息",不进任何一类)
- `threshold=0.0` → 严格不等式 (退化情况,等价于 `|k| == |c|` → balanced)
- `threshold < 0` → ValueError (caller 错)
- `threshold > 10` → 不拒绝,但警告 (clip 到 10)

### 3.2 Dashboard 改动

**只改 (2,1) scatter panel**:
- 颜色: 从 `Viridis` (连续 hit-rate) → 离散 3 色 regime
- 增加 legend(3 个 regime entry,k_dominant / c_dominant / balanced + 数量计数)
- hover 增加 `regime` 字段

**1,1 / 1,2 / 2,2 panel 不动** —— 业务价值主要是散点的批量模式识别,直方图和 CDF 不需要 regime 信息(再加 panel 会爆信息)。

### 3.3 新 API

```python
def classify_regime(
    k: float,
    c: float,
    threshold: float = 0.1,
) -> str:
    """[上文 3.1]"""
```

`build_full_market_oos_html` 签名 +1 关键参数:
```python
def build_full_market_oos_html(
    metrics_list: list[dict],
    output_path: str,
    title: str = '...',
    regime_threshold: float = 0.1,  # v5.12 NEW
) -> None:
```

**新 CLI flag**:
```python
p.add_argument('--regime-threshold', dest='regime_threshold', type=float, default=0.1,
               help='v5.12: |k|/|c| 平衡区相对差异容忍度 (默认 0.1)')
```

### 3.4 数据流

```
parameter_fit.py → kc_estimates.csv
  ↓
load_oos_predictions (v5.11 → 真实 (k̂, ĉ))
  ↓
compute_oos_metrics (v5.11 → 透传 k_used, c_used)
  ↓ metrics_list 含 {code, hit_rate, rmse, k_used, c_used}
build_full_market_oos_html (v5.10 + v5.12 → regime 颜色)
  ↓ classify_regime(k_used, c_used, threshold) → 'k_dominant' / 'c_dominant' / 'balanced'
  ↓ 散点按 regime 着色 + legend
HTML 输出
```

### 3.5 Regime 缺失处理

**关键场景**: v5.11 之前或 `k_used=c_used=0` (placeholder 状态) 时:
- `classify_regime(0, 0, 0.1)` → `balanced` (绿)
- 这意味着 placeholder 状态的股票会被画成绿色"平衡" —— 视觉上不区分真实平衡股

**对策** (v5.12 范围内):
- hover 模板里额外显示 `k̂=0.0000, ĉ=0.0000` (placeholder 提示)
- legend 里 balanced 计数包含 placeholder (总数对得上)
- **不在 v5.12 范围**: 加 4th color "unknown / placeholder" (留作 v5.13+)

## 4. 验证

### 4.1 单元测试

```python
# tests/test_dynamics_eigen.py 新增 test

def test_classify_regime():
    """v5.12 — classify_regime 5 cases: k_dominant / c_dominant / balanced / placeholder / 负值。"""
    # 1. k 主导
    assert classify_regime(0.5, 0.1, 0.1) == 'k_dominant'

    # 2. c 主导
    assert classify_regime(0.1, 0.5, 0.1) == 'c_dominant'

    # 3. 平衡 (|k|/|c| 在 [1/1.1, 1.1])
    assert classify_regime(0.5, 0.49, 0.1) == 'balanced'

    # 4. 占位符 (k=c=0) → balanced
    assert classify_regime(0.0, 0.0, 0.1) == 'balanced'

    # 5. 负 k(anti-restoring)也是 k_dominant
    assert classify_regime(-0.5, 0.1, 0.1) == 'k_dominant'

    # 6. 边界: |k|/|c|=1.10 → balanced (恰在阈值边界)
    assert classify_regime(0.55, 0.50, 0.1) == 'balanced'

    # 7. 边界: |k|/|c|=1.11 → k_dominant (刚超阈值)
    assert classify_regime(0.555, 0.50, 0.1) == 'k_dominant'

    # 8. threshold=0 → 严格不等式(只有 k=c 进 balanced)
    assert classify_regime(0.5, 0.5, 0.0) == 'balanced'
    assert classify_regime(0.5, 0.4, 0.0) == 'k_dominant'

    # 9. threshold < 0 → ValueError
    import pytest
    with pytest.raises(ValueError):
        classify_regime(0.5, 0.1, -0.1)

    # 10. threshold > 10 → clip 警告(测试不实现 clip,但保证不崩)
    assert classify_regime(0.5, 0.1, 100.0) in ('k_dominant',)
```

### 4.2 CLI 冒烟

```bash
# 1. 用真实 CSV 跑 v5.10 全市场(同 v5.11 冒烟)
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --days 60 --limit 5 --top-n 3 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --output backtrace/outputs/_smoke_v5_12.html

# 期望:HTML 散点 (2,1) 按 3 色 legend 显示,hover 含 regime

# 2. 调 threshold
PYTHONIOENCODING=utf-8 /c/Users/yellow/.conda/envs/venv/python.exe \
    backtrace/dynamics/dynamics_oos_batch.py --days 60 --limit 5 \
    --kc-estimates-csv data/projection/kc_estimates.csv \
    --regime-threshold 0.3 \
    --output backtrace/outputs/_smoke_v5_12_t03.html

# 期望:阈值变宽,balanced 数量变多
```

## 5. 约束

- **0 modifications to 11 protected files**:
  - `_projection_core.py` / `dynamics_forced_response.py` / `dynamics_system.py` /
    `dynamics_batch.py` (注意:**不是 `dynamics_oos_batch.py`**) / `dynamics_1step_oos.py` /
    `dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` /
    `dynamics_eigen_analysis.py` / `parameter_fit.py` / `dynamics_oos_viz.py`
- **0 modifications to `dynamics_oos_batch.py` core logic** (只允许在 `build_full_market_oos_html` 内部 + `classify_regime` 新 helper + main() argparse)
- **0 new dependencies** (pandas / numpy / plotly 已在)
- **1 modified file**: `backtrace/dynamics/dynamics_oos_batch.py` (build_full_market_oos_html + classify_regime + CLI flag)
- **1 modified file**: `tests/test_dynamics_eigen.py` (+1 test)
- **1 modified file**: `backtrace/dynamics/README.md` (§4.1.11)
- **不重跑 batch CSV** — dashboard 现算 regime,不需要新数据源

## 6. 与 v5.x 系列的关系

| 版 | commit | 主题 | (k̂, ĉ) 业务可见度 |
|---|---|---|---|
| v5.9 | d692860 | load_oos_predictions + 4-row plotly | placeholder (0.0) |
| v5.10 | c50b248 | 全市场分布 + top-N | placeholder + README caveat |
| v5.11 | 378d448 | kc_estimates 接入单股 | 真实 hover 内可见 |
| **v5.12** | (本次) | **regime 颜色编码** | **批量散点视觉化 (legend + 3 色)** |

v5.12 是 v5.11 真实 (k̂, ĉ) 的"视觉化"层 —— 数据已经在,只是没被业务方看到。

## 7. 不在范围 / 后续

- v5.13: dashboard 加 (3,1) panel 展示 (k̂, ĉ) 在 (k̂, ĉ) 平面上的散点(独立维度)
- v5.14: dominance 时序漂移(rolling 拟合窗口的 regime 切换热图)
- v5.15: top-N small multiples 按 regime 上色(目前只动 dashboard 散点)
- v5.16: 4th color "unknown / placeholder" 区分 k_used=c_used=0 状态

## Status: 📝 DRAFT — 2026-08-19

待 plan + implementer。