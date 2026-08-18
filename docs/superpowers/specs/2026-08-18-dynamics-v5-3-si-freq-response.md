# Spec v5.3 — Real SI Frequency Response (时序动画)

> **Date:** 2026-08-18
> **Base:** v5.2 parameter_fit integration with v5.1 overlay(`fce9532`)
> **Branch:** new(从 `main` HEAD = `fce9532`)

## 1. 问题

v5.2 把"industry G(ω) overlay"从手动输入 (k, c) 升级到**数据驱动**(`kc_estimates.csv` 行业聚合 + top-N 自动选取 + 单帧 overlay)。

但 v5.2 只画**一个时间点**的 overlay。v4.9 (`parameter_fit --rolling-time`) 已经能产生**每月滚动**的 `(k̂(t), ĉ(t))` 时序。业务场景真正的问题是"**频率响应如何随时间演化**":
- 哪个行业从**低通过滤器**(过阻尼)漂移到**共振区**(欠阻尼)?
- 哪个行业的 `|H(jω_n)|` 峰值在最近 N 个月翻倍?
- 哪些行业**始终稳定**,哪些**始终不稳定**?

v5.2 不回答这些 — 它画的是**一个 asof_date** 的快照。v5.3 把这层补上:**多 asof_date 的 Bode overlay 通过 plotly 动画 slider 联动**,业务可拖时间轴看漂移。

## 2. 目标

**核心**:新 CLI `dynamics_si_freq_response.py` 读 `kc_estimates_time.csv`(v4.9 rolling 输出),按 `asof_date` 切片 → 每片按 `index_code` 聚合 → 每片选 top-N 行业 → 每片用 v5.1 `bode_overlay` 风格的 G(ω) 曲线,但**通过 plotly `animation_frame` 联动多帧**。最终一个 HTML,时间滑块拖动可见漂移。

**非目标(YAGNI)**:
- ❌ 不读 `kc_estimates.csv`(full-sample, 无时序)— 只读 `kc_estimates_time.csv`(rolling 时序)
- ❌ 不做行业名称解析(申万二级中文名)— `index_code` 直接作 label(与 v5.2 一致)
- ❌ 不做预测/前瞻 — 只可视化过去时序
- ❌ 不重写 `bode_overlay` / `transfer_function` / `natural_frequency` — **v5+v5.1+v5.2 0 修改**
- ❌ 不耦合到 `parameter_fit` 的内部函数 — 只读 CSV(单一接口,与 v5.2 一致)
- ❌ 不做实时数据接入 — 离线分析(`kc_estimates_time.csv` 必须先存在)

**理由**:
- 与 v5.2 解耦 — v5.2 是单帧,v5.3 是动画;两者并存(用户跑 v5.2 看当前快照,跑 v5.3 看时序演化)
- 与 v4.9 解耦 — v4.9 算 SI(t),v5.3 算 G(ω)(t);两者可独立使用
- 单一职责 — v5.3 只做"时序 → 动画 overlay"
- 复用最大化 — `aggregate_by_industry` 内部按日期切片复用 v5.2 聚合函数(用 groupby 多列)

## 3. 设计

### 3.1 架构

```
┌────────────────────────────────────────────────────────────────┐
│  backtrace/dynamics/dynamics_si_freq_response.py (新文件)        │
│  ┌─────────────────────────────────────────────────────┐      │
│  │  [v5.3 新增] load_kc_time_series(path) → DataFrame    │      │
│  │  aggregate_by_industry_per_date(df, dates, group_col) │      │
│  │  select_top_n_per_date(per_date_dfs, criterion, n)    │      │
│  │  build_animated_overlay_html(per_date_pairs, ...)     │      │
│  └─────────────────────────────────────────────────────┘      │
│  ┌─────────────────────────────────────────────────────┐      │
│  │  [复用 v5.1] natural_frequency(k, c)                  │      │
│  │  [复用 v5.1] magnitude_phase(z_array, k, c)           │      │
│  └─────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────┘

输入:data/projection/kc_estimates_time.csv(v4.9 rolling 输出)
输出:1 个 HTML(plotly 动画 slider)+ 1 个 summary TXT + 1 个 audit CSV
```

### 3.2 v5.3 新 API

```python
def load_kc_time_series(csv_path: str) -> pd.DataFrame:
    """读 parameter_fit --rolling-time 输出 kc_estimates_time.csv。

    必需列:code, index_code, asof_date, k_hat, c_hat, status, n_valid_days

    Returns:
        DataFrame,只保留 status='ok' 的行 + n_valid_days >= 192(ramp-up filter,沿用 v4.9)

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    """
```

```python
def aggregate_by_industry_per_date(
    df: pd.DataFrame,
    dates: list[str],            # YYYY-MM-DD 列表
    group_col: str = "index_code",
    agg: str = "median",
) -> dict[str, pd.DataFrame]:
    """按 (asof_date, index_code) 聚合 (k̂, ĉ)。

    Returns:
        {asof_date: DataFrame [group_col, n_stocks, k_hat, c_hat]},每片按 group_col 排序

    YAGNI: 复用 v5.2 `aggregate_by_industry` 内部 groupby 逻辑(同函数签名兼容)
    """
```

```python
def select_top_n_per_date(
    per_date_dfs: dict[str, pd.DataFrame],
    criterion: str = "by_n_stocks",
    n: int = 5,
    group_col: str = "index_code",
) -> list[tuple[str, float, float, str]]:
    """每个 asof_date 选 top-N 行业,转动画 overlay 格式。

    Args:
        per_date_dfs: aggregate_by_industry_per_date 输出
        criterion: 排序标准(3 种,与 v5.2 select_top_n_industries 一致):
            - "by_n_stocks": 按股票数(最多成分股的行业)
            - "by_c_over_k": 按 c/k 比(最过阻尼)
            - "by_k_over_c": 按 k/c 比(最欠阻尼 / 最危险)
        n: top N(每个 date 最多选 n 个行业)
        group_col: label 用 group_col 值

    Returns:
        [(asof_date, k̂, ĉ, "Industry {group_col}"), ...]
        按 asof_date 排序(动画顺序)— 保证动画帧按时序演
    """
```

```python
def build_animated_overlay_html(
    pairs_per_date: list[tuple[str, float, float, str]],
    omega_grid: np.ndarray,
    output_path: str,
    title: str = "Industry G(ω) Frequency Response — Time Series",
) -> None:
    """构建 plotly 动画 slider:每帧一个 asof_date,每帧 N 条 industry Bode 曲线。

    行为:
        - 上子图 |H(jω)| dB vs ω(共享 omega_grid)
        - 下子图 arg H(jω) degrees vs ω
        - animation_frame = asof_date(可拖 slider)
        - Play/Pause 按钮(默认 500ms / frame)
        - 每帧 N 条曲线(N = top-n industries per date)
        - legend 显示 label + (k, c) at 当前帧
        - HTML 通过 plotly CDN 渲染(include_plotlyjs='cdn')

    Raises:
        ValueError: 空 pairs 列表
    """
```

### 3.3 CLI 扩展

```bash
# v5.2 数据驱动(单帧,不变)
python backtrace/dynamics/dynamics_forced_response.py --from-kc-estimates PATH

# v5.3 时序动画(新增)
python backtrace/dynamics/dynamics_si_freq_response.py
# 默认:kc_estimates_time.csv 路径 + top-5 industries + by_n_stocks + max 12 dates

python backtrace/dynamics/dynamics_si_freq_response.py \
    --kc-time-csv data/projection/kc_estimates_time.csv \
    --top-n-industries 5 \
    --industry-selection by_n_stocks \
    --max-dates 12
```

### 3.4 输出(全 gitignored)

| 路径 | 触发 | 内容 |
|---|---|---|
| `backtrace/outputs/dynsys_si_freq_response.html` | 默认 | plotly 动画 slider(slider 拖动看漂移) |
| `backtrace/outputs/dynsys_si_freq_response_summary.txt` | 默认 | UTF-8 中文汇总(每 date 一段,top-N industries + 业务解读) |
| `data/dynamics/si_freq_response_pairs.csv` | 默认 | 选中 (asof_date, industry) × (k̂, ĉ) pairs(审计用) |

### 3.5 动画帧细节(plotly 实现)

```python
# 初始 traces(第一帧)
fig = go.Figure(
    data=[go.Scatter(x=omega_grid, y=mag_db_first_frame, name=label_first_frame),
          ...],
    layout=go.Layout(
        title="Industry G(ω) — Time Series",
        xaxis_title="ω (rad/day)",
        yaxis_title="|H(jω)| dB",
        updatemenus=[dict(type="buttons", showactive=False,
                          y=1.15, x=0.5, xanchor="center",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 500}}])])]
    ),
    frames=[
        go.Frame(data=[go.Scatter(x=omega_grid, y=mag_db_date_i, name=label_date_i),
                       ...],
                 name=date_iso_i)
        for date_iso_i in dates
    ]
)
fig.update_layout(sliders=[dict(steps=[dict(method="animate",
                                            args=[[date_iso], {"mode": "immediate"}],
                                            label=date_iso)
                                    for date_iso in dates])])
```

## 4. 测试

### 4.1 单元测试(`tests/test_dynamics_eigen.py` 新增 5 个)

```python
def test_load_kc_time_series_filters_failed(tmp_path):
    """load_kc_time_series 过滤 status != 'ok' 行 + ramp-up filter。"""

def test_load_kc_time_series_validates_columns(tmp_path):
    """缺必需列 → ValueError。"""

def test_aggregate_by_industry_per_date():
    """按 (asof_date, index_code) 聚合,每个 date 一片 DataFrame。"""

def test_select_top_n_per_date():
    """每个 date 选 top-N,返 (asof_date, k, c, label) 元组列表。"""

def test_cli_si_freq_response_mode(tmp_path):
    """CLI 时序动画模式:合成 3 个 asof_date × 2 industries × 2 stocks = 12 行 CSV → 跑 → 验证 HTML + summary + pairs 3 个输出。"""
```

### 4.2 测试 fixture

合成 3 个 asof_date × 2 industries × 2 stocks = 12 行 CSV:
- Date 1: Industry A (k=0.5, c=2.0 过阻尼稳定), Industry B (k=3.5, c=0.5 共振)
- Date 2: Industry A (k=0.6, c=1.9 略过阻尼), Industry B (k=4.0, c=0.4 强共振)
- Date 3: Industry A (k=0.7, c=1.8 稳定), Industry B (k=3.0, c=0.6 中等共振)

总 12 行,所有 `status='ok'`,所有 `n_valid_days >= 192`。top-2 industries per date = [A, B] 全部。

### 4.3 回归保护

- v5 + v5.1 + v5.2 已有 67 个测试,**全部不动**
- **67 → 72 tests pass**(67 旧 + 5 新)

## 5. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ v5 + v5.1 + v5.2 已有函数(`transfer_function` / `natural_frequency` / `magnitude_phase` / `classify_response_type` / `is_in_schur_wedge` / `bode_plot` / `stability_heatmap` / `write_summary` / `bode_overlay` / `write_overlay_summary` / `parse_overlay_pairs` / `load_kc_estimates` / `aggregate_by_industry` / `select_top_n_industries` / `write_industry_pairs_csv`)签名 0 修改
- ❌ 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 修改
- ❌ 4 v4.x CLI (`dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py`) 0 修改
- ❌ `parameter_fit.py` 0 修改(只读 CSV,不调函数)
- ❌ v5 单对模式 main() 函数体 0 修改
- ❌ v5.1 `--overlay` 模式 main() 分支 0 修改
- ❌ v5.2 `--from-kc-estimates` 模式 main() 分支 0 修改
- ✓ v5.3 是**新文件** `dynamics_si_freq_response.py`,不动现有文件
- ✓ 所有新增输出 gitignored

## 6. 关键文件

- **新建**:`backtrace/dynamics/dynamics_si_freq_response.py` — 4 新函数 + CLI + main()
- 修改:`tests/test_dynamics_eigen.py` — 加 5 个 test
- 修改:`backtrace/dynamics/README.md` — §4.1 加 §4.1.2 v5.3 子节

## 7. 与 v5 / v5.1 / v5.2 / v4.x 的关系

| 版 | commit | 主题 |
|---|---|---|
| v5 | `0ce3014` | 受迫系统 + G(ω) 单对频率响应 |
| v5.1 | `e990fb3` | 多对 (k, c) overlay 对比(纯可视化扩展) |
| v5.2 | `fce9532` | 数据驱动 overlay(单帧,kc_estimates 行业聚合 + top-N) |
| v4.9 | `f2178a3` | SI(t) 时序 + 漂移检测(parameter_fit --rolling-time) |
| **v5.3** | **(本次)** | **时序动画 G(ω)(t) overlay** — 接 v4.9 rolling 时序 + v5.2 行业聚合 + v5.1 overlay,加 plotly 动画 slider |

v5.3 是**时序维度**的扩展:
- v5: 单对 (k, c),单 ω_grid,单 Bode
- v5.1: 多对 (k, c) overlay,单 ω_grid,单 Bode 多曲线
- v5.2: 多行业 overlay(单 asof_date 快照)
- **v5.3**: 多行业 overlay(**多 asof_date 动画**)

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| `kc_estimates_time.csv` 不存在(v4.9 没跑过) | main() 检测文件存在,不存在给清晰 error 提示用户跑 `parameter_fit --rolling-time` |
| asof_date 数 > `--max-dates` 12(性能) | 自动按 `--max-dates` 截断,默认 12 帧(动画流畅) |
| 同行业在不同 asof_date 的 (k̂, ĉ) 数值差异大 | 每帧单独画 trace,允许颜色重叠 + legend 重复 |
| 同行业在某 asof_date 整段 fail (无 status='ok') | 该 date 该行业跳过,top-N 不足时按实际数 |
| 动画 HTML 大 | plotly CDN 渲染,数据 ≤ 12 帧 × 5 industries × 200 ω points = 12k 点,~200KB |

## 9. 演示 / 复现

```bash
# 前置:v4.9 parameter_fit --rolling-time 已跑过,data/projection/kc_estimates_time.csv 存在
git log --oneline fce9532..HEAD  # 6 commits
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v   # 72 passed

# 端到端
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_si_freq_response.py
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_si_freq_response.html (plotly 动画 slider)
#   backtrace/outputs/dynsys_si_freq_response_summary.txt
#   data/dynamics/si_freq_response_pairs.csv

# v5.2 单帧模式 (向后兼容,0 修改)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py \
    --from-kc-estimates data/projection/kc_estimates.csv
```

## 10. 验证清单

- [ ] `_dynamics_core.py` 0 修改
- [ ] v5 + v5.1 + v5.2 已有函数签名 0 修改
- [ ] 3 caller + 4 v4.x CLI 0 修改
- [ ] `parameter_fit.py` 0 修改
- [ ] v5 单对模式 main() 函数体 0 修改
- [ ] v5.1 overlay 分支 0 修改
- [ ] v5.2 --from-kc-estimates 分支 0 修改
- [ ] 新建 `dynamics_si_freq_response.py` 是独立文件
- [ ] 5 新测试 + 67 旧测试 = 72 tests pass
- [ ] README §4.1 加 §4.1.2 v5.3 子节
- [ ] 3 个 gitignored 输出 (HTML + TXT + CSV)

## 11. 与 parameter_fit 的接口契约(只读)

```python
# v5.3 期望 kc_estimates_time.csv 的列(parameter_fit --rolling-time 输出):
# code: str — 股票代码
# index_code: str — 申万二级代码
# asof_date: str — YYYY-MM-DD 月末日期
# k_hat: float — 该窗口 OLS 拟合恢复系数
# c_hat: float — 该窗口 OLS 拟合阻尼系数
# n_valid_days: int — 该窗口有效天数(>= 192)
# status: str — "ok" / "fail" (过滤 fail 行)
#
# 其他列可选。**不调任何 parameter_fit 函数** — CSV 是 stable 接口
```

**理由**:即使 `parameter_fit.py` 内部重构 / 改函数签名,v5.3 仍能工作(只要 CSV schema 稳定)。