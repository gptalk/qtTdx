# Spec v5.2 — parameter_fit Integration with v5.1 Overlay

> **Date:** 2026-08-18
> **Base:** v5.1 Industry G(ω) Comparison(`e990fb3`)
> **Branch:** new(从 `main` HEAD = `e990fb3`)

## 1. 问题

v5.1 的多对 overlay 是**框架** — 用户手动传 `(k, c, label)` 列表。但业务场景的核心问题是"真实数据下哪些行业的频率响应最值得对比"。

**`parameter_fit.py`** 已经跑出每只票的 OLS 拟合 `(k̂, ĉ)`(`kc_estimates.csv`)。**v5.2 把 v5.1 的可视化框架接上真实数据**:读 parameter_fit 输出 → 行业级聚合 → 选 top-N → 喂给 v5.1 的 `bode_overlay` + `write_overlay_summary`。

## 2. 目标

**核心**:新 CLI flag `--from-kc-estimates PATH` 触发 v5.2 模式,读 `kc_estimates.csv`,按 `index_code` 分组,聚合 `(k̂, ĉ)`(中位数 / 均值),按业务标准排序,选 top-N 行业,自动喂给 v5.1 overlay 函数。

**非目标(YAGNI)**:
- ❌ 不读 `kc_rolling_*.csv` 滚动拟合输出 — v5.2 只用 full-sample `kc_estimates`
- ❌ 不做行业名称解析(申万二级中文名)— `index_code` 直接作 label
- ❌ 不做自动 top-N 推荐 — 用户传 `--top-n N`
- ❌ 不重写 `bode_overlay` / `write_overlay_summary` / `parse_overlay_pairs` — **v5.1 0 修改**
- ❌ 不耦合到 `parameter_fit` 的内部函数 — 只读 CSV 文件(单一接口)

**理由**:
- 与 v5.1 解耦 — v5.1 函数接收 `[(k, c, label), ...]`,v5.2 负责生成这个列表
- 与 v4.x 解耦 — 不碰 SI / IC 评估
- 单一职责 — v5.2 只做"数据 → overlay 输入列表"转换

## 3. 设计

### 3.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│  dynamics_forced_response.py (扩展 main + 加 3 helper)       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [v5.1 已有] bode_overlay / write_overlay_summary   │    │
│  │  parse_overlay_pairs                                │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [v5.2 新增] load_kc_estimates(path) → DataFrame    │    │
│  │  aggregate_by_industry(df, agg="median") → DataFrame│    │
│  │  select_top_n_industries(df, criterion, n) → pairs  │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  [v5.2 main 扩展] --from-kc-estimates CLI flag       │    │
│  │  if args.from_kc_estimates:                          │    │
│  │      pairs = load + aggregate + select_top_n     │    │
│  │      bode_overlay + write_overlay_summary         │    │
│  │      return                                          │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘

输入:data/projection/kc_estimates.csv(parameter_fit 输出)
输出:同 v5.1 overlay HTML + TXT + 1 个 gitignored 选中的 industries CSV
```

### 3.2 v5.2 新 API

```python
def load_kc_estimates(csv_path: str) -> pd.DataFrame:
    """读 parameter_fit kc_estimates.csv,验证必需列。

    必需列:code, index_code, k_hat, c_hat, status(其他列可选)

    Returns:
        DataFrame,只保留 status='ok' 的行(过滤拟合失败的)

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 必需列缺失
    """
```

```python
def aggregate_by_industry(
    df: pd.DataFrame,
    group_col: str = "index_code",
    agg: str = "median",
) -> pd.DataFrame:
    """按行业聚合 (k̂, ĉ)。

    Args:
        df: load_kc_estimates 输出
        group_col: 分组列(默认 index_code,可选 stock_tag)
        agg: 聚合方法("median" / "mean"),默认 median(抗极端值)

    Returns:
        DataFrame 列:[group_col, n_stocks, k_hat, c_hat]
        按 group_col 排序
    """
```

```python
def select_top_n_industries(
    df: pd.DataFrame,
    criterion: str = "by_n_stocks",
    n: int = 5,
    group_col: str = "index_code",
) -> list[tuple[float, float, str]]:
    """从聚合 DataFrame 选 top-N 行业,转 v5.1 overlay 格式。

    Args:
        df: aggregate_by_industry 输出
        criterion: 排序标准
            - "by_n_stocks": 按股票数(最多成分股的行业)
            - "by_c_over_k": 按 c/k 比(最过阻尼)
            - "by_k_over_c": 按 k/c 比(最欠阻尼 / 最危险)
        n: top N
        group_col: label 用 group_col 值

    Returns:
        [(k̂, ĉ, "Industry {group_col}"), ...] — 直接喂给 bode_overlay
    """
```

### 3.3 CLI 扩展

```bash
# v5.1 手动模式(不变)
python backtrace/dynamics/dynamics_forced_response.py --overlay "k1,c1,label1; ..."

# v5.2 数据驱动模式(新增)
python backtrace/dynamics/dynamics_forced_response.py \
    --from-kc-estimates data/projection/kc_estimates.csv \
    --top-n 5 \
    --industry-agg median \
    --select-criterion by_n_stocks

# 互斥:--from-kc-estimates 与 --overlay 不能同时用
# 优先级:--from-kc-estimates 优先(显式数据源)
```

### 3.4 输出(全 gitignored)

| 路径 | 触发 | 内容 |
|---|---|---|
| `backtrace/outputs/dynsys_bode_overlay.html` | `--from-kc-estimates` | 行业级 G(ω) overlay(同 v5.1) |
| `backtrace/outputs/dynsys_bode_overlay_summary.txt` | `--from-kc-estimates` | 行业级业务解读(同 v5.1) |
| `backtrace/outputs/dynsys_industry_overlay_pairs.csv` | `--from-kc-estimates` | 选中的 top-N 行业 (k̂, ĉ) + label + 行业股票数 |

新增第 3 个文件 — 记录 v5.2 实际喂给 v5.1 的输入列表,**审计用**(用户能看出选了哪些行业)。

## 4. 测试

### 4.1 单元测试(`tests/test_dynamics_eigen.py` 新增 5 个)

```python
def test_load_kc_estimates_filters_failed(tmp_path):
    """load_kc_estimates 过滤 status != 'ok' 的行。"""

def test_load_kc_estimates_validates_columns(tmp_path):
    """缺必需列 → ValueError。"""

def test_aggregate_by_industry_median():
    """agg='median' 对 (k̂, ĉ) 中位数聚合 + n_stocks 计数。"""

def test_select_top_n_by_n_stocks():
    """criterion='by_n_stocks' 按股票数降序排,选 top N。"""

def test_select_top_n_by_c_over_k():
    """criterion='by_c_over_k' 按 c/k 比降序排,选 top N。"""
```

### 4.2 测试 fixture

合成 3 个行业 × 4 股票 = 12 行 CSV:
- Industry A: 4 stocks, k=0.5, c=2.0 (Schur 内,过阻尼)
- Industry B: 3 stocks, k=2.0, c=1.5 (Schur 外,中等共振)
- Industry C: 2 stocks, k=3.5, c=0.5 (Schur 外,强共振)
- 1 个 status='fail' 的行(被过滤)

### 4.3 回归保护

- v5.1 已有 8 个测试,**全部不动**
- v5 53 个测试,**全部不动**
- **61 → 67 tests pass**(61 旧 + 6 新:5 单元测试 + 1 CLI 集成测试)

## 5. 约束兑现

- ❌ `_dynamics_core.py` 0 行修改
- ❌ v5 + v5.1 已有函数(`bode_overlay` / `write_overlay_summary` / `parse_overlay_pairs` / `transfer_function` / 等)0 行修改
- ❌ 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 修改
- ❌ 4 v4.x CLI 0 修改
- ❌ `parameter_fit.py` 0 修改(只读 CSV,不调函数)
- ❌ v5 单对模式 main() 函数体 0 修改
- ❌ v5.1 overlay 模式 main() 分支 0 修改
- ✓ 所有新增输出 gitignored

## 6. 关键文件

- 修改:[`backtrace/dynamics/dynamics_forced_response.py`](backtrace/dynamics/dynamics_forced_response.py) — 加 3 函数 + `--from-kc-estimates` / `--top-n` / `--industry-agg` / `--select-criterion` CLI flags + main() 中间分支(在 v5.1 if-return 之后)
- 修改:[`backtrace/dynamics/README.md`](backtrace/dynamics/README.md) — §4.1 加 §4.1.1 v5.2 子节
- 修改:[`tests/test_dynamics_eigen.py`](tests/test_dynamics_eigen.py) — 加 5 个 test

## 7. 与 v5.1 / v5 的关系

| 版 | commit | 主题 |
|---|---|---|
| v5 | `0ce3014` | 受迫系统 + G(ω) 单对频率响应 |
| v5.1 | `e990fb3` | 多对 (k, c) overlay 对比 |
| **v5.2** | **(本次)** | **数据驱动 overlay** — 接 parameter_fit 输出,自动选 top-N 行业 |

v5.2 是 v5.1 的**数据接入层**。v5.1 提供"对比框架",v5.2 提供"真实数据 → 框架输入"转换。

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| `kc_estimates.csv` 不存在 | main() 检测文件存在,不存在给清晰 error 提示用户跑 `parameter_fit.py` |
| `kc_estimates.csv` 缺必需列 | load_kc_estimates 验证列,缺则 ValueError(列出缺失列名) |
| `kc_estimates.csv` 全行 status='fail' | aggregate 后 n_stocks 全为 0,main() 报错退出 |
| `--from-kc-estimates` 与 `--overlay` 同时传 | argparse 互斥检查;同时传 → 报错退出 |
| 行业数 < --top-n | select_top_n 返回所有行业,main() print "只找到 X < N 个行业" 警告 |

## 9. 演示 / 复现

```bash
git log --oneline e990fb3..HEAD  # 6 commits
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v   # 66 passed

# 端到端(前提:parameter_fit.py 已跑过,data/projection/kc_estimates.csv 存在)
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py \
    --from-kc-estimates data/projection/kc_estimates.csv \
    --top-n 5 \
    --select-criterion by_n_stocks
# 期待:3 个 gitignored 输出
#   backtrace/outputs/dynsys_bode_overlay.html
#   backtrace/outputs/dynsys_bode_overlay_summary.txt
#   backtrace/outputs/dynsys_industry_overlay_pairs.csv

# v5.1 手动模式(不变)
python backtrace/dynamics/dynamics_forced_response.py --overlay "0.5,2.0,A; 2.0,1.5,B"
```

## 10. 验证清单

- [ ] `_dynamics_core.py` 0 修改
- [ ] v5 + v5.1 已有函数签名 0 修改
- [ ] 3 caller + 4 v4.x CLI 0 修改
- [ ] `parameter_fit.py` 0 修改
- [ ] v5 单对模式 main() 函数体 0 修改
- [ ] v5.1 overlay 模式 main() 分支 0 修改
- [ ] v5.1 --overlay 字符串模式仍可用(向后兼容)
- [ ] 新增 `dynsys_industry_overlay_pairs.csv` gitignored
- [ ] 5 新测试 + 61 旧测试 = 66 tests pass
- [ ] README §4.1 加 §4.1.1 v5.2 子节

## 11. 与 parameter_fit 的接口契约

**只读契约** — 不调任何 `parameter_fit` 函数:

```python
# v5.2 期望的 CSV 列:
# code: str — 股票代码
# index_code: str — 申万二级代码 (e.g., "801010")
# k_hat: float — OLS 拟合恢复系数
# c_hat: float — OLS 拟合阻尼系数
# status: str — "ok" / "fail" / ...
#
# 其他列(name, stock_tag, f_self_loss, n_valid_days 等)— 忽略
```

**理由**:即使 `parameter_fit.py` 内部重构 / 改函数签名,v5.2 仍能工作(只要 CSV schema 稳定)。CSV 是 stable interface。