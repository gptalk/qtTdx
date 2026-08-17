# `backtrace/dynamics` — Plan v4.3 全市场经验分布

> 2026-08-17 写。
> v4.1 (`a62fa9d`)+ v4.2 (`a62fa9d`) 完成了 ρ-primary 11 类分类 + 楔形距离几何 + Gram-Schmidt 能量自检;
> 但实证只在 4 只 smoke-test 票上跑过。本 spec 范围严格冻结:不动数学层,只把"单股结果"扩到"全 A 股横截面",
> 回答一个经验问题——**这套动力系统参数 (k̂, ĉ) 在全市场到底呈现什么样的分布结构?**

---

## 1. 范围(Scope)

**In scope**:
1. 跑 `parameter_fit.py` 全 A 股 (~5000 只,~20-40 分钟),补齐 `data/projection/kc_estimates.csv`(实际已跑出 5211 行,4972 ok)
2. 改 `dynamics_eigen_analysis.py`:
   - 多读 3 列 `industry_l1` / `industry_l2` / `exchange`(分别从 `data/sw2/members.csv` 和 `data/stock_basic.csv` 反查)
   - HTML 从 2×3 → **2×4**,新增 2 个聚合子图(申万二级行业 top10、交易所 SH / SZ / BJ)
   - 写出**纯文本汇总** `backtrace/outputs/dynsys_eigen_summary.txt`
3. 加 3 个新单元测试(`tests/test_dynamics_eigen.py`)
4. 更新 `backtrace/dynamics/README.md` 和 [`2026-08-16-dynamics-system-design.md`](2026-08-16-dynamics-system-design.md) §3.5

**Out of scope(冻结,显式不做)**:
- G(ω) 频率响应函数(独立 v5 工作包)
- 频率轴定义 / 共振峰提取(v5)
- 行业稳定性指数 SI(v4.4+)
- (k, c) 相图 + 7 状态颜色叠加(v4.4)
- 状态转移矩阵(v4.5)
- IC 评估 / basket 回测(走 `dynamics_state_backtest.py` 已有的路径)
- 任何交易信号(明确不做,描述层不进入预测)
- 跨期 `k̂, ĉ` 时序稳定性分析(v4.4+,要 `kc_rolling_summary.csv` 已有基础)
- 全 A 股 `simulate_trajectory` 批量模拟(v4.x 后续)
- 修改 `analyze_eigenvalues` / `simulate_trajectory` 数学
- 修改 3 个现有 caller:`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`

---

## 2. 数学层完全不变(冻结声明)

`analyze_eigenvalues(k, c)` 的:
- A 矩阵 = `[[1, 1], [-k, 1-c]]`
- 11 类分类法(ρ-primary)
- 楔形距离三件套(`distance_lower_boundary` / `distance_upper_boundary` / `distance_to_wedge`)
- Gram-Schmidt 能量自检(`energy_error`)

**全部冻结,本次 spec 一行不改**。v4.3 的全部价值在"经验分布"层,数学正确性已在 v4.1 + v4.2 的 23 个测试里被钉死。

---

## 3. 数据流与文件 IO

### 3.1 输入

| 来源 | 用途 | 备注 |
|---|---|---|
| `data/projection/kc_estimates.csv` | 单股 (k̂, ĉ) 主输入 | v4.3 之前只 4 只 smoke-test,本次跑全 A 股 (5211 行, 4972 ok) |
| `data/stock_basic.csv` | 反查 `exchange` (SH / SZ / BJ) | 列: `code, market, name, status`;`market` 即交易所 |
| `data/sw2/members.csv` | 反查 `industry_l1` / `industry_l2` (申万二级) | 5215 行;列: `sector_code, sector_name, member_code`。`sector_name` 作 `industry_l2`,`sector_code` 作 `industry_l1` |

**重要修正(2026-08-17)**:
- `stock_basic.csv` 不含行业列 — 行业来自 `data/sw2/members.csv`
- 原 v4.3 spec 误以为 `stock_basic.csv` 含 3 列,实施时已纠正
- 申万二级有 128 个行业,平均每行业 40 只 — `n >= 50` 可能不够 10 个,会触发降级到 `n >= 30`

### 3.2 输出(全部 gitignored)

| 路径 | 新增/修改 | 备注 |
|---|---|---|
| `data/dynamics/eigen_summary.csv` | **修改**:加 3 列 `industry_l1` / `industry_l2` / `exchange` | 现 18 列 → 21 列 |
| `backtrace/outputs/dynsys_eigen.html` | **修改**:2×3 → **2×4** | 新增 (1,4) 行业 top10 + (2,4) 交易所 SH vs SZ |
| `backtrace/outputs/dynsys_eigen_summary.txt` | **新增** | 纯文本汇总,Windows `cat` 直接读 |
| `data/dynamics/v43_eigen_top_industries.csv` | **新增** | 行业聚合表(下游可能用到) |
| `data/dynamics/v43_eigen_by_exchange.csv` | **新增** | 交易所聚合表 |

### 3.3 脚本结构

仍用 `backtrace/dynamics/dynamics_eigen_analysis.py`,签名 / 主要骨架不变,**只加**:
- 3 列读入(`industry_l1` / `industry_l2` / `exchange`,来自 `stock_basic.csv` 反查)
- 2 个新子图绘制函数
- 1 个文本汇总写出函数
- 3 个新测试

代码增量:**~120 行**(估计:3 列 IO ~10 行、2 子图 ~50 行、文本汇总 ~40 行、测试 ~20 行)。

---

## 4. 跨股票聚合统计量定义

### 4.1 关键决策:用 **median** 而非 **mean**

**Why**: ρ 分布偏态严重。发散股票 ρ 可以是 5 / 10 / 20(无限趋势强化 / 振荡发散),稳定股票 ρ 接近 0.5-0.9。
均值会被极端值拉飞,中位数更鲁棒。p25 / p75 作为误差棒让读者看出"组内是普遍稳定还是被少数票拖高"。

### 4.2 行业 ρ 中位数(申万一级, top 10 by 股票数)

```
group = kc_estimates.groupby('industry_l1')
agg_l1 = group.agg(
    n_stocks=('code', 'count'),
    rho_median=('spectral_radius', 'median'),
    rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
    rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
    k_hat_median=('k_hat', 'median'),
    c_hat_median=('c_hat', 'median'),
    schur_stable_pct=('schur_stable', 'mean'),   # 0~1
    in_wedge_pct=('in_wedge', 'mean'),
    dist_wedge_median=('distance_to_wedge', 'median'),
).reset_index()
```

筛选 `n_stocks >= 50` 的行业取 top 10(若 `n >= 50` 的不足 10 个,**硬降级**到 `n >= 30` 取所有;若 `n >= 30` 仍不足 5 个,HTML (1,4) 子图报"无足够行业"并跳到 (1,3) 占位),按 `rho_median` **降序**排。
误差棒用 `[p25, p75]`。

### 4.3 交易所 ρ 中位数(SH vs SZ)

```
group = kc_estimates.groupby('exchange')   # 'SH' / 'SZ'
agg_ex = group.agg(同上 9 个统计量)
```

柱状图:SH 蓝、SZ 橙(中国股市传统配色),各 1 个柱子 + 误差棒 [p25, p75]。

---

## 5. HTML 2×4 布局

| (行,列) | 子图 | 类型 | 数据源 | 新/旧 |
|---|---|---|---|---|
| (1,1) | (k̂, ĉ) 散点 + 楔形(分类着色) | scatter | eigen_summary | 旧 |
| (1,2) | ρ 直方图 + ρ=1 红虚线 | histogram | spectral_radius | 旧 |
| (1,3) | 11 类分类柱状(按数量降序) | bar | classification | 旧 |
| **(1,4)** | **行业 ρ 中位数 top10(误差棒 p25-p75)** | bar + error | agg_l1 | **新** |
| (2,1) | (k̂, ĉ) 散点(楔形距离着色 RdYlGn) | scatter | distance_to_wedge | 旧 |
| (2,2) | 楔形距离直方图 + 0 红虚线 | histogram | distance_to_wedge | 旧 |
| (2,3) | ρ vs 楔形距离(整体稳定性视角) | scatter | 双字段 | 旧 |
| **(2,4)** | **交易所 ρ 中位数对比(SH vs SZ)** | bar + error | agg_ex | **新** |

**性能预算**:
- ~5000 点 scatter × 3 = 15000 marker
- plotly 8 子图渲染 ~8-12 秒
- HTML 大小 ~2-4 MB(gitignored)

**legend / 颜色**:
- (1,4) 行业 top10 统一蓝色 + 黑色误差棒(简洁,数字说话)
- (2,4) SH 蓝 / SZ 橙(中国股市传统配色)

---

## 6. 文本汇总格式(`dynsys_eigen_summary.txt`)

```
=== v4.3 全市场 (k̂, ĉ) 经验分布报告 ===
样本数: N = 5000, 有效样本(status=ok*): N_eff = 4500
数据来源: data/projection/kc_estimates.csv
报告时间: 2026-08-17 14:32:00

--- 全市场 ---
ρ 中位数: 1.234 | p25: 0.87 | p75: 2.31
k̂ 中位数: 0.050 | p25: -0.030 | p75: 0.180
ĉ 中位数: 1.420 | p25: 0.890 | p75: 3.210
Schur 稳定(ρ<1): 1234/4500 (27.4%)
楔形内:         1456/4500 (32.4%)
ρ > 1(发散):    2789/4500 (62.0%)
distance_to_wedge 中位数: -0.150 (>0 在楔形内)

--- 11 类分布 ---
stable_oscillatory:        234 (5.2%)
stable_overdamped:        1000 (22.2%)
stable_critical_damping:     12 (0.3%)
oscillatory_divergent:     678 (15.1%)
monotonic_divergent:      450 (10.0%)
anti_restoring:            890 (19.8%)
critical_periodic:         156 (3.5%)
critical_period2:          234 (5.2%)
critical_real_unit:        123 (2.7%)
marginal_const:            567 (12.6%)
jordan_drift:              156 (3.5%)

--- 行业 ρ 中位数 top10 (n_stocks >= 50) ---
银行(申万):           n= 87, ρ_med=0.72, p25=0.55, p75=0.95, k̂_med=0.12, ĉ_med=1.05, 楔形内%= 65%
煤炭:                n= 32, ρ_med=0.85, ...
...

--- 交易所 ---
SH: n=2100, ρ_med=1.18, p25=0.85, p75=2.10, 楔形内%= 35.2%
SZ: n=2400, ρ_med=1.31, p25=0.88, p75=2.45, 楔形内%= 30.1%
```

格式纯 ASCII + UTF-8 中文章节标题,Windows 终端 `PYTHONIOENCODING=utf-8` 下能直接 `cat`。

---

## 7. 测试覆盖(3 个新测试,加到 `tests/test_dynamics_eigen.py`)

```python
def test_industry_aggregation_rho_median():
    """构造 100 只票 / 3 个行业的 dummy kc_estimates,验证 groupby median 正确。
    已知: industry_A 50 只 ρ∈[0.5,1.0], industry_B 30 只 ρ∈[1.0,2.0],
    industry_C 20 只 ρ∈[2.0,5.0]。
    断言: agg_l1.loc['A', 'rho_median'] ≈ 0.75, etc."""

def test_exchange_split_correctness():
    """构造 SH/SZ 各 50 只,验证 groupby 后 n_stocks=50/50,
    误差棒 p25/p75 计算正确(已知一组具体 ρ 值)。"""

def test_html_2x4_layout_and_text_summary():
    """构造 dummy kc_estimates(50 只,2 行业,SH/SZ 各 25),
    调 dynamics_eigen_analysis.main(临时改 CSV 路径),
    验证:
      - eigen_summary.csv 行数 = 50,新增 3 列 industry_l1/l2/exchange 全有限
      - dynsys_eigen.html 存在且大小 > 100KB
      - dynsys_eigen_summary.txt 存在,含 'ρ 中位数' / '行业 ρ 中位数' / '交易所' 三节标题
    """
```

**回归保护**:
- 不动 `analyze_eigenvalues` → 现 23 个测试照常通过
- 不动 `simulate_trajectory` → 现 3 个 caller 零修改

**总测试数**: 23 (v4.1+v4.2) + 3 (v4.3) = **26 个**

---

## 8. 关键文件

| 文件 | 改动 |
|---|---|
| [`backtrace/dynamics/dynamics_eigen_analysis.py`](../../backtrace/dynamics/dynamics_eigen_analysis.py) | +120 行: 3 列 IO + 2 子图 + 文本汇总 |
| [`backtrace/dynamics/README.md`](../../backtrace/dynamics/README.md) | +v4.3 节: 跑全 A 股命令、HTML 解读、文本汇总格式 |
| [`docs/superpowers/specs/2026-08-16-dynamics-system-design.md`](2026-08-16-dynamics-system-design.md) | §3.5 末尾补 v4.3 子节(2×4 布局 + 文本汇总 + 显式范围冻结) |
| [`tests/test_dynamics_eigen.py`](../../tests/test_dynamics_eigen.py) | +3 测试 |

**显式不改的文件**:
- `backtrace/dynamics/_dynamics_core.py`(数学层)
- `backtrace/dynamics/dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`(零改动)
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名

---

## 9. 验证路径

### 9.1 单元冒烟

```bash
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v
# 期望: 26 passed in ~5s
```

### 9.2 手工验收(全 A 股,需要 20-40 分钟)

```bash
# 1. 全 A 股参数拟合
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py
# 验收 1: kc_estimates.csv 行数 ≈ stocks.csv 行数(剔除 status != 'ok*')

# 2. 跑 v4.3 报告
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py
# 验收 2: eigen_summary.csv 多 3 列(industry_l1 / industry_l2 / exchange)
# 验收 3: dynsys_eigen.html 8 子图(2×4)能加载, ~2-4 MB
# 验收 4: dynsys_eigen_summary.txt 中文不乱码, 数字与 HTML 对得上

# 3. 完整性对照
# 验收 5: 文本汇总的 ρ 中位数 = HTML (1,2) 直方图的中位线
# 验收 6: 行业 top10 数字 = v43_eigen_top_industries.csv 内容
```

### 9.3 物理一致性(不写自动,人工)

- 行业 ρ 中位数应在 [0, ∞),观察是不是有些行业普遍 ρ<1(银行、公用事业风格)
- 交易所 SH vs SZ ρ 中位数差异 < 0.3(若差异大,要查是否数据偏差)
- 11 类占比总和 = 100%(四舍五入误差内)

---

## 10. 与 v4.1 / v4.2 / v4.4+ / v5 的关系

| 版 | 主题 | 数学增量 | 数据规模 |
|---|---|---|---|
| v4.1 | ρ-primary 11 类分类 + 边界 bug 修正 | 改 `analyze_eigenvalues` | 4 只 smoke |
| v4.2 | 楔形距离 + 能量自检 | 改 `simulate_trajectory` + 3 距离字段 | 4 只 smoke |
| **v4.3** | **全市场经验分布 + 聚合对比** | **不动** | **~5000 只全 A 股** |
| v4.4 | (k, c) 相图 + 7 状态颜色 + 行业 SI | 不动 + 新增相图 | 复用 v4.3 数据 |
| v4.5 | 状态转移矩阵 | 不动 | 复用 v4.3 + 跨期数据 |
| v5 | G(ω) 频率响应 | 新增 G(ω) 函数 + 共振峰 | 复用 v4.3 (k̂, ĉ) 作输入 |

**v4.3 是承上启下的一环**:数学定型(v4.1 + v4.2),数据铺底(v4.3),形态分析(v4.4 - v4.5),频域深化(v5)。

---

## 11. 决策记录

| 选择 | 替代 | 理由 |
|---|---|---|
| 用 median 而非 mean 聚合 | mean | ρ 分布偏态严重,mean 被极端值拉飞 |
| 申万一级行业 top10 | 申万二级 / GICS | 申万一级是 A 股最常用,股票数最均衡;二级过细(>100 个) |
| 文本汇总 + HTML 双输出 | 只 HTML | 自动化脚本常要 grep 数字;HTML 难 grep;txt 利于 CI / cron |
| 行业 top10 筛选 n≥50 | 不筛 | 避免小样本行业(可能 5 只)算出来的"中位数"误导 |
| 申万行业映射优先 `stock_basic.csv` | 直接 TQ 实时拉 | 本地 CSV 离线和有网都跑得快;TQ 仅在 CSV 缺失时补一次 |
| HTML 用 (1,4) + (2,4) | 拆两个 HTML | 8 子图 plotly 渲染 < 12s,2 文件找不全貌 |
| v4.3 不引入新数学 | 顺手把 G(ω) 做了 | 范围冻结;G(ω) 数学独立工作包更干净 |

---

## 12. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `parameter_fit.py` 全 A 股跑超时 | 报告延后 | 用 `run_in_background=true` 后台跑;验收脚本不阻塞 |
| `stock_basic.csv` 缺 industry_l1 | (1,4) 子图空 | TQ 拉一次申万二级映射补全;5 分钟;失败则子图报"无数据"不致命 |
| 行业 top10 实际不足 10 | 视觉稀疏 | 降级到 n≥30;README 注明筛选阈值 |
| 文本汇总中文 Windows GBK 乱码 | 用户 `cat` 失败 | 文件用 utf-8;README 提醒 `PYTHONIOENCODING=utf-8` |
| 8 子图 plotly 渲染慢 | 体验 | 已知 8-12s;后续可加采样(本次不做) |
| 全 A 股数据更新后 ρ 中位数变化大 | 与 v4.2 文档 25% 矛盾 | 这是经验事实更新,正是 v4.3 的目标;报告里写明"基于 2026-08-17 数据快照" |
