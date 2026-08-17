# `backtrace/dynamics` — v4.7 行业稳定性指数 SI(Sector Stability Index)

> 2026-08-18 写。v4.6 polish(`4e8c265`)已合,roadmap 未做项剩 4 个,本 spec 选 **行业稳定性指数 SI**(v4.4+ 第一个未做项)。

## 1. 背景与动机

v4.3 (`5b9e788`..`b08f627`) 把 (k̂, ĉ) 跑到了全 A 股 ~5000 只,产出 `eigen_summary.csv` 21 列 + 8 子图 HTML。
v4.4 / v4.5 / v4.6 集中在"个体分类绘制 + polish",但**整体行业层**仍只回答"行业 ρ 中位数 / 楔形距离"两类问题。

**业务问题** — 用户没看到现有产品形态,但读 v4.3 README §6.8.4 / §6.8.5 就能想到:
- 哪些行业**整体最稳定**?哪些最分裂?(银行 vs 半导体 vs 白酒 — 哪个 SI 最高?)
- 行业 SI 与 forward 20d / 60d return 是否有预测力?
- 行业 SI 时序是否稳定(滚动 60 日)?

本 spec **只回答第一个问题**(本期范围),后两个留 v4.7+ / v4.8。

## 2. 范围(Scope)

**In scope**:
1. 在 `backtrace/dynamics/dynamics_eigen_analysis.py` 新增 `compute_sector_stability(df)` 函数 — 输出 `sector_si.csv` (申万二级 × 1 个 SI 列 + 5 个分项列)
2. **单一指标** SI ∈ [0, 1] 本轮 = 加权评分(权重锁定 ρ-med 主导,见 §3)
3. 1 个新子图:行业 SI 直方图 + top 12 强 / top 12 弱行业横向棒
4. 5 个新单元测试(`tests/test_dynamics_eigen.py`)
5. `data/dynamics/sector_si.csv` + 1 个 industry_si.png(可选,持久化为 matplotlib PNG)
6. 更新 `backtrace/dynamics/README.md` §3.7 + spec footnote

**Out of scope(冻结,本轮显式不做)**:
- 行业 SI 时序(滚动 60 日)— v4.8 候选
- 行业 SI 与 forward return 的 IC 评估 — v4.8 候选,需要新增 `dynamics_state_backtest.py` 类的接口
- SI 阈值动态校准(把 SI 0.5 设为"稳/不稳"分界)— v4.8 候选
- 多维指标 dict(`rho_health` / `damping_health` / `wedge_inside`)的切分 — v4.8 候选
- 跨期 (k̂, ĉ) 时序稳定性(已在 v4.3 README §6.8 留过)
- 修改 `analyze_eigenvalues` / `simulate_trajectory` 数学
- 修改 3 个现有 caller:`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`
- 全 A 股 `simulate_trajectory` 批量模拟
- 任何交易信号(明确不做)

## 3. SI 定义(本轮核心)

### 3.1 数学定义

```
SI ∈ [0, 1]

3 个 0-1 子分(线性映射,所有极值有人工锚点):
  ρ_health      = clip(1 - ρ_med / 2,        0, 1)        # ρ=0 → 1, ρ≥2 → 0
  damping_health = clip(1 - |c_med - 1| / 2,  0, 1)        # c=1 → 1, |c-1|≥2 → 0
  wedge_health   = clip(in_wedge_pct,         0, 1)        # 100% 在楔形 → 1

固定权重(本轮锁定):
  ρ_health 权重 0.5
  damping_health 权重 0.2
  wedge_health 权重 0.3

  SI = 0.5 · ρ_health + 0.2 · damping_health + 0.3 · wedge_health
```

**权重说明**:
- ρ-med 主导(0.5):谱半径是"动态系统是否收敛"的最直接信号,经验上 A 股 ρ-med > 1.4 的行业偏多,ρ-med < 1.0 的是少数,差异化能力最强
- damping_health 权重 0.2:经验上 A 股 ĉ-med 普遍 > 1.0(过阻尼),damping_health 值偏中位,提供差异化但权重小
- wedge_health 权重 0.3:Schur 楔形是个"几何过滤"信号,行业若 80% 成员在楔形内,即使 ρ-med 高也未必"稳定";楔形占比是有意义的补充

### 3.2 锚点(为什么 ρ/2 = 0、c=1 锚点)

| 锚点 | 含义 | 估值依据 |
|---|---|---|
| ρ=0 → ρ_health = 1 | 完全无动态 | 理论极限 |
| ρ=2 → ρ_health = 0 | 严重发散 | 经验:A 股 anti_restoring 类 ρ 普遍 2-10 |
| c=1 → damping_health = 1 | 临界阻尼 | 理论最优(过阻尼 vs 欠阻尼的对称点) |
| \|c-1\| ≥ 2 → damping_health = 0 | 严重失配 | 经验:欠阻尼 c≈0.1 或过阻尼 c≈3 都差 |
| in_wedge_pct = 1 → 1 | 全部成员在楔形内 | 经验最强约束 |
| in_wedge_pct = 0 → 0 | 全部成员在楔形外 | 经验最弱约束 |

### 3.3 行业筛选阈值(沿用 v4.3)

`n_stocks >= 50` 优先,降级 `n_stocks >= 30`,都 < 5 行业时标"no data"。**与 `aggregate_by_industry` 共用聚合 helper**,不重复实现。

### 3.4 交易所维度(本轮不做)

SI 仅在行业层计算,交易所维度(SH/SZ/BJ)只用 N=3 数据,意义小,延后到 v4.8 + 多维 SI 一起做。

## 4. 数据流与文件 IO

### 4.1 输入

| 来源 | 用途 |
|---|---|
| `data/dynamics/eigen_summary.csv` | v4.3 产出,含 21 列(基础 14 + 楔形距离 3 + industry_l1/l2/exchange) |
| `data/sw2/members.csv` | code → sector_name mapping(已有,沿用 `_industry_name_lookup`) |

### 4.2 输出(全部 gitignored)

| 路径 | 内容 |
|---|---|
| `data/dynamics/sector_si.csv` | **新增** — 申万二级 × 9 列: `industry_l1, sector_name, n_stocks, rho_health, damping_health, wedge_health, SI, rho_median, c_median` |
| `backtrace/outputs/dynsys_sector_si.html` | **新增** — 1 个 plotly HTML,含 3 子图(详见 §5) |
| `backtrace/outputs/dynsys_sector_si_summary.txt` | **新增** — 纯文本 top 12 强 / 弱 行业,UTF-8 中文,Windows `cat` 可读 |

**注意**:`dynsys_eigen.html` 本轮**不修改**(v4.3 8 子图保留),SI 是独立 HTML,适合"按需跑" / "按需推送"。

### 4.3 脚本结构

仍用 `backtrace/dynamics/dynamics_eigen_analysis.py`,**只在末尾追加**:
- 1 个函数 `compute_sector_stability(df)` 返回 DataFrame
- 1 个函数 `build_sector_si_html(df_si, output_path)`
- 1 个函数 `write_sector_si_summary(df_si, output_path)`
- `main()` 末尾追加 SI 写出调用(默认运行,不增加 flag)
- 5 个新单测

代码增量: **~120 行**(compute_sector_stability ~30 行 / build_sector_si_html ~50 行 / write_sector_si_summary ~30 行 / test ~10 行算总)。

## 5. HTML 布局

```
┌───────────────────────────────────────────────────────┐
│  1. (1, 1) SI 分布直方图 + 0.5 阈值红虚线            │
│  2. (1, 2) SI vs ρ_med 散点(气泡 = n_stocks)         │
│  3. (2, 1) Top 12 强 SI 行业(横向棒,颜色 = SI)      │
│  4. (2, 2) Top 12 弱 SI 行业(横向棒,颜色 = ρ_health)│
└───────────────────────────────────────────────────────┘
```

文字汇总示例:
```
行业稳定性指数 SI(N=128, 阈值 >= 50 只,降级 >= 30)
================================================================
Top 12 强 SI:
  银行             SI=0.87  ρ_med=0.85  c_med=1.05  wedge=0.92
  公用事业         SI=0.84  ρ_med=0.91  c_med=1.12  wedge=0.88
  高速公路         SI=0.82  ρ_med=0.88  c_med=1.20  wedge=0.85
  ...

Top 12 弱 SI:
  半导体           SI=0.18  ρ_med=2.85  c_med=1.45  wedge=0.21
  元件             SI=0.21  ρ_med=2.42  c_med=1.30  wedge=0.25
  医疗器械          SI=0.24  ρ_med=2.15  c_med=1.18  wedge=0.30
  ...

SI 直方图:
  < 0.2: 18 行业
  0.2-0.4: 35 行业
  0.4-0.6: 42 行业
  0.6-0.8: 25 行业
  > 0.8: 8 行业
```

## 6. 测试设计

5 个新单测,放在 `tests/test_dynamics_eigen.py` 末尾:

| # | 名称 | 断言 |
|---|---|---|
| 1 | `test_sector_si_basic_shape` | 1 行业 100 只全稳定(ρ=0.5, c=1.0, in_wedge=True) → SI ≈ 0.5*0.75+0.2*1+0.3*1 = 0.875 |
| 2 | `test_sector_si_anti_restoring` | 1 行业 100 只全 anti_restoring(ρ=3.0, c=1.5, in_wedge=False) → SI ≈ 0.5*0 + 0.2*0.75 + 0.3*0 = 0.15 |
| 3 | `test_sector_si_clamps_extreme` | ρ=10.0 → ρ_health clipped to 0;c=10.0 → damping_health clipped to 0 |
| 4 | `test_sector_si_perfect` | ρ=0.0, c=1.0, in_wedge_pct=1.0 → SI = 1.0(边界) |
| 5 | `test_sector_si_summary_text` | `write_sector_si_summary` 包含 "Top 12 强" + 至少 1 个中文行业名 |

**总测试数**: 30 (v4.6) + 5 (v4.7) = **35 tests pass**

## 7. 与现有代码的关系

| 现有 | 关系 |
|---|---|
| `compute_dynamics` / `classify_states` (projection) | **不动** — 本轮不重写描述层 |
| `analyze_eigenvalues` (dynamics) | **不动** — 本轮不重写 11 类分类 |
| `simulate_trajectory` (dynamics) | **不动** — 本轮不重写模拟 |
| `aggregate_by_industry` (v4.6 T3.5) | **复用** — 在 SI 计算里降级显示 `n_stocks < 50` 的行业(仅展示,不进 top 12) |
| 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) | **0 行修改** — SI 是 `dynamics_eigen_analysis.py` 末尾追加,与其他 CLI 解耦 |
| `kc_estimates.csv` / `eigen_summary.csv` | **消费,不修改** — `compute_sector_stability` 只读 |

## 8. 已知风险

| 风险 | 缓解 |
|---|---|
| 权重 0.5/0.2/0.3 是经验性,未来 IC 评估发现错位 | 权重写在 `SI_WEIGHTS = (0.5, 0.2, 0.3)` 常量,集中 1 处,改 1 行即可 |
| ρ-med 主导若 A 股 ρ 普遍 > 1,SI 全行业偏低 | 阈值 0.5 在直方图上画红虚线,显示"行业中位"参考,不强行说"SI > 0.5 才稳" |
| 申万二级行业每年调整一次,样本跨期不一致 | `sector_si.csv` 留 `timestamp`(UTC 跑批时间),下游使用按时间戳筛选 |
| `eigen_summary.csv` 缺失(刚清缓存) | 函数开头 `if not os.path.exists(...): raise FileNotFoundError`,main() 端检查 |
| plotly 6 子图 HTML 旧 v4.5 5.8MB,SI 又加 1 个 HTML | SI HTML 子图仅 4 个,预估 ~300 KB,远小于 v4.3 8 子图;gitignored 可接受 |

## 9. 后续(本轮不做)

- v4.8 候选 1: 行业 SI 时序(滚动 60 日)+ 漂移检测
- v4.8 候选 2: SI 与 forward 20d / 60d return 的 IC 评估(走 `dynamics_state_backtest.py` 现有接口扩展)
- v4.8 候选 3: 多维 SI dict(替代单一指标,UI tooltip 展示 5 个分项)
- v4.8 候选 4: 交易所层 SI(SH/SZ/BJ)
- v4.9+: 受迫系统 + G(ω) 频率响应(独立 v5 工作包)

## 10. 关键文件

- 实现: `backtrace/dynamics/dynamics_eigen_analysis.py` (末尾追加 ~120 行)
- 测试: `tests/test_dynamics_eigen.py` (末尾追加 5 个测试)
- 文档: `backtrace/dynamics/README.md` (新增 §3.7 v4.7)
- spec: `docs/superpowers/specs/2026-08-18-dynamics-v4-7-sector-stability-index.md` (本文件)
- plan: `docs/superpowers/plans/2026-08-18-dynamics-v4-7-sector-stability-index.md` (下一步)

## 11. 验证清单

- [ ] 5 个新测试通过(35/35 total)
- [ ] `dynsys_sector_si.html` 4 子图正常渲染,11 类配色保持一致
- [ ] `sector_si.csv` 9 列正确,128 个行业(实际多数 < 50 只会被标 low-confidence)
- [ ] `dynsys_sector_si_summary.txt` 在 Windows `cat` 下能读,top 12 强 / 弱 行业名称中文正确
- [ ] 端到端:`--limit 50` 冒烟 exit 0,产出 3 个新文件
- [ ] 0 行修改:`backtrace/dynamics/_dynamics_core.py` / `dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`
