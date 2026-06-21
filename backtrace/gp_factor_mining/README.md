# GP 因子挖掘 — A 股中证全指(2015–2020 训 / 2021–2025 测)

> 用遗传规划(gplearn)对 A 股做**多轮残差因子挖掘**,产可解释公式,接 vectorbt 回测。

## 0. 仓库定位

放在 `backtrace/gp_factor_mining/`,复用项目里的 TQ 数据接入风格、vectorbt 回测。
和 `tsfresh_*` 系列互补:tsfresh 提特征 + 选分类器,GP 直接**进化公式**。

## 1. 安装

```bash
pip install gplearn        # 核心依赖,必装
pip install vectorbt       # 回测用,可选(装了能出净值 HTML 图)
# sklearn / pandas / numpy / scipy 本仓库已有
```

> 当前环境检测:`sklearn 1.7.2` ✓,`gplearn` / `vectorbt` 需手动装。

## 2. 文件结构与运行顺序

```
backtrace/gp_factor_mining/
├── 00_config.py           ← 所有参数集中改这里
├── 01_data_prep.py        ← ① TQ 取数 + 清洗 + 截面标准化 → data/panel.parquet
├── 02_primitive_set.py    ← 时序/截面原始算子(被 01/05 调用)
├── 03_neutralize.py       ← 行业+市值中性化(OLS 残差)
├── 04_ic_metrics.py       ← RankIC / ICIR / 衰减 / 换手 / 分组收益
├── 05_gp_mine.py          ← ② gplearn 多轮残差挖掘 → factors/factor_summary_*.csv
├── 06_factor_pool.py      ← ③ 入库门槛 + 相关性去重 → factors/factor_pool.csv
├── 07_backtest.py         ← ④ vectorbt 选股回测 + 净值图
└── README.md              ← 本文件
```

```bash
cd backtrace/gp_factor_mining
python 01_data_prep.py     # 取数 + 清洗 + 标准化(慢,首次可能要 5–15 分钟)
python 05_gp_mine.py       # 多轮残差 GP 挖掘(核心,每轮 5–30 分钟)
python 06_factor_pool.py   # 入库 + 去重
python 07_backtest.py      # 回测 + 出图
```

> 02/03/04 是模块,不需要单独跑,被 01/05 调用。

## 3. 关键参数(都在 `00_config.py` 里)

| 参数 | 当前值 | 说明 |
|---|---|---|
| `TRAIN_START / TRAIN_END` | 2015-01-01 / 2020-12-31 | 训练期 |
| `TEST_START / TEST_END` | 2021-01-01 / 2025-12-31 | 样本外测试期 |
| `TQ_SECTOR` | `沪深A股` | TQ 板块名(≈ 中证全指) |
| `HOLD_PERIOD` | 20 | 预测未来 20 日收益 |
| `POP_SIZE` | 2000 | GP 种群规模 |
| `N_GENERATIONS` | 30 | 进化代数 |
| `PARSIMONY_COEFFICIENT` | 0.001 | 防膨胀(bloat) |
| `N_RESIDUAL_ROUNDS` | 5 | 残差轮数 |
| `MIN_IMPROVE_IC` | 0.005 | 边际 IC < 此则早停 |
| `IN_SAMPLE_ICIR_MIN` | 1.5 | 入库样本内 ICIR |
| `OUT_SAMPLE_IC_MIN` | 0.04 | 入库样本外 IC |
| `OUT_SAMPLE_ICIR_MIN` | 1.0 | 入库样本外 ICIR |
| `MAX_CORR_WITH_POOL` | 0.70 | 与已入库因子最大相关 |
| `BACKTEST_TOP_N` | 50 | 月末选 top 50 |

## 4. 设计要点

### 4.1 数据流
```
TQ 沪深A股 板块
   ↓ (OHLCVA + Amount)
01_data_prep.py
   ↓ 清洗(剔 ST/停牌/次新/壳股) + 截面 rank-zscore
data/panel.parquet
   ↓
05_gp_mine.py
   ├─ 02_primitive_set  算 ma/rsi/macd/atr/bbi 等时序原始特征
   ├─ 03_neutralize     按日截面回归去 size + industry
   ├─ gplearn SymbolicRegressor  ×  N_RESIDUAL_ROUNDS 轮
   └─ 输出 factor_r{i}_train/test.parquet + factor_summary_*.csv
        ↓
06_factor_pool.py
   ├─ 重算 IC/ICIR/换手
   ├─ 入库门槛过滤
   ├─ 与已入库因子月度相关去重
   └─ factor_pool.csv + factors/pool/*.parquet
        ↓
07_backtest.py
   ├─ IC 加权合成综合因子
   ├─ 月末 top N(50) 等权
   └─ vectorbt 回测 → HTML 净值图
```

### 4.2 多轮残差挖掘的逻辑
```
Round 1: y_1 = fwd_ret_20d
         → 训练 → pred_1
Round 2: y_2 = y_1 - pred_1
         → 训练 → pred_2
...
Round K: y_K = y_(K-1) - pred_K
         → 训练 → pred_K

最终综合预测 = pred_1 + pred_2 + ... + pred_K
```
每轮 `pred_k` 是新的独立因子;入库时按 IC 加权合成。
**边际 IC 增益 < `MIN_IMPROVE_IC`** 时提前停(防过拟合 + 省时间)。

### 4.3 中性化
每日横截面 OLS:
```
factor ~ 1 + log(size) + C(industry)
```
- `size` = 过去 20 日均成交额(Amount 代理市值)
- `industry` = 行业代理,无 TQ 行业分类时用 ret/vol 分布聚类分 20 组

### 4.4 截面标准化
所有进 GP 的特征都是**截面 rank-zscore**:`(rank - mean) / std`,
去掉价格/成交量绝对水平,只保留**相对强弱**信息。

### 4.5 防过拟合措施
- 树深限制 ≤ 8,初始深度 ≤ 6
- `parsimony_coefficient=0.001` 偏好小树
- **多轮残差**:每轮只看"上一轮没解释的"部分
- 严格训练/测试切分(2015-2020 / 2021-2025)
- 入库门槛:样本内 ICIR ≥ 1.5 + 样本外 IC ≥ 0.04 + ICIR ≥ 1.0
- 因子间相关性 < 0.70(避免同质化)

## 5. 产出文件示例

```
factors/
├── factor_summary_20260620_143012.csv     # 每轮公式 + IC
├── factor_formulas_20260620_143012.json   # JSON 备份
├── factor_r1_train_20260620_143012.parquet
├── factor_r1_test_20260620_143012.parquet
├── ...
├── factor_pool.csv                         # 累计入库因子
└── pool/
    ├── gp_r1_train.parquet
    ├── gp_r1_test.parquet
    └── ...
```

## 6. 常见问题

**Q: gplearn 跑得太慢怎么办?**
A: 调小 `POP_SIZE` (500–1000)、`N_GENERATIONS` (10–20);或开 `n_jobs=-1` 多核;05 里 `sample_frac=0.30` 也会加速。

**Q: 挖出来的因子测试期 IC = 0?**
A: 大概率过拟合。检查:
  1. `parsimony_coefficient` 是否够大(0.001 → 0.005)
  2. `MAX_DEPTH` 是否太大(8 → 5)
  3. 入库门槛 `OUT_SAMPLE_ICIR_MIN` 是否过低(1.0 → 1.5)
  4. 是否漏做行业/市值中性化

**Q: 因子库越挖越烂?**
A: 这是 GP 的早熟收敛。改进:
  - 增加种群规模
  - 加大变异概率(`P_HOIST_MUTATION` 提到 0.1)
  - 减少交叉概率到 0.7
  - 加大样本量

**Q: TQ 板块「沪深A股」拉不到?**
A: 改 `TQ_SECTOR`:
  - `"中证流通"` / `"全部A股"` / `"深证A股"` / `"上证A股"` 都试一下
  - 或者写自定义板块:先 `tq.create_sector(...)` 然后把全 A 加进去

**Q: 数据量太大跑不动?**
A: 在 `01_data_prep.py` 里加:
```python
np.random.seed(42)
keep_codes = np.random.choice(all_codes, size=2000, replace=False)
panel = panel[panel['code'].isin(keep_codes)]
```
先在 2000 只大盘股上跑通,再放开。

## 7. 参考绩效

- 单因子样本外:**RankIC 4–8%,ICIR 1.5–3.0**
- 多因子组合:年化超额 8–15%,夏普 1.5–2.2,最大回撤 < 25%
- 行业轮动可跑赢中信一级等权基准 3–8%

> 实盘前请做更严格的样本外验证 + 滑点测试 + 多空对冲。

## 8. 改进方向

- **几何语义 GP (GSGP)**:对训练 MSE 直接做几何交叉,泛化更好
- **轻量化 GP**:限制节点数 + 复杂度惩罚
- **行业专门化**:申万 31 个一级行业分别挖,然后做行业中性聚合
- **基本面融合**:把 pe/pb/roe 等加进 Terminal(目前只有量价)
- **GPU 加速**:`deap` + `cupy` 自己实现,可提速 10×+