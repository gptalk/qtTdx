# Plan v4.3 — 全市场经验分布与跨股票聚合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `dynamics_eigen_analysis.py` 从 4 只 smoke-test 升级到 ~5000 只全 A 股,产出 8 子图 HTML (2×4) + 纯文本汇总,回答"动力系统参数 (k̂, ĉ) 在全市场呈什么分布"。

**Architecture:** 单 CLI 脚本 (`dynamics_eigen_analysis.py`) +120 行 — 多读 3 列 (`industry_l1` / `industry_l2` / `exchange`)、加 2 个聚合子图、写 1 个文本汇总;数学层 (`_dynamics_core.py`) 完全冻结;3 个新单测。

**Tech Stack:** Python 3.11+、pandas、plotly (`make_subplots` 2×4)、pytest。

**Spec:** [`2026-08-17-dynamics-v4-3-full-market-distribution.md`](../specs/2026-08-17-dynamics-v4-3-full-market-distribution.md)

---

## Global Constraints

- **数学层完全冻结**:`backtrace/dynamics/_dynamics_core.py` 的 `analyze_eigenvalues` / `simulate_trajectory` 一行不改
- **3 个 caller 零修改**:`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`
- **输出全部 gitignored**:`data/dynamics/eigen_summary.csv`、`backtrace/outputs/dynsys_eigen.html`、`backtrace/outputs/dynsys_eigen_summary.txt` 不进 git
- **平台**:Windows + `PYTHONIOENCODING=utf-8`(中文 print 兜底)
- **Python 命令**:`PYTHONIOENCODING=utf-8 python ...`(所有 bash)
- **聚合用 median**:ρ 分布偏态,mean 被极端值拉飞;中位数 + p25/p75 误差棒
- **行业映射来源**(关键修正,2026-08-17):
  - `industry_l1` / `industry_l2` 从 **`data/sw2/members.csv`** 反查 — 列: `sector_code, sector_name, member_code`;`sector_name` 作 `industry_l2` (字符串),`sector_code` 作 `industry_l1`
  - `exchange` 从 **`data/stock_basic.csv`** 反查 — 列: `code, market, name, status`;`market` 即交易所 (SH / SZ / BJ)
- **HTML 2×4 = 原 6 + 新 2**:保留 (1,1)-(2,3),加 (1,4) 行业 top10 + (2,4) 交易所
- **行业筛选**:硬阈值 `n_stocks >= 50` 取 top 10,不足则降级 `n >= 30`,仍不足 5 则子图占位
- **测试**:`tests/test_dynamics_eigen.py` 加 3 个,总 26 passed

---

## File Structure

| 文件 | 状态 | 职责 |
|---|---|---|
| `backtrace/dynamics/dynamics_eigen_analysis.py` | modify | CLI 入口 — 读 kc_estimates + 反查 stock_basic + 8 子图 + 文本汇总 |
| `tests/test_dynamics_eigen.py` | modify | +3 测试(行业聚合 / 交易所拆分 / HTML+文本输出) |
| `backtrace/dynamics/README.md` | modify | +v4.3 节:跑全 A 股命令、HTML 解读、文本汇总格式 |
| `docs/superpowers/specs/2026-08-16-dynamics-system-design.md` | modify | §3.5 末尾补 v4.3 子节 |
| `data/projection/kc_estimates.csv` | regen | `parameter_fit.py` 全 A 股输出(20-40 分钟) |
| `data/dynamics/eigen_summary.csv` | regen | CLI 输出(现 18 列 → 21 列) |
| `data/dynamics/v43_eigen_top_industries.csv` | regen | 行业聚合表(下游可能用) |
| `data/dynamics/v43_eigen_by_exchange.csv` | regen | 交易所聚合表 |
| `backtrace/outputs/dynsys_eigen.html` | regen | 2×4 plotly HTML(~2-4 MB) |
| `backtrace/outputs/dynsys_eigen_summary.txt` | regen | 纯文本汇总(便于 grep) |

显式不修改:`_dynamics_core.py`、`dynamics_system.py`、`dynamics_batch.py`、`dynamics_1step_oos.py`、`dynamics_state_backtest.py`、`analyze_eigenvalues`、`simulate_trajectory`。

---

## Task 1: 跑 `parameter_fit.py` 全 A 股,补齐 `kc_estimates.csv`

**Files:**
- Read: `data/projection/stocks.csv`(约 5000 行)
- Read: `data/stock_basic.csv`(反查用,已存在)
- Regen: `data/projection/kc_estimates.csv`(从 4 行 → ~5000 行)

**前置检查**:
- 确认 `data/projection/stocks.csv` 行数
- 确认 TQ 客户端已启动(否则 `parameter_fit.py` 在第一行 `tq.initialize` 就抛 RuntimeError)
- 确认 `data/stock_basic.csv` 存在(列: `code, market, name, status`)
- 确认 `data/sw2/members.csv` 存在(列: `sector_code, sector_name, member_code`,≥ 5000 行)
- **真正全 A 股前提**:先跑 `backtrace/projection/projection_batch.py --movement --input data/stock_basic.csv --limit 0 --days 240`(20-40 分钟),生成 ~5000 个 `data/projection/movement_*.csv`,否则 `parameter_fit.py` 只能扫到现有 ~7 个 movement 文件

**Step 1.0: 先跑 `projection_batch.py --movement` 生成全 A 股 movement 文件(关键前置)**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --movement \
    --input data/stock_basic.csv --limit 0 --days 240 2>&1 | tee /tmp/projection_batch_full.log
```

**为什么必须先做**:`parameter_fit.py` 扫描 `data/projection/movement_*.csv` 文件做 OLS;若这些文件不全,只能算到现有的 ~7 只,无法得到 ~5000 只的全市场 (k̂, ĉ)。20-40 分钟,TQ 数据拉取 + 投影计算。

**验收**:`ls data/projection/movement_*.csv | wc -l` 应 ≥ 5000(成功率高的话 ~5200)。

- [ ] **Step 1.1: 冒烟跑 5 只确认 pipeline 通畅**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 5
```

期望: 5-10 秒跑完,`data/projection/kc_estimates.csv` 至少 5 行,每行 status 以 "ok" 开头。

- [ ] **Step 1.2: 备份当前 kc_estimates.csv**

```bash
cp data/projection/kc_estimates.csv data/projection/kc_estimates.csv.smoke-bak
```

(回滚保险,跑全 A 股前保留 smoke-test 4 只版)

- [ ] **Step 1.3: 后台跑全 A 股**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 0 2>&1 | tee /tmp/parameter_fit_full.log
```

`--limit 0` = 全部。后台跑 20-40 分钟。

- [ ] **Step 1.4: 验收行数**

```bash
wc -l data/projection/kc_estimates.csv data/projection/stocks.csv
```

期望: `kc_estimates.csv` 行数 = `stocks.csv` 行数 + 1(表头);status 以 "ok" 开头的有效样本应 ≥ 70%。

- [ ] **Step 1.5: 抽样看 5 行**

```bash
head -6 data/projection/kc_estimates.csv
```

期望: 每行有 `code, name, index_code, index_tag, stock_tag, k_hat, c_hat, f_self_loss, n_valid_days, status`,k_hat/c_hat 是有限实数。

- [ ] **Step 1.6: 提交(数据 CSV gitignored,只提交日志)**

数据 CSV 不进 git(`data/` gitignored)。无需 git 提交 — 但**记录**该次全 A 股跑通的 commit 时间(在 v4.3 commit message 里引用)。

---

## Task 2: 在 `dynamics_eigen_analysis.py` 加 `industry_l1` / `industry_l2` / `exchange` 读入

**关键修正(2026-08-17)**:
- `data/stock_basic.csv` 列: `code, market, name, status` — **只**提供 `exchange`(=`market`)
- `data/sw2/members.csv` 列: `sector_code, sector_name, member_code` — 提供 `industry_l1`(=`sector_code`)、`industry_l2`(=`sector_name`)
- 不能合并到一个 `load_stock_basic` 函数,要分两个 load 函数

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py:81-88` (`load_kc_estimates` 函数 + 新增 `load_exchange_lookup` + `load_industry_lookup`)
- Test: `tests/test_dynamics_eigen.py`(Task 5 加测试,本 task 不写)

**Interfaces:**
- Consumes: `data/projection/kc_estimates.csv`(Task 1 输出)+ `data/stock_basic.csv` + `data/sw2/members.csv`
- Produces: `load_kc_estimates()` 返回 DataFrame 含 3 新列 `industry_l1` / `industry_l2` / `exchange`

- [ ] **Step 2.1: 读现状,定位修改点**

读 `backtrace/dynamics/dynamics_eigen_analysis.py:81-88`(现 `load_kc_estimates` 函数),确认要插入的位置。

- [ ] **Step 2.2: 加 `load_exchange_lookup` 函数(从 stock_basic.csv)**

在 `load_kc_estimates` 之后插入:

```python
def load_exchange_lookup(path: str = 'data/stock_basic.csv') -> pd.DataFrame:
    """读 stock_basic.csv,返回 code → exchange 反查表。

    stock_basic 列: code, market, name, status。`market` 即交易所(SH/SZ/BJ)。
    缺文件 / 缺列 → 返回空表,eigen_analysis 不致命。
    """
    if not os.path.exists(path):
        print(f'[eigen] ⚠ stock_basic 不存在: {path},exchange 列将留空')
        return pd.DataFrame(columns=['code', 'exchange'])
    df = pd.read_csv(path, dtype={'code': str})
    if 'market' not in df.columns:
        print(f'[eigen] ⚠ stock_basic 缺 market 列: {path},exchange 列将留空')
        return pd.DataFrame(columns=['code', 'exchange'])
    df['exchange'] = df['market'].fillna('').astype(str).str.strip()
    df.loc[df['exchange'].isin(['-', 'nan', 'None']), 'exchange'] = ''
    return df[['code', 'exchange']]
```

- [ ] **Step 2.3: 加 `load_industry_lookup` 函数(从 sw2/members.csv)**

```python
def load_industry_lookup(path: str = 'data/sw2/members.csv') -> pd.DataFrame:
    """读 sw2/members.csv,返回 code → {industry_l1, industry_l2} 反查表。

    sw2/members 列: sector_code, sector_name, member_code。
    - `sector_code`(881xxx.SH) → industry_l1(字符串,sector_code 本身即可,或行业名)
    - `sector_name`(中文,例如"银行") → industry_l2(人类可读)
    - `member_code` → code(要 join)

    注意:同 member_code 可能属于多个 industry_l1(同一只票同时是"银行"和"金融")。
    默认取第一条出现的行业(`.drop_duplicates('code', keep='first')`),
    允许出现多次的行业则需 join 多次(本次 v4.3 取首条,简单稳定)。
    """
    if not os.path.exists(path):
        print(f'[eigen] ⚠ sw2/members 不存在: {path},industry 列将留空')
        return pd.DataFrame(columns=['code', 'industry_l1', 'industry_l2'])
    df = pd.read_csv(path, dtype={'member_code': str})
    if 'member_code' not in df.columns or 'sector_name' not in df.columns:
        print(f'[eigen] ⚠ sw2/members 缺关键列: {path}')
        return pd.DataFrame(columns=['code', 'industry_l1', 'industry_l2'])
    df['industry_l1'] = df['sector_code'].fillna('').astype(str).str.strip() if 'sector_code' in df.columns else ''
    df['industry_l2'] = df['sector_name'].fillna('').astype(str).str.strip()
    df['code'] = df['member_code']
    df = df[['code', 'industry_l1', 'industry_l2']].copy()
    # 同 code 多行业 → 取首条
    df = df.drop_duplicates('code', keep='first')
    # 规范化
    for col in ['industry_l1', 'industry_l2']:
        df.loc[df[col].isin(['-', 'nan', 'None']), col] = ''
    return df
```

- [ ] **Step 2.4: 改 `load_kc_estimates` 合并 2 个 lookup**

```python
def load_kc_estimates(
    path: str, status_filter: str = 'ok', limit: int = 0,
    stock_basic_path: str = 'data/stock_basic.csv',
    sw2_members_path: str = 'data/sw2/members.csv',
) -> pd.DataFrame:
    """读 kc_estimates.csv,反查 stock_basic(exchange)+ sw2/members(industry_l1/l2)。"""
    df = pd.read_csv(path, dtype={'code': str})
    if status_filter:
        df = df[df['status'].astype(str).str.startswith(status_filter)].copy()
    if limit and len(df) > limit:
        df = df.head(limit).copy()
    # 先 drop(若 df 已含同名列,merge 会产生 _x / _y 后缀,污染数据)
    for col in ['industry_l1', 'industry_l2', 'exchange']:
        if col in df.columns:
            df = df.drop(columns=[col])
    # 反查 exchange
    ex_lookup = load_exchange_lookup(stock_basic_path)
    df = df.merge(ex_lookup, on='code', how='left')
    # 反查 industry
    ind_lookup = load_industry_lookup(sw2_members_path)
    df = df.merge(ind_lookup, on='code', how='left')
    for col in ['industry_l1', 'industry_l2', 'exchange']:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)
    return df
```

- [ ] **Step 2.5: 改 `rows.append` 把 3 列加进 summary dict**

在 `main()` 第 110-130 行的 `rows.append({...})` dict 里加 3 个 key:

```python
'industry_l1': row.get('industry_l1', ''),
'industry_l2': row.get('industry_l2', ''),
'exchange': row.get('exchange', ''),
```

- [ ] **Step 2.6: 改输出 CSV 路径加 fallback 文本输出路径**

在 `main()` 顶部加:

```python
TXT_OUT_PATH = os.path.join(os.path.dirname(args.output) or '.', 'dynsys_eigen_summary.txt')
AGG_INDUSTRY_CSV = os.path.join(CSV_OUT_DIR, 'v43_eigen_top_industries.csv')
AGG_EXCHANGE_CSV = os.path.join(CSV_OUT_DIR, 'v43_eigen_by_exchange.csv')
```

- [ ] **Step 2.7: 暂不运行(等 Task 4 文本汇总后整体跑)**

- [ ] **Step 2.8: Commit**

```bash
git add backtrace/dynamics/dynamics_eigen_analysis.py
git commit -m "feat(dynamics): v4.3 Task 2 — 读 industry_l1/l2 (sw2) + exchange (stock_basic)"
```

---

## Task 3: 加 2 个新子图 — (1,4) 行业 top10 + (2,4) 交易所

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py`(现 2×3 → 2×4)

**Interfaces:**
- Consumes: `summary_df`(含 3 新列)、`total` / `schur_n` / `wedge_n`(已有)
- Produces: 2 个新 `fig.add_trace(...)` 调用

- [ ] **Step 3.1: 改 make_subplots 布局 2×3 → 2×4**

```python
fig = make_subplots(
    rows=2, cols=4,   # ← 3 改 4
    subplot_titles=(
        '(k̂, ĉ) 散点 + 楔形(颜色=分类)',
        'ρ 分布直方图',
        '11 类分类分布',
        '行业 ρ 中位数 top10',            # ← 新
        '(k̂, ĉ) 散点(颜色=楔形距离)',
        '楔形距离分布',
        'ρ vs 楔形距离',
        '交易所 ρ 中位数(SH vs SZ)',        # ← 新
    ),
    specs=[[{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'bar'},    {'type': 'bar'}],
           [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}, {'type': 'bar'}]],
    horizontal_spacing=0.06,                # ← 0.08 改 0.06 (4 列更挤)
    vertical_spacing=0.18,
)
```

- [ ] **Step 3.2: 加行业聚合函数(放在 main() 之前)**

```python
def aggregate_by_industry(
    df: pd.DataFrame, min_stocks: int = 50, fallback_min: int = 30,
) -> tuple[pd.DataFrame, int]:
    """按 industry_l1 聚合 ρ 中位数。

    Returns:
        agg_df: top 10(降序),列: industry_l1, n_stocks, rho_median, rho_p25, rho_p75,
               k_hat_median, c_hat_median, schur_stable_pct, in_wedge_pct, dist_wedge_median
        threshold_used: 实际生效的 n_stocks 阈值(50 或 30,报告里写)
    """
    for thr in (min_stocks, fallback_min):
        agg = df.groupby('industry_l1').agg(
            n_stocks=('code', 'count'),
            rho_median=('spectral_radius', 'median'),
            rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
            rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
            k_hat_median=('k_hat', 'median'),
            c_hat_median=('c_hat', 'median'),
            schur_stable_pct=('schur_stable', 'mean'),
            in_wedge_pct=('in_wedge', 'mean'),
            dist_wedge_median=('distance_to_wedge', 'median'),
        ).reset_index()
        agg = agg[agg['n_stocks'] >= thr].sort_values('rho_median', ascending=False).head(10)
        if len(agg) >= 5:
            return agg, thr
    return agg, fallback_min   # 都凑不够 5,返回最后尝试的结果
```

- [ ] **Step 3.3: 加交易所聚合函数(放在 main() 之前)**

```python
def aggregate_by_exchange(df: pd.DataFrame) -> pd.DataFrame:
    """按 exchange 聚合(SH / SZ),列同行业聚合。"""
    agg = df.groupby('exchange').agg(
        n_stocks=('code', 'count'),
        rho_median=('spectral_radius', 'median'),
        rho_p25=('spectral_radius', lambda s: s.quantile(0.25)),
        rho_p75=('spectral_radius', lambda s: s.quantile(0.75)),
        k_hat_median=('k_hat', 'median'),
        c_hat_median=('c_hat', 'median'),
        schur_stable_pct=('schur_stable', 'mean'),
        in_wedge_pct=('in_wedge', 'mean'),
        dist_wedge_median=('distance_to_wedge', 'median'),
    ).reset_index().sort_values('rho_median')
    return agg
```

- [ ] **Step 3.4: 在 main() 里加 (1,4) 行业子图**

在现 (1,3) 子图代码之后插入:

```python
# (1,4) 行业 ρ 中位数 top10(误差棒 p25-p75)
agg_l1, l1_threshold = aggregate_by_industry(summary_df)
if len(agg_l1) >= 5:
    fig.add_trace(
        go.Bar(
            x=agg_l1['industry_l1'],
            y=agg_l1['rho_median'],
            error_y=dict(
                type='data',
                symmetric=False,
                array=agg_l1['rho_p75'] - agg_l1['rho_median'],
                arrayminus=agg_l1['rho_median'] - agg_l1['rho_p25'],
                color='black',
                thickness=1.5,
                width=4,
            ),
            marker_color='steelblue',
            name=f'行业 top10 (n≥{l1_threshold})',
            text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_l1['n_stocks'], agg_l1['rho_median'])],
            hovertemplate='<b>%{x}</b><br>ρ 中位数: %{y:.3f}<br>%{text}<extra></extra>',
            showlegend=False,
        ),
        row=1, col=4,
    )
else:
    # 占位:加一条文字说明
    fig.add_annotation(
        text=f'行业不足(n<{l1_threshold},仅 {len(agg_l1)} 个)',
        xref='x4 domain', yref='y4 domain', x=0.5, y=0.5,
        showarrow=False, font=dict(size=12, color='gray'),
        row=1, col=4,
    )
fig.update_xaxes(title_text=f'行业(n≥{l1_threshold})', row=1, col=4, tickangle=-30)
fig.update_yaxes(title_text='ρ 中位数', row=1, col=4)
```

- [ ] **Step 3.5: 在 main() 里加 (2,4) 交易所子图**

在现 (2,3) 子图代码之后插入:

```python
# (2,4) 交易所 ρ 中位数 SH vs SZ
agg_ex = aggregate_by_exchange(summary_df)
ex_colors = {'SH': '#1f77b4', 'SZ': '#ff7f0e', 'BJ': '#2ca02c'}
fig.add_trace(
    go.Bar(
        x=agg_ex['exchange'],
        y=agg_ex['rho_median'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=agg_ex['rho_p75'] - agg_ex['rho_median'],
            arrayminus=agg_ex['rho_median'] - agg_ex['rho_p25'],
            color='black',
            thickness=1.5,
            width=8,
        ),
        marker_color=[ex_colors.get(e, '#888888') for e in agg_ex['exchange']],
        name='交易所 ρ 中位数',
        text=[f"n={n}<br>ρ={r:.2f}" for n, r in zip(agg_ex['n_stocks'], agg_ex['rho_median'])],
        hovertemplate='<b>%{x}</b><br>ρ 中位数: %{y:.3f}<br>%{text}<extra></extra>',
        showlegend=False,
    ),
    row=2, col=4,
)
fig.update_xaxes(title_text='交易所', row=2, col=4)
fig.update_yaxes(title_text='ρ 中位数', row=2, col=4)
```

- [ ] **Step 3.6: 改 fig.update_layout 高度(2×3 → 2×4 需更高)**

```python
fig.update_layout(
    height=1000,        # ← 950 改 1000
    width=1800,         # ← 1500 改 1800
    title_text=f'动力系统特征值分析 v4.3 ({total} 只,Schur 稳定 {schur_n}/{total},楔形内 {wedge_n}/{total})',
    template='plotly_white',
    showlegend=True,
    legend=dict(orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.32),
)
```

- [ ] **Step 3.7: 暂不跑(等 Task 4 文本汇总后整体跑)**

- [ ] **Step 3.8: Commit**

```bash
git add backtrace/dynamics/dynamics_eigen_analysis.py
git commit -m "feat(dynamics): v4.3 Task 3 — HTML 2x4 + 行业/交易所聚合子图"
```

---

## Task 4: 写 `dynsys_eigen_summary.txt` 文本汇总

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py`(`main()` 末尾,`fig.write_html` 之后)

**Interfaces:**
- Consumes: `summary_df`、`total`、`schur_n`、`wedge_n`、`rho_gt1_n`、`wedge_close_n`、`cls_count`、`agg_l1`、`agg_ex`
- Produces: `backtrace/outputs/dynsys_eigen_summary.txt` + `data/dynamics/v43_eigen_top_industries.csv` + `data/dynamics/v43_eigen_by_exchange.csv`

- [ ] **Step 4.1: 加 `write_text_summary` 函数(放在 main() 之前)**

```python
def write_text_summary(
    summary_df: pd.DataFrame,
    cls_count: Counter,
    agg_l1: pd.DataFrame,
    l1_threshold: int,
    agg_ex: pd.DataFrame,
    path: str,
) -> None:
    """写 dynsys_eigen_summary.txt 纯文本汇总(UTF-8)。"""
    import datetime as _dt
    N = len(summary_df)
    rho = summary_df['spectral_radius']
    k_hat = summary_df['k_hat']
    c_hat = summary_df['c_hat']
    schur_n = int(summary_df['schur_stable'].sum())
    wedge_n = int(summary_df['in_wedge'].sum())
    rho_gt1_n = int((rho > 1.0 + 1e-8).sum())
    dist = summary_df['distance_to_wedge']
    timestamp = _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    lines = []
    lines.append('=== v4.3 全市场 (k̂, ĉ) 经验分布报告 ===')
    lines.append(f'样本数: N = {N}')
    lines.append(f'数据来源: data/projection/kc_estimates.csv')
    lines.append(f'报告时间: {timestamp}')
    lines.append('')
    lines.append('--- 全市场 ---')
    lines.append(f'ρ 中位数: {rho.median():.4f} | p25: {rho.quantile(0.25):.4f} | p75: {rho.quantile(0.75):.4f}')
    lines.append(f'k̂ 中位数: {k_hat.median():.4f} | p25: {k_hat.quantile(0.25):.4f} | p75: {k_hat.quantile(0.75):.4f}')
    lines.append(f'ĉ 中位数: {c_hat.median():.4f} | p25: {c_hat.quantile(0.25):.4f} | p75: {c_hat.quantile(0.75):.4f}')
    lines.append(f'Schur 稳定(ρ<1):   {schur_n}/{N} ({schur_n/N*100:.1f}%)')
    lines.append(f'楔形内:            {wedge_n}/{N} ({wedge_n/N*100:.1f}%)')
    lines.append(f'ρ > 1(发散):       {rho_gt1_n}/{N} ({rho_gt1_n/N*100:.1f}%)')
    lines.append(f'distance_to_wedge 中位数: {dist.median():+.4f} (>0 在楔形内)')
    lines.append('')
    lines.append('--- 11 类分布 ---')
    for cls, cnt in sorted(cls_count.items(), key=lambda x: -x[1]):
        lines.append(f'  {cls:<28} {cnt:>5} ({cnt/N*100:>5.1f}%)')
    lines.append('')
    lines.append(f'--- 行业 ρ 中位数 top10 (n_stocks >= {l1_threshold}) ---')
    if len(agg_l1) >= 5:
        for _, r in agg_l1.iterrows():
            lines.append(
                f'  {r["industry_l1"]:<20} n={int(r["n_stocks"]):>4}, '
                f'ρ_med={r["rho_median"]:.3f}, p25={r["rho_p25"]:.3f}, p75={r["rho_p75"]:.3f}, '
                f'k̂_med={r["k_hat_median"]:.3f}, ĉ_med={r["c_hat_median"]:.3f}, '
                f'楔形内%={r["in_wedge_pct"]*100:.1f}%'
            )
    else:
        lines.append(f'  (行业不足 5 个,n_stocks >= {l1_threshold} 仅 {len(agg_l1)} 个)')
    lines.append('')
    lines.append('--- 交易所 ---')
    for _, r in agg_ex.iterrows():
        lines.append(
            f'  {r["exchange"]:<5} n={int(r["n_stocks"]):>4}, '
            f'ρ_med={r["rho_median"]:.3f}, p25={r["rho_p25"]:.3f}, p75={r["rho_p75"]:.3f}, '
            f'楔形内%={r["in_wedge_pct"]*100:.1f}%'
        )
    lines.append('')

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[eigen] ✓ text summary: {path}')
```

- [ ] **Step 4.2: 在 main() 末尾 `fig.write_html` 之后插入调用 + 写 2 个聚合 CSV**

```python
# 行业/交易所聚合表落盘
os.makedirs(CSV_OUT_DIR, exist_ok=True)
agg_l1.to_csv(AGG_INDUSTRY_CSV, index=False, encoding='utf-8')
print(f'[eigen] ✓ industry agg: {AGG_INDUSTRY_CSV}({len(agg_l1)} 行)')
agg_ex.to_csv(AGG_EXCHANGE_CSV, index=False, encoding='utf-8')
print(f'[eigen] ✓ exchange agg: {AGG_EXCHANGE_CSV}({len(agg_ex)} 行)')

# 文本汇总
write_text_summary(summary_df, cls_count, agg_l1, l1_threshold, agg_ex, TXT_OUT_PATH)
```

- [ ] **Step 4.3: 改 print 头部声明(加 v4.3 标识)**

在 main() 顶部 print 段:

```python
print(f'[eigen] 输入: {args.input} ({len(df)} 行,status 前缀 {args.status_filter!r}) — v4.3 全市场经验分布')
```

- [ ] **Step 4.4: 跑全 A 股冒烟(若 Task 1 已跑过)或冒烟 50 只**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py --limit 50 2>&1 | head -60
```

期望:
- 打印 "v4.3 全市场经验分布"
- eigen_summary.csv 多 3 列(industry_l1/l2/exchange)
- dynsys_eigen.html 存在(>100KB)
- dynsys_eigen_summary.txt 存在,中文不乱码,含 "11 类分布" / "行业 ρ 中位数 top10" / "交易所" 三个章节
- v43_eigen_top_industries.csv / v43_eigen_by_exchange.csv 存在

- [ ] **Step 4.5: 验证文本汇总可读**

```bash
cat backtrace/outputs/dynsys_eigen_summary.txt | head -30
```

期望: 看到 "=== v4.3 全市场 (k̂, ĉ) 经验分布报告 ===" + "--- 全市场 ---" + "--- 11 类分布 ---" + "--- 行业 ρ 中位数 top10 ---" + "--- 交易所 ---"。

- [ ] **Step 4.6: Commit**

```bash
git add backtrace/dynamics/dynamics_eigen_analysis.py
git commit -m "feat(dynamics): v4.3 Task 4 — 文本汇总 dynsys_eigen_summary.txt"
```

---

## Task 5: 加 3 个单元测试

**Files:**
- Modify: `tests/test_dynamics_eigen.py`(现 23 个测试 → 26 个)

**Interfaces:**
- Consumes: `aggregate_by_industry` / `aggregate_by_exchange` / `dynamics_eigen_analysis.main`(从 `dynamics_eigen_analysis` 模块导入)
- Produces: 3 个新 pytest 函数

- [ ] **Step 5.1: 读现状,定位 import 段**

读 `tests/test_dynamics_eigen.py:1-30`,确认 `analyze_eigenvalues` import 写法,新测试沿用同样模式:

```python
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
from dynamics import dynamics_eigen_analysis as EA
```

- [ ] **Step 5.2: 加测试 1 — 行业聚合 ρ 中位数**

```python
def test_industry_aggregation_rho_median(tmp_path):
    """构造 100 只票 / 3 个行业的 dummy DataFrame,验证 groupby median 正确。"""
    import pandas as pd
    rng = np.random.default_rng(42)
    rows = []
    # industry_A 50 只 ρ∈[0.5, 1.0]
    for i in range(50):
        rho = rng.uniform(0.5, 1.0)
        rows.append({'code': f'A{i:03d}', 'industry_l1': 'A', 'spectral_radius': rho,
                     'k_hat': 0.1, 'c_hat': 1.0, 'schur_stable': rho < 1.0,
                     'in_wedge': True, 'distance_to_wedge': 0.1})
    # industry_B 30 只 ρ∈[1.0, 2.0]
    for i in range(30):
        rho = rng.uniform(1.0, 2.0)
        rows.append({'code': f'B{i:03d}', 'industry_l1': 'B', 'spectral_radius': rho,
                     'k_hat': 0.0, 'c_hat': 2.0, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -0.5})
    # industry_C 20 只 ρ∈[2.0, 5.0]
    for i in range(20):
        rho = rng.uniform(2.0, 5.0)
        rows.append({'code': f'C{i:03d}', 'industry_l1': 'C', 'spectral_radius': rho,
                     'k_hat': -0.1, 'c_hat': 5.0, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -2.0})
    df = pd.DataFrame(rows)

    agg, threshold = EA.aggregate_by_industry(df, min_stocks=10, fallback_min=5)
    # 100 只都在,A/B/C 三个行业都 n>=10,threshold=10
    assert threshold == 10
    assert len(agg) == 3
    # A ρ 中位数 ~0.75, B ~1.5, C ~3.5
    rho_med_by_industry = dict(zip(agg['industry_l1'], agg['rho_median']))
    assert 0.6 < rho_med_by_industry['A'] < 0.9
    assert 1.2 < rho_med_by_industry['B'] < 1.8
    assert 2.8 < rho_med_by_industry['C'] < 4.2
    # n_stocks 正确
    n_by_industry = dict(zip(agg['industry_l1'], agg['n_stocks']))
    assert n_by_industry['A'] == 50
    assert n_by_industry['B'] == 30
    assert n_by_industry['C'] == 20
    # 降序排
    assert agg['rho_median'].is_monotonic_decreasing
```

- [ ] **Step 5.3: 加测试 2 — 交易所拆分 + 误差棒**

```python
def test_exchange_split_correctness():
    """SH/SZ 各 50 只,验证 n_stocks=50/50,p25/p75 正确。"""
    import pandas as pd
    rng = np.random.default_rng(7)
    rows = []
    for i in range(50):
        rows.append({'code': f'SH{i:03d}', 'exchange': 'SH', 'spectral_radius': rng.uniform(0.8, 1.5),
                     'k_hat': 0.0, 'c_hat': 1.2, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -0.1})
    for i in range(50):
        rows.append({'code': f'SZ{i:03d}', 'exchange': 'SZ', 'spectral_radius': rng.uniform(1.0, 2.0),
                     'k_hat': 0.0, 'c_hat': 1.5, 'schur_stable': False,
                     'in_wedge': False, 'distance_to_wedge': -0.2})
    df = pd.DataFrame(rows)

    agg = EA.aggregate_by_exchange(df)
    assert set(agg['exchange']) == {'SH', 'SZ'}
    n_by_ex = dict(zip(agg['exchange'], agg['n_stocks']))
    assert n_by_ex['SH'] == 50
    assert n_by_ex['SZ'] == 50
    # SH ρ 中位数 ~1.15, SZ ~1.5
    rho_by_ex = dict(zip(agg['exchange'], agg['rho_median']))
    assert 1.0 < rho_by_ex['SH'] < 1.3
    assert 1.3 < rho_by_ex['SZ'] < 1.7
    # p25 <= median <= p75
    for _, r in agg.iterrows():
        assert r['rho_p25'] <= r['rho_median'] <= r['rho_p75']
```

- [ ] **Step 5.4: 加测试 3 — HTML + 文本输出(用 monkeypatch + tmp_path)**

```python
def test_html_2x4_layout_and_text_summary(tmp_path, monkeypatch):
    """构造 dummy kc_estimates + dummy stock_basic + dummy sw2/members,跑 main 验证 HTML + 文本汇总。"""
    import pandas as pd
    rng = np.random.default_rng(123)

    # --- 1. 构造 dummy kc_estimates(只含主字段,不含行业 / 交易所)
    rows = []
    for i in range(50):
        k = rng.uniform(-0.5, 0.5)
        c = rng.uniform(0.5, 3.0)
        eig = EA.analyze_eigenvalues(k, c)
        rows.append({
            'code': f'{i:06d}.SH' if i < 30 else f'{i:06d}.SZ',
            'name': f'Test{i}',
            'index_tag': '000',
            'stock_tag': f'{i:06d}',
            'k_hat': k,
            'c_hat': c,
            'lam1_real': float(eig['eigenvalues'][0].real),
            'lam1_imag': float(eig['eigenvalues'][0].imag),
            'lam2_real': float(eig['eigenvalues'][1].real),
            'lam2_imag': float(eig['eigenvalues'][1].imag),
            'spectral_radius': eig['spectral_radius'],
            'classification': eig['classification'],
            'stability': eig['stability'],
            'schur_stable': eig['schur_stable'],
            'in_wedge': eig['in_wedge'],
            'distance_lower_boundary': eig['distance_lower_boundary'],
            'distance_upper_boundary': eig['distance_upper_boundary'],
            'distance_to_wedge': eig['distance_to_wedge'],
        })
    df = pd.DataFrame(rows)
    csv_in = tmp_path / 'kc_estimates.csv'
    df.to_csv(csv_in, index=False)

    # --- 2. 构造 dummy stock_basic.csv(只含 exchange → market 列)
    sb_rows = []
    for i in range(50):
        market = 'SH' if i < 30 else 'SZ'
        sb_rows.append({'code': rows[i]['code'], 'market': market,
                        'name': f'Test{i}', 'status': 'active'})
    sb_path = tmp_path / 'stock_basic.csv'
    pd.DataFrame(sb_rows).to_csv(sb_path, index=False)

    # --- 3. 构造 dummy sw2/members.csv(提供 industry_l1/l2)
    # 前 30 只 → industry "A"(industry_l1=881xxx.SH, industry_l2="A组"),后 20 只 → industry "B"
    sw2_rows = []
    for i in range(30):
        sw2_rows.append({'sector_code': '881001.SH', 'sector_name': 'A组',
                         'member_code': rows[i]['code']})
    for i in range(30, 50):
        sw2_rows.append({'sector_code': '881002.SH', 'sector_name': 'B组',
                         'member_code': rows[i]['code']})
    sw2_path = tmp_path / 'sw2_members.csv'
    pd.DataFrame(sw2_rows).to_csv(sw2_path, index=False)

    # --- 4. 输出路径
    html_out = tmp_path / 'dynsys_eigen.html'
    txt_out = tmp_path / 'dynsys_eigen_summary.txt'
    csv_out = tmp_path / 'eigen_summary.csv'

    # --- 5. monkeypatch + 跑 main
    import sys
    sys.argv = ['dynamics_eigen_analysis',
                '--input', str(csv_in),
                '--output', str(html_out),
                '--stock-basic', str(sb_path),
                '--sw2-members', str(sw2_path),
                '--limit', '0']
    # 注意:实际参数名以 dynamics_eigen_analysis.py argparse 为准;
    # 若 main 不接受这俩 flag,可通过 monkeypatch 改 DEFAULT 路径

    EA.main()

    # --- 6. 验收
    assert html_out.exists() and html_out.stat().st_size > 50_000
    assert txt_out.exists()
    txt_content = txt_out.read_text(encoding='utf-8')
    assert '=== v4.3 全市场' in txt_content
    assert '--- 11 类分布 ---' in txt_content
    assert '--- 行业 ρ 中位数 top10' in txt_content
    assert '--- 交易所 ---' in txt_content
    # 行业 / 交易所 CSV 落盘
    assert (tmp_path / 'v43_eigen_top_industries.csv').exists()
    assert (tmp_path / 'v43_eigen_by_exchange.csv').exists()
```

- [ ] **Step 5.5: 跑全部测试**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py -v 2>&1 | tail -40
```

期望: 26 passed,0 failed(含 23 个老 + 3 个新)。

- [ ] **Step 5.6: 若失败,逐个修**

| 错误类型 | 修法 |
|---|---|
| `ModuleNotFoundError: No module named 'dynamics'` | 测试文件顶部 sys.path 注入逻辑缺;补 `BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` |
| `AttributeError: 'module' object has no attribute 'aggregate_by_industry'` | 命名空间不暴露;在 `dynamics_eigen_analysis.py` 顶部加 `__all__ = ['aggregate_by_industry', ...]` 或测试直接 `from dynamics.dynamics_eigen_analysis import aggregate_by_industry` |
| `eigenvalues tuple index` 错(老 `dynamics` 命名空间仍 export `analyze_eigenvalues`) | 沿用现 import 模式 |
| HTML < 50KB | 阈值太严,8 子图 plotly 实际 ~2-4 MB,改 `> 50_000` → 实际应远大于 |

- [ ] **Step 5.7: Commit**

```bash
git add tests/test_dynamics_eigen.py
git commit -m "test(dynamics): v4.3 — 行业聚合 / 交易所拆分 / HTML+文本输出(3 个)"
```

---

## Task 6: 更新 `backtrace/dynamics/README.md` 加 v4.3 节

**Files:**
- Modify: `backtrace/dynamics/README.md`(末尾加 v4.3 节;不动现有章节)

- [ ] **Step 6.1: 读 README 末尾,定位插入位置**

```bash
wc -l backtrace/dynamics/README.md
tail -20 backtrace/dynamics/README.md
```

在 "v4.2 楔形距离" 章节之后插入 "v4.3 全市场经验分布"。

- [ ] **Step 6.2: 加 v4.3 节**

```markdown
### 3.4 全市场经验分布(v4.3,2026-08-17)

把 `kc_estimates.csv` 从 4 只 smoke-test 扩张到全 A 股 (~5000 只),回答经验问题:
"动力系统参数 (k̂, ĉ) 在全市场到底呈什么分布?"

```bash
# 1. (前置)生成全 A 股 movement 文件(~20-40 分钟)
PYTHONIOENCODING=utf-8 python backtrace/projection/projection_batch.py --movement \
    --input data/stock_basic.csv --limit 0 --days 240

# 2. 跑全 A 股 parameter_fit(~20-40 分钟)
PYTHONIOENCODING=utf-8 python backtrace/projection/parameter_fit.py --limit 0

# 3. 跑 v4.3 报告
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py
```

**数据源**:
- `data/projection/kc_estimates.csv` — 主输入(全 A 股 ~5000 只 (k̂, ĉ))
- `data/stock_basic.csv` — 反查 `exchange`(`market` 字段 = SH/SZ/BJ)
- `data/sw2/members.csv` — 反查 `industry_l1`(sector_code)/ `industry_l2`(sector_name)

**输出**:
- `data/dynamics/eigen_summary.csv` — 21 列(18 + industry_l1/l2/exchange)
- `backtrace/outputs/dynsys_eigen.html` — **2×4 网格 8 子图 plotly**(~2-4 MB)
- `backtrace/outputs/dynsys_eigen_summary.txt` — 纯文本汇总(便于 CI/grep)
- `data/dynamics/v43_eigen_top_industries.csv` — 行业聚合表
- `data/dynamics/v43_eigen_by_exchange.csv` — 交易所聚合表

**HTML 8 子图布局**:
- (1,1) (k̂, ĉ) 散点 + 楔形(分类着色)
- (1,2) ρ 直方图 + ρ=1 红虚线
- (1,3) 11 类分类柱状
- (1,4) **行业 ρ 中位数 top10**(误差棒 [p25, p75])
- (2,1) (k̂, ĉ) 散点(楔形距离着色)
- (2,2) 楔形距离直方图
- (2,3) ρ vs 楔形距离
- (2,4) **交易所 ρ 中位数对比(SH vs SZ vs BJ)**

**关键决策**:
- 聚合用 **median** 而非 mean(ρ 偏态)
- 行业映射来自 `data/sw2/members.csv`,交易所来自 `data/stock_basic.csv`(两个独立 lookup)
- 行业筛选 `n_stocks >= 50` 硬阈值,不足 10 个降级到 `n >= 30`
- 行业用申万二级 sector_name(top10 视觉最均衡)

**v4.3 显式不做**:
- G(ω) 频率响应(独立 v5)
- 行业 SI / (k,c) 相图 / 状态转移矩阵
- IC / basket / 任何交易信号
```

- [ ] **Step 6.3: Commit**

```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README v4.3 节 — 全市场经验分布 + 2x4 HTML"
```

---

## Task 7: 更新上层 spec `2026-08-16-dynamics-system-design.md` §3.5

**Files:**
- Modify: `docs/superpowers/specs/2026-08-16-dynamics-system-design.md:178-323`(现 §3.5 末尾追加 v4.3 子节)

- [ ] **Step 7.1: 读 §3.5 现状**

读 `docs/superpowers/specs/2026-08-16-dynamics-system-design.md:175-325`,确认 §3.5 现状("analyze_eigenvalues (2026-08-17 v4.1 + v4.2)")。

- [ ] **Step 7.2: 追加 v4.3 子节**

在 §3.5 末尾(测试覆盖段落之后)追加:

```markdown
### 3.6 v4.3 全市场经验分布(2026-08-17)— 仅数据层扩张

数学层**完全不动**。`dynamics_eigen_analysis.py` 加 3 列读入 + 2 个聚合子图 + 1 个文本汇总:

**输入扩张**:
- `parameter_fit.py --limit 0` 跑全 A 股(~5000 只,~20-40 分钟;前置依赖: `projection_batch.py --movement` 先跑 ~20-40 分钟)
- `data/sw2/members.csv`(列: `sector_code, sector_name, member_code`)反查 `industry_l1`(sector_code)/ `industry_l2`(sector_name)
- `data/stock_basic.csv`(列: `code, market, name, status`)反查 `exchange`(`market` 字段 = SH/SZ/BJ)
- 缺文件 / 缺列 → 行业列 / exchange 留空,流程不致命

**聚合统计量**(用 median,不用 mean):
- 行业(申万二级 sector_name)top10: `n_stocks` / `rho_median` / `rho_p25` / `rho_p75` / `k_hat_median` / `c_hat_median` / `schur_stable_pct` / `in_wedge_pct` / `dist_wedge_median`
- 交易所 SH / SZ / BJ: 同 9 个统计量

**HTML 2×4**(现 2×3 + 新 2):
- (1,4) 行业 ρ 中位数 top10(误差棒 [p25, p75])
- (2,4) 交易所 ρ 中位数对比(SH / SZ / BJ)

**新增输出**:
- `data/dynamics/eigen_summary.csv` 18 → 21 列
- `backtrace/outputs/dynsys_eigen.html` 2×4
- `backtrace/outputs/dynsys_eigen_summary.txt` 纯文本汇总
- `data/dynamics/v43_eigen_top_industries.csv` 行业聚合
- `data/dynamics/v43_eigen_by_exchange.csv` 交易所聚合

**显式不做**(留 v4.4 - v5):
- G(ω) 频率响应
- 行业稳定性指数 SI
- (k, c) 相图 + 7 状态颜色
- 状态转移矩阵
- 任何 IC / basket / 交易信号

**测试**:`tests/test_dynamics_eigen.py` 加 3 个测试,总 26 passed。
```

- [ ] **Step 7.3: Commit**

```bash
git add docs/superpowers/specs/2026-08-16-dynamics-system-design.md
git commit -m "docs(dynamics): spec §3.5 末尾加 v4.3 子节 — 数据层扩张,数学不动"
```

---

## Task 8: 端到端验收 + 跑全 A 股

**Files:** 无代码改动,纯验收

- [ ] **Step 8.1: 跑全部单元测试**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python -m pytest tests/test_dynamics_eigen.py tests/test_projection_core.py -v 2>&1 | tail -50
```

期望: 26 + 25 = 51 passed(原 23 个 v4.1+v4.2 + 25 个 projection_core + 3 个 v4.3)。

- [ ] **Step 8.2: 跑 v4.3 全 A 股报告(若 Task 1 跑过,直接跑;否则先 Task 1)**

```bash
cd c:/Users/yellow/mcp/qtTdx
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_eigen_analysis.py 2>&1 | tail -50
```

期望: 不报错,打印 v4.3 标识 + 5 个 ✓ 输出路径。

- [ ] **Step 8.3: 验收 HTML**

```bash
ls -lh backtrace/outputs/dynsys_eigen.html
```

期望: 文件存在,大小 1-5 MB(8 子图 plotly)。

- [ ] **Step 8.4: 验收文本汇总(中文不乱码)**

```bash
cat backtrace/outputs/dynsys_eigen_summary.txt | head -40
```

期望: 看到 "=== v4.3 全市场 (k̂, ĉ) 经验分布报告 ===" + 各章节标题 + 数字(ρ 中位数、k̂、ĉ、11 类分布等)。

- [ ] **Step 8.5: 验收 eigen_summary.csv 21 列**

```bash
head -1 data/dynamics/eigen_summary.csv | tr ',' '\n' | tail -5
```

期望: 末尾 3 列是 `industry_l1`, `industry_l2`, `exchange`。

- [ ] **Step 8.6: 物理一致性自检(人工)**

- ρ 中位数应在 [0.5, 5.0] 范围(典型 A 股偏发散)
- 11 类分布总和 ≈ 100%(±0.1% 舍入误差)
- 行业 top10 中 n_stocks ≥ 50
- 交易所 SH / SZ 至少各出现一次

- [ ] **Step 8.7: 全部 commit 已在 Task 1-7 完成 — 无新增 commit**

---

## Self-Review

**1. Spec coverage:**

| Spec 章节 | 任务 |
|---|---|
| §1 范围 | Task 1 (input) + Task 2-4 (output) |
| §2 数学层冻结 | 显式声明于 Global Constraints + Task 8 验收 |
| §3 数据流 IO | Task 2 (3 列 IO) + Task 4 (3 个新输出) |
| §4 聚合统计量 | Task 3 (aggregate_by_industry/exchange) + Task 4 (write_text_summary) |
| §5 HTML 2×4 | Task 3 (make_subplots 改 + 2 个 add_trace) |
| §6 文本汇总格式 | Task 4 (write_text_summary) |
| §7 测试 | Task 5 (3 测试) |
| §8 关键文件 | Task 2-7 (modify 列表) |
| §9 验证路径 | Task 8 |
| §11 决策记录 | 在 README v4.3 节固化(不用代码) |

无遗漏。

**2. Placeholder scan:**

- 无 TBD / TODO / "implement later"
- 无 "appropriate error handling" 类泛述
- 每个代码块都完整可粘贴
- 无 "similar to Task N"(重复代码模式但每处独立)
- 所有用到的函数名(`aggregate_by_industry` / `aggregate_by_exchange` / `write_text_summary` / `TXT_OUT_PATH` / `AGG_INDUSTRY_CSV` / `AGG_EXCHANGE_CSV`)在定义的任务里有定义

**3. Type / name 一致性:**

- `aggregate_by_industry(df, min_stocks, fallback_min) -> (DataFrame, int)` 在 Task 3.2 定义,Task 4.1 消费,Task 5.2 测试。一致
- `aggregate_by_exchange(df) -> DataFrame` 在 Task 3.3 定义,Task 4.1 消费,Task 5.3 测试。一致
- `write_text_summary(...)` 在 Task 4.1 定义,Task 4.2 调用。一致
- `TXT_OUT_PATH` / `AGG_INDUSTRY_CSV` / `AGG_EXCHANGE_CSV` 在 Task 2.5 定义,Task 4.2 / 5.4 引用。一致
- `CSV_OUT_DIR` 已在原代码定义(第 37 行),Task 2.5 不重新定义,只引用

**4. Ambiguity:**

- 行业筛选阈值在 spec §4.2 已硬规则(50 → 30),Task 3.2 代码与之一致
- 行业映射"有效值 < 5"判定在 spec §3.1 明确,Task 2.2 代码与之一致
- HTML 高度 950→1000,宽度 1500→1800 在 Task 3.6 显式
- 测试 5.4 的 HTML 50KB 阈值(估计下限)在 Task 5.6 给了回退说明

OK。
