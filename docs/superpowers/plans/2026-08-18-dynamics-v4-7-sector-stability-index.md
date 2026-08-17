# v4.7 — 行业稳定性指数 SI(Sector Stability Index)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `backtrace/dynamics/dynamics_eigen_analysis.py` 末尾追加行业层稳定性评分(SI ∈ [0,1]),输出 `sector_si.csv` + 4 子图 HTML + 文本汇总,业务可回答"哪些行业整体最稳定 / 最分裂"。

**Architecture:** 复用 v4.3 `aggregate_by_industry` 的 groupby 框架,在 1 个新函数 `compute_sector_stability(df)` 内计算 3 个 0-1 子分 + 加权 SI,导出 1 个新 CSV + 1 个新 HTML + 1 个新文本汇总。数学层 0 改,3 caller 0 改。

**Tech Stack:** Python 3.13 / pandas / numpy / plotly / pytest / tsfresh 全栈环境(`/c/ProgramData/anaconda3/python.exe`)

## Global Constraints

- 数学层 `_dynamics_core.py` 0 行修改
- 3 caller(`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`)0 行修改
- `analyze_eigenvalues` / `simulate_trajectory` 函数签名不变
- 输出全部 gitignored(`data/dynamics/` + `backtrace/outputs/`)
- `PYTHONIOENCODING=utf-8` 必备;Windows 用 `/c/ProgramData/anaconda3/python.exe`
- 35 tests pass 目标(30 旧 + 5 新)
- 权重常量 `SI_WEIGHTS = (0.5, 0.2, 0.3)` 集中 1 处(`dynamics_eigen_analysis.py` 顶部),后续调权改 1 行

## 现状(实施前必读)

- `backtrace/dynamics/dynamics_eigen_analysis.py:163-189` — `aggregate_by_industry` 已有 groupby θ 框架,v4.7 SI 在此基础上加 3 列健康分 + 1 列 SI
- `backtrace/dynamics/dynamics_eigen_analysis.py:225-306` — `write_text_summary` 已有文本汇总模板,SI summary 复用其 UTF-8 / 中文处理
- `backtrace/dynamics/dynamics_eigen_analysis.py:400-` — `main()` 末尾追加 SI 调用(默认运行,不增加 flag)
- `tests/test_dynamics_eigen.py` 现有 30 测试(541 行),新 5 测试追加末尾
- 输入文件 `data/dynamics/eigen_summary.csv` 来自 v4.3,21 列(基础 14 + 楔形距离 3 + industry_l1/l2/exchange)

---

## Task 1: 5 测试 + 3 函数 + main() 末尾 hook

**Files:**
- Modify: `backtrace/dynamics/dynamics_eigen_analysis.py` (新增常量 / 3 函数 / main() hook,~120 行)
- Modify: `tests/test_dynamics_eigen.py` (新增 5 测试,~80 行)

**Interfaces:**
- `SI_WEIGHTS = (0.5, 0.2, 0.3)` 元组(ρ / damping / wedge 权重)
- `compute_sector_stability(df: pd.DataFrame, name_lookup: dict | None = None) -> pd.DataFrame` — 返回 9 列: `industry_l1, sector_name, n_stocks, rho_health, damping_health, wedge_health, SI, rho_median, c_median`
- `build_sector_si_html(df_si: pd.DataFrame, output_path: str) -> None` — 写 4 子图 plotly HTML
- `write_sector_si_summary(df_si: pd.DataFrame, output_path: str) -> None` — 写 UTF-8 文本汇总

### Step 1.1: 写 5 个失败测试

打开 `tests/test_dynamics_eigen.py`,在文件末尾追加:

```python
def test_sector_si_basic_shape():
    """SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health,锁定权重
    
    1 行业 100 只全稳定(ρ=0.5, c=1.0, in_wedge=True) → SI = 0.875
    """
    rng = np.random.default_rng(41)
    rows = []
    for i in range(100):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881999.SH',
            'spectral_radius': 0.5, 'c_hat': 1.0, 'in_wedge': True,
            'k_hat': 0.1, 'schur_stable': True, 'distance_to_wedge': 0.3,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    # ρ_health = clip(1 - 0.5/2, 0, 1) = 0.75
    # damping_health = clip(1 - |1-1|/2, 0, 1) = 1.0
    # wedge_health = clip(1.0, 0, 1) = 1.0
    # SI = 0.5*0.75 + 0.2*1.0 + 0.3*1.0 = 0.875
    assert np.isclose(si['SI'].iloc[0], 0.875, atol=1e-9)


def test_sector_si_anti_restoring():
    """anti_restoring 类(ρ=3.0, c=1.5, in_wedge=False) → SI = 0.15"""
    rows = []
    for i in range(100):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881888.SH',
            'spectral_radius': 3.0, 'c_hat': 1.5, 'in_wedge': False,
            'k_hat': -0.05, 'schur_stable': False, 'distance_to_wedge': -0.5,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    # ρ_health = clip(1 - 3/2, 0, 1) = 0
    # damping_health = clip(1 - 0.5/2, 0, 1) = 0.75
    # wedge_health = 0
    # SI = 0.5*0 + 0.2*0.75 + 0.3*0 = 0.15
    assert np.isclose(si['SI'].iloc[0], 0.15, atol=1e-9)


def test_sector_si_clamps_extreme():
    """极端 ρ=10, c=10 → ρ_health=0, damping_health=0,wedge 也 0 → SI=0"""
    rows = []
    for i in range(50):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881777.SH',
            'spectral_radius': 10.0, 'c_hat': 10.0, 'in_wedge': False,
            'k_hat': -1.0, 'schur_stable': False, 'distance_to_wedge': -2.0,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    assert si['rho_health'].iloc[0] == 0.0
    assert si['damping_health'].iloc[0] == 0.0
    assert si['SI'].iloc[0] == 0.0


def test_sector_si_perfect():
    """完美: ρ=0, c=1, in_wedge_pct=1 → SI = 1.0"""
    rows = []
    for i in range(50):
        rows.append({
            'code': f'X{i:03d}', 'industry_l1': '881666.SH',
            'spectral_radius': 0.0, 'c_hat': 1.0, 'in_wedge': True,
            'k_hat': 0.0, 'schur_stable': True, 'distance_to_wedge': 1.0,
        })
    df = pd.DataFrame(rows)
    si = EA.compute_sector_stability(df)
    assert len(si) == 1
    assert np.isclose(si['SI'].iloc[0], 1.0, atol=1e-9)


def test_sector_si_summary_text(tmp_path):
    """write_sector_si_summary 包含 "Top 12 强" + 至少 1 个中文行业名"""
    # 构造 3 个行业,确保前 12 名至少有 1 个中文
    rng = np.random.default_rng(42)
    rows = []
    industries = [
        ('881111.SH', '银行', 0.5, 1.0, 1.0),
        ('881222.SH', '半导体', 3.0, 1.5, 0.0),
        ('881333.SH', '公用事业', 0.7, 1.0, 0.8),
    ]
    for code, _, rho, c, wedge in industries:
        for i in range(60):
            rows.append({
                'code': f'{code}{i:03d}', 'industry_l1': code,
                'spectral_radius': rho, 'c_hat': c, 'in_wedge': wedge > 0.5,
                'k_hat': 0.1, 'schur_stable': rho < 1.0,
                'distance_to_wedge': 0.2 if wedge > 0.5 else -0.2,
            })
    df = pd.DataFrame(rows)
    df_si = EA.compute_sector_stability(df)
    out_path = tmp_path / 'si_summary.txt'
    EA.write_sector_si_summary(df_si, str(out_path))
    content = out_path.read_text(encoding='utf-8')
    assert 'Top 12 强' in content
    assert 'Top 12 弱' in content
    assert '银行' in content
    assert '半导体' in content
```

### Step 1.2: 跑测试,确认失败

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -k "sector_si" -v
```

**Expected:** 5 个 ImportError 或 AttributeError(函数未定义)

### Step 1.3: 实现 `compute_sector_stability`(核心)

打开 `backtrace/dynamics/dynamics_eigen_analysis.py`,在文件顶部 **`CLASS_LABEL_CN` dict 之后** 追加:

```python
# 行业稳定性指数 SI 权重(ρ / damping / wedge;总和 = 1.0)
SI_WEIGHTS = (0.5, 0.2, 0.3)
```

然后在 `aggregate_by_exchange` 函数之后(约 L207 之后)插入 `compute_sector_stability`:

```python
def compute_sector_stability(
    df: pd.DataFrame, name_lookup: dict | None = None,
) -> pd.DataFrame:
    """按申万二级行业计算稳定性指数 SI ∈ [0, 1]。

    3 个 0-1 子分(线性映射,clip 到 [0,1]):
      ρ_health      = clip(1 - ρ_med / 2,        0, 1)        # ρ=0 → 1, ρ≥2 → 0
      damping_health = clip(1 - |c_med - 1| / 2,  0, 1)        # c=1 → 1, |c-1|≥2 → 0
      wedge_health   = clip(in_wedge_pct,         0, 1)        # 100% 在楔形 → 1

    固定权重(在 SI_WEIGHTS):
      SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health

    Args:
        df: 含 industry_l1, spectral_radius, c_hat, in_wedge 列(v4.3 eigen_summary schema)
        name_lookup: 可选 sector_code → sector_name 反查表(默认无 → sector_name 留空)

    Returns:
        9 列 DataFrame: industry_l1, sector_name, n_stocks, rho_health, damping_health,
                       wedge_health, SI, rho_median, c_median
        按 SI 降序
    """
    if df.empty:
        return pd.DataFrame(columns=[
            'industry_l1', 'sector_name', 'n_stocks', 'rho_health',
            'damping_health', 'wedge_health', 'SI', 'rho_median', 'c_median',
        ])
    rho_w, damp_w, wedge_w = SI_WEIGHTS
    agg = df.groupby('industry_l1').agg(
        n_stocks=('code', 'count'),
        rho_median=('spectral_radius', 'median'),
        c_median=('c_hat', 'median'),
        wedge_pct=('in_wedge', 'mean'),
    ).reset_index()
    # §3.3 行业筛选阈值(沿用 v4.3): ≥50 强, ≥30 弱;两者都 < 5 行业时上层标 "no data"
    strong = agg[agg['n_stocks'] >= 50]
    weak = agg[(agg['n_stocks'] >= 30) & (agg['n_stocks'] < 50)]
    if len(strong) >= 5:
        agg = strong
    elif len(strong) + len(weak) >= 5:
        agg = pd.concat([strong, weak], ignore_index=True)
    # else: 保留全部,上层汇总会标 low-confidence
    # 3 个 0-1 子分
    agg['rho_health'] = (1.0 - agg['rho_median'] / 2.0).clip(0.0, 1.0)
    agg['damping_health'] = (1.0 - (agg['c_median'] - 1.0).abs() / 2.0).clip(0.0, 1.0)
    agg['wedge_health'] = agg['wedge_pct'].clip(0.0, 1.0)
    agg['SI'] = (
        rho_w * agg['rho_health']
        + damp_w * agg['damping_health']
        + wedge_w * agg['wedge_health']
    )
    # sector_name 反查(可选)
    if name_lookup:
        agg['sector_name'] = agg['industry_l1'].map(name_lookup).fillna('')
    else:
        agg['sector_name'] = ''
    # 排序 + 列顺序
    agg = agg.sort_values('SI', ascending=False).reset_index(drop=True)
    return agg[[
        'industry_l1', 'sector_name', 'n_stocks', 'rho_health',
        'damping_health', 'wedge_health', 'SI', 'rho_median', 'c_median',
    ]]
```

### Step 1.4: 跑 `test_sector_si_basic_shape` / `test_sector_si_anti_restoring` / `test_sector_si_clamps_extreme` / `test_sector_si_perfect`,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -k "sector_si_basic_shape or sector_si_anti_restoring or sector_si_clamps_extreme or sector_si_perfect" -v
```

**Expected:** 4 PASS(`test_sector_si_summary_text` 仍 fail,等 Step 1.5)

### Step 1.5: 实现 `write_sector_si_summary`

在 `compute_sector_stability` 之后追加:

```python
def write_sector_si_summary(df_si: pd.DataFrame, output_path: str) -> None:
    """写 UTF-8 文本汇总(top 12 强 / 弱 行业 + SI 直方图分布)。

    Args:
        df_si: `compute_sector_stability` 输出
        output_path: 文本文件路径
    """
    lines = []
    lines.append(f'行业稳定性指数 SI(N={len(df_si)} 各行业)')
    lines.append('=' * 70)
    if df_si.empty:
        lines.append('(无数据)')
        Path(output_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return
    # Top 12 强
    lines.append('')
    lines.append('Top 12 强 SI:')
    lines.append('-' * 70)
    top_strong = df_si.head(12)
    for _, row in top_strong.iterrows():
        name = row['sector_name'] or row['industry_l1']
        lines.append(
            f"  {name:<14s} SI={row['SI']:.3f}  "
            f"ρ_med={row['rho_median']:.3f}  "
            f"c_med={row['c_median']:.3f}  "
            f"wedge={row['wedge_health']:.2f}  "
            f"n={row['n_stocks']}"
        )
    # Top 12 弱
    lines.append('')
    lines.append('Top 12 弱 SI:')
    lines.append('-' * 70)
    top_weak = df_si.tail(12).iloc[::-1]  # 升序反转成从弱到最弱
    for _, row in top_weak.iterrows():
        name = row['sector_name'] or row['industry_l1']
        lines.append(
            f"  {name:<14s} SI={row['SI']:.3f}  "
            f"ρ_med={row['rho_median']:.3f}  "
            f"c_med={row['c_median']:.3f}  "
            f"wedge={row['wedge_health']:.2f}  "
            f"n={row['n_stocks']}"
        )
    # 直方图分布
    lines.append('')
    lines.append('SI 直方图分布:')
    lines.append('-' * 70)
    bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    for lo, hi in bins:
        count = ((df_si['SI'] >= lo) & (df_si['SI'] < hi)).sum()
        lines.append(f'  [{lo:.1f}, {hi:.1f}): {count:>3d} 行业')
    content = '\n'.join(lines) + '\n'
    Path(output_path).write_text(content, encoding='utf-8')
```

**注意**:在 `dynamics_eigen_analysis.py` 顶部 `import` 区域添加 `from pathlib import Path`(如果还没有)。

### Step 1.6: 跑全部 5 测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -k "sector_si" -v
```

**Expected:** 5 PASS

### Step 1.7: 实现 `build_sector_si_html`(4 子图 plotly)

在 `write_sector_si_summary` 之后追加:

```python
def build_sector_si_html(df_si: pd.DataFrame, output_path: str) -> None:
    """画 4 子图 plotly HTML:(1,1) SI 分布直方图 (1,2) SI vs ρ_med 散点 
    (2,1) Top 12 强 SI 行业 (2,2) Top 12 弱 SI 行业。

    Args:
        df_si: `compute_sector_stability` 输出
        output_path: HTML 输出路径
    """
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'SI 分布直方图', 'SI vs ρ_med(气泡 = n_stocks)',
            'Top 12 强 SI 行业', 'Top 12 弱 SI 行业',
        ),
        specs=[
            [{'type': 'xy'}, {'type': 'xy'}],
            [{'type': 'xy'}, {'type': 'xy'}],
        ],
        horizontal_spacing=0.12, vertical_spacing=0.18,
    )
    if df_si.empty:
        fig.add_annotation(text='无数据', xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        fig.write_html(output_path, include_plotlyjs='cdn')
        return
    # (1, 1) SI 分布直方图
    fig.add_trace(
        go.Histogram(x=df_si['SI'], nbinsx=20, name='SI', marker_color='#1f77b4'),
        row=1, col=1,
    )
    fig.add_vline(x=0.5, line_dash='dash', line_color='red', row=1, col=1)
    # (1, 2) SI vs ρ_med 散点
    fig.add_trace(
        go.Scatter(
            x=df_si['rho_median'], y=df_si['SI'],
            mode='markers',
            marker=dict(
                size=df_si['n_stocks'].clip(lower=5, upper=50),
                color=df_si['SI'], colorscale='RdYlGn', showscale=True,
                colorbar=dict(title='SI', x=0.46, len=0.5, y=0.78),
            ),
            text=df_si['sector_name'],
            hovertemplate='%{text}<br>ρ_med=%{x:.3f}<br>SI=%{y:.3f}<br>n=%{marker.size}<extra></extra>',
            name='行业',
        ),
        row=1, col=2,
    )
    # (2, 1) Top 12 强 SI 行业
    top_strong = df_si.head(12).iloc[::-1]  # 反转让最强在最上
    fig.add_trace(
        go.Bar(
            x=top_strong['SI'], y=top_strong['sector_name'],
            orientation='h',
            marker=dict(color=top_strong['SI'], colorscale='Greens', cmin=0, cmax=1),
            text=[f"SI={s:.2f}" for s in top_strong['SI']],
            textposition='outside',
            name='强 SI',
        ),
        row=2, col=1,
    )
    # (2, 2) Top 12 弱 SI 行业
    top_weak = df_si.tail(12)
    fig.add_trace(
        go.Bar(
            x=top_weak['SI'], y=top_weak['sector_name'],
            orientation='h',
            marker=dict(color=top_weak['SI'], colorscale='Reds', cmin=0, cmax=1),
            text=[f"SI={s:.2f}" for s in top_weak['SI']],
            textposition='outside',
            name='弱 SI',
        ),
        row=2, col=2,
    )
    fig.update_layout(
        height=900, width=1400,
        title_text='行业稳定性指数 SI(v4.7)',
        showlegend=False,
    )
    fig.update_xaxes(title_text='SI', row=1, col=1)
    fig.update_xaxes(title_text='ρ_med', row=1, col=2)
    fig.update_yaxes(title_text='频数', row=1, col=1)
    fig.update_yaxes(title_text='SI', row=1, col=2)
    fig.update_xaxes(title_text='SI', row=2, col=1)
    fig.update_xaxes(title_text='SI', row=2, col=2)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

### Step 1.8: main() 末尾追加 SI 写出调用

打开 `backtrace/dynamics/dynamics_eigen_analysis.py:400 main()`,在末尾追加(在 `tq.close()` 之前,如果有的话):

```python
    # --- v4.7 行业稳定性指数 SI ---
    from pathlib import Path
    print(f'[eigen] 计算行业稳定性指数 SI ...')
    name_lookup = _industry_name_lookup(args.sw2_members)
    df_si = compute_sector_stability(df, name_lookup=name_lookup)
    si_csv = os.path.join(CSV_OUT_DIR, 'sector_si.csv')
    df_si.to_csv(si_csv, index=False, encoding='utf-8-sig')
    print(f'[eigen] 💾 {si_csv} ({len(df_si)} 行业)')
    out_path = Path(args.output)
    si_html = str(out_path.with_name(out_path.stem + '_sector_si' + out_path.suffix))
    si_txt = str(out_path.with_name(out_path.stem + '_sector_si_summary.txt'))
    build_sector_si_html(df_si, si_html)
    print(f'[eigen] 🌐 {si_html}')
    write_sector_si_summary(df_si, si_txt)
    print(f'[eigen] 📝 {si_txt}')
```

**关键**: `si_html` / `si_txt` 用 `Path(args.output).with_name(stem + '_sector_si' + suffix)` 派生,避免 `replace` no-op。

### Step 1.9: 更新 `backtrace/dynamics/README.md` §3.7

打开 `backtrace/dynamics/README.md`,在 §3.6 之后追加 §3.7:

```markdown
### 3.7 v4.7 — 行业稳定性指数 SI (Sector Stability Index)

`dynamics_eigen_analysis.py` 默认运行后追加产出:
- `data/dynamics/sector_si.csv` (9 列)
- `backtrace/outputs/dynsys_sector_si.html` (4 子图 plotly)
- `backtrace/outputs/dynsys_sector_si_summary.txt` (UTF-8 中文 top 12 强/弱)

SI 定义: `SI = 0.5·ρ_health + 0.2·damping_health + 0.3·wedge_health`,权重集中在 `SI_WEIGHTS = (0.5, 0.2, 0.3)`(`dynamics_eigen_analysis.py` 顶部常量)。

行业筛选(沿用 v4.3): `n_stocks >= 50` 强 / `n_stocks >= 30` 弱;两者都 < 5 行业时上层标 `low-confidence`。
```

### Step 1.10: 跑全部 35 测试,确认通过

```bash
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

**Expected:** 35 passed (30 旧 + 5 新)

### Step 1.11: 端到端冒烟(2 路径)

```bash
# 1. 默认冒烟 — 验证 SI HTML / TXT / CSV 正常产出
cd "C:\Users\yellow\mcp\qtTdx"
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_eigen_analysis.py --limit 50
# 期待 exit 0,产出:
#   data/dynamics/sector_si.csv
#   backtrace/outputs/dynsys_sector_si.html
#   backtrace/outputs/dynsys_sector_si_summary.txt

# 2. 验证文本汇总可在 Windows cat 下读
cat backtrace/outputs/dynsys_sector_si_summary.txt | head -20
# 期待:中文行业名 + Top 12 强 / 弱 列表 + SI 直方图分布
```

### Step 1.12: Commit

```bash
cd "C:\Users\yellow\mcp\qtTdx"
git add backtrace/dynamics/dynamics_eigen_analysis.py tests/test_dynamics_eigen.py backtrace/dynamics/README.md
git commit -m "feat(dynamics): v4.7 — 行业稳定性指数 SI (Sector Stability Index)

- 新增 compute_sector_stability(df) 单一指标加权 SI ∈ [0, 1]
  - 3 个 0-1 子分: ρ_health (权重 0.5) / damping_health (0.2) / wedge_health (0.3)
  - SI_WEIGHTS = (0.5, 0.2, 0.3) 集中常量,改 1 行
  - §3.3 阈值: n_stocks >= 50 强 / >= 30 弱;<5 行业时上层标 low-confidence
- build_sector_si_html 4 子图 plotly (直方图 / SI vs ρ / Top 12 强 / Top 12 弱)
- write_sector_si_summary UTF-8 文本汇总 (Windows cat 可读)
- main() 末尾 hook,默认运行,无需 flag
- README.md §3.7 文档同步
- 5 新测试 (35/35 pass),数学层 + 3 caller 0 改"
```

---

## 显式不做

- ❌ 行业 SI 时序(滚动 60 日)— v4.8 候选
- ❌ SI 与 forward return 的 IC 评估 — v4.8 候选
- ❌ 多维 SI dict(`rho_health` / `damping_health` / `wedge_inside` 独立展示)— v4.8 候选
- ❌ 交易所层 SI(SH / SZ / BJ)
- ❌ v6 受迫系统 + G(ω) 频率响应
- ❌ 修改 `analyze_eigenvalues` / `simulate_trajectory` 数学
- ❌ 修改 3 个现有 caller

## 验证清单

- [ ] Step 1.1: 5 测试写入
- [ ] Step 1.2: 5 测试失败(ImportError)
- [ ] Step 1.3: `compute_sector_stability` 实现(含 §3.3 n_stocks 阈值)
- [ ] Step 1.4: 4 测试通过(`test_sector_si_basic_shape` 等)
- [ ] Step 1.5: `write_sector_si_summary` 实现
- [ ] Step 1.6: 5 测试全通过
- [ ] Step 1.7: `build_sector_si_html` 实现
- [ ] Step 1.8: main() 末尾 hook(Path.with_name 派生)
- [ ] Step 1.9: README.md §3.7 同步
- [ ] Step 1.10: 35 tests pass
- [ ] Step 1.11: 端到端冒烟(2 路径)
- [ ] Step 1.12: commit

## 风险

| 风险 | 缓解 |
|---|---|
| 权重 0.5/0.2/0.3 是经验性,未来 IC 评估发现错位 | `SI_WEIGHTS = (0.5, 0.2, 0.3)` 集中 1 处常量,改 1 行 |
| 申万二级行业每年调整,样本跨期不一致 | `sector_si.csv` 留 CSV 时间戳(JSON 字段缺失,下游消费时按 mtime 过滤) |
| `eigen_summary.csv` 缺失(刚清缓存) | `compute_sector_stability` 接受空 df(返回空表);main() 端依赖 `load_kc_estimates` 既有 read |
| HTML 5.8MB 累积(已有 v4.3 8 子图 + v4.5 phase) | SI HTML 4 子图,预估 ~300 KB,gitignored 可接受 |
| 50 只阈值覆盖后 < 5 行业,SI 直方图稀疏 | 文本汇总 bin 5 段合并,在主图右上加 (N=...) |

## 与 v4.3 / v4.6 的关系

| 版 | commit | 主题 |
|---|---|---|
| v4.3 | `b08f627` | 全市场经验分布 8 子图 HTML |
| v4.4 | `180f01c` | (1,4) bar chart label 增强 |
| v4.5 | `49ffd98` | (k, c) phase plot + 11 类颜色 |
| v4.6 | `4e8c265` | T3.3 / T3.4 / T3.5 polish |
| **v4.7** | (本次) | **行业稳定性指数 SI** |

v4.7 是"行业层最终的量化" — 之前 v4.3 行业聚合只到 ρ-med / 楔形距离,本轮加 1 个综合指标,把"行业整体稳不稳定"用一个 0-1 数字回答。
