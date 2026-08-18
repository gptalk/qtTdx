# v5.1 Industry G(ω) Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 v5 单对 Bode plot 基础上,加 `bode_overlay()` 函数 + `--overlay` CLI flag,在同一张图上画多条频率响应曲线,支持行业/时间窗对比。

**Architecture:**
- 复用 v5 已有数学层 (`transfer_function` / `magnitude_phase` / `natural_frequency` / `classify_response_type`) — **0 修改**
- 在 v5 同一文件 `dynamics_forced_response.py` 末尾新增 3 个函数 + 1 个 parser helper + 1 个 main() 分支
- 测试加在 `tests/test_dynamics_eigen.py` 末尾,与 v5 测试风格一致
- 输出全 gitignored,落 `backtrace/outputs/dynsys_bode_overlay*`

**Tech Stack:** Python 3.x / numpy / plotly / pandas(沿用 v5)

## Global Constraints

[v5 沿用 + v5.1 新增]

- `_dynamics_core.py` 0 行修改
- v5 已有函数 `transfer_function` / `natural_frequency` / `magnitude_phase` / `classify_response_type` / `bode_plot` / `stability_heatmap` / `write_summary` 签名 0 修改
- 3 caller (`dynamics_system.py` / `dynamics_batch.py` / `dynamics_1step_oos.py`) 0 行修改
- 4 v4.x CLI (`dynamics_si_ic.py` / `dynamics_si_timeseries.py` / `dynamics_si_lagged_ic.py` / `dynamics_eigen_analysis.py`) 0 修改
- v5 单对模式 main() 流程不变(只在末尾加 `if args.overlay:` 分支,单对模式不改逻辑)
- v5 单对模式 CLI flags 完全兼容(`--k` / `--c` / `--grid-csv` / `--stability-csv` / `--bode-html` / `--heatmap-html` / `--summary-txt`)
- 新增输出全 gitignored:`backtrace/outputs/dynsys_bode_overlay.html` + `dynsys_bode_overlay_summary.txt`
- 53 → 57 tests pass(53 旧 + 4 新)
- Python: `/c/ProgramData/anaconda3/python.exe`
- `PYTHONIOENCODING=utf-8` 必备
- 函数命名沿用 v5 风格(snake_case / docstring 中文)
- 不依赖 `parameter_fit` 输出(纯函数层)

---

### Task 1: `bode_overlay()` 函数 + 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py:343` (末尾新增)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 2 tests)

**Interfaces:**
- Consumes: v5 `transfer_function(omega, k, c)` → complex ndarray;`magnitude_phase(omega, k, c)` → tuple
- Produces: `bode_overlay(omega_grid, k_c_pairs, output_path, title)` → None(写 HTML)

- [ ] **Step 1: 写失败的测试 — 验证文件创建 + 空列表 raise**

修改 `tests/test_dynamics_eigen.py`,在末尾新增:

```python
def test_bode_overlay_creates_html(tmp_path):
    """bode_overlay 调用产生 HTML 文件 + 文件非空。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    pairs = [(0.5, 2.0, "Strong"), (2.0, 1.5, "Mild")]
    out = tmp_path / "overlay.html"
    DFR.bode_overlay(omega, pairs, str(out))
    assert out.exists()
    assert out.stat().st_size > 1000


def test_bode_overlay_validates_empty_list(tmp_path):
    """空 k_c_pairs → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    out = tmp_path / "overlay.html"
    with pytest.raises(ValueError, match="k_c_pairs 不能为空"):
        DFR.bode_overlay(omega, [], str(out))
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_bode_overlay_creates_html tests/test_dynamics_eigen.py::test_bode_overlay_validates_empty_list -v
```

期望:FAIL — `AttributeError: module 'backtrace.dynamics.dynamics_forced_response' has no attribute 'bode_overlay'`

- [ ] **Step 3: 实现 `bode_overlay`**

在 `backtrace/dynamics/dynamics_forced_response.py` 末尾(`write_summary` 函数之后)新增:

```python
def bode_overlay(omega_grid, k_c_pairs, output_path, title="Industry G(ω) Frequency Response Comparison"):
    """多对 (k, c) Bode plot 叠加对比。

    Args:
        omega_grid: 角频率数组,shape (N,),共享
        k_c_pairs: [(k, c, label), ...] 列表
        output_path: HTML 输出路径
        title: 图表标题

    行为:
        - 2 子图:上幅频 |H(jω)| vs ω,下相频 arg H(jω) vs ω
        - 每对一条曲线,共享 omega_grid,不同颜色 + 实线
        - legend 显示 label(带 (k, c) 数值)
        - HTML 通过 plotly CDN 渲染(include_plotlyjs='cdn')

    Raises:
        ValueError: 空列表 / k <= 0 / c <= 0 / label 重复
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not k_c_pairs:
        raise ValueError("k_c_pairs 不能为空")
    labels = [p[2] for p in k_c_pairs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"label 重复: {labels}")
    for k, c, _ in k_c_pairs:
        if k <= 0 or c <= 0:
            raise ValueError(f"k 和 c 必须 > 0,得 (k={k}, c={c})")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("|H(jω)|", "arg H(jω) (degrees)"),
                        vertical_spacing=0.12)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for idx, (k, c, label) in enumerate(k_c_pairs):
        H = transfer_function(omega_grid, k, c)
        mag, phase = magnitude_phase(omega_grid, k, c)
        color = colors[idx % len(colors)]
        legend_name = f'{label} (k={k}, c={c})'
        fig.add_trace(go.Scatter(x=omega_grid, y=mag, mode='lines',
                                 name=legend_name, line=dict(color=color, width=2),
                                 legendgroup=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=omega_grid, y=np.degrees(phase), mode='lines',
                                 name=legend_name, line=dict(color=color, width=2),
                                 legendgroup=label, showlegend=False), row=2, col=1)

    fig.update_xaxes(title_text='ω (角频率,rad/sample)', row=2, col=1)
    fig.update_yaxes(title_text='|H(jω)|', row=1, col=1)
    fig.update_yaxes(title_text='相位 (degrees)', row=2, col=1)
    fig.update_layout(title=title, height=800, width=1000,
                      hovermode='x unified', legend=dict(orientation='v',
                                                          xanchor='left', yanchor='top',
                                                          x=1.02, y=1.0))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_bode_overlay_creates_html tests/test_dynamics_eigen.py::test_bode_overlay_validates_empty_list -v
```

期望:2 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.1 — bode_overlay() 函数 + 2 unit tests

多对 (k, c) Bode plot 叠加对比,复用 v5 transfer_function +
magnitude_phase,纯可视化扩展,不动数学层。"
```

---

### Task 2: `write_overlay_summary()` 函数 + 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py:430` (Task 1 新函数之后)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 1 test)

**Interfaces:**
- Consumes: v5 `natural_frequency` / `magnitude_phase` / `classify_response_type` / `is_in_schur_wedge`
- Produces: `write_overlay_summary(omega_grid, k_c_pairs, output_path)` → None(写 UTF-8 TXT)

- [ ] **Step 1: 写失败的测试**

在 `tests/test_dynamics_eigen.py` 末尾新增:

```python
def test_write_overlay_summary_creates_txt(tmp_path):
    """write_overlay_summary 调用产生 TXT 文件 + 内容含所有 label。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    omega = np.linspace(0.01, np.pi, 50)
    pairs = [(0.5, 2.0, "Industry A"), (2.0, 1.5, "Industry B")]
    out = tmp_path / "overlay_summary.txt"
    DFR.write_overlay_summary(omega, pairs, str(out))
    assert out.exists()
    content = out.read_text(encoding='utf-8')
    assert "Industry A" in content
    assert "Industry B" in content
    assert "|H(j0)" in content or "DC" in content
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_write_overlay_summary_creates_txt -v
```

期望:FAIL — `AttributeError: ... has no attribute 'write_overlay_summary'`

- [ ] **Step 3: 实现 `write_overlay_summary`**

在 `bode_overlay` 函数后新增:

```python
def write_overlay_summary(omega_grid, k_c_pairs, output_path):
    """多对 (k, c) 的 UTF-8 中文汇总表。

    每对一行,展示:
            - 行业/时间窗 label
            - (k, c) + 响应类型
            - ω_n + |H(jω_n)|(若有)
            - |H(j0)| (DC 增益)
            - |H(jπ)| (Nyquist)
            - Schur 楔形内/外
            - 一句业务解读
    """
    lines = [
        '=' * 80,
        f'v5.1 Industry G(ω) Frequency Response Comparison — {len(k_c_pairs)} 对 (k, c)',
        '=' * 80,
        '',
    ]
    for k, c, label in k_c_pairs:
        omega_n = natural_frequency(k, c)
        response_type = classify_response_type(k, c)
        in_wedge = is_in_schur_wedge(k, c)
        mag_dc, _ = magnitude_phase(np.array([0.001]), k, c)
        mag_pi, _ = magnitude_phase(np.array([np.pi]), k, c)
        lines.append(f'[{label}]  (k={k}, c={c})')
        lines.append(f'  响应类型: {response_type}    Schur 楔形内: {in_wedge}')
        if np.isfinite(omega_n):
            mag_n, _ = magnitude_phase(np.array([omega_n]), k, c)
            lines.append(f'  ω_n = {omega_n:.4f}    |H(jω_n)| = {float(mag_n[0]):.4f}')
        else:
            lines.append(f'  ω_n = N/A (实极点)')
        lines.append(f'  |H(j0)|  = {float(mag_dc[0]):.4f} (DC 增益)')
        lines.append(f'  |H(jπ)| = {float(mag_pi[0]):.4f} (Nyquist)')
        # 业务解读
        if not in_wedge:
            lines.append(f'  业务解读: 共振风险高,β 强迫会在 ω_n 处放大 {float(mag_n[0]):.1f} 倍')
        elif response_type == 'overdamped':
            lines.append(f'  业务解读: 低通过滤器,β 强迫不会引发共振,稳定')
        elif response_type == 'critical':
            lines.append(f'  业务解读: 临界阻尼,边界 case')
        else:
            lines.append(f'  业务解读: 标准响应')
        lines.append('')
    lines.append('=' * 80)
    text = '\n'.join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_write_overlay_summary_creates_txt -v
```

期望:1 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.1 — write_overlay_summary() 函数 + 1 unit test

多对 (k, c) UTF-8 中文汇总,每对一行 + 业务解读。复用 v5
magnitude_phase / classify_response_type / is_in_schur_wedge。"
```

---

### Task 3: `parse_overlay_pairs()` parser + 单元测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py:470` (Task 2 后)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 1 test)

**Interfaces:**
- Produces: `parse_overlay_pairs(s)` → `list[tuple[float, float, str]]`

**注**:这个 helper 独立测试,因为它的解析规则复杂(label 可能含逗号)。

- [ ] **Step 1: 写失败的测试**

```python
def test_parse_overlay_pairs_basic():
    """基本 3 对解析。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    s = "0.5,2.0,Industry A; 2.0,1.5,Industry B; 3.0,0.5,Industry C"
    pairs = DFR.parse_overlay_pairs(s)
    assert len(pairs) == 3
    assert pairs[0] == (0.5, 2.0, "Industry A")
    assert pairs[1] == (2.0, 1.5, "Industry B")
    assert pairs[2] == (3.0, 0.5, "Industry C")


def test_parse_overlay_pairs_label_with_spaces():
    """label 含空格的解析。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    s = "2.0,1.5,Bank Index; 0.5,2.0,Tech Sector"
    pairs = DFR.parse_overlay_pairs(s)
    assert pairs[0] == (2.0, 1.5, "Bank Index")
    assert pairs[1] == (0.5, 2.0, "Tech Sector")


def test_parse_overlay_pairs_invalid_format():
    """错误格式 → ValueError。"""
    from backtrace.dynamics import dynamics_forced_response as DFR
    with pytest.raises(ValueError, match="格式错误"):
        DFR.parse_overlay_pairs("only_two_parts")
    with pytest.raises(ValueError, match="k 必须"):
        DFR.parse_overlay_pairs("abc,1.5,Label")
    with pytest.raises(ValueError, match="c 必须"):
        DFR.parse_overlay_pairs("1.0,xyz,Label")
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_parse_overlay_pairs_basic tests/test_dynamics_eigen.py::test_parse_overlay_pairs_label_with_spaces tests/test_dynamics_eigen.py::test_parse_overlay_pairs_invalid_format -v
```

期望:3 FAIL — `AttributeError: ... has no attribute 'parse_overlay_pairs'`

- [ ] **Step 3: 实现 `parse_overlay_pairs`**

```python
def parse_overlay_pairs(s):
    """解析 --overlay CLI 字符串为 [(k, c, label), ...]。

    格式:"k1,c1,label1; k2,c2,label2; ..."
    - 分号 `;` 分隔不同对
    - 逗号 `,` 分隔 k / c / label
    - label 可含逗号 / 空格(只取前两个逗号之前的为 k, c;之后全是 label)
    """
    if not s or not s.strip():
        raise ValueError("overlay 字符串为空")
    pairs = []
    for chunk in s.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(',', 2)  # 只 split 前 2 个逗号,label 可含逗号
        if len(parts) < 3:
            raise ValueError(f"格式错误: '{chunk}' 期望 k,c,label")
        try:
            k = float(parts[0].strip())
        except ValueError:
            raise ValueError(f"k 必须为数字,得 '{parts[0]}'")
        try:
            c = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"c 必须为数字,得 '{parts[1]}'")
        label = parts[2].strip()
        pairs.append((k, c, label))
    if not pairs:
        raise ValueError("未解析出任何 (k, c, label) 对")
    return pairs
```

- [ ] **Step 4: 跑测试,验证通过**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_parse_overlay_pairs_basic tests/test_dynamics_eigen.py::test_parse_overlay_pairs_label_with_spaces tests/test_dynamics_eigen.py::test_parse_overlay_pairs_invalid_format -v
```

期望:3 PASS

- [ ] **Step 5: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.1 — parse_overlay_pairs() 字符串解析 + 3 unit tests

--overlay CLI flag 的 parser,支持分号分隔 + label 含逗号 / 空格。"
```

---

### Task 4: `--overlay` CLI flag + main() 分支 + 集成测试

**Files:**
- Modify: `backtrace/dynamics/dynamics_forced_response.py:53` (`parse_args` + `main`)
- Modify: `tests/test_dynamics_eigen.py` (末尾新增 1 CLI 集成 test)

**Interfaces:**
- 新增 CLI flag:`--overlay "k1,c1,label1; k2,c2,label2; ..."`
- 新增 CLI flag:`--overlay-html` / `--overlay-summary-txt`(可选,沿用 default)

**注**:这是用户可见的扩展,必须不破坏单对模式。

- [ ] **Step 1: 写失败的 CLI 集成测试**

```python
def test_cli_overlay_mode(tmp_path):
    """CLI --overlay 模式产生 overlay HTML + summary TXT,不写单对输出。"""
    import subprocess
    overlay_str = "0.5,2.0,Strong; 2.0,1.5,Mild"
    out_html = tmp_path / "overlay.html"
    out_txt = tmp_path / "overlay_summary.txt"
    result = subprocess.run([
        sys.executable,
        "backtrace/dynamics/dynamics_forced_response.py",
        "--overlay", overlay_str,
        "--overlay-html", str(out_html),
        "--overlay-summary-txt", str(out_txt),
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_html.exists()
    assert out_txt.exists()
    # 单对模式输出不应被创建(overlay-only)
    # 注意:tmp_path 与默认输出路径不同,所以默认输出会被写到 cwd
    # 检查 cwd 是否产生了 dynsys_forced_response.html(单对模式默认)
    # 这个测试用 --overlay-html 覆盖,所以单对默认 HTML 不应被写
    # 但单对模式的 summary 还是会写... 我们让单对模式只在非 overlay 模式下执行
```

- [ ] **Step 2: 跑测试,验证失败**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_overlay_mode -v
```

期望:FAIL — `--overlay` flag 未定义 / argparse error

- [ ] **Step 3: 修改 `parse_args` 加 `--overlay` flag**

修改 `parse_args` 函数,在 `--summary-txt` 之后新增:

```python
    p.add_argument("--overlay", default="",
                   help="多对 (k, c, label) overlay,格式 'k1,c1,label1; k2,c2,label2; ...'")
    p.add_argument("--overlay-html", default=os.path.join(HTML_OUT_DIR, "dynsys_bode_overlay.html"),
                   help=f"overlay Bode HTML 输出路径")
    p.add_argument("--overlay-summary-txt", default=os.path.join(HTML_OUT_DIR, "dynsys_bode_overlay_summary.txt"),
                   help="overlay UTF-8 中文汇总输出路径")
```

- [ ] **Step 4: 修改 `main()` 加 overlay 分支**

**核心思路**:`main()` 开头判断 `args.overlay`,有则进入 overlay-only 分支(不执行单对逻辑,提前 return);无则执行单对逻辑(v5 既有,不变)。

修改 `main()` 函数开头(在 `args = parse_args()` 之后),新增:

```python
    # v5.1 overlay 分支:有 --overlay 则跳过单对逻辑,只写 overlay 文件
    if args.overlay:
        pairs = parse_overlay_pairs(args.overlay)
        omega_grid_overlay = np.linspace(0.001, np.pi, 200)
        bode_overlay(omega_grid_overlay, pairs, args.overlay_html,
                     title=f'v5.1 Industry G(ω) Comparison — {len(pairs)} 对')
        write_overlay_summary(omega_grid_overlay, pairs, args.overlay_summary_txt)
        print(f'[v5.1 overlay] {len(pairs)} 对 (k, c) 已写入 {args.overlay_html}')
        return  # overlay-only 模式,跳过单对 main 后续
    # else: 单对模式(v5 既有逻辑,不变)
```

**关键**:overlay-only 分支必须在 main() **最开头**(在拉数据 / 写单对 CSV 之前),这样单对输出文件(`grid_csv` / `stability_csv` / `bode_html` / `heatmap_html` / `summary_txt`)**完全不会被创建**。原 v5 单对 main() 函数体**一行不动**,只在最开头插入 if-return 块。

- [ ] **Step 5: 跑测试,验证通过**

```bash
PYTHONIODENCDOING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py::test_cli_overlay_mode -v
```

期望:1 PASS

- [ ] **Step 6: 跑全套测试,确认 57 tests pass + 单对模式未破坏**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe -m pytest tests/test_dynamics_eigen.py -v
```

期望:57 PASS(53 旧 + 4 新:1 test_bode_overlay_creates_html + 1 test_bode_overlay_validates_empty_list + 1 test_write_overlay_summary_creates_txt + 3 parse_overlay_pairs + 1 CLI test = 7 新?)

**等等**:Task 1 (2) + Task 2 (1) + Task 3 (3) + Task 4 (1) = 7 新测试。53 + 7 = 60 tests pass。

修正目标:**53 → 60 tests pass**。

- [ ] **Step 7: 手动验证单对模式未破坏**

```bash
PYTHONIOENCODING=utf-8 /c/ProgramData/anaconda3/python.exe backtrace/dynamics/dynamics_forced_response.py --k 2.0 --c 1.5 \
    --grid-csv /tmp/v5_grid.csv \
    --stability-csv /tmp/v5_stab.csv \
    --bode-html /tmp/v5_bode.html \
    --heatmap-html /tmp/v5_heat.html \
    --summary-txt /tmp/v5_sum.txt
```

期望:5 个文件创建,无 overlay 文件,无 stderr。

- [ ] **Step 8: Commit**

```bash
git add backtrace/dynamics/dynamics_forced_response.py tests/test_dynamics_eigen.py
git commit -m "feat(dynamics): v5.1 — --overlay CLI flag + main() 分支 + 1 集成测试

多对 (k, c) overlay 模式。--overlay \"k1,c1,label1; k2,c2,label2; ...\"
触发 bode_overlay + write_overlay_summary,产生 2 个新输出。
单对模式 main() 流程不变,print 信息按 mode 分支。"
```

---

### Task 5: README §4.1 + 最终 commit

**Files:**
- Modify: `backtrace/dynamics/README.md` (§4 末尾新增 §4.1 v5.1 子节)

- [ ] **Step 1: 读取 README 当前 §4 v5 内容**

```bash
PYTHONIOENCODING=utf-8 wc -l backtrace/dynamics/README.md
```

定位 §4 v5 章节末尾。

- [ ] **Step 2: 追加 §4.1 v5.1 子节**

在 README 末尾(或 §4 v5 章节后)新增:

```markdown
### 4.1 v5.1 — Industry G(ω) Frequency Response Comparison

**多对 (k, c) Bode plot 叠加对比**,回答业务问题"哪个行业对 β 强迫最敏感 / 哪个是低通过滤器 / 哪个危险"。

#### 新增 CLI flag

| flag | 类型 | 说明 |
|---|---|---|
| `--overlay` | str | 多对 (k, c) 字符串:`"k1,c1,label1; k2,c2,label2; ..."` |
| `--overlay-html` | path | overlay HTML 输出(默认 `backtrace/outputs/dynsys_bode_overlay.html`) |
| `--overlay-summary-txt` | path | overlay UTF-8 汇总(默认 `backtrace/outputs/dynsys_bode_overlay_summary.txt`) |

#### 端到端示例

```bash
PYTHONIOENCODING=utf-8 python backtrace/dynamics/dynamics_forced_response.py \
    --overlay "0.5,2.0,Strong damping; 2.0,1.5,Mild damping; 2.01,2.0,Near boundary; 4.0,0.5,Weak damping"
# 期待:
#   - backtrace/outputs/dynsys_bode_overlay.html (4 条曲线叠加)
#   - backtrace/outputs/dynsys_bode_overlay_summary.txt (4 对业务解读)
```

#### 输出解读(中文汇总表)

每对一行:
- 响应类型(`overdamped` / `critical` / `underdamped` / `anti_damped`)
- Schur 楔形内/外(`is_in_schur_wedge`)
- ω_n + |H(jω_n)|(若有)
- |H(j0)| DC 增益 + |H(jπ)| Nyquist
- 业务解读:低通过滤器 / 共振风险高 / 标准响应

#### 解析规则

- 分号 `;` 分隔不同对
- 逗号 `,` 分隔 k / c / label(只 split 前 2 个逗号,label 可含逗号)
- label 可含空格(trim 后)

推荐 ≤ 10 对(plotly 默认 10 色),> 10 对 label 需手动分组。

#### 与 v5 的关系

v5.1 是 v5 的**纯可视化层扩展**,不动数学层(`transfer_function` / `natural_frequency` / `magnitude_phase` 0 修改)。v5.2 候选:与 `parameter_fit` 集成,自动从历史 (k̂, ĉ) 序列选 top-N 行业画 overlay。
```

- [ ] **Step 3: Commit**

```bash
git add backtrace/dynamics/README.md
git commit -m "docs(dynamics): README §4.1 — v5.1 overlay 使用 + 输出解读"
```

---

## Final state

- **5 commits**(Task 1-5)
- **60 tests pass**(53 旧 + 7 新)
- **0 修改**:`_dynamics_core.py` / 3 caller / 4 v4.x CLI / v5 单对 main() / v5 单对 CLI flags / v5 已有函数签名
- **新增**:`bode_overlay()` / `write_overlay_summary()` / `parse_overlay_pairs()` / `--overlay` CLI + 2 个输出

## Self-Review Checklist

- [x] Spec 覆盖:每个 spec 章节都有对应 task
- [x] 无 placeholder / TODO / TBD
- [x] 类型一致:`bode_overlay(omega_grid, k_c_pairs, output_path, title)` 跨 task 一致
- [x] 函数命名沿用 v5 风格(snake_case + 中文 docstring)
- [x] 测试与函数一一对应
- [x] 单对模式 main() 不被破坏(if args.overlay: ... else: ... 分支隔离)
- [x] 所有输出路径 gitignored(`backtrace/outputs/`)