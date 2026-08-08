# `data/` 本地日线缓存 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立仓库根 `data/` 作为沪深全市场 500 交易日日线的本地镜像,并修复 `tsfresh_pipeline._try_local_csv` 因读写路径分家而失效的回退分支。

**Architecture:** 新增 `backtrace/common/data_store.py` 独占「本地 CSV 放哪、叫什么、怎么读写」这一件事(纯文件 IO,不 import TQ);新增 `backtrace/data_fetch/fetch_daily.py` 只做拉取编排(universe、分批、重试、manifest),落盘一律经由 `data_store`;`_try_local_csv` 退化为对 `data_store.load_daily` 的一行委托。读写共用 `csv_path()`,路径不一致在结构上不再可能发生。

**Tech Stack:** Python 3 / pandas 2.3.3 / pytest 8.4.2 / 通达信 TQ 接口(`tqcenter.tq`)

**Spec:** [docs/superpowers/specs/2026-08-08-data-daily-cache-design.md](../specs/2026-08-08-data-daily-cache-design.md)

## Global Constraints

- **解释器必须是** `C:/Users/yellow/.conda/envs/venv/python.exe` —— 只有这个环境装了 tsfresh 0.21.2 / vectorbt 1.0.0 / sklearn 1.8.0。`C:/ProgramData/anaconda3/python.exe` 缺 tsfresh,导入 `tsfresh_pipeline` 会直接 `ModuleNotFoundError`。下文所有命令用 `$PY` 代指该路径。
- **所有命令加 `PYTHONIOENCODING=utf-8`** —— Windows GBK 终端会让中文 print 直接 `UnicodeEncodeError`(CLAUDE.md 已知陷阱)。
- **从仓库根 `C:/Users/yellow/mcp/qtTdx` 运行**,不要 `cd` 进子目录。
- **`backtrace/common/` 没有 `__init__.py`**,靠 Python 3 隐式命名空间包工作。新增模块**不要**加 `__init__.py`,保持现状。
- **导入约定**:`backtrace/` 在 `sys.path` 上时 `from common import xxx`。子目录脚本用 CLAUDE.md 的最小模板把 `BACKTRACE_DIR` 插进 `sys.path`。
- **CSV schema 固定**:无名日期索引 + `Open,High,Low,Close,Volume,Amount`,`encoding='utf-8'`。**不发明新格式** —— 现有 `backtrace/outputs/*_daily.csv` 就是这个格式。
- **文件名固定**:`{code}_daily.csv`,`.` 替换为 `_`(`000001.SH` → `000001_SH_daily.csv`)。
- **改公开函数须同步更新** [docs/api.md](../../api.md) —— CLAUDE.md 顶部的硬性要求。
- **500 交易日 / 780 自然日**:实测交易日占比 0.670(`000001_SH_daily.csv` 181 行 / 270 天),500 / 0.670 ≈ 746,取 780 留余量。
- **TQ 空数据必须当错误**:TQ 客户端未启动时 `get_market_data` 会「假装成功」返回空数据。每批必须验 `df['Close'].shape[1] > 0`,为空则中止整轮非零退出 —— 静默写空 CSV 覆盖好数据比崩溃危险得多。

---

## File Structure

| 文件 | 职责 | 状态 |
|---|---|---|
| `backtrace/common/tsfresh_config.py` | 加 `DATA_DIR` 常量 | 修改(第 46 行后) |
| `backtrace/common/data_store.py` | **本地缓存唯一真相**:路径 / 读写 / manifest | 新建 |
| `backtrace/common/tsfresh_pipeline.py` | `_try_local_csv` 改为委托 | 修改(第 50-55 行) |
| `backtrace/data_fetch/fetch_daily.py` | 拉取编排:universe / 分批 / 重试 / CLI | 新建 |
| `tests/conftest.py` | 把 `backtrace/` 插入 `sys.path` | 新建 |
| `tests/test_data_store.py` | data_store 往返 + 路径同源防回归 | 新建 |
| `tests/test_fallback.py` | 回退真的修好了 | 新建 |
| `tests/test_fetch_helpers.py` | 纯函数:ST 过滤 / 分批 / 天数换算 / 截尾 | 新建 |
| `docs/api.md` | 补 `data_store` + `DATA_DIR` + `fetch_daily` | 修改 |
| `CLAUDE.md` | 补 `data/` 目录与 `data_fetch/` 说明 | 修改 |

**为什么 manifest 归 `data_store` 而非 `fetch_daily`**:spec §4.2 把 `manifest.json` 定义为磁盘布局的一部分,而 `data_store` 的职责就是「本地缓存长什么样」。放这里可让 manifest 路径与 CSV 路径同源,且能在无 TQ 环境下测试。这是对 spec §3.1 那 4 个函数的扩展(共 7 个),已核对无冲突。

---

## Task 1: `data_store.py` —— 本地缓存的唯一真相

**Files:**
- Modify: `backtrace/common/tsfresh_config.py:46`(在 `OUTPUTS_DIR` 后追加)
- Create: `backtrace/common/data_store.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data_store.py`
- Modify: `docs/api.md`

**Interfaces:**
- Consumes: `tsfresh_config.BACKTRACE_DIR`(已存在,`backtrace/` 绝对路径)
- Produces:
  - `tsfresh_config.DATA_DIR: str` —— 仓库根 `data/` 绝对路径
  - `data_store.KINDS: tuple = ('stocks', 'sectors', 'indices')`
  - `data_store.DATA_DIR: str` —— 模块级变量,测试可 monkeypatch
  - `data_store.csv_path(code: str, kind: str = 'stocks') -> str`
  - `data_store.save_daily(code: str, df: pd.DataFrame, kind: str = 'stocks') -> str`
  - `data_store.load_daily(code: str) -> pd.DataFrame | None`
  - `data_store.has_daily(code: str) -> bool`
  - `data_store.manifest_path() -> str`
  - `data_store.load_manifest() -> dict`
  - `data_store.save_manifest(man: dict) -> str`

- [ ] **Step 1: 给 `tsfresh_config.py` 加 `DATA_DIR`**

在 `backtrace/common/tsfresh_config.py` 第 46 行 `OUTPUTS_DIR = ...` 之后追加:

```python
DATA_DIR = os.path.join(os.path.dirname(BACKTRACE_DIR), 'data')                 # 仓库根 data/(gitignored) — 本地日线缓存,见 data_store.py
```

同时把第 12 行的 `- 输出:14 个公开常量` 改为 `- 输出:15 个公开常量`。

同时把第 26 行 `LOCAL_FALLBACK_CODES` 的行尾注释改掉(旧路径已废弃):

```python
LOCAL_FALLBACK_CODES = ['000001.SH', '002475.SZ']    # TQ 拉取失败时回退的本地 CSV 代码(须有对应 data/<kind>/<code>_daily.csv,见 data_store)
```

- [ ] **Step 2: 建 `tests/conftest.py`**

创建 `tests/conftest.py`:

```python
# -*- coding: utf-8 -*-
"""把 backtrace/ 插进 sys.path,使测试能 `from common import ...`(与脚本同一套导入约定)。"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE_DIR = os.path.join(REPO_ROOT, 'backtrace')
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
```

- [ ] **Step 3: 写失败的测试**

创建 `tests/test_data_store.py`:

```python
# -*- coding: utf-8 -*-
import os

import pandas as pd
import pytest

from common import data_store


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """把缓存根指向临时目录,避免测试污染真实 data/"""
    monkeypatch.setattr(data_store, 'DATA_DIR', str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_df():
    idx = pd.to_datetime(['2026-08-05', '2026-08-06', '2026-08-07'])
    return pd.DataFrame({
        'Open':   [10.0, 10.5, 10.2],
        'High':   [10.8, 10.9, 10.6],
        'Low':    [9.9, 10.1, 10.0],
        'Close':  [10.5, 10.2, 10.4],
        'Volume': [1000000.0, 1200000.0, 900000.0],
        'Amount': [10000000.0, 12000000.0, 9000000.0],
    }, index=idx)


def test_csv_path_uses_underscore_and_kind_subdir(tmp_store):
    p = data_store.csv_path('000001.SH', 'indices')
    assert p.endswith(os.path.join('indices', '000001_SH_daily.csv'))


def test_csv_path_rejects_unknown_kind(tmp_store):
    with pytest.raises(ValueError):
        data_store.csv_path('000001.SH', 'bogus')


def test_save_then_load_roundtrip(tmp_store, sample_df):
    data_store.save_daily('000001.SZ', sample_df, 'stocks')
    got = data_store.load_daily('000001.SZ')
    pd.testing.assert_frame_equal(got, sample_df)


def test_load_searches_all_kinds(tmp_store, sample_df):
    data_store.save_daily('399001.SZ', sample_df, 'indices')
    assert data_store.load_daily('399001.SZ') is not None


def test_load_missing_returns_none(tmp_store):
    assert data_store.load_daily('999999.SZ') is None


def test_has_daily(tmp_store, sample_df):
    assert data_store.has_daily('600000.SH') is False
    data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert data_store.has_daily('600000.SH') is True


def test_read_write_share_one_path(tmp_store, sample_df):
    """防回归:写入路径必须正是 load 查找的路径 —— 这正是当前 bug 的根因"""
    written = data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert os.path.exists(written)
    assert data_store.load_daily('600000.SH') is not None


def test_save_leaves_no_tmp_file(tmp_store, sample_df):
    data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert list(tmp_store.rglob('*.tmp')) == []


def test_manifest_roundtrip(tmp_store):
    man = data_store.load_manifest()
    assert man['entries'] == {}
    man['entries']['000001.SZ'] = {'kind': 'stocks', 'rows': 500, 'status': 'ok'}
    data_store.save_manifest(man)
    assert data_store.load_manifest()['entries']['000001.SZ']['rows'] == 500


def test_manifest_leaves_no_tmp_file(tmp_store):
    data_store.save_manifest(data_store.load_manifest())
    assert list(tmp_store.rglob('*.tmp')) == []
```

- [ ] **Step 4: 跑测试确认失败**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_data_store.py -v
```

Expected: 收集阶段就 FAIL —— `ModuleNotFoundError: No module named 'common.data_store'`(模块还没建)。

- [ ] **Step 5: 写实现**

创建 `backtrace/common/data_store.py`:

```python
# -*- coding: utf-8 -*-
"""
本地日线缓存的唯一真相源:定义 CSV 放哪、叫什么、怎么读写。

约定/做法:
  - 读写共用 csv_path(),杜绝"写在 A、读在 B"的路径分家 bug
    (历史教训:_try_local_csv 读 backtrace/,而 CSV 实际写在 backtrace/outputs/,回退长期失效)
  - 纯文件 IO,不 import TQ —— 离线读取无需 TQ 客户端在场,也使本模块可独立测试
  - 原子写(.tmp + os.replace),中途 Ctrl-C 不留半截文件被下次当成有效缓存

磁盘布局:
  data/stocks/000001_SZ_daily.csv     沪深 A 股(去 ST/退市)
  data/sectors/880xxx_SH_daily.csv    申万二级 128 行业指数
  data/indices/000001_SH_daily.csv    上证综指 / 深证成指
  data/manifest.json                  每只票的行数/首末日期/拉取时间/失败原因

输入/输出:
  - 输入:股票代码 code、DataFrame
  - 输出:7 个公开函数

依赖:pandas, common.tsfresh_config

用法:
  from common import data_store
  data_store.save_daily('000001.SZ', df, 'stocks')
  df = data_store.load_daily('000001.SZ')      # 跨 kind 查找
"""
import json
import os

import pandas as pd

from common import tsfresh_config as C

# 模块级变量而非常量:测试用 monkeypatch.setattr(data_store, 'DATA_DIR', tmp) 重定向
DATA_DIR = C.DATA_DIR

KINDS = ('stocks', 'sectors', 'indices')

# CSV schema —— 与 backtrace/outputs/*_daily.csv 既有格式一致,不要改
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']


def _filename(code):
    """000001.SH -> 000001_SH_daily.csv"""
    return f"{code.replace('.', '_')}_daily.csv"


def csv_path(code, kind='stocks'):
    """路径的唯一真相。读和写都必须经过这里。"""
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一,收到 {kind!r}")
    return os.path.join(DATA_DIR, kind, _filename(code))


def save_daily(code, df, kind='stocks'):
    """原子写(先 .tmp 再 os.replace),返回落盘路径。"""
    path = csv_path(code, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, encoding='utf-8')
    os.replace(tmp, path)
    return path


def load_daily(code):
    """按 stocks -> sectors -> indices 顺序查找;都没有返回 None。

    跨目录查找是必要的:调用方(如 _try_local_csv)只有 code,不知道它是个股还是指数。
    """
    for kind in KINDS:
        p = csv_path(code, kind)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None


def has_daily(code):
    """任一 kind 目录下存在该 code 的 CSV(供断点续传判断)"""
    return any(os.path.exists(csv_path(code, k)) for k in KINDS)


def manifest_path():
    return os.path.join(DATA_DIR, 'manifest.json')


def load_manifest():
    """不存在时返回空骨架,调用方无需处理 None。"""
    p = manifest_path()
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'generated_at': None, 'trading_days': None, 'entries': {}}


def save_manifest(man):
    """原子写,返回落盘路径。"""
    p = manifest_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p
```

- [ ] **Step 6: 跑测试确认通过**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_data_store.py -v
```

Expected: 10 passed。

若 `test_save_then_load_roundtrip` 因 index dtype 报错,检查 `sample_df` 的索引确为 `DatetimeIndex`(`pd.to_datetime` 已保证)。

- [ ] **Step 7: 更新 `docs/api.md`**

在 `docs/api.md` 中新增一节(位置紧跟 `tsfresh_config` 那节之后):

```markdown
### backtrace/common/data_store.py

本地日线缓存的唯一真相源。纯文件 IO,不依赖 TQ。

| 函数 | 签名 | 说明 |
|---|---|---|
| `csv_path` | `(code, kind='stocks') -> str` | 路径唯一真相;`kind ∈ ('stocks','sectors','indices')`,非法值抛 `ValueError` |
| `save_daily` | `(code, df, kind='stocks') -> str` | 原子写(.tmp + os.replace),返回落盘路径 |
| `load_daily` | `(code) -> DataFrame \| None` | 按 stocks→sectors→indices 查找,都没有返回 None |
| `has_daily` | `(code) -> bool` | 任一 kind 下存在该 code |
| `manifest_path` | `() -> str` | `data/manifest.json` |
| `load_manifest` | `() -> dict` | 不存在时返回 `{'generated_at':None,'trading_days':None,'entries':{}}` |
| `save_manifest` | `(man) -> str` | 原子写 |

模块级 `DATA_DIR`(默认 `tsfresh_config.DATA_DIR`)可 monkeypatch 用于测试。
```

并在 `tsfresh_config` 常量表补一行:

```markdown
| `DATA_DIR` | 仓库根 `data/` 绝对路径 | 本地日线缓存根目录 |
```

- [ ] **Step 8: 提交**

```bash
cd C:/Users/yellow/mcp/qtTdx
git add backtrace/common/data_store.py backtrace/common/tsfresh_config.py tests/conftest.py tests/test_data_store.py docs/api.md
git commit -m "feat(data): 新增 data_store —— 本地日线缓存读写同源

读写共用 csv_path(),原子写防半截文件,manifest 归口同一模块。
纯文件 IO 不依赖 TQ,10 个测试全部离线可跑。"
```

---

## Task 2: 修复失效的本地 CSV 回退

**Files:**
- Modify: `backtrace/common/tsfresh_pipeline.py:37`(加 import)、`:50-55`(改函数体)
- Create: `tests/test_fallback.py`

**Interfaces:**
- Consumes: `data_store.load_daily(code) -> DataFrame | None`(Task 1)
- Produces: 无新公开接口 —— `load_ohlcva` / `load_sector` 的对外契约保持不变(有数据返回 DataFrame,没有返回 `None`)

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_fallback.py`:

```python
# -*- coding: utf-8 -*-
"""验证 TQ 不可用时,load_ohlcva 真能从 data/ 回退拿到数据。

修复前 _try_local_csv 读 backtrace/{code}_daily.csv,而 CSV 实际在别处,
所以这些测试在修复前必然失败(返回 None)。
"""
import pandas as pd
import pytest

from common import data_store
from common import tsfresh_pipeline as P


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(data_store, 'DATA_DIR', str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_df():
    idx = pd.to_datetime(['2026-08-05', '2026-08-06', '2026-08-07'])
    return pd.DataFrame({
        'Open':   [10.0, 10.5, 10.2],
        'High':   [10.8, 10.9, 10.6],
        'Low':    [9.9, 10.1, 10.0],
        'Close':  [10.5, 10.2, 10.4],
        'Volume': [1000000.0, 1200000.0, 900000.0],
        'Amount': [10000000.0, 12000000.0, 9000000.0],
    }, index=idx)


def test_load_ohlcva_falls_back_to_data_dir(tmp_store, sample_df):
    data_store.save_daily('000001.SH', sample_df, 'indices')
    got = P.load_ohlcva('000001.SH', use_tq=False)
    assert got is not None, "回退失效:data/indices/ 下有 CSV 却拿到 None"
    assert list(got.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    assert len(got) == 3


def test_load_ohlcva_returns_none_when_absent(tmp_store):
    assert P.load_ohlcva('999999.SZ', use_tq=False) is None


def test_fallback_finds_stock_kind_too(tmp_store, sample_df):
    data_store.save_daily('002475.SZ', sample_df, 'stocks')
    assert P.load_ohlcva('002475.SZ', use_tq=False) is not None
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_fallback.py -v
```

Expected: `test_load_ohlcva_falls_back_to_data_dir` 与 `test_fallback_finds_stock_kind_too` FAIL,断言消息为 "回退失效:data/indices/ 下有 CSV 却拿到 None"。
`test_load_ohlcva_returns_none_when_absent` 会 PASS(修复前后都返回 None)—— 这是预期的,它守的是另一个方向。

- [ ] **Step 3: 加 import**

`backtrace/common/tsfresh_pipeline.py` 第 37 行 `from common import tsfresh_config as C` 之后追加一行:

```python
from common import data_store
```

- [ ] **Step 4: 改 `_try_local_csv`**

把第 50-55 行整个函数替换为:

```python
def _try_local_csv(code):
    """回退路径:委托给 data_store(仓库根 data/)。

    历史教训:这里曾自己拼 backtrace/{code}_daily.csv,而写入方落在 backtrace/outputs/,
    路径分家导致回退长期静默失效。现在读写共用 data_store.csv_path()。
    """
    return data_store.load_daily(code)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_fallback.py tests/test_data_store.py -v
```

Expected: 13 passed。

- [ ] **Step 6: 把两个历史 CSV 搬进 `data/`(手工,spec §9 明确不写迁移脚本)**

```bash
cd C:/Users/yellow/mcp/qtTdx
mkdir -p data/indices data/stocks data/sectors
cp backtrace/outputs/000001_SH_daily.csv data/indices/
cp backtrace/outputs/002475_SZ_daily.csv data/stocks/
ls -la data/indices data/stocks
```

Expected: 两个文件各就各位。

- [ ] **Step 7: 用真实文件验证端到端回退**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backtrace'))
from common import tsfresh_pipeline as P
df = P.load_ohlcva('000001.SH', use_tq=False, verbose=True)
print('type:', type(df).__name__)
print('shape:', None if df is None else df.shape)
assert df is not None, 'FAIL: 回退仍然失效'
print('OK: 回退可用')
"
```

Expected: 打印 `[CSV] 000001.SH  181 行 ...`、`shape: (181, 6)`、`OK: 回退可用`。

- [ ] **Step 8: 提交**

```bash
cd C:/Users/yellow/mcp/qtTdx
git add backtrace/common/tsfresh_pipeline.py tests/test_fallback.py
git commit -m "fix(data): 修复失效的本地 CSV 回退路径

_try_local_csv 原先读 backtrace/{code}_daily.csv,而 CSV 实际写在
backtrace/outputs/,路径对不上导致回退分支永远返回 None,把
'TQ 客户端没启动' 伪装成下游的 NoneType 错误。

改为委托 data_store.load_daily,读写共用 csv_path。"
```

---

## Task 3: `fetch_daily.py` 纯函数层

**Files:**
- Create: `backtrace/data_fetch/fetch_daily.py`(仅纯函数部分,TQ 编排在 Task 4)
- Create: `tests/test_fetch_helpers.py`

**Interfaces:**
- Consumes: 无(纯函数,不依赖 Task 1/2)
- Produces:
  - `fetch_daily.TRADING_DAYS: int = 500`
  - `fetch_daily.BATCH_SIZE: int = 250`
  - `fetch_daily.TRADING_DAY_RATIO: float = 0.670`
  - `fetch_daily.INDEX_CODES: list = ['000001.SH', '399001.SZ']`
  - `fetch_daily.filter_st(items: list[dict]) -> list[str]`
  - `fetch_daily.chunked(seq: list, size: int = BATCH_SIZE) -> Iterator[list]`
  - `fetch_daily.calendar_days_for(trading_days: int = TRADING_DAYS) -> int`
  - `fetch_daily.trim_tail(df: pd.DataFrame, n: int = TRADING_DAYS) -> pd.DataFrame`

- [ ] **Step 1: 写失败的测试**

创建 `tests/test_fetch_helpers.py`:

```python
# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd

# data_fetch/ 不是 common/,需单独加进 path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'backtrace', 'data_fetch'))

import fetch_daily as FD


def test_filter_st_drops_st_and_delisted():
    items = [
        {'Code': '000001.SZ', 'Name': '平安银行'},
        {'Code': '000002.SZ', 'Name': 'ST康美'},
        {'Code': '000003.SZ', 'Name': '*ST夏利'},
        {'Code': '000004.SZ', 'Name': '乐视退'},
        {'Code': '000005.SZ', 'Name': '万科A'},
    ]
    assert FD.filter_st(items) == ['000001.SZ', '000005.SZ']


def test_filter_st_skips_malformed_entries():
    items = [None, {}, {'Name': '无代码'}, {'Code': '600000.SH', 'Name': '浦发银行'}]
    assert FD.filter_st(items) == ['600000.SH']


def test_chunked_splits_evenly():
    assert list(FD.chunked([1, 2, 3, 4, 5], size=2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty():
    assert list(FD.chunked([], size=250)) == []


def test_calendar_days_covers_500_trading_days():
    days = FD.calendar_days_for(500)
    # 实测交易日占比 0.670 -> 500/0.670 ≈ 746;必须留余量但别夸张
    assert 746 <= days <= 850, f"天数 {days} 不合理"


def test_trim_tail_keeps_last_n_sorted():
    idx = pd.to_datetime(['2026-08-07', '2026-08-05', '2026-08-06'])
    df = pd.DataFrame({'Close': [3.0, 1.0, 2.0]}, index=idx)
    got = FD.trim_tail(df, n=2)
    assert list(got['Close']) == [2.0, 3.0]      # 先排序再取尾


def test_trim_tail_keeps_short_series_intact():
    idx = pd.to_datetime(['2026-08-05', '2026-08-06'])
    df = pd.DataFrame({'Close': [1.0, 2.0]}, index=idx)
    assert len(FD.trim_tail(df, n=500)) == 2     # 不补齐、不丢弃


def test_index_codes_are_sse_and_szse():
    assert FD.INDEX_CODES == ['000001.SH', '399001.SZ']
```

- [ ] **Step 2: 跑测试确认失败**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_fetch_helpers.py -v
```

Expected: 收集阶段 FAIL —— `ModuleNotFoundError: No module named 'fetch_daily'`。

- [ ] **Step 3: 建目录与纯函数实现**

创建 `backtrace/data_fetch/fetch_daily.py`(此步只写到纯函数为止,TQ 部分 Task 4 追加):

```python
# -*- coding: utf-8 -*-
"""
拉取沪深全市场 + 申万二级行业指数 + 两大盘指数的日线,落盘到仓库根 data/。

职责边界:本模块只做「编排」—— universe、分批、重试、进度。
落盘一律经由 common.data_store,自己不拼任何路径。

用法:
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py            # 全量
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --limit 20 # 冒烟
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --force    # 忽略 manifest 重拉
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --probe    # 只探测 TQ 列表接口
"""
import os
import sys

import pandas as pd

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# ========================= 配置 =========================
TRADING_DAYS = 500        # 每只票保留的交易日数
BATCH_SIZE = 250          # 每批喂给 get_market_data 的代码数
                          # 依据:CLAUDE.md 记录 6000 只 timeout、~600 只可行,250 留足余量
TRADING_DAY_RATIO = 0.670 # 实测交易日/自然日占比(000001_SH_daily.csv 181 行 / 270 天)
CALENDAR_MARGIN = 1.05    # 自然日请求余量
INDEX_CODES = ['000001.SH', '399001.SZ']   # 上证综指 / 深证成分指数
SW2_LIST_ARG = '11'       # get_stock_list('11', list_type=1) -> 128 申万二级行业
FIELDS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
# ======================================================


def filter_st(items):
    """[{'Code','Name'}, ...] -> [code],剔除 ST/*ST/SST 与退市标的。

    条目可能是 None 或缺 Code(TQ 返回偶有脏数据),一律跳过。
    """
    out = []
    for it in items or []:
        if not it or not it.get('Code'):
            continue
        name = it.get('Name') or ''
        if 'ST' in name.upper() or '退' in name:
            continue
        out.append(it['Code'])
    return out


def chunked(seq, size=BATCH_SIZE):
    """把列表切成每块 size 个,末块可短。"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def calendar_days_for(trading_days=TRADING_DAYS):
    """交易日数 -> 需向 TQ 请求的自然日数。

    多请求的成本几乎为零(TQ 按区间返回),少拉却要整轮重来,所以宁可多留余量。
    """
    return int(trading_days / TRADING_DAY_RATIO * CALENDAR_MARGIN)


def trim_tail(df, n=TRADING_DAYS):
    """排序后取尾部 n 行。不足 n 行的原样返回 —— 次新股照收,不补齐、不丢弃。"""
    return df.sort_index().tail(n)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_fetch_helpers.py -v
```

Expected: 8 passed。`calendar_days_for(500)` 应返回 `783`。

- [ ] **Step 5: 提交**

```bash
cd C:/Users/yellow/mcp/qtTdx
git add backtrace/data_fetch/fetch_daily.py tests/test_fetch_helpers.py
git commit -m "feat(data): fetch_daily 纯函数层(ST 过滤/分批/天数换算/截尾)

8 个测试全部离线可跑,不需要 TQ 客户端。"
```

---

## Task 4: `fetch_daily.py` TQ 编排层

**Files:**
- Modify: `backtrace/data_fetch/fetch_daily.py`(在 Task 3 的纯函数之后追加)

**Interfaces:**
- Consumes:
  - Task 1: `data_store.save_daily(code, df, kind) -> str`、`data_store.load_manifest() -> dict`、`data_store.save_manifest(man) -> str`
  - Task 3: `filter_st`、`chunked`、`calendar_days_for`、`trim_tail`、`TRADING_DAYS`、`BATCH_SIZE`、`INDEX_CODES`、`SW2_LIST_ARG`、`FIELDS`
- Produces:
  - `fetch_daily.probe_lists(tq) -> None` —— 打印 TQ 列表接口返回结构供人工判读
  - `fetch_daily.build_sector_universe(tq) -> tuple[list[str], dict[str, str]]` —— (行业代码, {代码: 中文名})
  - `fetch_daily.build_stock_universe(tq, sector_codes) -> list[str]`
  - `fetch_daily.fetch_batch(tq, codes, start, end) -> dict[str, pd.DataFrame]` —— 空数据时抛 `RuntimeError`
  - `fetch_daily.main() -> int` —— 退出码

- [ ] **Step 1: 追加 TQ 编排代码**

在 `backtrace/data_fetch/fetch_daily.py` 末尾追加:

```python
# ==================== 以下需要 TQ 客户端 ====================
import argparse
import json
import traceback
from datetime import datetime, timedelta

from common import data_store
from common import tsfresh_config as C


def _tq():
    """懒加载 TQ —— 纯函数层的测试不该被这个 import 拖累。"""
    sys.path.insert(0, C.TQ_PLUGINS_DIR)
    from tqcenter import tq
    return tq


def probe_lists(tq):
    """打印 TQ 列表接口在若干实参下的返回结构,供人工判读全市场列表怎么取。

    只读、只打印,不写任何文件。
    """
    print("=" * 70)
    print("探测 get_stock_list / get_sector_list 返回结构")
    print("=" * 70)
    for arg in ['1', '2', '11', '12', '21', '22']:
        try:
            got = tq.get_stock_list(arg, list_type=1)
            n = len(got or [])
            sample = (got or [])[:3]
            print(f"  get_stock_list({arg!r}, list_type=1) -> {n} 条  样例={sample}")
        except Exception as e:
            print(f"  get_stock_list({arg!r}, list_type=1) -> {type(e).__name__}: {e}")
    for lt in [0, 1]:
        try:
            got = tq.get_sector_list(list_type=lt)
            print(f"  get_sector_list(list_type={lt}) -> {len(got or [])} 条  样例={(got or [])[:2]}")
        except Exception as e:
            print(f"  get_sector_list(list_type={lt}) -> {type(e).__name__}: {e}")


def build_sector_universe(tq):
    """128 申万二级行业。返回 (代码列表, {代码: 中文名})。

    已由 tsfresh_top1_industry.py:46-56 验证:这批 Code 可直接喂 get_market_data。
    """
    items = tq.get_stock_list(SW2_LIST_ARG, list_type=1) or []
    codes, names = [], {}
    for it in items:
        if it and it.get('Code'):
            codes.append(it['Code'])
            names[it['Code']] = it.get('Name') or ''
    if not codes:
        raise RuntimeError(f"get_stock_list({SW2_LIST_ARG!r}) 返回空 —— TQ 客户端可能未启动")
    print(f"  申万二级行业: {len(codes)} 个")
    return codes, names


def build_stock_universe(tq, sector_codes):
    """个股 universe = 128 行业成分股并集,再剔除 ST/退市。

    为什么用行业并集而非 get_stock_list 全市场:get_stock_list 取沪深两市的实参
    未经验证(见 --probe),而 get_stock_list_in_sector 对这批行业码已由
    tsfresh_top1_industry.py:69 跑通。覆盖面接近全市场,且顺带拿到行业归属。
    探明全市场实参后可在此替换。
    """
    seen = {}
    for i, code in enumerate(sector_codes, 1):
        try:
            members = tq.get_stock_list_in_sector(code) or []
        except Exception as e:
            print(f"  [WARN] 行业 {code} 成分股拉取失败: {type(e).__name__}: {e}")
            continue
        for m in members:
            seen.setdefault(m, None)
        if i % 20 == 0:
            print(f"  行业成分股进度 {i}/{len(sector_codes)}  累计去重 {len(seen)} 只")

    # get_stock_list_in_sector 只给代码不给名称,需要名称才能过滤 ST
    all_codes = sorted(seen)
    if not all_codes:
        raise RuntimeError("行业成分股并集为空 —— TQ 客户端可能未启动")

    items = []
    for c in all_codes:
        try:
            # get_stock_info 返回 dict 含 'name' 字段(小写) — 已由
            # backtrace/gp_factor_mining/01_data_prep.py:189 验证可用
            name = tq.get_stock_info(c).get('name', '')
        except Exception:
            name = ''
        items.append({'Code': c, 'Name': name})
    kept = filter_st(items)
    print(f"  个股 universe: 并集 {len(all_codes)} 只 -> 去 ST/退市后 {len(kept)} 只")
    return kept


def fetch_batch(tq, codes, start, end):
    """拉一批,返回 {code: DataFrame}。

    TQ 客户端未启动时会「假装成功」返回空数据 —— 这里必须当成硬错误抛出,
    否则会用空 CSV 覆盖掉上一轮的好数据(静默的数据损坏比崩溃危险得多)。
    """
    raw = tq.get_market_data(
        field_list=FIELDS, stock_list=list(codes),
        start_time=start, end_time=end,
        dividend_type='front', period='1d', fill_data=True,
    )
    if raw is None or 'Close' not in raw or raw['Close'].shape[1] == 0:
        raise RuntimeError("TQ 返回空列 —— 客户端可能未启动")

    out = {}
    for c in codes:
        if c not in raw['Close'].columns:
            continue
        cols = {}
        for f in FIELDS:
            if f in raw and c in raw[f].columns:
                cols[f] = pd.to_numeric(raw[f][c], errors='coerce')
        if 'Close' not in cols:
            continue
        df = trim_tail(pd.DataFrame(cols))
        if len(df) == 0:
            continue
        out[c] = df
    return out


def _record(man, code, kind, df, name=None):
    entry = {
        'kind': kind,
        'rows': int(len(df)),
        'start': str(df.index[0].date()),
        'end': str(df.index[-1].date()),
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'status': 'ok',
    }
    if name:
        entry['name'] = name
    man['entries'][code] = entry


def _run_group(tq, codes, kind, start, end, man, names=None, force=False):
    """拉一组(个股/行业/指数),逐批落盘 + 更新 manifest。返回 (成功数, 失败数)。"""
    today = datetime.now().strftime('%Y-%m-%d')
    todo = []
    for c in codes:
        e = man['entries'].get(c)
        done_today = (not force and e and e.get('status') == 'ok'
                      and str(e.get('fetched_at', '')).startswith(today))
        if done_today:
            continue
        todo.append(c)
    skipped = len(codes) - len(todo)
    if skipped:
        print(f"  [{kind}] 断点续传跳过今日已完成 {skipped} 只")

    ok = fail = 0
    batches = list(chunked(todo))
    for bi, batch in enumerate(batches, 1):
        got = None
        for attempt in (1, 2):
            try:
                got = fetch_batch(tq, batch, start, end)
                break
            except RuntimeError:
                raise                      # 空数据 = 环境问题,不重试,直接上抛中止整轮
            except Exception as e:
                print(f"  [{kind}] 批 {bi}/{len(batches)} 第 {attempt} 次失败: {type(e).__name__}: {e}")
                if attempt == 2:
                    for c in batch:
                        man['entries'][c] = {'kind': kind, 'status': 'failed',
                                             'reason': f'{type(e).__name__}: {e}'}
                    fail += len(batch)
        if got is None:
            continue
        for c, df in got.items():
            data_store.save_daily(c, df, kind)
            _record(man, c, kind, df, (names or {}).get(c))
            ok += 1
        for c in batch:
            if c not in got:
                man['entries'][c] = {'kind': kind, 'status': 'failed', 'reason': 'TQ 无该代码数据'}
                fail += 1
        data_store.save_manifest(man)      # 每批存盘,崩了也不白跑
        print(f"  [{kind}] 批 {bi}/{len(batches)} 完成  累计 ok={ok} fail={fail}")
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description='拉取日线到仓库根 data/')
    ap.add_argument('--limit', type=int, default=0, help='每组只取前 N 个代码(冒烟用)')
    ap.add_argument('--force', action='store_true', help='忽略 manifest,全量重拉')
    ap.add_argument('--probe', action='store_true', help='只探测 TQ 列表接口后退出')
    args = ap.parse_args()

    tq = _tq()
    tq.initialize(os.path.abspath(__file__))
    try:
        if args.probe:
            probe_lists(tq)
            return 0

        cal_days = calendar_days_for()
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=cal_days)).strftime('%Y%m%d')
        print(f"目标 {TRADING_DAYS} 交易日 -> 请求 {cal_days} 自然日 ({start} ~ {end})")

        man = data_store.load_manifest()
        man['trading_days'] = TRADING_DAYS

        print("\n[1/3] 行业指数")
        sector_codes, sector_names = build_sector_universe(tq)
        if args.limit:
            sector_codes = sector_codes[:args.limit]
        s_ok, s_fail = _run_group(tq, sector_codes, 'sectors', start, end, man,
                                  names=sector_names, force=args.force)

        print("\n[2/3] 大盘指数")
        i_ok, i_fail = _run_group(tq, INDEX_CODES, 'indices', start, end, man, force=args.force)

        print("\n[3/3] 个股")
        stock_codes = build_stock_universe(tq, sector_codes)
        if args.limit:
            stock_codes = stock_codes[:args.limit]
        k_ok, k_fail = _run_group(tq, stock_codes, 'stocks', start, end, man, force=args.force)

        man['generated_at'] = datetime.now().isoformat(timespec='seconds')
        data_store.save_manifest(man)

        print("\n" + "=" * 70)
        print(f"行业 ok={s_ok} fail={s_fail} | 指数 ok={i_ok} fail={i_fail} | 个股 ok={k_ok} fail={k_fail}")
        print(f"manifest: {data_store.manifest_path()}")
        return 0

    except RuntimeError as e:
        print(f"\n[FATAL] {e}")
        print("请确认通达信客户端已启动,然后重跑(已落盘的数据不会丢,会自动续传)")
        return 1
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        try:
            tq.close()
        except Exception:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] **Step 2: 确认纯函数测试没被 TQ import 打断**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/test_fetch_helpers.py -v
```

Expected: 仍然 8 passed。若报 `ModuleNotFoundError: tqcenter`,说明 TQ import 没做成懒加载 —— 检查 `_tq()` 里的 `from tqcenter import tq` 确实在函数体内。

- [ ] **Step 3: 探测 TQ 列表接口(需 TQ 客户端启动)**

先启动通达信客户端,然后:

```bash
cd C:/Users/yellow/mcp/qtTdx
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY backtrace/data_fetch/fetch_daily.py --probe
```

Expected: 打印各实参下的条数与样例。`'11'` 应约 128 条。

**判读:** 若某实参返回 >4000 条且样例是股票(非板块),记下它 —— 可把 `build_stock_universe` 换成直接用该实参 + `filter_st`,比行业并集更直接。若没有这样的实参,保持行业并集实现不动。

- [ ] **Step 4: 冒烟跑(需 TQ 客户端启动)**

```bash
cd C:/Users/yellow/mcp/qtTdx
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY backtrace/data_fetch/fetch_daily.py --limit 5
```

Expected: 退出码 0;`data/sectors/` 出现 5 个 CSV、`data/indices/` 出现 2 个、`data/stocks/` 出现 5 个;`data/manifest.json` 存在。

验证落盘内容:

```bash
ls data/sectors data/indices data/stocks
PYTHONIOENCODING=utf-8 $PY -c "
import json, glob, pandas as pd
man = json.load(open('data/manifest.json', encoding='utf-8'))
print('manifest 条目:', len(man['entries']))
f = sorted(glob.glob('data/indices/*.csv'))[0]
df = pd.read_csv(f, index_col=0, parse_dates=True)
print(f, df.shape, list(df.columns))
assert list(df.columns)[:5] == ['Open','High','Low','Close','Volume'], df.columns
assert len(df) <= 500
print('OK')
"
```

- [ ] **Step 5: 验证空数据保护(不需要真的关客户端)**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -c "
import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'backtrace', 'data_fetch'))
sys.path.insert(0, os.path.join(os.getcwd(), 'backtrace'))
import pandas as pd, fetch_daily as FD

class FakeTQ:
    def get_market_data(self, **kw):
        return {'Close': pd.DataFrame()}      # 模拟'假装成功'的空返回

try:
    FD.fetch_batch(FakeTQ(), ['000001.SZ'], '20250101', '20260101')
    print('FAIL: 空数据没被拦住')
    raise SystemExit(1)
except RuntimeError as e:
    print('OK: 空数据被拦截 ->', e)
"
```

Expected: `OK: 空数据被拦截 -> TQ 返回空列 —— 客户端可能未启动`

- [ ] **Step 6: 断点续传验证(需 TQ 客户端启动)**

再跑一次同样的冒烟命令:

```bash
PYTHONIOENCODING=utf-8 $PY backtrace/data_fetch/fetch_daily.py --limit 5
```

Expected: 打印 `断点续传跳过今日已完成 N 只`,且 `data/` 下无 `.tmp` 残留:

```bash
find data -name "*.tmp" | wc -l    # 期望 0
```

- [ ] **Step 7: 提交**

```bash
cd C:/Users/yellow/mcp/qtTdx
git add backtrace/data_fetch/fetch_daily.py
git commit -m "feat(data): fetch_daily TQ 编排层(universe/分批/重试/断点续传)

- 空返回当硬错误中止整轮,避免空 CSV 覆盖好数据
- 每批落盘 + 存 manifest,崩溃不白跑
- universe 走行业成分股并集(已验证路径);--probe 供探明全市场实参后替换"
```

---

## Task 5: 全量实跑与文档收尾

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/api.md`

**Interfaces:**
- Consumes: Task 1-4 全部
- Produces: 无代码接口

- [ ] **Step 1: 全量拉取(需 TQ 客户端启动,20-40 分钟)**

用后台跑,别阻塞:

```bash
cd C:/Users/yellow/mcp/qtTdx
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY backtrace/data_fetch/fetch_daily.py
```

(agentic worker:用 `run_in_background=true`,`TaskOutput` 监控)

Expected: 退出码 0,末行打印三组 ok/fail 统计。

- [ ] **Step 2: 核对落盘规模**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -c "
import glob, json
for k in ['stocks','sectors','indices']:
    print(f'{k:8s}', len(glob.glob(f'data/{k}/*.csv')))
man = json.load(open('data/manifest.json', encoding='utf-8'))
ok = sum(1 for v in man['entries'].values() if v.get('status')=='ok')
bad = [c for c,v in man['entries'].items() if v.get('status')!='ok']
print('manifest ok =', ok, ' failed =', len(bad))
print('失败样例:', bad[:10])
"
du -sh data
```

Expected: `sectors` ≈ 128,`indices` = 2,`stocks` 数千;`du` 约 150-200 MB。

**若 `stocks` 远低于 4000**,说明行业并集覆盖不全 —— 回到 Task 4 Step 3 的探测结果,改用全市场实参。这属于预期内的分支,不是失败。

- [ ] **Step 3: 验证离线回退真的可用(关掉 TQ 客户端)**

关闭通达信客户端后:

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -c "
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'backtrace'))
from common import tsfresh_pipeline as P
for code in ['000001.SH', '399001.SZ']:
    df = P.load_ohlcva(code, use_tq=False, verbose=True)
    assert df is not None, f'FAIL {code}'
    print(code, df.shape)
print('OK: 离线回退可用')
"
```

Expected: 两个代码都返回 DataFrame,打印 `OK: 离线回退可用`。

- [ ] **Step 4: 全测试回归**

```bash
PY=C:/Users/yellow/.conda/envs/venv/python.exe
PYTHONIOENCODING=utf-8 $PY -m pytest tests/ -v
```

Expected: 21 passed(10 + 3 + 8)。

- [ ] **Step 5: 更新 `CLAUDE.md`**

在「## 仓库本质」段落后、「## 跑脚本的最快姿势」前插入新节:

```markdown
## 本地日线缓存 — `data/`

TQ 客户端没启动时,`P.load_ohlcva` / `P.load_sector` 会回退到仓库根 `data/`:

```
data/stocks/    沪深 A 股(去 ST/退市)日线,~5000 只
data/sectors/   申万二级 128 行业指数
data/indices/   000001.SH 上证综指 / 399001.SZ 深证成指
data/manifest.json   每只票的行数/首末日期/拉取时间/失败原因
```

每只票保留 **500 个交易日**。刷新数据:

```bash
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py           # 全量,20-40 分钟
PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --limit 5 # 冒烟
```

**路径只认 [backtrace/common/data_store.py](backtrace/common/data_store.py)** —— 读写都走 `csv_path()`。
不要在别处硬拼 `data/...` 路径,历史上正是「写在 A、读在 B」让回退静默失效了半年。
```

同时在「## backtrace/ 目录结构」的树里加一行:

```
├── data_fetch/              ← 日线批量拉取(写 data/)
│   └── fetch_daily.py
```

- [ ] **Step 6: 更新 `docs/api.md`**

新增 `fetch_daily` 一节:

```markdown
### backtrace/data_fetch/fetch_daily.py

日线批量拉取编排。落盘经由 `data_store`,自身不拼路径。

| 函数 | 签名 | 说明 |
|---|---|---|
| `filter_st` | `(items) -> list[str]` | 剔除 ST/*ST/退市;跳过 None 与缺 Code 条目 |
| `chunked` | `(seq, size=250) -> Iterator[list]` | 分批 |
| `calendar_days_for` | `(trading_days=500) -> int` | 交易日→自然日(占比 0.670,余量 5%) |
| `trim_tail` | `(df, n=500) -> DataFrame` | 排序取尾;不足不补齐 |
| `build_sector_universe` | `(tq) -> (list[str], dict)` | 128 申万二级行业码 + 中文名 |
| `build_stock_universe` | `(tq, sector_codes) -> list[str]` | 行业成分股并集去 ST |
| `fetch_batch` | `(tq, codes, start, end) -> dict` | 一批 OHLCVA;**空返回抛 RuntimeError** |
| `main` | `() -> int` | CLI 入口,退出码 |

CLI:`--limit N` 冒烟 / `--force` 忽略 manifest / `--probe` 探测 TQ 列表接口。

常量:`TRADING_DAYS=500`、`BATCH_SIZE=250`、`INDEX_CODES=['000001.SH','399001.SZ']`。
```

- [ ] **Step 7: 确认 `data/` 未被 git 追踪**

```bash
cd C:/Users/yellow/mcp/qtTdx
git status --short | grep '^?? data/' && echo "警告: data/ 未被忽略" || echo "OK: data/ 已被 .gitignore 覆盖"
git check-ignore -v data/indices/000001_SH_daily.csv
```

Expected: 打印 `OK: data/ 已被 .gitignore 覆盖`,`check-ignore` 显示命中 `.gitignore` 的 `*.csv` 规则。

若 `manifest.json` 被列为未追踪(它不是 `.csv`),在 `.gitignore` 追加:

```
# 本地日线缓存(仓库根 data/)
data/
```

- [ ] **Step 8: 提交**

```bash
cd C:/Users/yellow/mcp/qtTdx
git add CLAUDE.md docs/api.md .gitignore
git commit -m "docs: 补 data/ 本地日线缓存与 data_fetch/ 说明

同步 api.md 的 data_store / fetch_daily 函数表(CLAUDE.md 要求)。"
```

---

## Self-Review

**Spec 覆盖核对:**

| Spec 节 | 落点 |
|---|---|
| §1 动机(修回退) | Task 2 全部 |
| §2 六项决策 | Task 1(布局/位置)、Task 3(500 日)、Task 4(范围/板块) |
| §3.1 `data_store` API | Task 1 Step 5(7 个函数,较 spec 的 4 个多出 manifest 三件套,理由见 File Structure) |
| §3.2 `fetch_daily` | Task 3 + Task 4 |
| §3.3 `_try_local_csv` | Task 2 Step 4 |
| §4 磁盘布局 / 文件名 / schema | Task 1 Step 5(`KINDS`/`_filename`/`COLUMNS`) |
| §4.2 manifest | Task 1 Step 5 + Task 4 `_record` |
| §5.1 universe(主+备) | Task 4 Step 3(`--probe`)+ `build_stock_universe`(备路径为默认) |
| §5.1 分批 250 | Task 3 `BATCH_SIZE` |
| §5.1 健康检查 | Task 4 `fetch_batch` + Task 4 Step 5 验证 |
| §5.1 断点续传 | Task 4 `_run_group` + Task 4 Step 6 验证 |
| §5.2 780 自然日 | Task 3 `calendar_days_for` |
| §5.3 原子写 | Task 1 `save_daily`/`save_manifest` + 两个 `.tmp` 测试 |
| §6 规模 | Task 5 Step 2 |
| §7 错误处理 7 类 | Task 4 `_run_group`/`main` 的 try 分支 |
| §8 五项验证 | Task 1 Step 6、Task 2 Step 7、Task 4 Step 4-6、Task 5 Step 3 |
| §9 范围外 | 未出现在任何任务 ✓(历史 CSV 用 `cp` 手工搬,未写脚本) |

**已知偏离 spec 之处(有意):**

1. `data_store` 从 4 个函数扩到 7 个(加 manifest 三件套)—— 理由见 File Structure 段。
2. `build_stock_universe` 把 spec 的「备路径」(行业并集)作为**默认实现**,「主路径」(`get_stock_list` 全市场)降为 `--probe` 后的可选替换。理由:备路径已被 `tsfresh_top1_industry.py:69` 验证可用,主路径实参未知;先交付能跑的,再按探测结果优化。Task 4 Step 3 与 Task 5 Step 2 都留了切换判据。

**类型一致性:** `csv_path/save_daily/load_daily/has_daily/manifest_path/load_manifest/save_manifest` 在 Task 1 定义,Task 2 用 `load_daily`、Task 4 用 `save_daily`/`load_manifest`/`save_manifest`/`manifest_path`,签名一致。`filter_st/chunked/calendar_days_for/trim_tail` 在 Task 3 定义,Task 4 用 `chunked`/`calendar_days_for`/`trim_tail`/`filter_st`,一致。

**未验证的假设(实施时会撞上):**

- ~~`tq.get_instrument_detail(code)['InstrumentName']` 用于取股票名以过滤 ST~~ — **已验证修正**:此函数在 `tqcenter.py` 中不存在。已查阅仓库内已有调用 [backtrace/gp_factor_mining/01_data_prep.py:189](backtrace/gp_factor_mining/01_data_prep.py),真实 API 是 `tq.get_stock_info(c).get('name', '')`(小写 `name`,已替换)。Task 4 Step 1 的 `build_stock_universe` 已改用此调用,空字符串回退保留(单票失败不连累整组)。
- 板块指数是否有 `Amount` 字段未经验证 —— `fetch_batch` 按字段存在性逐列取,缺列即少一列,不会崩。
- `calendar_days_for` 用 0.670 交易日占比启发式 + 5% 余量 → 500 交易日取 783 自然日。`tqcenter.py` 提供 `get_trading_dates(market, start, end, count=-1)` 可精确换算,但会让 Task 3 的纯函数绑上 TQ、破坏离线可测性。**保留启发式**;若后续要更精确,在 `_run_group` 入口处加一次 bootstrap 调用,算出实际日历天后覆盖常量即可,Task 3 的纯函数层不动。
