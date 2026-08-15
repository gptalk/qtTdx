# -*- coding: utf-8 -*-
"""从 data/sw2/union.csv 拉每只股的 name,产出 data/stock_basic.csv。

为什么从 union.csv 走而不是再问一次 TQ 全市场:
  - union.csv 已经持久化 128 申万二级行业成分股并集(~5200 只),与 fetch_daily.py 一致
  - 再问 TQ get_stock_list 的全市场实参未验证(见 fetch_daily.py:152 注释)
  - TQ 客户端启动时 union.csv 是最新,覆盖范围与日线 fetch 一致

输出列:code, market, name, status
  - code:  6位代码.交易所(000001.SH / 000001.SZ / 000xxx.BJ)
  - market: 交易所缩写(SH / SZ / BJ),由 code 后缀派生,不调 TQ
  - name:  TQ get_stock_info(c).get('name', '') — 与 fetch_daily.py:212 同源
  - status:
      active    正常沪深 A 股
      st        ST/*ST/SST(名字含 ST/ST)
      delisted  名字含「退」
      bj        北交所
      unknown   空名/异常

用法:
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_stock_basic.py             # 全量
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_stock_basic.py --limit 10  # 冒烟

输出:
  data/stock_basic.csv(~5200 行,UTF-8,gitignored,与 manifest.json 同层)
"""
import argparse
import os
import sys

import pandas as pd

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

from common import tsfresh_config as C  # noqa: E402

UNION_CSV = os.path.join(C.DATA_DIR, 'sw2', 'union.csv')
BASIC_CSV = os.path.join(C.DATA_DIR, 'stock_basic.csv')


def _tq():
    """懒加载 TQ — 不在 import 期触发 tqcenter,允许 --help 等纯命令行场景。"""
    sys.path.insert(0, C.TQ_PLUGINS_DIR)
    from tqcenter import tq
    return tq


def classify_status(name: str, market: str) -> str:
    """判定 status。优先级:bj > delisted > st > unknown > active。

    判定规则与 fetch_daily.filter_st 一致:
      - market=BJ → bj
      - name 含「退」→ delisted
      - name.upper() 含 'ST' → st(覆盖 ST/*ST/SST)
      - 空名 → unknown
      - 其它 → active
    """
    if market == 'BJ':
        return 'bj'
    if '退' in (name or ''):
        return 'delisted'
    if 'ST' in (name or '').upper():
        return 'st'
    if not name:
        return 'unknown'
    return 'active'


def build_basic(tq, union_path: str = UNION_CSV, limit: int = 0) -> pd.DataFrame:
    """读 union.csv → 拉 name → 组装 DataFrame。limit=0 表示全部。"""
    if not os.path.exists(union_path):
        raise FileNotFoundError(
            f"{union_path} 缺失。先跑:\n"
            f"  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py\n"
            f"(fetch_daily 会写 union.csv 作为副产品)"
        )
    union_df = pd.read_csv(union_path, dtype={'code': str})
    codes = union_df['code'].tolist()
    if limit > 0:
        codes = codes[:limit]

    print(f"输入: {union_path}  共 {len(union_df)} 只  本次处理 {len(codes)} 只")
    rows = []
    for i, code in enumerate(codes, 1):
        try:
            info = tq.get_stock_info(code) or {}
            name = info.get('name', '') or ''
        except Exception as e:
            print(f"  [{i}/{len(codes)}] [WARN] {code} get_stock_info 失败: {type(e).__name__}: {e}")
            name = ''
        market = code.split('.')[-1].upper() if '.' in code else ''
        status = classify_status(name, market)
        rows.append({'code': code, 'market': market, 'name': name, 'status': status})
        if i % 500 == 0:
            print(f"  进度 {i}/{len(codes)}")

    out = pd.DataFrame(rows).sort_values('code').reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser(description='从 sw2/union.csv 生成 stock_basic.csv')
    p.add_argument('--union', default=UNION_CSV, help=f'输入 union.csv 路径。默认 {UNION_CSV}')
    p.add_argument('--limit', type=int, default=0, help='最多处理多少只;0 表示全部。默认 0(冒烟用 --limit 10)')
    p.add_argument('--out', default=BASIC_CSV, help=f'输出 CSV 路径。默认 {BASIC_CSV}')
    args = p.parse_args()

    tq = _tq()
    tq.initialize(__file__)

    try:
        df = build_basic(tq, union_path=args.union, limit=args.limit)
    finally:
        tq.close()

    # 落盘
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + '.tmp'
    df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, args.out)

    # 统计
    print(f"\n=== 写入 {args.out} ===")
    print(f"  总行数: {len(df)}")
    if len(df):
        counts = df['status'].value_counts()
        for st in ['active', 'st', 'delisted', 'bj', 'unknown']:
            n = counts.get(st, 0)
            print(f"  {st:<10s} {n:>5d}")
        unknowns = df[df['status'] == 'unknown']
        if len(unknowns):
            sample = unknowns.head(10)[['code', 'name']].values.tolist()
            print(f"  unknown 样例(前 10): {sample}")


if __name__ == '__main__':
    main()