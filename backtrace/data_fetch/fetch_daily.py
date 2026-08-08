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