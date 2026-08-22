# -*- coding: utf-8 -*-
"""
拉取沪深全市场 + 申万二级行业指数 + 两大盘指数的日线,落盘到仓库根 data/。

排除范围:
  - 北交所(.BJ)标的:历史短(2021 开业)、流动性差、交易规则与沪深不同(30% 涨跌幅),
    策略框架不覆盖。在 build_stock_universe 收 sector members 时源头剔除,不进入
    members.csv / union.csv,fetch 阶段自然也不会拉到。
  - ST/*ST/SST、退市:由 build_stock_universe 写 stock_basic.csv 后,按
    status 列过滤(同一轮里复用 fetch_stock_basic.build_basic,不再重复判)。

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
SW2_OUT_DIR = 'sw2'  # 128 行业 + 成分股 long-format + 并集清单,fetch 过程中落盘(路径经 C.DATA_DIR 拼,便于测试切根)
# ======================================================


def filter_st(items):
    """[{'Code','Name'}, ...] -> [code],剔除 ST/*ST/SST、退市标的。

    ⚠️ 仅文档/测试参考用 — build_stock_universe 已改为读 stock_basic.status
    做过滤(避免对每只股重复跑 ST 名字判定)。新逻辑请走 fetch_stock_basic。

    北交所(.BJ)已在更上游(build_stock_universe 收 sector members 时)排除,
    这里不再重复过滤,避免出现「members.csv 有 .BJ、union.csv 无 .BJ」的不一致。
    条目可能是 None 或缺 Code(TQ 返回偶有脏数据),一律跳过。
    """
    out = []
    for it in items or []:
        if not it or not it.get('Code'):
            continue
        code = it['Code']
        name = it.get('Name') or ''
        if 'ST' in name.upper() or '退' in name:
            continue
        out.append(code)
    return out


def is_bj(code: str) -> bool:
    """北交所股票代码后缀 .BJ(400xxx/920xxx 等),策略框架不覆盖,直接排除。"""
    return bool(code) and code.upper().endswith('.BJ')


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
    Side effect:持久化 data/sw2/industries.csv (sector_code, sector_name)。
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

    os.makedirs(os.path.join(C.DATA_DIR, SW2_OUT_DIR), exist_ok=True)
    industries_path = os.path.join(C.DATA_DIR, SW2_OUT_DIR, 'industries.csv')
    pd.DataFrame(
        [{'sector_code': c, 'sector_name': names[c]} for c in codes]
    ).to_csv(industries_path, index=False, encoding='utf-8')
    print(f"  → {industries_path} ({len(codes)} 行)")
    return codes, names


def build_stock_universe(tq, sector_codes, sector_names):
    """个股 universe = 128 行业成分股并集,再剔除 ST/退市。

    为什么用行业并集而非 get_stock_list 全市场:get_stock_list 取沪深两市的实参
    未经验证(见 --probe),而 get_stock_list_in_sector 对这批行业码已由
    tsfresh_top1_industry.py:69 跑通。覆盖面接近全市场,且顺带拿到行业归属。
    探明全市场实参后可在此替换。

    Side effect:持久化
      - data/sw2/members.csv (long-format: sector_code, sector_name, member_code)
      - data/sw2/union.csv   (去重后、ST 过滤前的所有代码: code, name)
    """
    seen = set()
    sector_to_members = {}  # sector_code -> [member_code] 保留每行业归属
    bj_excluded = 0
    for i, code in enumerate(sector_codes, 1):
        try:
            members = tq.get_stock_list_in_sector(code) or []
        except Exception as e:
            print(f"  [WARN] 行业 {code} 成分股拉取失败: {type(e).__name__}: {e}")
            continue
        # 上游剔除北交所(.BJ):不进入 sector_to_members / seen,避免污染 members.csv
        # 北交所历史短(2021 开业)、流动性差、交易规则与沪深不同,策略框架不覆盖
        kept_members = []
        for m in members:
            if is_bj(m):
                bj_excluded += 1
                continue
            kept_members.append(m)
        sector_to_members[code] = kept_members
        for m in kept_members:
            seen.add(m)
        if i % 20 == 0:
            print(f"  行业成分股进度 {i}/{len(sector_codes)}  累计去重 {len(seen)} 只")

    if bj_excluded:
        print(f"  [剔除北证] {bj_excluded} 条 .BJ 成员已在源头过滤(未进入 members.csv / union.csv)")

    # 持久化 long-format 成分股(sector_code + sector_name + member_code)
    os.makedirs(os.path.join(C.DATA_DIR, SW2_OUT_DIR), exist_ok=True)
    member_rows = []
    for s_code in sector_codes:
        s_name = sector_names.get(s_code, '')
        for m in sector_to_members.get(s_code, []):
            member_rows.append({
                'sector_code': s_code,
                'sector_name': s_name,
                'member_code': m,
            })
    members_path = os.path.join(C.DATA_DIR, SW2_OUT_DIR, 'members.csv')
    pd.DataFrame(member_rows).to_csv(members_path, index=False, encoding='utf-8')
    print(f"  → {members_path} ({len(member_rows)} 行 long-format)")

    # get_stock_list_in_sector 只给代码不给名称,需要名称才能过滤 ST
    all_codes = sorted(seen)
    if not all_codes:
        raise RuntimeError("行业成分股并集为空 —— TQ 客户端可能未启动")

    items = []
    empty_name_count = 0
    for c in all_codes:
        try:
            # TQ 返回字段名是 'Name'(大写 N),实测于 2026-08-15
            # 容错小写 'name' 以备将来 TQ 改大小写
            info = tq.get_stock_info(c) or {}
            name = info.get('Name', '') or info.get('name', '') or ''
        except Exception:
            name = ''
        if not name:
            empty_name_count += 1
        items.append({'Code': c, 'Name': name})

    # 持久化 union(去重后、ST 过滤前的所有代码)
    union_path = os.path.join(C.DATA_DIR, SW2_OUT_DIR, 'union.csv')
    pd.DataFrame(
        [{'code': it['Code'], 'name': it['Name']} for it in items]
    ).to_csv(union_path, index=False, encoding='utf-8')
    print(f"  → {union_path} ({len(items)} 行,ST 过滤前)")

    # 复用 fetch_stock_basic:同样要拉 5200+ 次 get_stock_info,在 union 写完后
    # 直接调一遍,把分类结果落 data/stock_basic.csv。之后任何脚本/stocks_info 都能
    # 直接读这张表,不用再问 TQ,也省了 filter_st 重新跑 ST 名字判断。
    from fetch_stock_basic import build_basic as build_basic_table
    basic_df = build_basic_table(tq, union_path=union_path)
    basic_path = os.path.join(C.DATA_DIR, 'stock_basic.csv')
    os.makedirs(os.path.dirname(basic_path), exist_ok=True)
    tmp = basic_path + '.tmp'
    basic_df.to_csv(tmp, index=False, encoding='utf-8')
    os.replace(tmp, basic_path)
    print(f"  → {basic_path} ({len(basic_df)} 行,4 列:code/market/name/status)")

    # 用 stock_basic.status 列做过滤 — 一次性查表,不再重复跑 ST 名字判定
    status_by_code = dict(zip(basic_df['code'], basic_df['status']))
    kept = [c for c in all_codes if status_by_code.get(c) == 'active']
    dropped_st = sum(1 for c in all_codes if status_by_code.get(c) == 'st')
    dropped_del = sum(1 for c in all_codes if status_by_code.get(c) == 'delisted')
    dropped_bj = sum(1 for c in all_codes if status_by_code.get(c) == 'bj')
    dropped_unk = sum(1 for c in all_codes if status_by_code.get(c) == 'unknown')
    print(f"  个股 universe: 并集 {len(all_codes)} 只 -> 去 ST({dropped_st}) "
          f"退市({dropped_del}) 北证({dropped_bj}) 异常({dropped_unk}) 后剩 {len(kept)} 只")
    return kept


def fetch_batch(tq, codes, start, end):
    """拉一批,返回 ({code: DataFrame}, {missing_code: reason})。

    三种空数据情形,必须区分:
      A) 整批返回空(raw 是 None 或 'Close' 列宽=0)→ 客户端/环境问题 → RuntimeError,中止整轮
      B) 整批返回空但只有 1 只代码 → 可能是该代码本身无行情(新股/暂停)
         → 当成"个股缺数据",记 failed 继续,避免 1 只新股把整轮 5214 只卡住
      C) 个别代码在返回里缺失 → 记 failed,其它正常落盘

    (B) 判定:raw 非空 + Close 列宽=0 + batch 只有 1 只代码 → 视为个股缺数据
    """
    raw = tq.get_market_data(
        field_list=FIELDS, stock_list=list(codes),
        start_time=start, end_time=end,
        dividend_type='front', period='1d', fill_data=True,
    )
    # raw 是 None / 不是 dict / 缺 Close 字段:
    #   - batch 多只 → 客户端问题,中止整轮
    #   - batch 只 1 只 → 该代码无行情(新股未上市/暂停),记 failed 继续
    if raw is None or not isinstance(raw, dict) or 'Close' not in raw:
        if len(codes) == 1:
            return {}, {codes[0]: 'TQ 返回空(可能新股未上市或已暂停)'}
        raise RuntimeError(f"TQ 返回空 —— 客户端可能未启动 (请求 {len(codes)} 只)")
    if raw['Close'].shape[1] == 0:
        if len(codes) == 1:
            return {}, {codes[0]: 'TQ 返回无该代码行情数据(可能新股未上市或已暂停)'}
        raise RuntimeError(f"TQ 返回空列 —— 客户端可能未启动 (请求 {len(codes)} 只)")

    out = {}
    missing = {}
    for c in codes:
        if c not in raw['Close'].columns:
            missing[c] = 'TQ 返回里无该代码'
            continue
        cols = {}
        for f in FIELDS:
            if f in raw and c in raw[f].columns:
                cols[f] = pd.to_numeric(raw[f][c], errors='coerce')
        if 'Close' not in cols:
            missing[c] = 'TQ 返回无 Close 字段'
            continue
        df = trim_tail(pd.DataFrame(cols))
        if len(df) == 0:
            missing[c] = 'TQ 返回数据为空'
            continue
        out[c] = df
    return out, missing


def fetch_one_stock(code, period, lookback_days):
    """Single TQ pull for one stock at given period (intraday only; daily uses fetch_batch)."""
    import sys as _sys
    _sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')
    from tqcenter import tq as _tq
    _tq.initialize(__file__)
    if period == 'daily':
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=int(lookback_days * 1.5) + 30)).strftime('%Y%m%d') \
                if lookback_days else \
                (datetime.now() - timedelta(days=int(C.LOOKBACK_YEARS * 365 + 30))).strftime('%Y%m%d')
        tq_period = '1d'
    else:
        days = lookback_days or C.DEFAULT_INTRADAY_LOOKBACK_DAYS
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=int(days * 1.8) + 10)).strftime('%Y%m%d')
        tq_period = C.TQ_PERIOD_MAP[period]
    fields = FIELDS  # existing
    raw = _tq.get_market_data(
        field_list=fields, stock_list=[code],
        start_time=start, end_time=end,
        dividend_type='front', period=tq_period, fill_data=True,
    )
    if raw is None or raw.get('Close') is None or raw['Close'].empty:
        raise RuntimeError(f"TQ empty for {code} {tq_period}")
    df = pd.DataFrame({
        'Open':   pd.to_numeric(raw['Open'][code],   errors='coerce'),
        'High':   pd.to_numeric(raw['High'][code],   errors='coerce'),
        'Low':    pd.to_numeric(raw['Low'][code],    errors='coerce'),
        'Close':  pd.to_numeric(raw['Close'][code],  errors='coerce'),
        'Volume': pd.to_numeric(raw['Volume'][code], errors='coerce'),
        'Amount': pd.to_numeric(raw['Amount'][code], errors='coerce'),
    }).dropna(subset=['Close']).sort_index()
    _tq.close()
    return df


def _record(man, code, kind, df, name=None, period='daily'):
    entry = {
        'kind': kind,
        'rows': int(len(df)),
        'start': str(df.index[0].date()),
        'end': str(df.index[-1].date()),
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'status': 'ok',
        'period': period,
    }
    if name:
        entry['name'] = name
    man['entries'][code] = entry


def _run_group(tq, codes, kind, start, end, man, names=None, force=False,
                period='daily', lookback_days=0):
    """拉一组(个股/行业/指数),逐批落盘 + 更新 manifest。返回 (成功数, 失败数)。

    For daily: uses batched fetch_batch + save_daily (existing behavior).
    For intraday: calls TQ per-stock via fetch_one_stock + save_df.
    """
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

    if period == 'daily':
        # === daily path: batched TQ (unchanged behavior) ====================
        batches = list(chunked(todo))
        for bi, batch in enumerate(batches, 1):
            got = None
            missing = {}
            for attempt in (1, 2):
                try:
                    got, missing = fetch_batch(tq, batch, start, end)
                    break
                except RuntimeError:
                    raise                      # 空数据 = 环境问题,不重试,直接上抛中止整轮
                except Exception as e:
                    print(f"  [{kind}] 批 {bi}/{len(batches)} 第 {attempt} 次失败: {type(e).__name__}: {e}")
                    if attempt == 2:
                        for c in batch:
                            man['entries'][c] = {'kind': kind, 'status': 'failed',
                                                 'reason': f'{type(e).__name__}: {e}', 'period': period}
                        fail += len(batch)
            if got is None:
                continue
            for c, df in got.items():
                data_store.save_daily(c, df, kind)
                _record(man, c, kind, df, (names or {}).get(c), period=period)
                ok += 1
            # 记入 missing 的代码 → 标 failed,不抛错
            for c, reason in missing.items():
                man['entries'][c] = {'kind': kind, 'status': 'failed', 'reason': reason, 'period': period}
                fail += 1
                print(f"  [{kind}] {c} 跳过:{reason}")
            if not got and missing and len(missing) == len(batch):
                pass
            data_store.save_manifest(man)      # 每批存盘,崩了也不白跑
            print(f"  [{kind}] 批 {bi}/{len(batches)} 完成  累计 ok={ok} fail={fail}")
    else:
        # === intraday path: per-stock TQ (one-at-a-time) ===================
        for bi, code in enumerate(todo, 1):
            for attempt in (1, 2):
                try:
                    df = fetch_one_stock(code, period, lookback_days)
                    break
                except Exception as e:
                    if attempt == 2:
                        man['entries'][code] = {'kind': kind, 'status': 'failed',
                                                'reason': f'{type(e).__name__}: {e}', 'period': period}
                        fail += 1
                        data_store.save_manifest(man)
                        print(f"  [{kind}] {code} 第{attempt}次失败:{type(e).__name__}: {e}")
                        break
                    continue
            else:
                # second attempt also failed (break without success)
                continue
            # success
            data_store.save_df(code, df, period, kind)
            _record(man, code, kind, df, (names or {}).get(code), period=period)
            ok += 1
            data_store.save_manifest(man)
            if bi % 50 == 0 or bi == len(todo):
                print(f"  [{kind}] {bi}/{len(todo)} 完成  累计 ok={ok} fail={fail}")

    return ok, fail


def main():
    ap = argparse.ArgumentParser(description='拉取日线到仓库根 data/')
    ap.add_argument('--limit', type=int, default=0, help='每组只取前 N 个代码(冒烟用)')
    ap.add_argument('--force', action='store_true', help='忽略 manifest,全量重拉')
    ap.add_argument('--probe', action='store_true', help='只探测 TQ 列表接口后退出')
    ap.add_argument('--period', choices=['daily', '15m', '5m', '1m'],
                    default='daily',
                    help='缓存粒度(daily = 现有默认;intraday = TQ 直拉)')
    ap.add_argument('--lookback-days', type=int, default=0,
                    help='intraday 回看天数(daily 忽略)。0 = C.DEFAULT_INTRADAY_LOOKBACK_DAYS')
    args = ap.parse_args()

    try:
        tq = _tq()
        tq.initialize(os.path.abspath(__file__))
        if args.probe:
            probe_lists(tq)
            return 0

        cal_days = calendar_days_for()
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=cal_days)).strftime('%Y%m%d')
        print(f"目标 {TRADING_DAYS} 交易日 -> 请求 {cal_days} 自然日 ({start} ~ {end})")

        man = data_store.load_manifest()
        man['trading_days'] = TRADING_DAYS
        man['period'] = args.period
        man['lookback_days'] = args.lookback_days or (
            int(C.LOOKBACK_YEARS * 365) if args.period == 'daily'
            else C.DEFAULT_INTRADAY_LOOKBACK_DAYS
        )

        print("\n[1/3] 行业指数")
        sector_codes, sector_names = build_sector_universe(tq)
        if args.limit:
            sector_codes = sector_codes[:args.limit]
        s_ok, s_fail = _run_group(tq, sector_codes, 'sectors', start, end, man,
                                  names=sector_names, force=args.force,
                                  period=args.period, lookback_days=args.lookback_days)

        print("\n[2/3] 大盘指数")
        i_ok, i_fail = _run_group(tq, INDEX_CODES, 'indices', start, end, man,
                                   force=args.force,
                                   period=args.period, lookback_days=args.lookback_days)

        print("\n[3/3] 个股")
        stock_codes = build_stock_universe(tq, sector_codes, sector_names)
        if args.limit:
            stock_codes = stock_codes[:args.limit]
        k_ok, k_fail = _run_group(tq, stock_codes, 'stocks', start, end, man,
                                   force=args.force,
                                   period=args.period, lookback_days=args.lookback_days)

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