def test_granularity_constants_present():
    from common import tsfresh_config as C
    assert C.VALID_GRANULARITIES == ('daily', '15m', '5m', '1m')
    assert C.DEFAULT_INTRADAY_GRANULARITY == '5m'
    assert C.DEFAULT_INTRADAY_LOOKBACK_DAYS == 60
    assert C.TQ_PERIOD_MAP == {'daily': '1d', '15m': '15m', '5m': '5m', '1m': '1m'}
    assert C.GRANULARITY_DT_SEC == {'daily': 86400, '15m': 900, '5m': 300, '1m': 60}