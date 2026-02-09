"""
测试数据模块

验证数据流：
    ParquetStorage.save() → HistoricFeed.subscribe() → Feed.next() → Bar
"""
import sys
import os
import shutil

import numpy as np
import pandas as pd

# 确保可以 import 项目模块
sys.path.insert(0, os.path.dirname(__file__))

from data import ParquetStorage, HistoricFeed, Bar


def generate_test_data(storage: ParquetStorage):
    """生成测试用的 Parquet 数据（模拟 AAPL 和 SPY 日线）"""
    dates = pd.bdate_range('2023-01-03', '2023-01-31')  # 工作日

    # AAPL
    np.random.seed(42)
    aapl_close = 130.0 + np.cumsum(np.random.randn(len(dates)) * 2)
    aapl_df = pd.DataFrame({
        'open': aapl_close - np.random.rand(len(dates)),
        'high': aapl_close + np.abs(np.random.randn(len(dates))) * 1.5,
        'low': aapl_close - np.abs(np.random.randn(len(dates))) * 1.5,
        'close': aapl_close,
        'volume': np.random.randint(50_000_000, 100_000_000, len(dates)),
    }, index=dates)
    # 确保 high >= max(open, close) 和 low <= min(open, close)
    aapl_df['high'] = aapl_df[['open', 'high', 'close']].max(axis=1)
    aapl_df['low'] = aapl_df[['open', 'low', 'close']].min(axis=1)

    # SPY
    spy_close = 390.0 + np.cumsum(np.random.randn(len(dates)) * 3)
    spy_df = pd.DataFrame({
        'open': spy_close - np.random.rand(len(dates)),
        'high': spy_close + np.abs(np.random.randn(len(dates))) * 2,
        'low': spy_close - np.abs(np.random.randn(len(dates))) * 2,
        'close': spy_close,
        'volume': np.random.randint(80_000_000, 150_000_000, len(dates)),
    }, index=dates)
    spy_df['high'] = spy_df[['open', 'high', 'close']].max(axis=1)
    spy_df['low'] = spy_df[['open', 'low', 'close']].min(axis=1)

    storage.save('AAPL', aapl_df, '1d')
    storage.save('SPY', spy_df, '1d')
    print(f"  Generated test data: AAPL ({len(aapl_df)} bars), SPY ({len(spy_df)} bars)")
    return aapl_df, spy_df


def test_parquet_storage():
    """测试 1: ParquetStorage 读写"""
    print("\n" + "=" * 60)
    print("Test 1: ParquetStorage read/write")
    print("=" * 60)

    storage = ParquetStorage('./test_data_cache')
    aapl_df, spy_df = generate_test_data(storage)

    # 读回来验证
    loaded = storage.load('AAPL', frequency='1d')
    assert len(loaded) == len(aapl_df), f"Length mismatch: {len(loaded)} vs {len(aapl_df)}"
    assert list(loaded.columns) == list(aapl_df.columns), "Column mismatch"

    # 日期范围过滤
    start = pd.Timestamp('2023-01-10')
    end = pd.Timestamp('2023-01-20')
    filtered = storage.load('AAPL', start, end, '1d')
    assert all(filtered.index >= start), "Filter failed: data before start"
    assert all(filtered.index <= end), "Filter failed: data after end"
    print(f"  Date filter: {len(filtered)} bars in [{start.date()}, {end.date()}]")

    # exists
    assert storage.exists('AAPL', '1d'), "exists() should return True"
    assert not storage.exists('GOOG', '1d'), "exists() should return False"
    print("  ParquetStorage PASSED")


def test_historic_feed():
    """测试 2: HistoricFeed Iterator"""
    print("\n" + "=" * 60)
    print("Test 2: HistoricFeed Iterator")
    print("=" * 60)

    storage = ParquetStorage('./test_data_cache')
    feed = HistoricFeed(storage, frequency='1d')

    start = pd.Timestamp('2023-01-03')
    end = pd.Timestamp('2023-01-31')
    feed.subscribe(['AAPL', 'SPY'], start, end)

    # 验证按时间顺序输出
    bars = []
    prev_ts = None
    while feed.has_next():
        bar = feed.next()
        bars.append(bar)
        if prev_ts is not None:
            assert bar.timestamp >= prev_ts, f"Out of order: {prev_ts} -> {bar.timestamp}"
        prev_ts = bar.timestamp

    print(f"  Total bars consumed: {len(bars)}")
    print(f"  Time range: {bars[0].timestamp.date()} -> {bars[-1].timestamp.date()}")

    # 验证多标的交错
    symbols_seen = set()
    for bar in bars[:10]:
        symbols_seen.add(bar.symbol)
        print(f"    {bar.timestamp.date()} {bar.symbol:5s} close={bar.close:.2f}")
    assert len(symbols_seen) == 2, "First 10 bars should contain 2 symbols"
    print("  HistoricFeed PASSED")


def test_bar_validation():
    """测试 3: Bar 数据验证"""
    print("\n" + "=" * 60)
    print("Test 3: Bar validation")
    print("=" * 60)

    # 正常 bar
    bar = Bar(
        timestamp=pd.Timestamp('2023-01-03'),
        symbol='AAPL',
        open=130.0, high=132.0, low=128.0, close=131.0, volume=50000000
    )
    print(f"  Normal Bar: {bar.symbol} {bar.timestamp.date()} O={bar.open} H={bar.high} L={bar.low} C={bar.close}")

    # high < low 应报错
    try:
        Bar(timestamp=pd.Timestamp('2023-01-03'), symbol='BAD',
            open=130.0, high=125.0, low=128.0, close=131.0, volume=1000)
        assert False, "Should raise ValueError"
    except ValueError as e:
        print(f"  Rejected invalid bar (high < low): {e}")

    # frozen 不可变
    try:
        bar.close = 999.0
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        print("  Bar is immutable (frozen)")

    print("  Bar validation PASSED")


def cleanup():
    """清理测试数据"""
    if os.path.exists('./test_data_cache'):
        shutil.rmtree('./test_data_cache')
        print("\nCleaned up test data directory")


if __name__ == '__main__':
    try:
        test_bar_validation()
        test_parquet_storage()
        test_historic_feed()
        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
    finally:
        cleanup()
