"""
测试数据模块

验证完整数据流：
    ParquetStorage.save() → ParquetFeed.subscribe() → DataHandler.update() → MarketEvent
"""
import sys
import os
import queue
import shutil

import numpy as np
import pandas as pd

# 确保可以 import 项目模块
sys.path.insert(0, os.path.dirname(__file__))

from data import ParquetStorage, HistoricFeed, DataHandler, Bar
from event import MarketEvent


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
    print(f"✅ 生成测试数据: AAPL ({len(aapl_df)} bars), SPY ({len(spy_df)} bars)")
    return aapl_df, spy_df


def test_parquet_storage():
    """测试 1: ParquetStorage 读写"""
    print("\n" + "=" * 60)
    print("测试 1: ParquetStorage 读写")
    print("=" * 60)

    storage = ParquetStorage('./test_data_cache')
    aapl_df, spy_df = generate_test_data(storage)

    # 读回来验证
    loaded = storage.load('AAPL', frequency='1d')
    assert len(loaded) == len(aapl_df), f"长度不匹配: {len(loaded)} vs {len(aapl_df)}"
    assert list(loaded.columns) == list(aapl_df.columns), "列不匹配"

    # 日期范围过滤
    start = pd.Timestamp('2023-01-10')
    end = pd.Timestamp('2023-01-20')
    filtered = storage.load('AAPL', start, end, '1d')
    assert all(filtered.index >= start), "过滤失败: 有早于 start 的数据"
    assert all(filtered.index <= end), "过滤失败: 有晚于 end 的数据"
    print(f"✅ 日期过滤: {len(filtered)} bars in [{start.date()}, {end.date()}]")

    # exists
    assert storage.exists('AAPL', '1d'), "exists() 应返回 True"
    assert not storage.exists('GOOG', '1d'), "exists() 应返回 False"
    print("✅ ParquetStorage 测试通过")


def test_historic_feed():
    """测试 2: HistoricFeed Iterator"""
    print("\n" + "=" * 60)
    print("测试 2: HistoricFeed Iterator")
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
            assert bar.timestamp >= prev_ts, f"时间线乱序: {prev_ts} -> {bar.timestamp}"
        prev_ts = bar.timestamp

    print(f"✅ 总共消费 {len(bars)} bars")
    print(f"   时间范围: {bars[0].timestamp.date()} → {bars[-1].timestamp.date()}")

    # 验证多标的交错
    symbols_seen = set()
    for bar in bars[:10]:
        symbols_seen.add(bar.symbol)
        print(f"   {bar.timestamp.date()} {bar.symbol:5s} close={bar.close:.2f}")
    assert len(symbols_seen) == 2, "前 10 个 bar 应包含两个标的"
    print("✅ 多标的时间线自动合并、按时间排序")
    print("✅ HistoricFeed 测试通过")


def test_data_handler():
    """测试 3: DataHandler 混合设计（Iterator + Query）"""
    print("\n" + "=" * 60)
    print("测试 3: DataHandler (Iterator + Query)")
    print("=" * 60)

    events = queue.Queue()
    storage = ParquetStorage('./test_data_cache')
    feed = HistoricFeed(storage, frequency='1d')

    start = pd.Timestamp('2023-01-03')
    end = pd.Timestamp('2023-01-31')
    feed.subscribe(['AAPL', 'SPY'], start, end)

    handler = DataHandler(feed, events, cache_size=200)

    # 消费所有数据
    count = 0
    while handler.update():
        count += 1

    print(f"✅ handler.update() 共调用 {count} 次")

    # 验证事件队列
    event_count = 0
    while not events.empty():
        event = events.get()
        assert isinstance(event, MarketEvent), f"期望 MarketEvent, 得到 {type(event)}"
        event_count += 1
    assert event_count == count, f"事件数量不匹配: {event_count} vs {count}"
    print(f"✅ 事件队列收到 {event_count} 个 MarketEvent")

    # 测试查询接口
    latest_1 = handler.get_latest_bar('AAPL')
    assert latest_1 is not None, "get_latest_bar 应返回 Bar"
    print(f"✅ get_latest_bar('AAPL'): {latest_1.timestamp.date()} close={latest_1.close:.2f}")

    latest_5 = handler.get_latest_bars('AAPL', N=5)
    assert len(latest_5) == 5, f"期望 5 个 bar, 得到 {len(latest_5)}"
    # 验证时间正序
    for i in range(1, len(latest_5)):
        assert latest_5[i].timestamp >= latest_5[i-1].timestamp, "get_latest_bars 应返回时间正序"
    print(f"✅ get_latest_bars('AAPL', 5): {[b.timestamp.date() for b in latest_5]}")

    # 测试 DataFrame 查询
    df = handler.get_latest_bars_df('SPY', N=3)
    assert len(df) == 3, f"期望 3 行, 得到 {len(df)}"
    assert 'close' in df.columns, "DataFrame 应包含 close 列"
    print(f"✅ get_latest_bars_df('SPY', 3):\n{df[['close', 'volume']]}")

    # 验证 continue_backtest
    assert not handler.continue_backtest, "数据耗尽后 continue_backtest 应为 False"
    print("✅ DataHandler 测试通过")


def test_bar_validation():
    """测试 4: Bar 数据验证"""
    print("\n" + "=" * 60)
    print("测试 4: Bar 数据验证")
    print("=" * 60)

    # 正常 bar
    bar = Bar(
        timestamp=pd.Timestamp('2023-01-03'),
        symbol='AAPL',
        open=130.0, high=132.0, low=128.0, close=131.0, volume=50000000
    )
    print(f"✅ 正常 Bar: {bar.symbol} {bar.timestamp.date()} O={bar.open} H={bar.high} L={bar.low} C={bar.close}")

    # high < low 应报错
    try:
        Bar(timestamp=pd.Timestamp('2023-01-03'), symbol='BAD',
            open=130.0, high=125.0, low=128.0, close=131.0, volume=1000)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"✅ 拒绝非法 bar (high < low): {e}")

    # frozen 不可变
    try:
        bar.close = 999.0
        assert False, "应该抛出 FrozenInstanceError"
    except AttributeError:
        print("✅ Bar 不可变 (frozen)")

    print("✅ Bar 数据验证测试通过")


def cleanup():
    """清理测试数据"""
    if os.path.exists('./test_data_cache'):
        shutil.rmtree('./test_data_cache')
        print("\n🧹 清理测试数据目录")


if __name__ == '__main__':
    try:
        test_bar_validation()
        test_parquet_storage()
        test_historic_feed()
        test_data_handler()
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("=" * 60)
    finally:
        cleanup()
