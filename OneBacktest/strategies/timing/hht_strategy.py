"""
HHT 择时策略

基于改进 Hilbert-Huang Transform 的择时策略。

核心思路:
    1. 对收盘价做 MA 平滑 + 一阶差分（去趋势、平稳化）
    2. 在滚动窗口上做 Hilbert 变换，提取瞬时相位
    3. 相位 ∈ [-π/2, π/2] 表示上升周期 → 做多
       相位在此区间外 表示下降周期 → 空仓

on_market_close 返回 forecast:
    1.0 = 上升周期 (做多)
    0.0 = 下降周期 (空仓)
"""
from collections import deque
from typing import Dict, List
import numpy as np
from scipy.signal import hilbert

from strategy.base import Strategy
from data.types import Bar


class HHTTimingStrategy(Strategy):
    """
    HHT 择时策略

    Parameters:
        symbols: 标的列表
        ma_period: 移动平均周期（平滑噪声）
        ht_period: Hilbert 变换滚动窗口
    """

    def __init__(self, symbols: List[str], ma_period=60, ht_period=30):
        self.symbols = symbols
        self.ma_period = ma_period
        self.ht_period = ht_period

        buf_size = ma_period + ht_period + 5
        self._closes = {s: deque(maxlen=buf_size) for s in symbols}

    def _calc_signal(self, symbol: str) -> int:
        """
        计算 HT 二值信号

        Returns:
            1 = 做多, 0 = 空仓, -1 = 数据不足
        """
        closes = np.array(self._closes[symbol])
        if len(closes) < self.ma_period + self.ht_period:
            return -1

        ma = np.convolve(closes, np.ones(self.ma_period) / self.ma_period,
                         mode='valid')
        diff = np.diff(ma)
        if len(diff) < self.ht_period:
            return -1

        window = diff[-self.ht_period:]
        analytic = hilbert(window)
        phase = np.angle(analytic)

        current_phase = phase[-1]
        threshold = np.pi * 0.5

        if -threshold <= current_phase <= threshold:
            return 1
        else:
            return 0

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s in self._closes:
            self._closes[s].append(bar.close)

    def on_market_close(self, dt) -> Dict[str, float]:
        forecasts = {}
        for s in self.symbols:
            signal = self._calc_signal(s)
            if signal >= 0:
                forecasts[s] = float(signal)
        return forecasts
