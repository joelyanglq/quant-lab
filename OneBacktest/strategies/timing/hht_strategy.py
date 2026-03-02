"""
HHT 择时策略

基于改进 Hilbert-Huang Transform 的择时策略。
参考: hugo2046/QuantsPlaybook - 结合改进HHT模型和分类算法的交易策略

核心思路:
    1. 对收盘价做 MA 平滑 + 一阶差分（去趋势、平稳化）
    2. 在滚动窗口上做 Hilbert 变换，提取瞬时相位
    3. 相位 ∈ [-π/2, π/2] 表示上升周期 → 做多
       相位在此区间外 表示下降周期 → 平仓

相位象限含义（考虑 Hilbert 变换的 π/2 相位延迟）:
    [0, π/2]     Q1: 上升初期（底部回升）
    [π/2, π]     Q2: 上升末期（接近顶部）
    [-π, -π/2]   Q3: 下降初期
    [-π/2, 0]    Q4: 下降末期（接近底部）
    做多区间 = Q1 + Q4 = [-π/2, π/2]
"""
from collections import deque
from typing import List
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
        position_frac: 每个标的最大仓位占比
    """

    def __init__(self, symbols: List[str], ma_period=60, ht_period=30,
                 position_frac=0.3):
        self.symbols = symbols
        self.ma_period = ma_period
        self.ht_period = ht_period
        self.position_frac = position_frac

        # 需要缓存 ma_period + ht_period 根 close 来计算信号
        buf_size = ma_period + ht_period + 5
        self._closes = {s: deque(maxlen=buf_size) for s in symbols}

    def _calc_signal(self, symbol: str) -> int:
        """
        计算 HT 二值信号

        Returns:
            1 = 做多, 0 = 空仓
        """
        closes = np.array(self._closes[symbol])
        if len(closes) < self.ma_period + self.ht_period:
            return -1  # 数据不足

        # Step 1: MA 平滑
        ma = np.convolve(closes, np.ones(self.ma_period) / self.ma_period, mode='valid')

        # Step 2: 一阶差分（去趋势）
        diff = np.diff(ma)
        if len(diff) < self.ht_period:
            return -1

        # Step 3: 取最近 ht_period 个差分值，做 Hilbert 变换
        window = diff[-self.ht_period:]
        analytic = hilbert(window)
        phase = np.angle(analytic)

        # Step 4: 最后一个相位值判断方向
        current_phase = phase[-1]
        threshold = np.pi * 0.5

        if -threshold <= current_phase <= threshold:
            return 1  # 上升周期 → 做多
        else:
            return 0  # 下降周期 → 空仓

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s not in self._closes:
            return

        self._closes[s].append(bar.close)
        signal = self._calc_signal(s)

        if signal < 0:
            return  # 数据不足，不操作

        pos = self.get_position(s)

        if signal == 1 and pos == 0:
            # 开仓：按 portfolio 比例配仓
            cash_for_trade = self.get_portfolio_value() * self.position_frac
            qty = int(cash_for_trade / bar.close)
            if qty > 0:
                self.buy(s, qty)

        elif signal == 0 and pos > 0:
            # 平仓
            self.sell(s, pos)
