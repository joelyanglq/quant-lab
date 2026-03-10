"""
QRS 择时策略

基于 Quantitative Resistance-Support 指标的择时策略。

核心思路:
    1. 在滚动窗口内，对 high/low 序列计算 beta = std(high)/std(low)*corr(low,high)
    2. 对 beta 时间序列做 z-score 标准化
    3. 乘以 R^2 = corr(low,high)^2 作为置信权重
    4. 最终信号 = zscore_beta * R^2

on_market_close 返回 forecast:
    归一化到 [-1, +1] 的连续信号
"""
from collections import deque
from typing import Dict, List
import numpy as np

from strategy.base import Strategy
from data.types import Bar


class QRSTimingStrategy(Strategy):
    """
    QRS 择时策略

    Parameters:
        symbols: 标的列表
        regression_window: 回归窗口（计算 beta 的滚动周期）
        zscore_window: z-score 标准化窗口
        upper_bound: 强看多阈值
        lower_bound: 强看空阈值
    """

    def __init__(self, symbols: List[str], regression_window=18,
                 zscore_window=600, upper_bound=0.7, lower_bound=-0.7):
        self.symbols = symbols
        self.regression_window = regression_window
        self.zscore_window = zscore_window
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        buf_size = regression_window + 5
        self._highs = {s: deque(maxlen=buf_size) for s in symbols}
        self._lows = {s: deque(maxlen=buf_size) for s in symbols}
        self._betas = {s: deque(maxlen=zscore_window) for s in symbols}

    def _calc_beta(self, symbol: str) -> float:
        highs = np.array(self._highs[symbol])
        lows = np.array(self._lows[symbol])

        if len(highs) < self.regression_window:
            return np.nan

        h = highs[-self.regression_window:]
        l = lows[-self.regression_window:]

        std_h = np.std(h)
        std_l = np.std(l)

        if std_l == 0:
            return np.nan

        corr = np.corrcoef(l, h)[0, 1]
        return std_h / std_l * corr

    def _calc_signal(self, symbol: str) -> float:
        betas = self._betas[symbol]
        if len(betas) < self.zscore_window:
            return np.nan

        beta_arr = np.array(betas)
        mean = np.nanmean(beta_arr)
        std = np.nanstd(beta_arr)

        if std == 0:
            return np.nan

        zscore = (beta_arr[-1] - mean) / std

        highs = np.array(self._highs[symbol])
        lows = np.array(self._lows[symbol])
        h = highs[-self.regression_window:]
        l = lows[-self.regression_window:]
        corr = np.corrcoef(l, h)[0, 1]
        r_squared = corr ** 2

        return zscore * r_squared

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s not in self._highs:
            return

        self._highs[s].append(bar.high)
        self._lows[s].append(bar.low)

        beta = self._calc_beta(s)
        if not np.isnan(beta):
            self._betas[s].append(beta)

    def on_market_close(self, dt) -> Dict[str, float]:
        forecasts = {}
        for s in self.symbols:
            signal = self._calc_signal(s)
            if not np.isnan(signal):
                forecasts[s] = float(np.clip(signal, -1.0, 1.0))
        return forecasts
