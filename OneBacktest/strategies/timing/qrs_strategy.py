"""
QRS 择时策略

基于 Quantitative Resistance-Support 指标的择时策略。
参考: hugo2046/QuantsPlaybook - QRS择时信号

核心思路:
    1. 在滚动窗口 (N) 内，对 high/low 序列计算:
       beta = std(high) / std(low) * corr(low, high)
       beta 衡量阻力位相对支撑位的强度
    2. 对 beta 时间序列做 z-score 标准化 (窗口 M)，得到标准化信号
    3. 乘以修正项 R^2 = corr(low, high)^2 作为置信权重
       线性关系越强，信号越可信
    4. 最终信号 = zscore_beta * R^2
       信号上穿 upper_bound → 做多（支撑强于阻力，看涨）
       信号下穿 lower_bound → 平仓（阻力增强，看跌）
"""
from collections import deque
from typing import List
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
        upper_bound: 开仓阈值（信号上穿此值做多）
        lower_bound: 平仓阈值（信号下穿此值平仓）
        position_frac: 每个标的最大仓位占比
    """

    def __init__(self, symbols: List[str], regression_window=18,
                 zscore_window=600, upper_bound=0.7, lower_bound=-0.7,
                 position_frac=0.3):
        self.symbols = symbols
        self.regression_window = regression_window
        self.zscore_window = zscore_window
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.position_frac = position_frac

        # 缓存 high/low 用于计算 beta
        buf_size = regression_window + 5
        self._highs = {s: deque(maxlen=buf_size) for s in symbols}
        self._lows = {s: deque(maxlen=buf_size) for s in symbols}

        # 缓存 beta 序列用于 z-score
        self._betas = {s: deque(maxlen=zscore_window) for s in symbols}

        # 上一期信号，用于判断穿越
        self._prev_signal = {s: None for s in symbols}

    def _calc_beta(self, symbol: str) -> float:
        """
        计算当前窗口的 beta 值

        beta = std(high) / std(low) * corr(low, high)

        Returns:
            beta value, or NaN if insufficient data
        """
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
        """
        计算 QRS 信号

        signal = zscore(beta) * R^2

        Returns:
            signal value, or NaN if insufficient data
        """
        betas = self._betas[symbol]
        if len(betas) < self.zscore_window:
            return np.nan

        beta_arr = np.array(betas)
        mean = np.nanmean(beta_arr)
        std = np.nanstd(beta_arr)

        if std == 0:
            return np.nan

        zscore = (beta_arr[-1] - mean) / std

        # R^2: 当前窗口的 corr^2
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

        # 计算 beta 并缓存
        beta = self._calc_beta(s)
        if not np.isnan(beta):
            self._betas[s].append(beta)

        # 计算信号
        signal = self._calc_signal(s)
        if np.isnan(signal):
            return

        prev = self._prev_signal[s]
        self._prev_signal[s] = signal

        if prev is None:
            return

        pos = self.get_position(s)

        # 上穿 upper_bound → 开仓
        if prev <= self.upper_bound < signal and pos == 0:
            cash_for_trade = self.get_portfolio_value() * self.position_frac
            qty = int(cash_for_trade / bar.close)
            if qty > 0:
                self.buy(s, qty)

        # 下穿 lower_bound → 平仓
        elif prev >= self.lower_bound > signal and pos > 0:
            self.sell(s, pos)
