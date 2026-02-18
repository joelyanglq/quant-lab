"""
周频因子调仓策略 (WeeklyFactorStrategy)

演示 on_week_end + self.history.panel + self.rebalance_to 的用法:
  - 每周五收盘后, 取最近 N 天 close 面板
  - 计算动量 + 反波动率因子, z-score 合成
  - 持有得分最高的 top_n 只股票, 等权
"""
from typing import List

import numpy as np
import pandas as pd

from strategy.base import Strategy
from data.types import Bar


def _zscore(s: pd.Series) -> pd.Series:
    """截面 z-score (drop NaN)."""
    s = s.dropna()
    mu, sd = s.mean(), s.std()
    if sd > 0:
        return (s - mu) / sd
    return s * 0.0


class WeeklyFactorStrategy(Strategy):
    """
    周频因子调仓.

    Args:
        symbols: 股票池
        top_n: 持仓数量
        lookback: 因子计算回看天数
        min_history: 首次调仓前至少需要的 bar 数
    """

    def __init__(self, symbols: List[str], top_n: int = 10,
                 lookback: int = 60, min_history: int = 80):
        self.symbols = symbols
        self.top_n = top_n
        self.lookback = lookback
        self.min_history = min_history

    def on_bar(self, bar: Bar):
        pass

    def on_week_end(self, dt):
        close = self.history.panel('close', self.min_history)
        if len(close) < self.min_history:
            return

        # 动量: trailing return
        mom = close.iloc[-1] / close.iloc[-self.lookback] - 1

        # 反波动率: lower vol → higher score
        log_ret = np.log(close / close.shift(1)).iloc[-self.lookback:]
        vol = log_ret.std()
        inv_vol = 1.0 / vol.replace(0, np.nan)

        # z-score 合成
        score = _zscore(mom) + _zscore(inv_vol)
        score = score.dropna()

        if len(score) < self.top_n:
            return

        top = score.nlargest(self.top_n).index.tolist()
        self.rebalance_to(top)
