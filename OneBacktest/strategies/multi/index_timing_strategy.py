"""
IndexTimingStrategy — SPY 级别 HHT + QRS 择时, 广播到全 universe

调用 signals.py 中的纯函数 compute_hht / compute_qrs,
在 SPY 上计算择时信号, 广播到所有持仓股票.

与 per-symbol HHT/QRS Strategy 不同:
    - 这里只跟踪 index_symbol (SPY)
    - 结果广播到全 universe (乘性缩放)

on_market_close 返回 forecast:
    HHT=BUY + QRS=BUY  → 1.0
    HHT=BUY + QRS=HOLD → 0.3~1.0 (插值)
    HHT=BUY + QRS=SELL → 0.3
    HHT=FLAT           → 0.0
"""
from typing import Dict

import numpy as np

from strategy.base import Strategy
from strategies.timing.signals import compute_hht, compute_qrs


class IndexTimingStrategy(Strategy):
    """
    SPY 择时 → 全 universe 广播.

    Parameters:
        index_symbol:  择时标的 (默认 SPY)
        hht_ma:        HHT MA 平滑窗口
        hht_ht:        HHT Hilbert 变换窗口
        qrs_reg_w:     QRS 回归窗口
        qrs_zscore_w:  QRS z-score 窗口
        qrs_upper:     QRS 买入阈值
        qrs_lower:     QRS 卖出阈值
    """

    def __init__(self,
                 index_symbol: str = 'SPY',
                 hht_ma: int = 60,
                 hht_ht: int = 30,
                 qrs_reg_w: int = 18,
                 qrs_zscore_w: int = 250,
                 qrs_upper: float = 0.7,
                 qrs_lower: float = -0.7):
        self.index_symbol = index_symbol
        self.hht_ma = hht_ma
        self.hht_ht = hht_ht
        self.qrs_reg_w = qrs_reg_w
        self.qrs_zscore_w = qrs_zscore_w
        self.qrs_upper = qrs_upper
        self.qrs_lower = qrs_lower
        self._min_bars = max(hht_ma + hht_ht, qrs_reg_w + qrs_zscore_w)

    def on_market_close(self, dt) -> Dict[str, float]:
        closes = self.history.get(
            self.index_symbol, 'close', self._min_bars + 10)
        if len(closes) < self.hht_ma + self.hht_ht:
            return {}

        hht_sig, _ = compute_hht(closes, self.hht_ma, self.hht_ht)

        if hht_sig == -1:
            return {}

        if hht_sig == 0:
            timing_forecast = 0.0
        else:
            timing_forecast = self._refine_with_qrs()

        forecasts = {}
        for sym in self.latest_prices:
            if sym != self.index_symbol:
                forecasts[sym] = timing_forecast
        return forecasts

    def _refine_with_qrs(self) -> float:
        """HHT=BUY 时用 QRS 细化, 输出 [0.3, 1.0]."""
        highs = self.history.get(
            self.index_symbol, 'high', self._min_bars + 10)
        lows = self.history.get(
            self.index_symbol, 'low', self._min_bars + 10)

        if len(highs) < self.qrs_reg_w + self.qrs_zscore_w:
            return 0.7

        qrs_code, qrs_val, _ = compute_qrs(
            highs, lows,
            self.qrs_reg_w, self.qrs_zscore_w,
            self.qrs_upper, self.qrs_lower,
        )

        if qrs_code == -99:
            return 0.7
        if qrs_code == 1:
            return 1.0
        if qrs_code == -1:
            return 0.3

        # QRS=HOLD → interpolate
        if qrs_val is not None:
            t = float(np.clip(
                (qrs_val - self.qrs_lower) /
                (self.qrs_upper - self.qrs_lower),
                0.0, 1.0,
            ))
            return 0.3 + 0.7 * t
        return 0.7
