"""
VolTargetSizer -- 波动率目标仓位计算器

Rob Carver 式:
    target_position = forecast × (target_vol / realized_vol) × (capital / price)
"""
from typing import Dict

import numpy as np

from data.types import Bar
from data.history import HistoryManager


class VolTargetSizer:
    """
    波动率目标仓位计算器.

    将综合 forecast ([-1, +1]) 转化为目标股数,
    通过 realized vol 缩放使组合波动率趋近 target_vol.
    """

    def __init__(self,
                 target_vol: float = 0.15,
                 vol_lookback: int = 20,
                 max_leverage: float = 1.0,
                 min_forecast: float = 0.05):
        """
        Args:
            target_vol:    年化目标波动率 (default 15%)
            vol_lookback:  realized vol 回看窗口 (交易日, default 20)
            max_leverage:  最大杠杆倍数 (default 1.0)
            min_forecast:  |forecast| 低于此阈值不持仓 (default 0.05)
        """
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.max_leverage = max_leverage
        self.min_forecast = min_forecast

    def size(self,
             combined_forecasts: Dict[str, float],
             history: HistoryManager,
             capital: float,
             latest_prices: Dict[str, Bar]
             ) -> Dict[str, int]:
        """
        Args:
            combined_forecasts: {symbol: forecast} from Combiner
            history:            用于计算 realized vol
            capital:            组合总市值
            latest_prices:      当前价格

        Returns:
            {symbol: target_shares} (正=多头, 负=空头, 0 不出现)
        """
        # 1. 过滤弱信号
        active = {s: f for s, f in combined_forecasts.items()
                  if abs(f) >= self.min_forecast}
        if not active:
            return {}

        # 2. 计算每个 symbol 的 raw allocation
        raw = {}
        for sym, forecast in active.items():
            closes = history.get(sym, 'close', self.vol_lookback + 1)
            if len(closes) < 2:
                continue

            returns = np.diff(closes) / closes[:-1]
            if len(returns) < 2:
                continue
            realized_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
            if realized_vol < 1e-8 or np.isnan(realized_vol):
                continue

            vol_scalar = min(self.target_vol / realized_vol, self.max_leverage)
            raw[sym] = forecast * vol_scalar

        if not raw:
            return {}

        # 3. 归一化: 总 |raw| 不超过 max_leverage
        total_abs = sum(abs(v) for v in raw.values())
        if total_abs > self.max_leverage:
            scale = self.max_leverage / total_abs
            raw = {s: v * scale for s, v in raw.items()}

        # 4. 转换为目标股数
        targets = {}
        for sym, alloc in raw.items():
            bar = latest_prices.get(sym)
            if bar is None or bar.close <= 0:
                continue
            shares = int(capital * alloc / bar.close)
            if shares != 0:
                targets[sym] = shares

        return targets
