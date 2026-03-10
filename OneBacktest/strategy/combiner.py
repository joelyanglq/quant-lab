"""
ForecastCombiner -- 信号合成器

将多个 Alpha 的 forecast 合并为单一综合 forecast.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class ForecastCombiner(ABC):
    """将多个 Alpha 的 forecast 合并为单一综合 forecast."""

    @abstractmethod
    def combine(self,
                forecasts: List[Dict[str, float]],
                weights: List[float]
                ) -> Dict[str, float]:
        """
        Args:
            forecasts: 每个 Alpha 的输出 [{symbol: forecast}, ...]
            weights:   对应的权重 [w1, w2, ...]

        Returns:
            {symbol: combined_forecast}, combined_forecast ∈ [-1, +1]
        """
        ...


class WeightedAvgCombiner(ForecastCombiner):
    """
    加权平均合成.

    combined[sym] = clip(Σ(w_i × f_i[sym]) / Σw_i, -1, +1)

    只有对该 symbol 有观点的 Alpha 参与加权平均
    (缺失 key 的 Alpha 不参与, 而非视为 0).
    """

    def combine(self, forecasts, weights):
        all_symbols = set()
        for f in forecasts:
            all_symbols.update(f.keys())

        combined = {}
        for sym in all_symbols:
            num = 0.0
            den = 0.0
            for f, w in zip(forecasts, weights):
                if sym in f:
                    num += w * f[sym]
                    den += w
            if den > 0:
                combined[sym] = float(np.clip(num / den, -1.0, 1.0))
        return combined


class MultiplicativeCombiner(ForecastCombiner):
    """
    乘性合成.

    combined[sym] = clip(Π f_i[sym], -1, +1)

    适用于 Alpha 之间是"门控"关系 (如 timing × regime).
    缺失 sym 的 Alpha 视为 1.0 (identity), weights 忽略.
    """

    def combine(self, forecasts, weights):
        all_symbols = set()
        for f in forecasts:
            all_symbols.update(f.keys())

        combined = {}
        for sym in all_symbols:
            product = 1.0
            for f in forecasts:
                if sym in f:
                    product *= f[sym]
            combined[sym] = float(np.clip(product, -1.0, 1.0))
        return combined


class LayeredCombiner(ForecastCombiner):
    """
    分层合成: 组内加权平均 → 组间相乘.

    Example:
        groups=[[0,1,2], [3]]
        group_weights=[[0.4, 0.3, 0.3], None]  # None → 组内等权

      Layer 1: avg(alpha[0..2]) → group_forecast_0
      Layer 2: group_forecast_0 × alpha[3] → final
    """

    def __init__(self,
                 groups: List[List[int]],
                 group_weights: Optional[List[Optional[List[float]]]] = None,
                 require_all_groups: bool = False):
        """
        Args:
            groups:             Alpha 索引分组, e.g. [[0,1,2], [3]]
            group_weights:      组内权重. None → 全部等权.
                               单组 None → 该组等权.
            require_all_groups: 若为 True, symbol 必须在所有组中都有 forecast
                               才会出现在最终结果中. 适用于"选股 × 择时"场景,
                               避免非选股 symbol 仅凭择时信号被交易.
        """
        self.groups = groups
        self.require_all_groups = require_all_groups
        if group_weights is None:
            self.group_weights: List[Optional[List[float]]] = [None] * len(groups)
        else:
            self.group_weights = group_weights

    def combine(self, forecasts, weights):
        all_symbols = set()
        for f in forecasts:
            all_symbols.update(f.keys())

        # 计算每组的组内加权平均
        avg = WeightedAvgCombiner()
        group_forecasts = []
        for g_idx, group in enumerate(self.groups):
            gw = self.group_weights[g_idx]
            if gw is None:
                gw = [1.0 / len(group)] * len(group)
            group_f = [forecasts[i] for i in group]
            group_forecasts.append(avg.combine(group_f, gw))

        # 组间相乘
        combined = {}
        n_groups = len(group_forecasts)
        for sym in all_symbols:
            product = 1.0
            n_present = 0
            for gf in group_forecasts:
                if sym in gf:
                    product *= gf[sym]
                    n_present += 1
            if self.require_all_groups and n_present < n_groups:
                continue
            if n_present > 0:
                combined[sym] = float(np.clip(product, -1.0, 1.0))
        return combined
