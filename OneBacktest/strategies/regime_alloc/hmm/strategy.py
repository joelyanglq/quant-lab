"""
HMMRegimeStrategy — HMM regime 仓位缩放

调用已有 model.py 中的 fit_hmm / forecast_return_variance,
月频重拟合 (expanding window), 输出 forecast ∈ [0, 1].

支持两种模式:
    1. model_path=None: 月频重拟合 (默认)
    2. model_path='xxx.joblib': 加载预训练模型, 仅做推断 (无 refit)

on_market_close 返回 forecast:
    w* = (1/γ) × (μ/σ²) → clip → 归一化到 [0, 1]
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from strategy.base import Strategy
from .model import fit_hmm, forecast_return_variance, load_model


class HMMRegimeStrategy(Strategy):
    """
    HMM Regime 仓位缩放策略.

    每月重拟合 Gaussian HMM (expanding window),
    根据状态概率加权的预期收益/方差计算最优权重,
    作为 forecast 返回给 CompositeStrategy.

    Parameters:
        index_symbol:  拟合标的 (默认 SPY)
        universe:      forecast 广播到的 symbol 列表 (None → 用 latest_prices)
        n_states:      HMM 隐状态数 (默认 2)
        gamma:         风险厌恶系数 (默认 6)
        n_init:        HMM 随机初始化次数
        min_history:   最少需要的历史天数
        model_path:    预训练模型路径 (.joblib); 若提供则跳过 refit
    """

    def __init__(self,
                 index_symbol: str = 'SPY',
                 universe: List[str] = None,
                 n_states: int = 2,
                 gamma: float = 6.0,
                 n_init: int = 10,
                 min_history: int = 252,
                 model_path: Optional[str] = None):
        self.index_symbol = index_symbol
        self.universe = universe
        self.n_states = n_states
        self.gamma = gamma
        self.n_init = n_init
        self.min_history = min_history
        self.model_path = model_path
        self._last_forecast = None
        self._pretrained_model = None

    def on_init(self):
        if self.model_path is not None:
            p = Path(self.model_path)
            if p.exists():
                model, meta = load_model(p)
                self._pretrained_model = model
                if meta:
                    self.n_states = meta.get('n_states', self.n_states)
                    self.gamma = meta.get('gamma', self.gamma)

    def on_market_close(self, dt) -> Dict[str, float]:
        # 使用缓存的 forecast (月频重算, 日频复用)
        if self._last_forecast is not None:
            return self._broadcast(self._last_forecast)
        return {}

    def on_month_end(self, dt) -> Dict[str, float]:
        closes = self.history.get(self.index_symbol, 'close', 504)
        if len(closes) < self.min_history:
            return {}

        returns = np.diff(closes) / closes[:-1]
        if len(returns) < self.min_history:
            return {}

        try:
            if self._pretrained_model is not None:
                # 预训练模型: 仅推断, 不 refit
                model = self._pretrained_model
            else:
                model = fit_hmm(returns, n_states=self.n_states,
                                n_init=self.n_init)
            exp_ret, exp_var = forecast_return_variance(model, returns)
        except Exception:
            return {}

        if exp_var <= 0:
            self._last_forecast = 0.0
        else:
            w = (1.0 / self.gamma) * (exp_ret / exp_var)
            # w 原始范围 [0, 1.5] → 归一化到 [0, 1]
            self._last_forecast = float(np.clip(w / 1.5, 0.0, 1.0))

        return self._broadcast(self._last_forecast)

    def _broadcast(self, forecast: float) -> Dict[str, float]:
        """将 forecast 广播到 universe 中所有非 index symbol."""
        symbols = self.universe or list(self.latest_prices.keys())
        return {
            s: forecast for s in symbols
            if s != self.index_symbol
        }
