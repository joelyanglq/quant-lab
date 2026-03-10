"""
CrossSectionAlpha — 截面多因子选股 Alpha

从 history panel 计算价格因子 (技术、反转、alpha101),
横截面 z-score 等权打分, 选出 top_n 股票, 返回正向 forecast.

基本面因子可通过 DataContext 注入:
    data_context.register('EPS_Score', panel)
    data_context.register('PS', panel)

forecast 范围: [0, 1] 对入选股票, 0 对未入选.
"""
import warnings
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from strategy.base import Strategy
from strategies.cross_section.neutralize import (
    VOLUME_BIASED_FACTORS, load_gics_sector_map, neutralize_factors,
)


class CrossSectionAlpha(Strategy):
    """
    截面多因子选股 Alpha.

    Pipeline:
        1. 从 history.panel() 取 OHLCV 面板 (最近 min_history 根 bar)
        2. 计算价格因子 (technical + reversal + alpha101)
        3. 查询 DataContext 中的基本面因子 (可选)
        4. MAD winsorize + z-score 等权打分
        5. 选出 top_n 股票, 返回 forecast ∈ (0, 1]
    """

    def __init__(self,
                 top_n: int = 10,
                 min_history: int = 252,
                 use_alpha101: bool = True,
                 neutralize_sectors: bool = True,
                 factors_config: Optional[Dict[str, int]] = None):
        """
        Args:
            top_n:              选股数量
            min_history:        因子计算需要的最少历史 bar 数 (RS_12M 需 252)
            use_alpha101:       是否使用 alpha101 因子
            neutralize_sectors: 对基本面因子做行业中性化 (减行业均值)
            factors_config:     {factor_compute_key: ic_direction}
                               None → 从 factor_registry.csv 自动加载
        """
        self.top_n = top_n
        self.min_history = min_history
        self.use_alpha101 = use_alpha101
        self.neutralize_sectors = neutralize_sectors
        self._factors_config = factors_config
        self._selected: Dict[str, int] = {}
        self._factors_alpha101: Set[str] = set()
        self._sector_map: Optional[pd.Series] = None
        self._context_factor_names: Set[str] = set()

    def on_init(self):
        if self._factors_config is not None:
            self._selected = dict(self._factors_config)
            self._factors_alpha101 = {
                k for k in self._selected if k.startswith('alpha_')
            }
        else:
            from strategies.cross_section.pick_stocks import load_active_factors
            self._selected, _, self._factors_alpha101, _ = load_active_factors(
                use_1min=False, use_alpha101=self.use_alpha101)
        if not self.use_alpha101:
            self._factors_alpha101 = set()

        # Load GICS sector mapping for industry neutralization
        if self.neutralize_sectors:
            self._sector_map = load_gics_sector_map()

    def on_market_close(self, dt) -> Dict[str, float]:
        n = min(self.min_history, 504)
        close = self.history.panel('close', n)
        if len(close) < 60:
            return {}

        high = self.history.panel('high', n)
        low = self.history.panel('low', n)
        open_p = self.history.panel('open', n)
        volume = self.history.panel('volume', n)

        factors = self._compute_factors(close, high, low, open_p, volume)

        # Query DataContext for additional factors
        if self.data is not None:
            self._add_context_factors(dt, close, factors)

        # Filter to selected only
        factors = {k: v for k, v in factors.items() if k in self._selected}
        if not factors:
            return {}

        # Score and rank
        scored = self._score_stocks(factors)
        if scored.empty:
            return {}

        return self._to_forecasts(scored)

    # ── Internal ──

    def _compute_factors(self, close, high, low, open_p, volume):
        """Compute price-based factors from OHLCV panels."""
        factors = {}

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            try:
                from strategies.cross_section.factors import (
                    compute_technical_factors,
                )
                factors.update(compute_technical_factors(close, high, low))
            except Exception:
                pass

            try:
                from strategies.cross_section.factors import (
                    compute_reversal_factors,
                )
                factors.update(compute_reversal_factors(close, open_p))
            except Exception:
                pass

            if self.use_alpha101 and self._factors_alpha101:
                try:
                    from strategies.cross_section.alpha101 import Alphas
                    alphas_obj = Alphas(close, open_p, high, low, volume)
                    for fid in self._factors_alpha101:
                        method = getattr(alphas_obj, fid, None)
                        if method is None:
                            continue
                        val = method()
                        if isinstance(val, pd.DataFrame) and not val.empty:
                            factors[fid] = val.replace(
                                [np.inf, -np.inf], np.nan)
                except Exception:
                    pass

        return factors

    def _add_context_factors(self, dt, close, factors):
        """Query DataContext for fundamental factors not in price factors."""
        self._context_factor_names = set()
        for name in self._selected:
            if name in factors:
                continue
            if not self.data.has(name):
                continue
            try:
                val = self.data.as_of(dt, name)
                if isinstance(val, pd.Series):
                    df = pd.DataFrame([val])
                    df.index = [close.index[-1]]
                    factors[name] = df
                    self._context_factor_names.add(name)
            except (ValueError, KeyError):
                pass

    def _score_stocks(self, factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Cross-sectional z-score scoring.

        Takes latest row from each factor panel, MAD winsorizes,
        z-scores, adjusts for IC direction, equal-weight average.

        Sector-biased factors (fundamentals + volume-biased) are
        industry-neutralized before z-scoring if neutralize_sectors=True.
        """
        latest = {}
        for name, panel in factors.items():
            row = panel.iloc[-1].dropna()
            if len(row) > 0:
                latest[name] = row

        if not latest:
            return pd.DataFrame()

        df = pd.DataFrame(latest)

        # Industry neutralization: subtract sector mean for biased factors
        if self.neutralize_sectors and self._sector_map is not None:
            cols = self._context_factor_names | (VOLUME_BIASED_FACTORS & set(df.columns))
            if cols:
                df = neutralize_factors(df, self._sector_map, cols)

        # MAD winsorize + z-score per factor
        for col in df.columns:
            direction = self._selected.get(col, +1)
            s = df[col]
            median = s.median()
            mad = (s - median).abs().median()
            if mad > 0:
                cutoff = 3 * 1.4826 * mad
                s = s.clip(median - cutoff, median + cutoff)
            mu, sd = s.mean(), s.std()
            if sd > 0:
                df[col] = direction * (s - mu) / sd
            else:
                df[col] = 0.0

        df = df.fillna(0.0)
        df['score'] = df.mean(axis=1)
        return df.sort_values('score', ascending=False)

    def _to_forecasts(self, scored: pd.DataFrame) -> Dict[str, float]:
        """Convert top_n scores to forecasts in (0, 1]."""
        top = scored.head(self.top_n)
        max_score = top['score'].iloc[0]
        if max_score <= 0:
            return {}

        forecasts = {}
        for sym in top.index:
            score = top.loc[sym, 'score']
            if score > 0:
                forecasts[sym] = float(np.clip(score / max_score, 0.05, 1.0))
        return forecasts
