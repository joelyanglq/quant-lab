"""
101 Formulaic Alphas (Kakushadze, 2016)

复现论文全部 101 个公式化 Alpha 因子.
输入: OHLCV 日频面板 (dates × symbols)
输出: 每个 alpha 为 dates × symbols 截面面板

Reference:
  Kakushadze, Z. (2016). 101 Formulaic Alphas.
  Available at SSRN: https://ssrn.com/abstract=2701346

Usage:
    from strategies.cross_section.alpha101 import Alphas
    a = Alphas(close, open_, high, low, volume)
    all_alphas = a.compute_all()
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# Part 1: Operators
# ═══════════════════════════════════════════════════════════════

def rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank (per date, across symbols)."""
    return x.rank(axis=1, pct=True)


def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Value of x d days ago."""
    return x.shift(int(d))


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Today's value minus d days ago."""
    return x - x.shift(int(d))


def ts_sum(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=int(d)).sum()


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=int(d)).mean()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=max(2, int(d))).std()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=int(d)).min()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=int(d)).max()


def ts_argmax(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Which position in the last d days had the max (0=oldest, d-1=most recent)."""
    d = int(d)
    arr = x.values.astype(float)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(d - 1, rows):
        window = arr[i - d + 1:i + 1, :]
        all_nan = np.isnan(window).all(axis=0)
        # np.nanargmax raises on all-NaN slices, so mask them
        safe_window = window.copy()
        safe_window[:, all_nan] = 0
        out[i, :] = np.nanargmax(safe_window, axis=0).astype(float)
        out[i, all_nan] = np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_argmin(x: pd.DataFrame, d: int) -> pd.DataFrame:
    d = int(d)
    arr = x.values.astype(float)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(d - 1, rows):
        window = arr[i - d + 1:i + 1, :]
        all_nan = np.isnan(window).all(axis=0)
        safe_window = window.copy()
        safe_window[:, all_nan] = 0
        out[i, :] = np.nanargmin(safe_window, axis=0).astype(float)
        out[i, all_nan] = np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Percentile rank of current value among past d values (vectorized)."""
    d = int(d)
    arr = x.values.astype(float)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    min_obs = max(2, d // 2)  # Allow partial windows
    for i in range(d - 1, rows):
        window = arr[i - d + 1:i + 1, :]  # (d, cols)
        current = window[-1, :]
        valid = ~np.isnan(window)
        n_valid = valid.sum(axis=0)
        # count(x_j <= x_current) / n_valid
        le = (window <= current[np.newaxis, :]) & valid
        pct = le.sum(axis=0).astype(float) / np.maximum(n_valid, 1)
        pct[n_valid < min_obs] = np.nan
        pct[np.isnan(current)] = np.nan
        out[i, :] = pct
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_product(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(int(d), min_periods=int(d)).apply(np.prod, raw=True)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Element-wise rolling correlation over d days."""
    return x.rolling(int(d), min_periods=int(d)).corr(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    """Element-wise rolling covariance over d days."""
    return x.rolling(int(d), min_periods=int(d)).cov(y)


def decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    """Linearly decaying weighted moving average (weights 1, 2, ..., d), vectorized."""
    d = int(d)
    weights = np.arange(1, d + 1, dtype=float)
    weights /= weights.sum()
    arr = x.values.astype(float)
    rows, cols = arr.shape
    out = np.full((rows, cols), np.nan)
    for i in range(d - 1, rows):
        window = arr[i - d + 1:i + 1, :]  # (d, cols)
        has_nan = np.isnan(window).any(axis=0)
        result = (weights[:, np.newaxis] * window).sum(axis=0)
        result[has_nan] = np.nan
        out[i, :] = result
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """Rescale x such that sum(abs(x)) = a per date."""
    abs_sum = x.abs().sum(axis=1)
    abs_sum = abs_sum.replace(0, 1)
    return x.mul(a / abs_sum, axis=0)


def signedpower(x, a: float):
    """sign(x) * abs(x)^a."""
    return np.sign(x) * np.abs(x) ** a


def log(x):
    return np.log(x.replace(0, np.nan))


def sign(x):
    return np.sign(x)


def indneutralize(x: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Cross-sectionally demean x within each industry group."""
    result = x.copy()
    for g in groups.dropna().unique():
        syms = groups[groups == g].index.tolist()
        common = [s for s in syms if s in x.columns]
        if len(common) > 1:
            means = x[common].mean(axis=1)
            result[common] = x[common].sub(means, axis=0)
    return result


def _where(cond, x, y):
    """Element-wise ternary: cond ? x : y."""
    if isinstance(cond, pd.DataFrame):
        idx, cols = cond.index, cond.columns
    elif isinstance(x, pd.DataFrame):
        idx, cols = x.index, x.columns
    else:
        idx, cols = y.index, y.columns
    return pd.DataFrame(
        np.where(cond, x, y), index=idx, columns=cols)


# ═══════════════════════════════════════════════════════════════
# Part 2: Alphas Class
# ═══════════════════════════════════════════════════════════════

class Alphas:
    """101 Formulaic Alphas from Kakushadze (2016)."""

    def __init__(
        self,
        close: pd.DataFrame,
        open_: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
        vwap: Optional[pd.DataFrame] = None,
        cap: Optional[pd.DataFrame] = None,
        sector: Optional[pd.Series] = None,
    ):
        self.close = close
        self.open = open_
        self.high = high
        self.low = low
        self.volume = volume
        self.returns = close.pct_change()
        self.vwap = vwap if vwap is not None else (high + low + close) / 3
        self.cap = cap
        self.sector = sector  # Series: symbol -> sector_name

        # Pre-compute adv (average daily dollar volume)
        dv = close * volume
        self._adv = {}
        for d in [5, 10, 15, 20, 30, 40, 50, 60, 81, 120, 150, 180]:
            self._adv[d] = dv.rolling(d, min_periods=d).mean()

    def _adv_d(self, d: int) -> pd.DataFrame:
        if d not in self._adv:
            self._adv[d] = (self.close * self.volume).rolling(d, min_periods=d).mean()
        return self._adv[d]

    def _ind(self, x: pd.DataFrame) -> pd.DataFrame:
        """Industry-neutralize if sector data available, else return x."""
        if self.sector is not None:
            return indneutralize(x, self.sector)
        return x

    # ───────────────────────────────────────────────────────
    # Alpha implementations
    # ───────────────────────────────────────────────────────

    def alpha_001(self):
        inner = _where(self.returns < 0, ts_std(self.returns, 20), self.close)
        return rank(ts_argmax(signedpower(inner, 2.0), 5)) - 0.5

    def alpha_002(self):
        return -1 * ts_corr(rank(delta(log(self.volume), 2)),
                            rank((self.close - self.open) / self.open), 6)

    def alpha_003(self):
        return -1 * ts_corr(rank(self.open), rank(self.volume), 10)

    def alpha_004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha_005(self):
        return rank(self.open - ts_mean(self.vwap, 10)) * (-1 * rank(self.close - self.vwap).abs())

    def alpha_006(self):
        return -1 * ts_corr(self.open, self.volume, 10)

    def alpha_007(self):
        adv20 = self._adv_d(20)
        d7 = delta(self.close, 7)
        return _where(adv20 < self.volume,
                      -1 * ts_rank(d7.abs(), 60) * sign(d7),
                      -1.0)

    def alpha_008(self):
        x = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -1 * rank(x - delay(x, 10))

    def alpha_009(self):
        d1 = delta(self.close, 1)
        return _where(0 < ts_min(d1, 5), d1,
                      _where(ts_max(d1, 5) < 0, d1, -1 * d1))

    def alpha_010(self):
        d1 = delta(self.close, 1)
        inner = _where(0 < ts_min(d1, 4), d1,
                       _where(ts_max(d1, 4) < 0, d1, -1 * d1))
        return rank(inner)

    def alpha_011(self):
        return (rank(ts_max(self.vwap - self.close, 3)) +
                rank(ts_min(self.vwap - self.close, 3))) * rank(delta(self.volume, 3))

    def alpha_012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha_013(self):
        return -1 * rank(ts_cov(rank(self.close), rank(self.volume), 5))

    def alpha_014(self):
        return -1 * rank(delta(self.returns, 3)) * ts_corr(self.open, self.volume, 10)

    def alpha_015(self):
        return -1 * ts_sum(rank(ts_corr(rank(self.high), rank(self.volume), 3)), 3)

    def alpha_016(self):
        return -1 * rank(ts_cov(rank(self.high), rank(self.volume), 5))

    def alpha_017(self):
        return (-1 * rank(ts_rank(self.close, 10)) *
                rank(delta(delta(self.close, 1), 1)) *
                rank(ts_rank(self.volume / self._adv_d(20), 5)))

    def alpha_018(self):
        return -1 * rank(ts_std((self.close - self.open).abs(), 5) +
                         (self.close - self.open) +
                         ts_corr(self.close, self.open, 10))

    def alpha_019(self):
        d7 = self.close - delay(self.close, 7)
        return (-1 * sign(d7 + delta(self.close, 7)) *
                (1 + rank(1 + ts_sum(self.returns, 250))))

    def alpha_020(self):
        return (-1 * rank(self.open - delay(self.high, 1)) *
                rank(self.open - delay(self.close, 1)) *
                rank(self.open - delay(self.low, 1)))

    def alpha_021(self):
        sma8 = ts_mean(self.close, 8)
        std8 = ts_std(self.close, 8)
        sma2 = ts_mean(self.close, 2)
        vol_ratio = self.volume / self._adv_d(20)
        return _where(sma8 + std8 < sma2, -1.0,
                      _where(sma2 < sma8 - std8, 1.0,
                             _where((vol_ratio >= 1), 1.0, -1.0)))

    def alpha_022(self):
        return -1 * delta(ts_corr(self.high, self.volume, 5), 5) * rank(ts_std(self.close, 20))

    def alpha_023(self):
        return _where(ts_mean(self.high, 20) < self.high,
                      -1 * delta(self.high, 2), 0.0)

    def alpha_024(self):
        cond = delta(ts_mean(self.close, 100), 100) / delay(self.close, 100)
        return _where(cond <= 0.05,
                      -1 * (self.close - ts_min(self.close, 100)),
                      -1 * delta(self.close, 3))

    def alpha_025(self):
        return rank(-1 * self.returns * self._adv_d(20) * self.vwap * (self.high - self.close))

    def alpha_026(self):
        return -1 * ts_max(ts_corr(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def alpha_027(self):
        x = ts_mean(ts_corr(rank(self.volume), rank(self.vwap), 6), 2)
        return _where(rank(x) > 0.5, -1.0, 1.0)

    def alpha_028(self):
        return scale(ts_corr(self._adv_d(20), self.low, 5) +
                     (self.high + self.low) / 2 - self.close)

    def alpha_029(self):
        x = rank(rank(scale(log(ts_sum(ts_min(rank(rank(-1 * rank(delta(self.close - 1, 5)))), 2), 1)))))
        y = ts_rank(delay(-1 * self.returns, 6), 5)
        return ts_min(ts_product(x, 1), 5) + y

    def alpha_030(self):
        x = (sign(self.close - delay(self.close, 1)) +
             sign(delay(self.close, 1) - delay(self.close, 2)) +
             sign(delay(self.close, 2) - delay(self.close, 3)))
        return (1.0 - rank(x)) * ts_sum(self.volume, 5) / ts_sum(self.volume, 20)

    def alpha_031(self):
        return (rank(rank(rank(decay_linear(-1 * rank(rank(delta(self.close, 10))), 10)))) +
                rank(-1 * delta(self.close, 3)) +
                sign(scale(ts_corr(self._adv_d(20), self.low, 12))))

    def alpha_032(self):
        return (scale(ts_mean(self.close, 7) - self.close) +
                20 * scale(ts_corr(self.vwap, delay(self.close, 5), 230)))

    def alpha_033(self):
        return rank(-1 * (1 - self.open / self.close))

    def alpha_034(self):
        return rank(1 - rank(ts_std(self.returns, 2) / ts_std(self.returns, 5)) +
                    (1 - rank(delta(self.close, 1))))

    def alpha_035(self):
        return (ts_rank(self.volume, 32) *
                (1 - ts_rank(self.close + self.high - self.low, 16)) *
                (1 - ts_rank(self.returns, 32)))

    def alpha_036(self):
        return (2.21 * rank(ts_corr(self.close - self.open, delay(self.volume, 1), 15)) +
                0.7 * rank(self.open - self.close) +
                0.73 * rank(ts_rank(delay(-1 * self.returns, 6), 5)) +
                rank(ts_corr(self.vwap, self._adv_d(20), 6).abs()) +
                0.6 * rank((ts_mean(self.close, 200) - self.open) * (self.close - self.open)))

    def alpha_037(self):
        return (rank(ts_corr(delay(self.open - self.close, 1), self.close, 200)) +
                rank(self.open - self.close))

    def alpha_038(self):
        return -1 * rank(ts_rank(self.close, 10)) * rank(self.close / self.open)

    def alpha_039(self):
        return (-1 * rank(delta(self.close, 7) *
                         (1 - rank(decay_linear(self.volume / self._adv_d(20), 9)))) *
                (1 + rank(ts_sum(self.returns, 250))))

    def alpha_040(self):
        return -1 * rank(ts_std(self.high, 10)) * ts_corr(self.high, self.volume, 10)

    def alpha_041(self):
        return (self.high * self.low) ** 0.5 - self.vwap

    def alpha_042(self):
        return rank(self.vwap - self.close) / rank(self.vwap + self.close)

    def alpha_043(self):
        return ts_rank(self.volume / self._adv_d(20), 20) * ts_rank(-1 * delta(self.close, 7), 8)

    def alpha_044(self):
        return -1 * ts_corr(self.high, rank(self.volume), 5)

    def alpha_045(self):
        return (-1 * rank(ts_mean(delay(self.close, 5), 20)) *
                ts_corr(self.close, self.volume, 2) *
                rank(ts_corr(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)))

    def alpha_046(self):
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10
        y = (delay(self.close, 10) - self.close) / 10
        cond1 = x - y
        return _where(cond1 > 0.25, -1.0,
                      _where(cond1 < 0, 1.0,
                             -1 * delta(self.close, 1)))

    def alpha_047(self):
        return ((rank(1 / self.close) * self.volume / self._adv_d(20)) *
                (self.high * rank(self.high - self.close) / (ts_mean(self.high, 5))) -
                rank(self.vwap - delay(self.vwap, 5)))

    def alpha_048(self):
        if self.sector is None:
            return None
        x = ts_corr(delta(self.close, 1), delta(delay(self.close, 1), 1), 250)
        x = indneutralize(x * delta(self.close, 1) / self.close, self.sector)
        denom = ts_sum((delta(self.close, 1) / delay(self.close, 1)) ** 2, 250)
        return x / denom

    def alpha_049(self):
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10
        y = (delay(self.close, 10) - self.close) / 10
        cond = x - y
        return _where(cond < -0.1, 1.0, -1 * delta(self.close, 1))

    def alpha_050(self):
        return -1 * ts_max(rank(ts_corr(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha_051(self):
        x = (delay(self.close, 20) - delay(self.close, 10)) / 10
        y = (delay(self.close, 10) - self.close) / 10
        cond = x - y
        return _where(cond < -0.05, 1.0, -1 * delta(self.close, 1))

    def alpha_052(self):
        return (((-1 * ts_min(self.low, 5) + delay(ts_min(self.low, 5), 5)) *
                 rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)) *
                ts_rank(self.volume, 5))

    def alpha_053(self):
        return -1 * delta(((self.close - self.low) - (self.high - self.close)) /
                          (self.close - self.low + 1e-8), 9)

    def alpha_054(self):
        return -1 * (self.low - self.close) * (self.open ** 5) / ((self.low - self.high + 1e-8) * (self.close ** 5))

    def alpha_055(self):
        hl_range = ts_max(self.high, 12) - ts_min(self.low, 12)
        hl_range = hl_range.replace(0, 1e-8)
        x = (self.close - ts_min(self.low, 12)) / hl_range
        return -1 * ts_corr(rank(x), rank(self.volume), 6)

    def alpha_056(self):
        if self.cap is None:
            return None
        return -1 * rank(ts_sum(self.returns, 10) /
                         ts_sum(ts_sum(self.returns, 2), 3)) * rank(self.returns * self.cap)

    def alpha_057(self):
        return -(self.close - self.vwap) / decay_linear(rank(ts_argmax(self.close, 30)), 2)

    def alpha_058(self):
        if self.sector is None:
            return None
        return -1 * ts_rank(decay_linear(
            ts_corr(indneutralize(self.vwap, self.sector), self.volume, 3), 7), 5)

    def alpha_059(self):
        if self.sector is None:
            return None
        return -1 * ts_rank(decay_linear(
            ts_corr(indneutralize(self.vwap, self.sector), self.volume, 4), 16), 8)

    def alpha_060(self):
        x = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-8)
        return -(2 * scale(rank(x * self.volume)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha_061(self):
        a = rank(self.vwap - ts_min(self.vwap, 16))
        b = rank(ts_corr(self.vwap, self._adv_d(180), 17))
        return (a < b).astype(float)

    def alpha_062(self):
        a = rank(ts_corr(self.vwap, ts_sum(self._adv_d(20), 22), 9))
        b = rank((rank(self.open) + rank(self.open)) <
                 (rank((self.high + self.low) / 2) + rank(self.high)))
        return _where(a < b, -1.0, 0.0)

    def alpha_063(self):
        if self.sector is None:
            return None
        a = rank(decay_linear(delta(indneutralize(self.close, self.sector), 2), 8))
        b = rank(decay_linear(ts_corr(
            self.vwap * 0.318108 + self.open * 0.681892,
            ts_sum(self._adv_d(180), 37), 13), 12))
        return (a - b) * -1

    def alpha_064(self):
        a = rank(ts_corr(ts_sum(self.open * 0.178404 + self.low * 0.821596, 12),
                         ts_sum(self._adv_d(120), 12), 16))
        b = rank(delta((self.high + self.low) / 2 * 0.178404 + self.vwap * 0.821596, 3))
        return _where(a < b, -1.0, 0.0)

    def alpha_065(self):
        a = rank(ts_corr(self.open * 0.00817205 + self.vwap * 0.99182795,
                         ts_sum(self._adv_d(60), 8), 6))
        b = rank(self.open - ts_min(self.open, 13))
        return _where(a < b, -1.0, 0.0)

    def alpha_066(self):
        a = rank(decay_linear(delta(self.vwap, 3), 7))
        b = ts_rank(decay_linear(
            (self.low - self.vwap) / (self.open - (self.high + self.low) / 2 + 1e-8), 11), 6)
        return (a + b) * -1

    def alpha_067(self):
        if self.sector is None:
            return None
        a = rank(self.high - ts_min(self.high, 2))
        b = rank(ts_corr(indneutralize(self.vwap, self.sector),
                         indneutralize(self._adv_d(20), self.sector), 6))
        return signedpower(a, b) * -1

    def alpha_068(self):
        a = ts_rank(ts_corr(rank(self.high), rank(self._adv_d(15)), 8), 13)
        b = rank(delta(self.close * 0.518371 + self.low * 0.481629, 1))
        return _where(a < b, -1.0, 0.0)

    def alpha_069(self):
        if self.sector is None:
            return None
        a = rank(ts_max(delta(indneutralize(self.vwap, self.sector), 2), 4))
        b = ts_rank(ts_corr(self.close * 0.490655 + self.vwap * 0.509345,
                            self._adv_d(20), 4), 9)
        return signedpower(a, b) * -1

    def alpha_070(self):
        if self.sector is None:
            return None
        a = rank(delta(self.vwap, 1))
        b = ts_rank(ts_corr(indneutralize(self.close, self.sector),
                            self._adv_d(50), 17), 17)
        return signedpower(a, b) * -1

    def alpha_071(self):
        a = ts_rank(decay_linear(ts_corr(ts_rank(self.close, 3),
                                         ts_rank(self._adv_d(180), 12), 18), 4), 15)
        b = ts_rank(decay_linear(
            rank((self.low + self.open - 2 * self.vwap) ** 2), 16), 4)
        return pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns)

    def alpha_072(self):
        a = rank(decay_linear(ts_corr((self.high + self.low) / 2,
                                      self._adv_d(40), 8), 10))
        b = rank(decay_linear(ts_corr(ts_rank(self.vwap, 3),
                                      ts_rank(self.volume, 18), 6), 2))
        return a / b.replace(0, np.nan)

    def alpha_073(self):
        a = rank(decay_linear(delta(self.vwap, 4), 2))
        b = ts_rank(decay_linear(
            delta(self.open * 0.147155 + self.low * 0.852845, 2) /
            (self.open * 0.147155 + self.low * 0.852845 + 1e-8) * -1, 3), 16)
        return pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns) * -1

    def alpha_074(self):
        a = rank(ts_corr(self.close, ts_sum(self._adv_d(30), 37), 15))
        b = rank(ts_corr(rank(self.high * 0.0261661 + self.vwap * 0.9738339),
                         rank(self.volume), 11))
        return _where(a < b, -1.0, 0.0)

    def alpha_075(self):
        a = rank(ts_corr(self.vwap, self.volume, 4))
        b = rank(ts_corr(rank(self.low), rank(self._adv_d(50)), 12))
        return (a < b).astype(float)

    def alpha_076(self):
        if self.sector is None:
            return None
        a = rank(decay_linear(delta(self.vwap, 1), 11))
        b = ts_rank(decay_linear(ts_rank(ts_corr(
            indneutralize(self.low, self.sector), self._adv_d(81), 8), 19), 17), 19)
        return pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns) * -1

    def alpha_077(self):
        a = rank(decay_linear((self.high + self.low) / 2 + self.high - self.vwap - self.high, 20))
        b = rank(decay_linear(ts_corr((self.high + self.low) / 2,
                                      self._adv_d(40), 3), 5))
        return pd.DataFrame(np.minimum(a, b), index=a.index, columns=a.columns)

    def alpha_078(self):
        a = rank(ts_corr(ts_sum(self.low * 0.352233 + self.vwap * 0.647767, 19),
                         ts_sum(self._adv_d(40), 19), 6))
        b = rank(ts_corr(rank(self.vwap), rank(self.volume), 5))
        return signedpower(a, b)

    def alpha_079(self):
        if self.sector is None:
            return None
        a = rank(delta(indneutralize(
            self.close * 0.60733 + self.open * 0.39267, self.sector), 1))
        b = rank(ts_corr(ts_rank(self.vwap, 3), ts_rank(self._adv_d(150), 9), 14))
        return (a < b).astype(float)

    def alpha_080(self):
        if self.sector is None:
            return None
        a = rank(sign(delta(indneutralize(
            self.open * 0.868128 + self.high * 0.131872, self.sector), 4)))
        b = ts_rank(ts_corr(self.high, self._adv_d(10), 5), 5)
        return signedpower(a, b) * -1

    def alpha_081(self):
        if self.sector is None:
            return None
        x = rank(ts_corr(self.vwap, ts_sum(self._adv_d(10), 49), 8))
        a = rank(log(ts_product(rank(signedpower(x, 4)), 14)))
        b = rank(ts_corr(rank(self.vwap), rank(self.volume), 5))
        return _where(a < b, -1.0, 0.0)

    def alpha_082(self):
        if self.sector is None:
            return None
        a = rank(decay_linear(delta(self.open, 1), 14))
        b = ts_rank(decay_linear(ts_corr(
            indneutralize(self.volume, self.sector), self.open, 17), 6), 13)
        return pd.DataFrame(np.minimum(a, b), index=a.index, columns=a.columns) * -1

    def alpha_083(self):
        x = (self.high - self.low) / (ts_mean(self.close, 5) + 1e-8)
        return rank(delay(x, 2)) * rank(rank(self.volume)) / (x / (self.vwap - self.close + 1e-8))

    def alpha_084(self):
        return signedpower(ts_rank(self.vwap - ts_max(self.vwap, 15), 20),
                           delta(self.close, 4))

    def alpha_085(self):
        a = rank(ts_corr(self.high * 0.876703 + self.close * 0.123297,
                         self._adv_d(30), 9))
        b = rank(ts_corr(ts_rank((self.high + self.low) / 2, 3),
                         ts_rank(self.volume, 10), 7))
        return signedpower(a, b)

    def alpha_086(self):
        a = ts_rank(ts_corr(self.close, ts_sum(self._adv_d(20), 14), 6), 20)
        b = rank(self.close - self.vwap)  # simplified: (open+close) - (vwap+open) = close-vwap
        return _where(a < b, -1.0, 0.0)

    def alpha_087(self):
        if self.sector is None:
            return None
        a = rank(decay_linear(delta(self.close * 0.369701 + self.vwap * 0.630299, 1), 2))
        b = ts_rank(decay_linear(ts_corr(
            indneutralize(self._adv_d(81), self.sector), self.close, 13).abs(), 4), 14)
        return pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns) * -1

    def alpha_088(self):
        a = rank(decay_linear(
            rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close), 8))
        b = ts_rank(decay_linear(ts_corr(ts_rank(self.close, 8),
                                         ts_rank(self._adv_d(60), 20), 8), 6), 2)
        return pd.DataFrame(np.minimum(a, b), index=a.index, columns=a.columns)

    def alpha_089(self):
        if self.sector is None:
            return None
        a = ts_rank(decay_linear(ts_corr(self.low, self._adv_d(10), 6), 5), 3)
        b = ts_rank(decay_linear(delta(
            indneutralize(self.vwap, self.sector), 3), 10), 15)
        return a - b

    def alpha_090(self):
        if self.sector is None:
            return None
        a = rank(self.close - ts_max(self.close, 4))
        b = ts_rank(ts_corr(indneutralize(self._adv_d(40), self.sector),
                            self.low, 5), 3)
        return signedpower(a, b) * -1

    def alpha_091(self):
        if self.sector is None:
            return None
        a = ts_rank(decay_linear(decay_linear(ts_corr(
            indneutralize(self.close, self.sector), self.volume, 9), 16), 3), 4)
        b = rank(decay_linear(ts_corr(self.vwap, self._adv_d(30), 4), 2))
        return (a - b) * -1

    def alpha_092(self):
        a = ts_rank(decay_linear(
            ((self.high + self.low) / 2 + self.close < self.low + self.open).astype(float), 14), 18)
        b = ts_rank(decay_linear(ts_corr(rank(self.low), rank(self._adv_d(30)), 7), 6), 6)
        return pd.DataFrame(np.minimum(a, b), index=a.index, columns=a.columns)

    def alpha_093(self):
        if self.sector is None:
            return None
        a = ts_rank(decay_linear(ts_corr(
            indneutralize(self.vwap, self.sector), self._adv_d(81), 17), 19), 7)
        b = rank(decay_linear(delta(
            self.close * 0.524434 + self.vwap * 0.475566, 2), 16))
        return a / b.replace(0, np.nan)

    def alpha_094(self):
        a = rank(self.vwap - ts_min(self.vwap, 11))
        b = ts_rank(ts_corr(ts_rank(self.vwap, 19),
                            ts_rank(self._adv_d(60), 4), 18), 2)
        return signedpower(a, b) * -1

    def alpha_095(self):
        a = rank(self.open - ts_min(self.open, 12))
        b = ts_rank(signedpower(
            rank(ts_corr(ts_sum((self.high + self.low) / 2, 19),
                         ts_sum(self._adv_d(40), 19), 12)), 5), 11)
        return (a < b).astype(float)

    def alpha_096(self):
        a = ts_rank(decay_linear(ts_corr(rank(self.vwap), rank(self.volume), 3), 4), 8)
        b = ts_rank(decay_linear(ts_argmax(
            ts_corr(ts_rank(self.close, 7), ts_rank(self._adv_d(60), 4), 3), 12), 14), 13)
        return pd.DataFrame(np.maximum(a, b), index=a.index, columns=a.columns) * -1

    def alpha_097(self):
        if self.sector is None:
            return None
        a = rank(decay_linear(delta(indneutralize(
            self.low * 0.721001 + self.vwap * 0.278999, self.sector), 3), 20))
        b = ts_rank(decay_linear(ts_rank(ts_corr(
            ts_rank(self.low, 7), ts_rank(self._adv_d(60), 17), 4), 18), 15), 6)
        return (a - b) * -1

    def alpha_098(self):
        a = rank(decay_linear(ts_corr(self.vwap, ts_sum(self._adv_d(5), 26), 4), 7))
        b = rank(decay_linear(ts_rank(ts_argmin(
            ts_corr(rank(self.open), rank(self._adv_d(15)), 20), 8), 6), 8))
        return a - b

    def alpha_099(self):
        a = rank(ts_corr(ts_sum((self.high + self.low) / 2, 19),
                         ts_sum(self._adv_d(60), 19), 8))
        b = rank(ts_corr(self.low, self.volume, 6))
        return _where(a < b, -1.0, 0.0)

    def alpha_100(self):
        if self.sector is None:
            return None
        x = (self.close - self.low) - (self.high - self.close)
        x = x / (self.high - self.low + 1e-8) * self.volume
        a = 1.5 * scale(indneutralize(indneutralize(rank(x), self.sector), self.sector))
        b = scale(indneutralize(
            ts_corr(self.close, rank(self._adv_d(20)), 5) -
            rank(ts_argmin(self.close, 30)), self.sector))
        return -(a - b) * (self.volume / self._adv_d(20))

    def alpha_101(self):
        return (self.close - self.open) / (self.high - self.low + 0.001)

    # ───────────────────────────────────────────────────────
    # Batch computation
    # ───────────────────────────────────────────────────────

    def compute_all(self, verbose: bool = True) -> Dict[str, pd.DataFrame]:
        """Compute all 101 alphas. Returns {name: panel}."""
        results = {}
        for i in range(1, 102):
            name = f'alpha_{i:03d}'
            method = getattr(self, name, None)
            if method is None:
                continue
            try:
                val = method()
                if val is None:
                    if verbose:
                        print(f'  {name}: SKIP (missing data)')
                    continue
                if isinstance(val, pd.DataFrame) and not val.empty:
                    # Replace inf with NaN
                    val = val.replace([np.inf, -np.inf], np.nan)
                    results[name] = val
                    if verbose and i % 10 == 0:
                        print(f'  {name}: OK')
            except Exception as e:
                if verbose:
                    print(f'  {name}: ERROR ({e})')
        return results
