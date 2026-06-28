"""Statistical performance metrics with Carver-style severity flags.

Default report includes:
    - Annualized return, vol, max drawdown, calmar
    - Sharpe Ratio + 95% t-distribution CI + bootstrap CI
    - Skewness (with negative-skew warning < -0.5)
    - 252-day rolling Sharpe series
    - Carver plausibility flags (single-instrument SR > 0.5; portfolio SR > 1.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats as _stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if len(returns) == 0:
        return float("nan")
    return float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    if returns.std(ddof=1) == 0:
        return float("nan")
    excess = returns - rf / periods_per_year
    return float(excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ci_t(returns: pd.Series, periods_per_year: int = 252,
                alpha: float = 0.05) -> tuple[float, float]:
    """95% t-distribution confidence interval for SR.

    Uses standard formula: SE(SR) = sqrt((1 + 0.5 * SR^2) / N).
    """
    n = len(returns)
    if n < 30:
        return (float("nan"), float("nan"))
    sr = sharpe_ratio(returns, periods_per_year)
    se = np.sqrt((1 + 0.5 * sr ** 2) / n) * np.sqrt(periods_per_year)
    if HAS_SCIPY:
        t_crit = _stats.t.ppf(1 - alpha / 2, df=n - 1)
    else:
        t_crit = 1.96
    return (float(sr - t_crit * se), float(sr + t_crit * se))


def sharpe_ci_bootstrap(
    returns: pd.Series,
    n_resamples: int = 1000,
    block_size: int = 5,
    periods_per_year: int = 252,
    seed: int = 42,
) -> tuple[float, float]:
    """Block-bootstrap 95% CI for SR (preserves serial correlation)."""
    arr = returns.dropna().values
    n = len(arr)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    sr_samples = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[s:s + block_size] for s in starts])[:n]
        s = pd.Series(sample)
        if s.std(ddof=1) == 0:
            continue
        sr_samples.append(sharpe_ratio(s, periods_per_year))
    if not sr_samples:
        return (float("nan"), float("nan"))
    arr = np.array(sr_samples)
    return (float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975)))


def skewness(returns: pd.Series) -> float:
    n = len(returns)
    if n < 3:
        return float("nan")
    if HAS_SCIPY:
        return float(_stats.skew(returns.dropna()))
    r = returns.dropna()
    m = r.mean()
    s = r.std(ddof=1)
    if s == 0:
        return float("nan")
    return float(((r - m) ** 3).mean() / (s ** 3))


def rolling_sharpe(returns: pd.Series, window: int = 252,
                   periods_per_year: int = 252) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    return (mu / sigma) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def calmar_ratio(returns: pd.Series, equity: pd.Series,
                 periods_per_year: int = 252) -> float:
    ar = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(equity))
    return float(ar / mdd) if mdd > 0 else float("nan")


@dataclass
class PerformanceReport:
    n_obs: int
    annual_return: float
    annual_volatility: float
    sharpe: float
    sharpe_ci_t: tuple[float, float]
    sharpe_ci_bootstrap: tuple[float, float]
    skewness: float
    max_drawdown: float
    calmar: float
    rolling_sharpe: pd.Series = field(repr=False, default_factory=lambda: pd.Series(dtype=float))
    flags: list[str] = field(default_factory=list)


def compute_metrics(
    equity: pd.Series,
    is_portfolio: bool = True,
    periods_per_year: int = 252,
    bootstrap: bool = True,
) -> PerformanceReport:
    returns = equity.pct_change().dropna()
    sr = sharpe_ratio(returns, periods_per_year)
    flags = []

    sk = skewness(returns)
    if sk < -0.5:
        flags.append(f"NEGATIVE_SKEW (skew={sk:.2f}) — verify against tail events")

    if is_portfolio and sr > 1.0:
        flags.append(f"SR_ABOVE_PORTFOLIO_PLAUSIBILITY (SR={sr:.2f}) — Carver Ch2 threshold; verify out-of-sample")
    if (not is_portfolio) and sr > 0.5:
        flags.append(f"SR_ABOVE_SINGLE_INSTRUMENT_PLAUSIBILITY (SR={sr:.2f}) — Carver Ch2; check for overfit")

    return PerformanceReport(
        n_obs=len(returns),
        annual_return=annualized_return(returns, periods_per_year),
        annual_volatility=annualized_volatility(returns, periods_per_year),
        sharpe=sr,
        sharpe_ci_t=sharpe_ci_t(returns, periods_per_year),
        sharpe_ci_bootstrap=sharpe_ci_bootstrap(returns, periods_per_year=periods_per_year)
                             if bootstrap else (float("nan"), float("nan")),
        skewness=sk,
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(returns, equity, periods_per_year),
        rolling_sharpe=rolling_sharpe(returns, periods_per_year=periods_per_year),
        flags=flags,
    )


def print_report(report: PerformanceReport, name: str = "Strategy") -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name} — Performance Report")
    print(f"{'=' * 60}")
    print(f"  N observations         : {report.n_obs}")
    print(f"  Annual return          : {report.annual_return:>8.2%}")
    print(f"  Annual volatility      : {report.annual_volatility:>8.2%}")
    print(f"  Sharpe Ratio           : {report.sharpe:>8.3f}")
    lo, hi = report.sharpe_ci_t
    print(f"  SR 95% CI (t-dist)     : [{lo:>6.3f}, {hi:>6.3f}]")
    lo, hi = report.sharpe_ci_bootstrap
    print(f"  SR 95% CI (bootstrap)  : [{lo:>6.3f}, {hi:>6.3f}]")
    print(f"  Skewness               : {report.skewness:>8.3f}")
    print(f"  Max drawdown           : {report.max_drawdown:>8.2%}")
    print(f"  Calmar                 : {report.calmar:>8.3f}")
    if report.flags:
        print(f"  ⚠ FLAGS:")
        for flag in report.flags:
            print(f"      - {flag}")
    print(f"{'=' * 60}\n")
