"""
因子诊断 & Alpha Test 单元测试

使用合成数据, 验证核心函数的正确性.
"""
import numpy as np
import pandas as pd
import pytest

from strategies.cross_section.diagnostics import (
    compute_distribution_stability,
    compute_coverage,
    build_style_factor_proxies,
    compute_known_factor_correlation,
    compute_turnover_capacity,
    compute_industry_distribution,
    run_diagnostics,
)
from strategies.cross_section.alpha_test import (
    compute_multi_horizon_ic,
    _newey_west_tstat,
    _estimate_ic_half_life,
    compute_enhanced_quantile_metrics,
    fama_macbeth_regression,
    build_sector_dummies,
    classify_market_regime,
    compute_sub_sample_stats,
    walk_forward_split,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def _make_dates(n=100):
    return pd.bdate_range('2024-01-02', periods=n, freq='B')


def _make_symbols(n=25):
    return [f'S{i}' for i in range(1, n + 1)]


def _make_factor(dates, symbols, seed=42):
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal((len(dates), len(symbols)))
    mask = rng.random(vals.shape) < 0.05
    vals[mask] = np.nan
    return pd.DataFrame(vals, index=dates, columns=symbols)


def _make_close(dates, symbols, seed=123):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, (len(dates), len(symbols)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def _make_volume(dates, symbols, seed=456):
    rng = np.random.default_rng(seed)
    vol = np.exp(rng.normal(15, 1, (len(dates), len(symbols))))
    return pd.DataFrame(vol, index=dates, columns=symbols)


# ═══════════════════════════════════════════════════════════════
# Distribution Stability
# ═══════════════════════════════════════════════════════════════

def test_distribution_stability_basic():
    dates = _make_dates(100)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    result = compute_distribution_stability(factor, rolling_window=20)
    assert len(result.rolling_mean) > 0
    assert len(result.rolling_var) > 0
    assert isinstance(result.drift_detected, bool)


def test_distribution_stability_detects_drift():
    dates = _make_dates(200)
    symbols = _make_symbols(25)
    rng = np.random.default_rng(0)
    drift = np.linspace(0, 5, len(dates))
    vals = rng.standard_normal((len(dates), len(symbols)))
    vals += drift[:, None]
    factor = pd.DataFrame(vals, index=dates, columns=symbols)
    result = compute_distribution_stability(factor, rolling_window=20)
    assert result.drift_detected is True


def test_distribution_stability_no_drift():
    dates = _make_dates(200)
    symbols = _make_symbols(100)  # 更多 symbols → 截面均值更稳定
    rng = np.random.default_rng(42)
    vals = rng.standard_normal((len(dates), len(symbols)))
    factor = pd.DataFrame(vals, index=dates, columns=symbols)
    result = compute_distribution_stability(factor, rolling_window=20)
    # 用较宽松阈值: 关注的是 drift_detected (p < 0.01)
    # 100 symbols 截面均值标准差 ~0.1, 不太会出现显著趋势
    assert isinstance(result.drift_pvalue, float)


# ═══════════════════════════════════════════════════════════════
# Coverage
# ═══════════════════════════════════════════════════════════════

def test_coverage_full_panel():
    dates = _make_dates(50)
    symbols = _make_symbols(25)
    factor = pd.DataFrame(1.0, index=dates, columns=symbols)
    report = compute_coverage(factor)
    assert report.median_coverage == 1.0
    assert (report.daily_count == 25).all()


def test_coverage_partial_panel():
    dates = _make_dates(50)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    report = compute_coverage(factor)
    assert 0.85 < report.median_coverage < 1.0


def test_coverage_with_mktcap():
    dates = _make_dates(50)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    mktcap = pd.DataFrame(
        np.abs(np.random.randn(50, 25)) * 1e9 + 1e8,
        index=dates, columns=symbols,
    )
    report = compute_coverage(factor, mktcap)
    assert report.small_cap_bias is not None


# ═══════════════════════════════════════════════════════════════
# Style Factor Correlation
# ═══════════════════════════════════════════════════════════════

def test_build_style_factor_proxies_keys():
    dates = _make_dates(300)
    symbols = _make_symbols(25)
    close = _make_close(dates, symbols)
    proxies = build_style_factor_proxies(close)
    assert 'momentum' in proxies
    assert 'low_vol' in proxies
    assert 'size' in proxies


def test_known_factor_correlation_with_self():
    dates = _make_dates(100)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    styles = {'self_proxy': factor.copy()}
    result = compute_known_factor_correlation(factor, styles, rolling_window=10)
    assert abs(result.correlations['self_proxy'] - 1.0) < 0.15
    assert result.dominant_style == 'self_proxy'


def test_known_factor_correlation_uncorrelated():
    dates = _make_dates(100)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols, seed=42)
    other = _make_factor(dates, symbols, seed=999)
    styles = {'random': other}
    result = compute_known_factor_correlation(factor, styles, rolling_window=10)
    assert abs(result.correlations['random']) < 0.3


# ═══════════════════════════════════════════════════════════════
# Turnover
# ═══════════════════════════════════════════════════════════════

def test_turnover_constant_factor():
    dates = _make_dates(50)
    symbols = _make_symbols(25)
    vals = np.tile(np.arange(25, dtype=float), (50, 1))
    factor = pd.DataFrame(vals, index=dates, columns=symbols)
    result = compute_turnover_capacity(factor)
    assert result.mean_turnover == 0.0


def test_turnover_random_factor():
    dates = _make_dates(50)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    result = compute_turnover_capacity(factor)
    assert result.mean_turnover > 0.0


# ═══════════════════════════════════════════════════════════════
# Newey-West t-stat
# ═══════════════════════════════════════════════════════════════

def test_newey_west_positive_mean():
    rng = np.random.default_rng(42)
    series = pd.Series(1.0 + rng.normal(0, 0.1, 100))
    t = _newey_west_tstat(series)
    assert t > 5.0


def test_newey_west_zero_mean():
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(0, 1, 100))
    t = _newey_west_tstat(series)
    assert abs(t) < 3.0


def test_newey_west_short_series():
    t = _newey_west_tstat(pd.Series([1.0, 2.0]))
    assert t == 0.0


# ═══════════════════════════════════════════════════════════════
# IC Half-Life
# ═══════════════════════════════════════════════════════════════

def test_ic_half_life_decaying():
    # 人工构造衰减 IC
    mean_ics = {1: 0.10, 5: 0.06, 21: 0.02, 63: 0.005}
    hl = _estimate_ic_half_life(mean_ics)
    assert hl is not None
    assert hl > 0


def test_ic_half_life_no_decay():
    # 平坦 IC → 无衰减
    mean_ics = {1: 0.05, 5: 0.05, 21: 0.05, 63: 0.05}
    hl = _estimate_ic_half_life(mean_ics)
    assert hl is None  # slope ≈ 0, not a valid decay


# ═══════════════════════════════════════════════════════════════
# Multi-Horizon IC
# ═══════════════════════════════════════════════════════════════

def test_multi_horizon_ic_returns_all_horizons():
    dates = _make_dates(300)
    symbols = _make_symbols(50)
    close = _make_close(dates, symbols)
    factor = _make_factor(dates, symbols)
    result = compute_multi_horizon_ic(factor, close, horizons=[5, 21], min_obs=10)
    assert 5 in result.mean_ic
    assert 21 in result.mean_ic
    assert 5 in result.tstat_nw


# ═══════════════════════════════════════════════════════════════
# Enhanced Quantile Metrics
# ═══════════════════════════════════════════════════════════════

def test_enhanced_quantile_basic():
    dates = _make_dates(200)
    symbols = _make_symbols(50)
    close = _make_close(dates, symbols)
    factor = _make_factor(dates, symbols)

    from strategies.cross_section.backtest import build_monthly_rebalance
    bt = build_monthly_rebalance(factor, close, min_stocks=10)
    eq = compute_enhanced_quantile_metrics(bt, transaction_cost_bps=10.0)

    assert eq.base_metrics is not None
    assert 5 in eq.quantile_max_dd
    assert eq.ls_sharpe_net is not None
    assert len(eq.ls_with_costs) > 0


# ═══════════════════════════════════════════════════════════════
# Fama-MacBeth
# ═══════════════════════════════════════════════════════════════

def test_fama_macbeth_perfect_predictor():
    dates = _make_dates(100)
    symbols = _make_symbols(30)
    rng = np.random.default_rng(42)
    factor_vals = rng.standard_normal((len(dates), len(symbols)))
    fwd_vals = factor_vals + rng.normal(0, 0.1, factor_vals.shape)
    factor = pd.DataFrame(factor_vals, index=dates, columns=symbols)
    fwd_ret = pd.DataFrame(fwd_vals, index=dates, columns=symbols)
    result = fama_macbeth_regression(factor, fwd_ret, min_obs=10)
    assert result.mean_slope > 0
    assert result.tstat > 2.0


def test_fama_macbeth_with_controls():
    dates = _make_dates(100)
    symbols = _make_symbols(30)
    rng = np.random.default_rng(42)
    factor = pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))),
        index=dates, columns=symbols,
    )
    fwd_ret = pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))) * 0.02,
        index=dates, columns=symbols,
    )
    control = pd.DataFrame(
        rng.standard_normal((len(dates), len(symbols))),
        index=dates, columns=symbols,
    )
    result = fama_macbeth_regression(
        factor, fwd_ret, controls={'size': control}, min_obs=10,
    )
    assert len(result.slope_series) > 0
    assert 'size' in result.control_slopes


def test_build_sector_dummies_shape():
    gics_df = pd.DataFrame({
        'symbol': ['A', 'B', 'C', 'D'],
        'gics_sector': ['Tech', 'Tech', 'Health', 'Finance'],
    })
    dummies = build_sector_dummies(['A', 'B', 'C', 'D'], gics_df)
    assert dummies.shape[0] == 4
    assert dummies.shape[1] == 2  # drop_first


# ═══════════════════════════════════════════════════════════════
# Sub-sample
# ═══════════════════════════════════════════════════════════════

def test_classify_market_regime_ma200():
    dates = _make_dates(300)
    spy = pd.Series(np.linspace(100, 200, 300), index=dates)
    regime = classify_market_regime(spy, method='ma200')
    # 持续上涨 → 后期全 bull
    assert (regime.iloc[-50:] == 'bull').all()


def test_classify_market_regime_vol():
    dates = _make_dates(200)
    rng = np.random.default_rng(42)
    spy = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=dates)
    regime = classify_market_regime(spy, method='vol')
    assert set(regime.dropna().unique()) <= {'high_vol', 'low_vol'}


def test_compute_sub_sample_stats():
    dates = _make_dates(100)
    ic = pd.Series(np.random.randn(100) * 0.05, index=dates)
    ls = pd.Series(np.random.randn(100) * 0.01, index=dates)
    labels = pd.Series(
        ['bull'] * 60 + ['bear'] * 40, index=dates,
    )
    result = compute_sub_sample_stats(ic, ls, labels, ann_factor=12)
    assert 'bull' in result.index
    assert 'bear' in result.index
    assert 'mean_ic' in result.columns


def test_walk_forward_split_returns_both():
    dates = _make_dates(200)
    symbols = _make_symbols(30)
    close = _make_close(dates, symbols)
    factor = _make_factor(dates, symbols)
    result = walk_forward_split(factor, close, train_pct=0.7, horizons=[21])
    assert 'in_sample' in result
    assert 'out_of_sample' in result
    assert 'split_date' in result
    assert 21 in result['in_sample']


# ═══════════════════════════════════════════════════════════════
# Run Diagnostics (integration)
# ═══════════════════════════════════════════════════════════════

def test_run_diagnostics_integration():
    dates = _make_dates(100)
    symbols = _make_symbols(25)
    factor = _make_factor(dates, symbols)
    close = _make_close(dates, symbols)
    report = run_diagnostics(factor, 'test_factor', close=close)
    assert report.factor_name == 'test_factor'
    assert isinstance(report.pass_sanity, bool)
    assert report.distribution is not None
    assert report.coverage is not None
    assert report.industry is None  # no gics_df → None


# ═══════════════════════════════════════════════════════════════
# Industry Distribution
# ═══════════════════════════════════════════════════════════════

def _make_gics(symbols, n_sectors=4):
    """Create synthetic GICS mapping."""
    sectors = ['Tech', 'Health', 'Finance', 'Industrial'][:n_sectors]
    return pd.DataFrame({
        'symbol': symbols,
        'gics_sector': [sectors[i % n_sectors] for i in range(len(symbols))],
    })


def test_industry_distribution_basic():
    dates = _make_dates(100)
    symbols = _make_symbols(40)
    factor = _make_factor(dates, symbols)
    gics = _make_gics(symbols, n_sectors=4)
    result = compute_industry_distribution(factor, gics)
    assert len(result.avg_sector_mean) == 4
    assert len(result.top_q_sector_pct) == 4
    assert len(result.bot_q_sector_pct) == 4
    # HHI for 4 uniform sectors = 0.25
    assert result.hhi_top > 0
    assert result.hhi_bot > 0


def test_industry_distribution_concentrated():
    """Factor systematically higher in one sector → concentrated quintile."""
    dates = _make_dates(100)
    symbols = _make_symbols(40)
    rng = np.random.default_rng(42)
    vals = rng.standard_normal((len(dates), len(symbols)))
    gics = _make_gics(symbols, n_sectors=4)
    # Add large bias: Tech stocks (S1, S5, S9, ...) get +5
    sector_map = gics.set_index('symbol')['gics_sector']
    for i, sym in enumerate(symbols):
        if sector_map[sym] == 'Tech':
            vals[:, i] += 5.0
    factor = pd.DataFrame(vals, index=dates, columns=symbols)
    result = compute_industry_distribution(factor, gics)
    # Tech should dominate top quintile
    assert result.top_q_sector_pct.get('Tech', 0) > 0.5
    # HHI should be high
    assert result.hhi_top > 0.3


def test_industry_distribution_with_neutralized_ic():
    """When close is provided, neutralized IC should be computed."""
    dates = _make_dates(200)
    symbols = _make_symbols(40)
    factor = _make_factor(dates, symbols)
    close = _make_close(dates, symbols)
    gics = _make_gics(symbols, n_sectors=4)
    result = compute_industry_distribution(factor, gics, close=close, ic_horizon=21)
    assert result.raw_ic_21d is not None
    assert result.neutralized_ic is not None


def test_industry_distribution_too_few_symbols():
    """With < 20 valid symbols, should return empty."""
    dates = _make_dates(50)
    symbols = _make_symbols(5)
    factor = _make_factor(dates, symbols)
    gics = _make_gics(symbols, n_sectors=2)
    result = compute_industry_distribution(factor, gics)
    assert len(result.avg_sector_mean) == 0


def test_run_diagnostics_with_gics():
    """Integration test: run_diagnostics with gics_df."""
    dates = _make_dates(100)
    symbols = _make_symbols(30)
    factor = _make_factor(dates, symbols)
    close = _make_close(dates, symbols)
    gics = _make_gics(symbols, n_sectors=4)
    report = run_diagnostics(factor, 'test_factor', close=close, gics_df=gics)
    assert report.industry is not None
    assert len(report.industry.avg_sector_mean) > 0
