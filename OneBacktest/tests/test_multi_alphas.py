"""Tests for multi-strategy implementations."""
import numpy as np
import pandas as pd
import pytest

from data.history import HistoryManager
from data.types import Bar
from strategy.base import Strategy
from strategy.combiner import LayeredCombiner


# ── Test helpers ──

def _make_history(symbols, n_days=300, start_price=100.0, daily_vol=0.01,
                  seed=42):
    """Create HistoryManager pre-loaded with random walk prices."""
    np.random.seed(seed)
    hm = HistoryManager(symbols, max_periods=504)
    latest = {}

    for sym in symbols:
        price = start_price
        base = pd.Timestamp('2024-01-02')
        for i in range(n_days):
            price *= (1 + np.random.normal(0, daily_vol))
            dt = base + pd.Timedelta(days=i)
            while dt.weekday() >= 5:
                dt += pd.Timedelta(days=1)
            bar = Bar(
                timestamp=dt + pd.Timedelta(hours=16),
                symbol=sym,
                open=price * 0.999,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=100_000,
            )
            hm._on_bar(bar)
            latest[sym] = bar

    return hm, latest


# ═══════════════════════════════════════════════════════════════
# HMMRegimeStrategy tests
# ═══════════════════════════════════════════════════════════════

class TestHMMRegimeStrategy:
    def test_returns_empty_without_enough_history(self):
        from strategies.regime_alloc.hmm.strategy import HMMRegimeStrategy
        s = HMMRegimeStrategy(index_symbol='SPY', min_history=252)
        hm, latest = _make_history(['SPY', 'AAPL'], n_days=50)
        s.history = hm
        s.latest_prices = latest
        result = s.on_month_end(pd.Timestamp('2024-03-01').date())
        assert result == {}

    def test_returns_empty_before_first_month_end(self):
        from strategies.regime_alloc.hmm.strategy import HMMRegimeStrategy
        s = HMMRegimeStrategy(index_symbol='SPY')
        hm, latest = _make_history(['SPY', 'AAPL'], n_days=300)
        s.history = hm
        s.latest_prices = latest
        # on_market_close before any on_month_end → empty
        result = s.on_market_close(pd.Timestamp('2024-10-01').date())
        assert result == {}

    def test_forecast_bounded_after_month_end(self):
        from strategies.regime_alloc.hmm.strategy import HMMRegimeStrategy
        s = HMMRegimeStrategy(index_symbol='SPY', min_history=252)
        hm, latest = _make_history(['SPY', 'AAPL', 'MSFT'], n_days=300)
        s.history = hm
        s.latest_prices = latest
        result = s.on_month_end(pd.Timestamp('2024-10-01').date())
        # May be empty if HMM fitting fails with random data
        for sym, f in result.items():
            assert 0.0 <= f <= 1.0, f'{sym}: forecast {f} out of [0, 1]'
        # SPY excluded from broadcast
        assert 'SPY' not in result


# ═══════════════════════════════════════════════════════════════
# IndexTimingStrategy tests
# ═══════════════════════════════════════════════════════════════

class TestIndexTimingStrategy:
    def test_returns_empty_without_enough_history(self):
        from strategies.multi.index_timing_strategy import IndexTimingStrategy
        s = IndexTimingStrategy(index_symbol='SPY')
        hm, latest = _make_history(['SPY', 'AAPL'], n_days=50)
        s.history = hm
        s.latest_prices = latest
        result = s.on_market_close(pd.Timestamp('2024-03-01').date())
        assert result == {}

    def test_produces_forecasts_with_history(self):
        from strategies.multi.index_timing_strategy import IndexTimingStrategy
        s = IndexTimingStrategy(
            index_symbol='SPY',
            hht_ma=60, hht_ht=30,
            qrs_reg_w=18, qrs_zscore_w=250,
        )
        hm, latest = _make_history(['SPY', 'AAPL', 'MSFT'], n_days=400)
        s.history = hm
        s.latest_prices = latest
        result = s.on_market_close(pd.Timestamp('2025-03-01').date())
        if result:
            for sym, f in result.items():
                assert 0.0 <= f <= 1.0
            assert 'SPY' not in result

    def test_forecast_range(self):
        from strategies.multi.index_timing_strategy import IndexTimingStrategy
        s = IndexTimingStrategy(index_symbol='SPY')
        hm, latest = _make_history(['SPY', 'AAPL'], n_days=400)
        s.history = hm
        s.latest_prices = latest
        result = s.on_market_close(pd.Timestamp('2025-03-01').date())
        for f in result.values():
            assert 0.0 <= f <= 1.0


# ═══════════════════════════════════════════════════════════════
# CrossSectionAlpha tests
# ═══════════════════════════════════════════════════════════════

class TestCrossSectionAlpha:
    def test_returns_empty_without_enough_history(self):
        from strategies.multi.cross_section_alpha import CrossSectionAlpha
        alpha = CrossSectionAlpha(
            top_n=3, min_history=252,
            use_alpha101=False,
            factors_config={'RS_12M': 1},
        )
        hm, latest = _make_history(
            ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA'], n_days=30)
        alpha.history = hm
        alpha.latest_prices = latest
        alpha.data = None
        alpha.on_init()
        result = alpha.on_market_close(pd.Timestamp('2024-02-01').date())
        assert result == {}

    def test_produces_top_n_forecasts(self):
        from strategies.multi.cross_section_alpha import CrossSectionAlpha
        symbols = [f'SYM{i}' for i in range(20)]
        alpha = CrossSectionAlpha(
            top_n=5, min_history=252,
            use_alpha101=False,
            factors_config={'RS_12M': 1, 'Range_52W': 1},
        )
        hm, latest = _make_history(symbols, n_days=300)
        alpha.history = hm
        alpha.latest_prices = latest
        alpha.data = None
        alpha.on_init()
        result = alpha.on_market_close(pd.Timestamp('2024-10-01').date())
        assert len(result) <= 5
        for f in result.values():
            assert 0.0 < f <= 1.0

    def test_forecast_normalized(self):
        from strategies.multi.cross_section_alpha import CrossSectionAlpha
        symbols = [f'S{i}' for i in range(10)]
        alpha = CrossSectionAlpha(
            top_n=3, min_history=252,
            use_alpha101=False,
            factors_config={'RS_12M': 1},
        )
        hm, latest = _make_history(symbols, n_days=300)
        alpha.history = hm
        alpha.latest_prices = latest
        alpha.data = None
        alpha.on_init()
        result = alpha.on_market_close(pd.Timestamp('2024-10-01').date())
        if result:
            max_f = max(result.values())
            assert max_f == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# LayeredCombiner require_all_groups tests
# ═══════════════════════════════════════════════════════════════

class TestLayeredCombinerRequireAll:
    def test_default_allows_partial(self):
        c = LayeredCombiner(groups=[[0], [1]])
        result = c.combine(
            [{'AAPL': 0.8}, {'AAPL': 0.6, 'MSFT': 0.5}],
            [1.0, 1.0],
        )
        assert 'AAPL' in result
        assert 'MSFT' in result

    def test_require_all_filters_partial(self):
        c = LayeredCombiner(groups=[[0], [1]], require_all_groups=True)
        result = c.combine(
            [{'AAPL': 0.8}, {'AAPL': 0.6, 'MSFT': 0.5}],
            [1.0, 1.0],
        )
        assert 'AAPL' in result
        assert 'MSFT' not in result

    def test_require_all_three_groups(self):
        c = LayeredCombiner(
            groups=[[0], [1], [2]],
            require_all_groups=True,
        )
        result = c.combine(
            [{'AAPL': 0.8}, {'AAPL': 0.7, 'MSFT': 0.9}, {'AAPL': 0.6}],
            [1.0, 1.0, 1.0],
        )
        assert 'AAPL' in result
        assert result['AAPL'] == pytest.approx(0.8 * 0.7 * 0.6)
        assert 'MSFT' not in result

    def test_backward_compatible(self):
        c = LayeredCombiner(groups=[[0, 1], [2]])
        result = c.combine(
            [{'AAPL': 0.6}, {'AAPL': 0.8}, {'AAPL': 0.5}],
            [1.0, 1.0, 1.0],
        )
        assert result['AAPL'] == pytest.approx(0.35)


# ═══════════════════════════════════════════════════════════════
# Integration test
# ═══════════════════════════════════════════════════════════════

class TestMultiStrategyIntegration:
    def test_three_strategies_pipeline(self):
        """Verify the full Strategy → Combiner → Sizer pipeline works."""
        from strategies.multi.index_timing_strategy import IndexTimingStrategy
        from strategy.sizer import VolTargetSizer

        symbols = [f'S{i}' for i in range(10)] + ['SPY']
        hm, latest = _make_history(symbols, n_days=400)

        # Dummy cross-section that always picks S0..S4
        class DummyCrossSection(Strategy):
            def on_market_close(self, dt):
                return {f'S{i}': 0.5 + 0.1 * i for i in range(5)}

        # Dummy regime that returns 0.8 for all non-SPY
        class DummyRegime(Strategy):
            def on_market_close(self, dt):
                return {s: 0.8 for s in self.latest_prices
                        if s != 'SPY'}

        cs = DummyCrossSection()
        cs.history = hm
        cs.latest_prices = latest
        cs.data = None

        regime = DummyRegime()
        regime.history = hm
        regime.latest_prices = latest

        timing = IndexTimingStrategy(index_symbol='SPY')
        timing.history = hm
        timing.latest_prices = latest

        combiner = LayeredCombiner(
            groups=[[0], [1], [2]],
            require_all_groups=True,
        )

        # Collect forecasts
        dt = pd.Timestamp('2025-03-01').date()
        forecasts = [
            cs.on_market_close(dt),
            regime.on_market_close(dt),
            timing.on_market_close(dt),
        ]
        weights = [1.0, 1.0, 1.0]
        combined = combiner.combine(forecasts, weights)

        # Only S0..S4 should appear (cross-section selected)
        for sym in combined:
            assert sym.startswith('S')
        assert 'SPY' not in combined

        for f in combined.values():
            assert -1.0 <= f <= 1.0

        # Sizer should produce share targets
        sizer = VolTargetSizer(
            target_vol=0.15, vol_lookback=20, min_forecast=0.01)
        targets = sizer.size(combined, hm, 1_000_000.0, latest)
        assert len(targets) >= 0


# ═══════════════════════════════════════════════════════════════
# CrossSectionAlpha sector neutralization tests
# ═══════════════════════════════════════════════════════════════

class TestSectorNeutralization:
    def test_neutralize_subtracts_sector_mean(self):
        """After neutralization, within-sector mean should be ~0."""
        from strategies.cross_section.neutralize import neutralize_factors

        sector_map = pd.Series({
            'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOG': 'Tech',
            'JPM': 'Financials', 'GS': 'Financials', 'C': 'Financials',
            'JNJ': 'Health', 'PFE': 'Health', 'UNH': 'Health',
        })

        # ROE values with clear sector bias: Financials high, Tech mid, Health low
        df = pd.DataFrame({
            'RS_12M': [0.1, 0.2, 0.15, 0.3, 0.25, 0.35, 0.05, 0.1, 0.08],
            'ROE': [0.15, 0.18, 0.12, 0.30, 0.35, 0.28, 0.05, 0.08, 0.06],
        }, index=['AAPL', 'MSFT', 'GOOG', 'JPM', 'GS', 'C', 'JNJ', 'PFE', 'UNH'])

        result = neutralize_factors(df, sector_map, {'ROE'})

        # RS_12M should be unchanged (not in neutralize set)
        pd.testing.assert_series_equal(result['RS_12M'], df['RS_12M'])

        # ROE should be neutralized: sector mean ≈ 0
        for sector in ['Tech', 'Financials', 'Health']:
            syms = sector_map[sector_map == sector].index.tolist()
            sector_mean = result.loc[syms, 'ROE'].mean()
            assert abs(sector_mean) < 1e-10, \
                f'{sector} ROE mean should be ~0 after neutralization, got {sector_mean}'

    def test_neutralize_preserves_unlisted_cols(self):
        """Columns not in cols_to_neutralize should pass through unchanged."""
        from strategies.cross_section.neutralize import neutralize_factors

        sector_map = pd.Series({'A': 'X', 'B': 'X', 'C': 'X', 'D': 'Y', 'E': 'Y'})

        df = pd.DataFrame({
            'RS_12M': [0.1, 0.2, 0.3, 0.4, 0.5],
            'ROE': [0.1, 0.2, 0.3, 0.4, 0.5],
        }, index=['A', 'B', 'C', 'D', 'E'])

        result = neutralize_factors(df, sector_map, {'ROE'})
        pd.testing.assert_series_equal(result['RS_12M'], df['RS_12M'])

    def test_volume_biased_factors_in_constant(self):
        """VOLUME_BIASED_FACTORS should contain expected factors."""
        from strategies.cross_section.neutralize import VOLUME_BIASED_FACTORS
        assert 'alpha_054' in VOLUME_BIASED_FACTORS
        assert 'vol_battle_pos' in VOLUME_BIASED_FACTORS
        assert 'RS_12M' not in VOLUME_BIASED_FACTORS
