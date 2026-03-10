"""Tests for ForecastCombiner implementations."""
import pytest
from strategy.combiner import (
    WeightedAvgCombiner, MultiplicativeCombiner, LayeredCombiner,
)


class TestWeightedAvg:
    def test_equal_weight(self):
        c = WeightedAvgCombiner()
        result = c.combine([{'AAPL': 0.8}, {'AAPL': 0.4}], [1.0, 1.0])
        assert result['AAPL'] == pytest.approx(0.6)

    def test_unequal_weight(self):
        c = WeightedAvgCombiner()
        result = c.combine([{'AAPL': 1.0}, {'AAPL': 0.0}], [0.7, 0.3])
        assert result['AAPL'] == pytest.approx(0.7)

    def test_clips_to_bounds(self):
        c = WeightedAvgCombiner()
        # Weighted sum can exceed 1 if weights are not normalized
        result = c.combine([{'A': 1.0}, {'A': 1.0}], [2.0, 2.0])
        assert result['A'] <= 1.0

    def test_partial_coverage(self):
        c = WeightedAvgCombiner()
        result = c.combine(
            [{'AAPL': 0.8}, {'AAPL': 0.4, 'MSFT': 0.6}],
            [1.0, 1.0],
        )
        assert result['AAPL'] == pytest.approx(0.6)
        assert result['MSFT'] == pytest.approx(0.6)  # only alpha 1

    def test_empty(self):
        assert WeightedAvgCombiner().combine([{}, {}], [1.0, 1.0]) == {}

    def test_negative_forecast(self):
        c = WeightedAvgCombiner()
        result = c.combine([{'A': -0.5}, {'A': 0.5}], [1.0, 1.0])
        assert result['A'] == pytest.approx(0.0)


class TestMultiplicative:
    def test_basic(self):
        c = MultiplicativeCombiner()
        result = c.combine([{'AAPL': 0.8}, {'AAPL': 0.5}], [1.0, 1.0])
        assert result['AAPL'] == pytest.approx(0.4)

    def test_gating_zero(self):
        c = MultiplicativeCombiner()
        result = c.combine([{'AAPL': 1.0}, {'AAPL': 0.0}], [1.0, 1.0])
        assert result['AAPL'] == pytest.approx(0.0)

    def test_missing_is_identity(self):
        c = MultiplicativeCombiner()
        result = c.combine([{'AAPL': 0.7}, {}], [1.0, 1.0])
        assert result['AAPL'] == pytest.approx(0.7)

    def test_two_symbols(self):
        c = MultiplicativeCombiner()
        result = c.combine(
            [{'A': 0.5, 'B': 0.8}, {'A': 0.6}],
            [1.0, 1.0],
        )
        assert result['A'] == pytest.approx(0.3)
        assert result['B'] == pytest.approx(0.8)


class TestLayered:
    def test_two_layers(self):
        # Group 0 avg: (0.6+0.8)/2 = 0.7, Group 1: 0.5
        # Final: 0.7 * 0.5 = 0.35
        c = LayeredCombiner(groups=[[0, 1], [2]])
        result = c.combine(
            [{'AAPL': 0.6}, {'AAPL': 0.8}, {'AAPL': 0.5}],
            [1.0, 1.0, 1.0],
        )
        assert result['AAPL'] == pytest.approx(0.35)

    def test_custom_group_weights(self):
        # Group 0 weighted: 0.7*1.0 + 0.3*0.0 = 0.7
        # Group 1: 0.5
        # Final: 0.7 * 0.5 = 0.35
        c = LayeredCombiner(
            groups=[[0, 1], [2]],
            group_weights=[[0.7, 0.3], None],
        )
        result = c.combine(
            [{'AAPL': 1.0}, {'AAPL': 0.0}, {'AAPL': 0.5}],
            [1.0, 1.0, 1.0],
        )
        assert result['AAPL'] == pytest.approx(0.35)

    def test_single_group_no_multiply(self):
        c = LayeredCombiner(groups=[[0, 1]])
        result = c.combine([{'A': 0.4}, {'A': 0.6}], [1.0, 1.0])
        assert result['A'] == pytest.approx(0.5)
