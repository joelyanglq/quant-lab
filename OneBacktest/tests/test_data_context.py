"""Tests for DataContext: register, as_of PIT, panel_between."""
import pandas as pd
import pytest
from data.context import DataContext


class TestRegister:
    def test_register_multi_column(self):
        ctx = DataContext()
        panel = pd.DataFrame(
            {'AAPL': [10.0, 12.0], 'MSFT': [5.0, 6.0]},
            index=pd.to_datetime(['2025-03-31', '2025-06-30']),
        )
        ctx.register('roe', panel)
        assert ctx.has('roe')
        assert 'roe' in ctx.names

    def test_register_single_column(self):
        ctx = DataContext()
        panel = pd.DataFrame(
            {'rate': [0.04, 0.045]},
            index=pd.to_datetime(['2025-01-02', '2025-02-03']),
        )
        ctx.register('tbill_3m', panel)
        assert ctx.has('tbill_3m')

    def test_register_duplicate_raises(self):
        ctx = DataContext()
        panel = pd.DataFrame({'A': [1.0]}, index=pd.to_datetime(['2025-01-01']))
        ctx.register('x', panel)
        with pytest.raises(ValueError, match="already registered"):
            ctx.register('x', panel)

    def test_register_non_datetime_index_raises(self):
        ctx = DataContext()
        panel = pd.DataFrame({'A': [1.0]}, index=[0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ctx.register('bad', panel)

    def test_names_empty(self):
        assert DataContext().names == []


class TestAsOf:
    def _make_ctx(self):
        ctx = DataContext()
        panel = pd.DataFrame(
            {'AAPL': [10.0, 12.0, 15.0], 'MSFT': [5.0, 6.0, 7.0]},
            index=pd.to_datetime(['2025-03-31', '2025-06-30', '2025-09-30']),
        )
        ctx.register('roe', panel)
        return ctx

    def test_exact_date(self):
        result = self._make_ctx().as_of(pd.Timestamp('2025-06-30'), 'roe')
        assert result['AAPL'] == 12.0
        assert result['MSFT'] == 6.0

    def test_pit_returns_earlier(self):
        result = self._make_ctx().as_of(pd.Timestamp('2025-08-15'), 'roe')
        assert result['AAPL'] == 12.0  # not 15.0

    def test_before_any_data_raises(self):
        with pytest.raises(ValueError, match="No data"):
            self._make_ctx().as_of(pd.Timestamp('2025-01-01'), 'roe')

    def test_single_column_returns_float(self):
        ctx = DataContext()
        panel = pd.DataFrame(
            {'rate': [0.04, 0.045]},
            index=pd.to_datetime(['2025-01-02', '2025-02-03']),
        )
        ctx.register('tbill', panel)
        result = ctx.as_of(pd.Timestamp('2025-01-15'), 'tbill')
        assert isinstance(result, float)
        assert result == 0.04

    def test_accepts_date_object(self):
        """Engine passes datetime.date, not Timestamp."""
        from datetime import date
        result = self._make_ctx().as_of(date(2025, 6, 30), 'roe')
        assert result['AAPL'] == 12.0

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            DataContext().as_of(pd.Timestamp('2025-01-01'), 'nope')


class TestPanelBetween:
    def test_slice_inclusive(self):
        ctx = DataContext()
        panel = pd.DataFrame(
            {'AAPL': [10.0, 12.0, 15.0]},
            index=pd.to_datetime(['2025-03-31', '2025-06-30', '2025-09-30']),
        )
        ctx.register('roe', panel)
        result = ctx.panel_between('roe', '2025-03-31', '2025-06-30')
        assert len(result) == 2

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            DataContext().panel_between('x', '2025-01-01', '2025-12-31')
