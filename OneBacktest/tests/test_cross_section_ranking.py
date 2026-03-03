import numpy as np
import pandas as pd

from strategies.cross_section.ranking import assign_quantiles, cross_sectional_zscore


def test_cross_sectional_zscore_rowwise_standardization():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
        columns=["A", "B", "C"],
    )

    z = cross_sectional_zscore(factor)

    for dt in z.index:
        row = z.loc[dt].dropna()
        assert np.isclose(row.mean(), 0.0)
        assert np.isclose(row.std(), 1.0)


def test_assign_quantiles_returns_expected_buckets():
    factor = pd.DataFrame(
        [list(range(10))],
        index=pd.to_datetime(["2025-01-01"]),
        columns=[f"S{i}" for i in range(10)],
    )

    q = assign_quantiles(factor, n_quantiles=5, min_stocks=10)
    counts = q.iloc[0].value_counts().to_dict()

    assert counts == {1.0: 2, 2.0: 2, 3.0: 2, 4.0: 2, 5.0: 2}


def test_assign_quantiles_returns_nan_when_not_enough_stocks():
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=pd.to_datetime(["2025-01-01"]),
        columns=["A", "B", "C"],
    )

    q = assign_quantiles(factor, n_quantiles=3, min_stocks=5)
    assert q.iloc[0].isna().all()
