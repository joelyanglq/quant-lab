import numpy as np
import pandas as pd

from strategies.cross_section.composite import build_composite_factor


def test_build_composite_factor_weighted_average_on_common_index_and_columns():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
    f1 = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=dates, columns=["A", "B"])
    f2 = pd.DataFrame([[2.0, 0.0], [4.0, 2.0]], index=dates, columns=["A", "B"])
    factors = {"ROE": f1, "ROIC": f2}
    weights = {"ROE": 0.7, "ROIC": 0.3}

    comp = build_composite_factor(factors, weights=weights, min_factor_pct=0.5)

    assert list(comp.index) == list(dates)
    assert list(comp.columns) == ["A", "B"]
    assert comp.notna().all().all()


def test_build_composite_factor_applies_min_factor_coverage():
    dates = pd.to_datetime(["2025-01-01"])
    base = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    missing = pd.DataFrame(
        [[1.0, 2.0, 3.0, np.nan]],
        index=dates,
        columns=["A", "B", "C", "D"],
    )
    factors = {"ROE": base, "ROIC": base, "EV_EBITDA": missing, "FCF_Yield": missing}
    weights = {"ROE": 0.25, "ROIC": 0.25, "EV_EBITDA": 0.25, "FCF_Yield": 0.25}

    comp = build_composite_factor(factors, weights=weights, min_factor_pct=0.75)

    assert pd.notna(comp.loc[dates[0], "A"])
    assert pd.isna(comp.loc[dates[0], "D"])
