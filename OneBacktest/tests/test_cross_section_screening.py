import numpy as np
import pandas as pd

from strategies.cross_section.screening import (
    compute_ic_summary,
    compute_rank_ic,
    correlation_dedup,
    ic_filter,
    orthogonalize,
    orthogonalize_sequential,
    screen_factors,
)


def _panel(rows, dates, symbols):
    return pd.DataFrame(rows, index=pd.to_datetime(dates), columns=symbols)


def test_compute_rank_ic_perfect_rank_correlation():
    symbols = [f"S{i}" for i in range(1, 26)]
    dates = ["2025-01-03", "2025-01-10", "2025-01-17"]
    seq = list(range(1, 26))
    factor = _panel(
        [
            seq,
            seq,
            seq,
        ],
        dates,
        symbols,
    )
    fwd = _panel(
        [
            [x * 10 for x in seq],
            [x * 10 for x in seq],
            [x * 10 for x in seq],
        ],
        dates,
        symbols,
    )

    ic = compute_rank_ic(factor, fwd, min_obs=3)
    assert len(ic) == 3
    assert np.allclose(ic.values, 1.0)


def test_ic_filter_and_correlation_dedup_keep_higher_icir():
    symbols = [f"S{i}" for i in range(1, 26)]
    dates = ["2025-01-03", "2025-01-10", "2025-01-17", "2025-01-24"]
    seq = list(range(1, 26))
    f1 = _panel(
        [
            seq,
            seq,
            seq,
            seq,
        ],
        dates,
        symbols,
    )
    # Almost identical to f1 -> high correlation.
    f2 = f1 + 0.01
    # Distinct pattern.
    f3 = _panel(
        [
            seq[::-1],
            seq[::2] + seq[1::2],
            seq[5:] + seq[:5],
            seq[10:] + seq[:10],
        ],
        dates,
        symbols,
    )
    # Forward returns mostly align with f1.
    fwd = _panel(
        [
            [x * 10 for x in seq],
            [x * 10 for x in seq],
            [x * 10 for x in seq],
            [x * 10 for x in seq],
        ],
        dates,
        symbols,
    )
    factors = {"f1": f1, "f2": f2, "f3": f3}

    ic_df = compute_ic_summary(factors, fwd, min_periods=3)
    passed = ic_filter(ic_df, min_abs_ic=0.01)
    selected = correlation_dedup(factors, ic_df, passed, max_corr=0.95)

    assert "f1" in selected
    assert not ("f1" in selected and "f2" in selected)


def test_screen_factors_end_to_end():
    symbols = [f"S{i}" for i in range(1, 26)]
    dates = ["2025-01-03", "2025-01-10", "2025-01-17"]
    seq = list(range(1, 26))
    good = _panel(
        [
            seq,
            seq,
            seq,
        ],
        dates,
        symbols,
    )
    bad = _panel(
        [
            seq[::-1],
            seq[::-1],
            seq[::-1],
        ],
        dates,
        symbols,
    )
    fwd = _panel(
        [
            seq,
            seq,
            seq,
        ],
        dates,
        symbols,
    )

    selected, ic_df = screen_factors(
        {"good": good, "bad": bad},
        fwd,
        min_abs_ic=0.5,
        max_corr=0.7,
    )

    assert not ic_df.empty
    assert "good" in selected
    # "bad" passes abs(IC) but is removed by correlation dedup due to high |corr| with "good".
    assert "bad" not in selected


def test_orthogonalize_reduces_linear_dependency():
    symbols = [f"S{i}" for i in range(20)]
    dt = pd.Timestamp("2025-01-03")
    x = pd.Series(np.arange(1, 21), index=symbols)
    y = 2.0 * x + 1.0

    new_factor = pd.DataFrame([y.values], index=[dt], columns=symbols)
    existing = {"x": pd.DataFrame([x.values], index=[dt], columns=symbols)}

    resid = orthogonalize(new_factor, existing)
    assert dt in resid.index
    # Residual should be numerically near zero for exact linear relation.
    assert np.allclose(resid.loc[dt].fillna(0).values, 0.0, atol=1e-10)


def test_orthogonalize_sequential_respects_order():
    symbols = [f"S{i}" for i in range(20)]
    dt = pd.Timestamp("2025-01-03")
    base = pd.Series(np.arange(1, 21), index=symbols)
    f1 = pd.DataFrame([base.values], index=[dt], columns=symbols)
    f2 = pd.DataFrame([(2 * base + 1).values], index=[dt], columns=symbols)

    out = orthogonalize_sequential({"f1": f1, "f2": f2}, order=["f1", "f2"])
    assert "f1" in out and "f2" in out
    assert out["f1"].equals(f1)
