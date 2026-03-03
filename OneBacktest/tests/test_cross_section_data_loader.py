import pandas as pd

from strategies.cross_section.data_loader import (
    _compute_ttm,
    _fix_filing_dates,
    NON_FLOW_OVERRIDES,
    build_fundamental_panel,
)


def test_fix_filing_dates_fill_q4_from_fy_then_fallback():
    raw = pd.DataFrame(
        {
            "fiscal_period": ["Q4", "FY", "Q1"],
            "end_date": ["2024-12-31", "2024-12-31", "2025-03-31"],
            "filing_date": [None, "2025-02-28", None],
        }
    )

    out = _fix_filing_dates(raw)

    q4 = out.loc[out["fiscal_period"] == "Q4", "filing_date"].iloc[0]
    q1 = out.loc[out["fiscal_period"] == "Q1", "filing_date"].iloc[0]
    assert str(q4).startswith("2025-02-28")
    assert str(q1).startswith("2025-06-29")


def test_compute_ttm_flow_vs_stock():
    q = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    flow = _compute_ttm(q, "v", is_flow=True)
    stock = _compute_ttm(q, "v", is_flow=False)

    assert pd.isna(flow.iloc[0]) and pd.isna(flow.iloc[1]) and pd.isna(flow.iloc[2])
    assert flow.iloc[3] == 10.0
    assert flow.iloc[4] == 14.0
    assert stock.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_build_fundamental_panel_uses_ttm_and_non_flow_override(monkeypatch):
    trading_dates = pd.to_datetime(
        ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01"]
    )
    fields = [
        "income_statement__revenues",
        "income_statement__diluted_average_shares",
    ]
    symbols = ["AAA"]

    df = pd.DataFrame(
        {
            "timeframe": ["quarterly"] * 5,
            "fiscal_year": [2024, 2024, 2024, 2024, 2025],
            "fiscal_period": ["Q1", "Q2", "Q3", "Q4", "Q1"],
            "end_date": [
                "2024-03-31",
                "2024-06-30",
                "2024-09-30",
                "2024-12-31",
                "2025-03-31",
            ],
            "filing_date": [
                "2025-01-01",
                "2025-02-01",
                "2025-03-01",
                "2025-04-01",
                "2025-05-01",
            ],
            "income_statement__revenues": [10.0, 20.0, 30.0, 40.0, 50.0],
            # Q4 should be ignored for NON_FLOW_OVERRIDES.
            "income_statement__diluted_average_shares": [100.0, 110.0, 120.0, 999.0, 130.0],
        }
    )

    assert "income_statement__diluted_average_shares" in NON_FLOW_OVERRIDES

    def fake_load(symbol):
        return df if symbol == "AAA" else None

    monkeypatch.setattr(
        "strategies.cross_section.data_loader._load_single_fundamental",
        fake_load,
    )

    panels = build_fundamental_panel(symbols, fields, trading_dates)

    revenues = panels["income_statement__revenues"]["AAA"]
    shares = panels["income_statement__diluted_average_shares"]["AAA"]

    # TTM first appears at Q4 filing date: 10+20+30+40=100
    assert revenues.loc[pd.Timestamp("2025-04-01")] == 100.0
    # Next quarter rolls: 20+30+40+50=140
    assert revenues.loc[pd.Timestamp("2025-05-01")] == 140.0

    # NON_FLOW_OVERRIDES should exclude Q4(999) and forward-fill Q3 value until Q1(130).
    assert shares.loc[pd.Timestamp("2025-04-01")] == 120.0
    assert shares.loc[pd.Timestamp("2025-05-01")] == 130.0
