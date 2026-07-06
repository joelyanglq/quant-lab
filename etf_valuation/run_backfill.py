"""
Full pipeline: fetch holdings → fetch ratios → compute valuations → persist → report.

Holdings source: FMP /stable/etf/holdings (daily, real-time weights).
Historical data: N-PORT quarterly filings (for history reconstruction only).

Priority: Tier 2 (11 sector ETFs) + Tier 3 (14 theme ETFs) + Tier 1 broadbase for context.
"""
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Setup
logging.basicConfig(level=logging.WARNING, format="%(message)s")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etf_valuation.config import load_config, METRICS
from etf_valuation.fetcher import RatiosBulkFetcher, FMPHoldingsFetcher
from etf_valuation.metrics import aggregate_etf_metrics
from etf_valuation.scoring import score_etf, percentile_to_signal
from etf_valuation.history import append_today_snapshot

config = load_config()

# Priority tickers: sectors + themes + broad for context
PRIORITY_TICKERS = [
    # Tier 1 broad (for context/ERP reference)
    "SPY",
    # Tier 2 sectors
    "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC",
    # Tier 3 themes
    "SMH", "IBB", "XBI", "ICLN", "KRE", "XOP", "IYR", "XHB", "ITB",
    "HACK", "BOTZ", "ARKK", "TAN",
]


def step1_fetch_holdings():
    """Fetch current ETF holdings from FMP for all priority ETFs."""
    print("=" * 70)
    print("STEP 1: Fetch ETF Holdings (FMP /stable/etf/holdings)")
    print("=" * 70)

    from data_sync.config import load_config as load_ds_config
    ds_config = load_ds_config()
    fmp_key = ds_config.get_api_key("fmp")

    fetcher = FMPHoldingsFetcher(fmp_key)
    holdings_map = {}

    for i, ticker in enumerate(PRIORITY_TICKERS, 1):
        try:
            df = fetcher.fetch(ticker)
            if df.empty:
                print(f"  [{i}/{len(PRIORITY_TICKERS)}] {ticker:<5} — empty")
                continue

            n = len(df)
            mapped = df["ticker"].notna().sum()
            wt = df["weight"].sum() if "weight" in df.columns else 0
            holdings_map[ticker] = df
            print(f"  [{i}/{len(PRIORITY_TICKERS)}] {ticker:<5} — {n} holdings, {mapped} tickers, wt={wt:.1f}%")
        except Exception as e:
            print(f"  [{i}/{len(PRIORITY_TICKERS)}] {ticker:<5} — ERROR: {e}")

        if i < len(PRIORITY_TICKERS):
            time.sleep(0.15)

    print(f"\n  Total: {len(holdings_map)}/{len(PRIORITY_TICKERS)} ETFs with holdings")
    print()
    return holdings_map


def step3_fetch_ratios():
    """Fetch today's FMP ratios-ttm-bulk."""
    print("=" * 70)
    print("STEP 3: Fetch FMP Ratios TTM Bulk")
    print("=" * 70)

    from data_sync.config import load_config as load_ds_config
    ds_config = load_ds_config()
    fmp_key = ds_config.get_api_key("fmp")

    ratios_dir = config.get_storage_path("ratios_bulk")
    fetcher = RatiosBulkFetcher(ratios_dir, fmp_key)
    n = fetcher.sync()
    if n:
        print(f"  Fetched {n} stock ratios")
    else:
        print("  Already up-to-date for today")
    print()
    return fetcher.load_latest()


def step4_treasury():
    """Get latest 10Y treasury rate."""
    print("=" * 70)
    print("STEP 4: Treasury Rates")
    print("=" * 70)

    try:
        from data_sync.config import load_config as load_ds_config
        from data_sync.providers.treasury import FMPTreasuryProvider

        ds_config = load_ds_config()
        treasury = FMPTreasuryProvider(ds_config)
        treasury.sync_with_status(None)

        rate = treasury.get_rate(date.today())
        if rate:
            print(f"  10Y Treasury: {rate:.2%}")
        else:
            print("  10Y Treasury: not available, using fallback")
            rate = 0.0449  # fallback
        print()
        return rate
    except Exception as e:
        print(f"  Treasury error: {e}, using fallback 4.49%")
        print()
        return 0.0449


def step5_compute_and_report(holdings_map, ratios_df, treasury_10y):
    """Compute valuations from FMP holdings + ratios and print report."""
    print("=" * 70)
    print("STEP 5: Compute ETF Valuations")
    print("=" * 70)

    results = []
    for ticker in PRIORITY_TICKERS:
        etf_def = config.etfs.get(ticker)
        if not etf_def or not etf_def.primary:
            continue

        holdings = holdings_map.get(ticker)
        if holdings is None or holdings.empty or "ticker" not in holdings.columns:
            print(f"  {ticker:<5} — no holdings data")
            continue

        matched = holdings[holdings["ticker"].notna()]
        if matched.empty:
            print(f"  {ticker:<5} — no mapped tickers")
            continue

        # Compute all relevant metrics
        metrics_to_compute = [m for m in METRICS.keys()]
        agg = aggregate_etf_metrics(
            matched, ratios_df,
            metrics=metrics_to_compute,
            treasury_10y=treasury_10y,
        )

        pri_val = agg.get(etf_def.primary, {}).get("value", np.nan)
        pri_cov = agg.get(etf_def.primary, {}).get("coverage", 0)
        sec_val = agg.get(etf_def.secondary, {}).get("value", np.nan) if etf_def.secondary else np.nan
        sec_cov = agg.get(etf_def.secondary, {}).get("coverage", 0) if etf_def.secondary else 0

        n_matched = len(matched)
        n_total = len(holdings)
        w_cov = matched["weight"].sum()

        row = {
            "ticker": ticker,
            "name": etf_def.name,
            "tier": etf_def.tier,
            "primary_metric": etf_def.primary,
            "primary_value": pri_val,
            "primary_coverage": pri_cov,
            "secondary_metric": etf_def.secondary,
            "secondary_value": sec_val,
            "secondary_coverage": sec_cov,
            "holdings_matched": n_matched,
            "holdings_total": n_total,
            "weight_coverage": w_cov,
        }

        # Add all computed metrics
        for m in METRICS.keys():
            row[f"{m}_value"] = agg.get(m, {}).get("value", np.nan)

        results.append(row)
        status = "OK" if pri_cov > 0.7 else "LOW COV"
        print(f"  {ticker:<5} {etf_def.name:<16} matched={n_matched}/{n_total} weight={w_cov:.0f}% [{status}]")

    print()
    return results


def step6_persist_snapshots(results):
    """Save today's ETF metrics to snapshot history files."""
    print("=" * 70)
    print("STEP 6: Persist Snapshots")
    print("=" * 70)

    snapshots_dir = config.get_storage_path("snapshots")
    saved = 0
    for r in results:
        ticker = r["ticker"]
        today_metrics = {}
        for m in METRICS.keys():
            val = r.get(f"{m}_value")
            if val is not None:
                today_metrics[m] = val
                cov_key = f"{r['primary_metric']}_coverage" if m == r["primary_metric"] else None
                if cov_key:
                    today_metrics["coverage"] = r.get("primary_coverage", 0)

        if today_metrics:
            append_today_snapshot(ticker, today_metrics, snapshots_dir)
            saved += 1

    print(f"  Saved {saved} ETF snapshots to {snapshots_dir}")
    print()


def load_percentiles() -> dict:
    """Load historical percentiles for all ETFs from snapshot files."""
    snapshots_dir = config.get_storage_path("snapshots")
    percentiles = {}

    for hist_file in snapshots_dir.glob("*_history.parquet"):
        ticker = hist_file.stem.replace("_history", "")
        df = pd.read_parquet(hist_file)
        if len(df) < 4:
            continue

        pct = {}
        for metric in METRICS.keys():
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            if len(series) < 4:
                continue
            current = series.iloc[-1]
            pct[metric] = float(np.mean(series.values <= current))

        if pct:
            percentiles[ticker] = pct

    return percentiles


def print_final_report(results, treasury_10y):
    """Print the final formatted valuation report with percentile-based signals."""
    if not results:
        print("No results to report.")
        return

    # Load historical percentiles
    pct_data = load_percentiles()

    print()
    print("=" * 100)
    print("                     US ETF VALUATION REPORT")
    print(f"                     {date.today().isoformat()}  |  10Y Treasury: {treasury_10y:.2%}")
    print("=" * 100)

    # Group by tier
    tiers_order = ["broad", "sectors", "themes"]
    tier_labels = {"broad": "Tier 1: Broad Market", "sectors": "Tier 2: GICS Sectors", "themes": "Tier 3: Sub-Industry / Themes"}

    for tier in tiers_order:
        tier_results = [r for r in results if r["tier"] == tier]
        if not tier_results:
            continue

        print(f"\n{'─' * 100}")
        print(f"  {tier_labels.get(tier, tier)}")
        print(f"{'─' * 100}")
        print(f"  {'ETF':<6} {'Name':<16} {'P/E':>6} {'P/B':>6} {'P/S':>6} {'DivYld':>7} {'EV/EB':>6} {'FCF.Y':>6} {'ERP':>7} | {'Cov':>4} {'#':>4}")
        print(f"  {'─'*6} {'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*6} {'─'*6} {'─'*7}   {'─'*4} {'─'*4}")

        for r in tier_results:
            pe = r.get("pe_ttm_value", np.nan)
            pb = r.get("pb_lf_value", np.nan)
            ps = r.get("ps_ttm_value", np.nan)
            dy = r.get("div_yield_value", np.nan)
            ev = r.get("ev_ebitda_value", np.nan)
            fcf = r.get("fcf_yield_value", np.nan)
            erp = r.get("erp_value", np.nan)
            cov = r.get("weight_coverage", 0)
            n = r.get("holdings_matched", 0)

            pe_s = f"{pe:6.1f}" if not np.isnan(pe) else "   N/A"
            pb_s = f"{pb:6.1f}" if not np.isnan(pb) else "   N/A"
            ps_s = f"{ps:6.1f}" if not np.isnan(ps) else "   N/A"
            dy_s = f"{dy*100:6.2f}%" if not np.isnan(dy) else "    N/A"
            ev_s = f"{ev:6.1f}" if not np.isnan(ev) else "   N/A"
            fcf_s = f"{fcf*100:5.1f}%" if not np.isnan(fcf) else "   N/A"
            erp_s = f"{erp*100:6.2f}%" if not np.isnan(erp) else "    N/A"

            name = r["name"][:14]
            print(f"  {r['ticker']:<6} {name:<16} {pe_s} {pb_s} {ps_s} {dy_s} {ev_s} {fcf_s} {erp_s} | {cov:3.0f}% {n:4d}")

    # Cross-comparison with percentile-based signals
    print(f"\n{'=' * 100}")
    print("  VALUATION SIGNALS (Percentile-Based)")
    print(f"{'=' * 100}")
    print(f"  {'ETF':<6} {'Name':<16} {'Primary':<10} {'Value':>10} {'Pct%':>6} {'Signal':<13} {'History'}")
    print(f"  {'─'*6} {'─'*16} {'─'*10} {'─'*10} {'─'*6} {'─'*13} {'─'*20}")

    for r in results:
        ticker = r["ticker"]
        pri = r["primary_metric"]
        val = r["primary_value"]
        meta = METRICS.get(pri, {})
        direction = meta.get("direction", "")

        etf_pct = pct_data.get(ticker, {})
        pct_val = etf_pct.get(pri)

        if np.isnan(val):
            val_s = "N/A"
            pct_s = "  N/A"
            signal = "—"
            hist_s = ""
        else:
            if pri in ("div_yield", "fcf_yield", "erp"):
                val_s = f"{val*100:.2f}%"
            else:
                val_s = f"{val:.1f}"

            if pct_val is not None:
                # For "higher_cheap" metrics, invert the percentile for signal
                if direction == "higher_cheap":
                    signal_pct = 1.0 - pct_val
                else:
                    signal_pct = pct_val

                pct_s = f"{pct_val*100:5.0f}%"
                signal = _percentile_to_signal(signal_pct)
                n_hist = len(etf_pct)
                hist_s = f"{n_hist} metrics, {_get_n_quarters(ticker)} Q"
            else:
                pct_s = "  N/A"
                signal = _fallback_signal(pri, val)
                hist_s = "no history"

        print(f"  {ticker:<6} {r['name'][:14]:<16} {pri:<10} {val_s:>10} {pct_s} {signal:<13} {hist_s}")

    # Save report
    output_dir = Path("D:/04_Project/quant-lab/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Enrich results with percentile data
    for r in results:
        ticker = r["ticker"]
        etf_pct = pct_data.get(ticker, {})
        for metric, pct_val in etf_pct.items():
            r[f"{metric}_percentile"] = pct_val

    df = pd.DataFrame(results)
    csv_path = output_dir / f"etf_valuation_{date.today().isoformat()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Report saved to: {csv_path}")

    json_path = output_dir / f"etf_valuation_{date.today().isoformat()}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"  JSON saved to: {json_path}")


def _percentile_to_signal(pct: float) -> str:
    """Convert percentile to a signal label."""
    if pct <= 0.10:
        return "STRONG_BUY"
    elif pct <= 0.25:
        return "BUY"
    elif pct <= 0.40:
        return "LEAN_BUY"
    elif pct <= 0.60:
        return "HOLD"
    elif pct <= 0.75:
        return "LEAN_SELL"
    elif pct <= 0.90:
        return "SELL"
    else:
        return "STRONG_SELL"


def _fallback_signal(metric: str, val: float) -> str:
    """Fallback signal using hardcoded thresholds when no history is available."""
    if metric == "pe_ttm":
        return "Expensive" if val > 30 else "Moderate" if val > 20 else "Cheap"
    elif metric == "pb_lf":
        return "Expensive" if val > 4 else "Moderate" if val > 2 else "Cheap"
    elif metric == "ps_ttm":
        return "Expensive" if val > 5 else "Moderate" if val > 2 else "Cheap"
    elif metric == "ev_ebitda":
        return "Expensive" if val > 15 else "Moderate" if val > 10 else "Cheap"
    return "—"


def _get_n_quarters(ticker: str) -> int:
    """Get number of historical quarters for an ETF."""
    snapshots_dir = config.get_storage_path("snapshots")
    hist_file = snapshots_dir / f"{ticker}_history.parquet"
    if hist_file.exists():
        df = pd.read_parquet(hist_file)
        return len(df)
    return 0


def step7_timeseries_charts():
    """Generate time-series trajectory charts for visual inspection."""
    print("=" * 70)
    print("STEP 7: Time-Series Charts")
    print("=" * 70)

    from etf_valuation.report_charts import generate_timeseries_charts
    paths = generate_timeseries_charts()
    for p in paths:
        print(f"  Saved: {p}")
    print()
    return paths


def main():
    t0 = time.time()

    holdings_map = step1_fetch_holdings()
    ratios = step3_fetch_ratios()
    treasury_10y = step4_treasury()
    results = step5_compute_and_report(holdings_map, ratios, treasury_10y)
    step6_persist_snapshots(results)
    print_final_report(results, treasury_10y)
    step7_timeseries_charts()

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed/60:.1f} minutes")
    print("  Done.")


if __name__ == "__main__":
    main()
