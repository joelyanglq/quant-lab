"""
Generate time-series trajectory charts for ETF valuation metrics.

Called at the end of each pipeline run to produce per-tier charts
showing historical trajectories with mean/σ bands, current percentile,
and signal annotation — the visual counterpart to the text report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from etf_valuation.config import load_config, METRICS

# ── Theme ──
DARK_THEME = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.size": 9,
}

METRIC_DISPLAY = {
    "pe_ttm": "PE TTM",
    "pb_lf": "P/B",
    "ps_ttm": "P/S",
    "div_yield": "Div Yield",
    "ev_ebitda": "EV/EBITDA",
    "erp": "ERP",
    "fcf_yield": "FCF Yield",
}

TIER_LAYOUT = {
    "broad":   {"title": "Broad Market", "color": "#58a6ff"},
    "sectors": {"title": "GICS Sectors", "color": "#bc8cff"},
    "themes":  {"title": "Theme / Sub-Industry", "color": "#f778ba"},
}

PALETTE = ["#58a6ff", "#bc8cff", "#f778ba", "#3fb950", "#d29922", "#ff7b72"]


def _signal_color(pct: float, direction: str) -> str:
    signal_pct = (1.0 - pct) if direction == "higher_cheap" else pct
    if signal_pct <= 0.10:
        return "#238636"
    if signal_pct <= 0.25:
        return "#3fb950"
    if signal_pct <= 0.60:
        return "#d29922"
    if signal_pct <= 0.90:
        return "#f85149"
    return "#da3633"


def _signal_label(pct: float, direction: str) -> str:
    sp = (1.0 - pct) if direction == "higher_cheap" else pct
    if sp <= 0.10:
        return "STRONG_BUY"
    if sp <= 0.25:
        return "BUY"
    if sp <= 0.40:
        return "LEAN_BUY"
    if sp <= 0.60:
        return "HOLD"
    if sp <= 0.75:
        return "LEAN_SELL"
    if sp <= 0.90:
        return "SELL"
    return "STRONG_SELL"


def _fmt_val(metric: str, val: float) -> str:
    if np.isnan(val):
        return "N/A"
    if metric in ("div_yield", "erp", "fcf_yield"):
        return f"{val * 100:.2f}%"
    return f"{val:.1f}"


def generate_timeseries_charts(
    output_dir: Optional[Path] = None,
    snapshots_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate time-series charts grouped by tier. Returns list of saved paths."""
    plt.rcParams.update(DARK_THEME)
    try:
        plt.rcParams["font.family"] = "Microsoft YaHei"
    except Exception:
        pass

    config = load_config()
    if snapshots_dir is None:
        snapshots_dir = config.get_storage_path("snapshots")
    if output_dir is None:
        output_dir = Path("D:/04_Project/quant-lab/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    today_str = date.today().isoformat()
    saved = []

    # Group ETFs by tier
    tier_tickers = {}
    for ticker in [
        "SPY",
        "XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLU", "XLB", "XLRE", "XLC",
        "SMH", "IBB", "XBI", "ICLN", "KRE", "XOP", "IYR", "XHB", "ITB",
        "HACK", "BOTZ", "ARKK", "TAN",
    ]:
        etf_def = config.etfs.get(ticker)
        if not etf_def:
            continue
        tier_tickers.setdefault(etf_def.tier, []).append(ticker)

    for tier_key, tickers in tier_tickers.items():
        layout = TIER_LAYOUT.get(tier_key, {"title": tier_key, "color": "#58a6ff"})

        # Determine grid: each ETF gets a row with primary + secondary metric
        n_etfs = len(tickers)
        n_cols = 2
        fig, axes = plt.subplots(n_etfs, n_cols, figsize=(16, 2.8 * n_etfs + 1.2))
        fig.suptitle(
            f'{layout["title"]} — Valuation Trajectories  ({today_str})',
            fontsize=14, fontweight="bold", color=layout["color"], y=0.995,
        )

        if n_etfs == 1:
            axes = axes.reshape(1, -1)

        for row, ticker in enumerate(tickers):
            etf_def = config.etfs.get(ticker)
            hist_file = snapshots_dir / f"{ticker}_history.parquet"
            if not hist_file.exists() or not etf_def:
                for c in range(n_cols):
                    axes[row, c].set_visible(False)
                continue

            df = pd.read_parquet(hist_file)
            metrics_to_plot = [etf_def.primary, etf_def.secondary]

            for col, metric in enumerate(metrics_to_plot):
                ax = axes[row, col]
                if metric is None or metric not in df.columns:
                    ax.set_visible(False)
                    continue

                series = df[metric].dropna()
                if len(series) < 3:
                    ax.set_visible(False)
                    continue

                _draw_metric_panel(ax, series, ticker, metric, etf_def.name,
                                   is_primary=(col == 0), color=PALETTE[col])

        plt.tight_layout(rect=[0, 0, 1, 0.99])
        out_path = output_dir / f"ts_{tier_key}_{today_str}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

    # SPY all-metrics deep view
    spy_path = _generate_spy_deep(snapshots_dir, output_dir, today_str)
    if spy_path:
        saved.append(spy_path)

    return saved


def _draw_metric_panel(
    ax, series: pd.Series, ticker: str, metric: str,
    name: str, is_primary: bool, color: str,
):
    """Draw one metric's time-series panel."""
    dates = series.index
    values = series.values
    meta = METRICS.get(metric, {})
    direction = meta.get("direction", "")

    # Display values: multiply yield metrics by 100
    is_yield = metric in ("div_yield", "erp", "fcf_yield")
    dv = values * 100 if is_yield else values

    mean_val = np.mean(dv)
    std_val = np.std(dv)
    current = dv[-1]
    pct = float(np.mean(dv <= current))

    # ±1σ band
    ax.fill_between(dates, mean_val - std_val, mean_val + std_val,
                     alpha=0.12, color=color)

    # P10/P90 lines
    p10, p90 = np.percentile(dv, 10), np.percentile(dv, 90)
    ax.axhline(p10, color="#3fb950", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axhline(p90, color="#f85149", linewidth=0.7, linestyle=":", alpha=0.5)

    # Mean line
    ax.axhline(mean_val, color="#3fb950", linewidth=1, linestyle="--", alpha=0.6)

    # Trajectory
    ax.plot(dates, dv, color=color, linewidth=1.8, marker="o", markersize=3.5, zorder=3)

    # Current value dot
    dot_color = _signal_color(pct, direction)
    ax.scatter([dates[-1]], [current], color=dot_color, s=70, zorder=5,
               edgecolors="white", linewidth=1.5)

    # COVID shading
    covid_mask = (dates >= "2020-02-01") & (dates <= "2020-06-30")
    if covid_mask.any():
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-01"),
                    alpha=0.08, color="#f85149")

    # Title
    role = "PRIMARY" if is_primary else "secondary"
    label = METRIC_DISPLAY.get(metric, metric)
    ax.set_title(f"{ticker} — {name}  [{label}, {role}]",
                 color="#f0f6fc", fontsize=10, fontweight="bold")

    # Info box
    signal = _signal_label(pct, direction)
    unit = "%" if is_yield else ""
    info = (
        f"Now: {current:.2f}{unit}\n"
        f"Mean: {mean_val:.2f}{unit}\n"
        f"Pct: {pct * 100:.0f}%\n"
        f"{signal}"
    )
    sig_color = _signal_color(pct, direction)
    ax.text(
        0.02, 0.97, info, transform=ax.transAxes,
        fontsize=7.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                  edgecolor=sig_color, alpha=0.9),
        color=sig_color, fontweight="bold",
    )

    ax.grid(True, alpha=0.2)
    ax.tick_params(axis="x", rotation=30, labelsize=7)


def _generate_spy_deep(
    snapshots_dir: Path, output_dir: Path, today_str: str,
) -> Optional[Path]:
    """SPY 6-metric deep view."""
    hist_file = snapshots_dir / "SPY_history.parquet"
    if not hist_file.exists():
        return None

    df = pd.read_parquet(hist_file)
    metrics = ["pe_ttm", "pb_lf", "ps_ttm", "div_yield", "ev_ebitda", "erp"]
    available = [m for m in metrics if m in df.columns and df[m].dropna().shape[0] >= 3]
    if not available:
        return None

    n = len(available)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows + 1))
    fig.suptitle(f"SPY — All Metrics Deep View  ({today_str})",
                 fontsize=14, fontweight="bold", color="#58a6ff", y=0.995)

    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, metric in enumerate(available):
        ax = axes[i // cols][i % cols]
        series = df[metric].dropna()
        _draw_metric_panel(ax, series, "SPY", metric, "S&P 500",
                           is_primary=(metric == "pe_ttm"),
                           color=PALETTE[i % len(PALETTE)])

    # Hide unused
    for i in range(len(available), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = output_dir / f"ts_spy_deep_{today_str}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    paths = generate_timeseries_charts()
    for p in paths:
        print(f"  Saved: {p}")
