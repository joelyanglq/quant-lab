"""
因子评估图表模块

华泰金工风格：12+ 张标准图表。
- IC: rank_ic 时序(+21MA), cumsum_ic, monthly ICIR, monthly IC mean, yearly ICIR, yearly IC mean
- 回归: factor return 时序, 累计 factor return (max/min cumsum)
- 换手: 分组换手率柱状图, 行业换手率
- 收益: 分层回测折线, top 组合收益+回撤
"""
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from .ic_analysis import ICReport
from .regression_analysis import RegressionReport
from .turnover_analysis import TurnoverReport
from .return_analysis import ReturnReport

# ── 风格 ──

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
POS_COLOR = "#1f77b4"
NEG_COLOR = "#d62728"
NEUTRAL_COLOR = "#7f7f7f"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
    })


# ═══════════════════════════════════════════════════════════════
# IC 图表
# ═══════════════════════════════════════════════════════════════

def plot_ic_charts(report: ICReport, save_dir: Path, factor_name: str = ""):
    """生成全部 6 张 IC 图表。"""
    _setup_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{factor_name}_" if factor_name else ""

    _plot_rank_ic_timeseries(report, save_dir / f"{prefix}rank_ic_timeseries.png")
    _plot_cumsum_ic(report, save_dir / f"{prefix}cumsum_ic.png")
    _plot_monthly_icir(report, save_dir / f"{prefix}monthly_icir.png")
    _plot_monthly_ic_mean(report, save_dir / f"{prefix}monthly_ic_mean.png")
    _plot_yearly_icir(report, save_dir / f"{prefix}yearly_icir.png")
    _plot_yearly_ic_mean(report, save_dir / f"{prefix}yearly_ic_mean.png")


def _plot_rank_ic_timeseries(report: ICReport, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ic = report.daily_rank_ic
    ma = report.daily_rank_ic_ma
    if ic.empty:
        plt.close(); return
    ax.bar(ic.index, ic.values, width=1, alpha=0.3, color=POS_COLOR, label="Daily Rank IC")
    ax.plot(ma.index, ma.values, color=NEG_COLOR, linewidth=1.2, label="21-day MA")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"Daily Rank IC (horizon={report.best_horizon}d)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cumsum_ic(report: ICReport, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ic = report.daily_rank_ic
    if ic.empty:
        plt.close(); return
    cumsum = ic.cumsum()
    ax.plot(cumsum.index, cumsum.values, color=POS_COLOR, linewidth=1.2)
    ax.fill_between(cumsum.index, cumsum.values, alpha=0.15, color=POS_COLOR)
    ax.set_title(f"Cumulative Rank IC (horizon={report.best_horizon}d)")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_monthly_icir(report: ICReport, path: Path):
    """月度 ICIR 柱状图 — 每月一个 bar。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    icir = report.monthly_icir
    if icir.empty:
        plt.close(); return
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in icir.values]
    ax.bar(icir.index.astype(str), icir.values, color=colors, alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("ICIR")
    ax.set_title("Monthly ICIR")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_monthly_ic_mean(report: ICReport, path: Path):
    """月度 IC 均值 — monthly_ic_mean 是 pivot table (month × year)。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    mic = report.monthly_ic_mean
    if mic.empty:
        plt.close(); return
    # 每月均值 across years
    mean_by_month = mic.mean(axis=1)
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in mean_by_month.values]
    ax.bar(mean_by_month.index.astype(str), mean_by_month.values, color=colors, alpha=0.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean IC")
    ax.set_title("Monthly Mean IC")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_yearly_icir(report: ICReport, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    yicir = report.yearly_icir
    if yicir.empty:
        plt.close(); return
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in yicir.values]
    ax.bar(yicir.index.astype(str), yicir.values, color=colors, alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("ICIR")
    ax.set_title("Yearly ICIR")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_yearly_ic_mean(report: ICReport, path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    yic = report.yearly_ic_mean
    if yic.empty:
        plt.close(); return
    colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in yic.values]
    ax.bar(yic.index.astype(str), yic.values, color=colors, alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean IC")
    ax.set_title("Yearly Mean IC")
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 回归图表
# ═══════════════════════════════════════════════════════════════

def plot_regression_charts(report: RegressionReport, save_dir: Path, factor_name: str = ""):
    _setup_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{factor_name}_" if factor_name else ""

    _plot_factor_return(report, save_dir / f"{prefix}factor_return.png")
    _plot_cum_factor_return(report, save_dir / f"{prefix}cum_factor_return.png")


def _plot_factor_return(report: RegressionReport, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    fr = report.factor_returns
    if fr.empty:
        plt.close(); return
    ax.plot(fr.index, fr.values, color=NEUTRAL_COLOR, alpha=0.5, linewidth=0.5)
    ma = fr.rolling(21, min_periods=5).mean()
    ax.plot(ma.index, ma.values, color=POS_COLOR, linewidth=1.2, label="21-day MA")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Factor Return (regression coefficient)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cum_factor_return(report: RegressionReport, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    if report.cumsum.empty:
        plt.close(); return
    ax.plot(report.cumsum.index, report.cumsum.values, color=POS_COLOR, linewidth=1.2, label="Cumsum")
    ax.plot(report.max_cumsum.index, report.max_cumsum.values, color="#2ca02c",
            linewidth=0.8, linestyle="--", label="Max cumsum")
    ax.plot(report.min_cumsum.index, report.min_cumsum.values, color=NEG_COLOR,
            linewidth=0.8, linestyle="--", label="Min cumsum")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative Factor Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 换手率图表
# ═══════════════════════════════════════════════════════════════

def plot_turnover_charts(report: TurnoverReport, save_dir: Path, factor_name: str = ""):
    _setup_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{factor_name}_" if factor_name else ""

    _plot_group_turnover(report, save_dir / f"{prefix}group_turnover.png")
    if not report.industry_turnover.empty:
        _plot_industry_turnover(report, save_dir / f"{prefix}industry_turnover.png")


def _plot_group_turnover(report: TurnoverReport, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    gt = report.group_turnover
    colors = [COLORS[i % len(COLORS)] for i in range(len(gt))]
    ax.bar(gt.index, gt.values, color=colors, alpha=0.8)
    ax.set_xlabel("Quantile Group")
    ax.set_ylabel("Avg Turnover")
    ax.set_title("Average Turnover by Quantile Group")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_industry_turnover(report: TurnoverReport, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    it = report.industry_turnover.head(15)  # top 15 industries
    ax.barh(it.index[::-1], it.values[::-1], color=POS_COLOR, alpha=0.8)
    ax.set_xlabel("Avg Industry Weight Change")
    ax.set_title("Top Group: Industry Turnover")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 收益图表
# ═══════════════════════════════════════════════════════════════

def plot_return_charts(report: ReturnReport, save_dir: Path, factor_name: str = ""):
    _setup_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{factor_name}_" if factor_name else ""

    _plot_stratified_returns(report, save_dir / f"{prefix}stratified_returns.png")
    _plot_top_group(report, save_dir / f"{prefix}top_group.png")


def _plot_stratified_returns(report: ReturnReport, path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    df = report.quantile_cumrets
    if df.empty:
        plt.close(); return
    for i, col in enumerate(df.columns):
        lw = 2.0 if col == "L/S" else 1.0
        ls = "--" if col == "L/S" else "-"
        ax.plot(df.index, df[col].values, label=col, linewidth=lw, linestyle=ls,
                color=COLORS[i % len(COLORS)])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Quantile Stratified Cumulative Returns")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_top_group(report: ReturnReport, path: Path):
    """双面板: 上=cumulative return, 下=drawdown, 右侧注释指标表。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1], sharex=True)

    # 上: cumulative return
    cumret = report.quantile_cumrets
    if "Q5" in cumret.columns:
        top_cum = cumret["Q5"]
        ax1.plot(top_cum.index, top_cum.values, color=POS_COLOR, linewidth=1.5)
        ax1.fill_between(top_cum.index, top_cum.values, alpha=0.1, color=POS_COLOR)
    ax1.set_title("Top Group (Q5) — Cumulative Return & Drawdown")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # 下: drawdown
    dd = report.top_drawdown
    if not dd.empty:
        ax2.fill_between(dd.index, dd.values, alpha=0.5, color=NEG_COLOR)
        ax2.plot(dd.index, dd.values, color=NEG_COLOR, linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # 指标注释
    metrics = report.top_metrics
    if metrics:
        text_lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                if "return" in k or "vol" in k or "drawdown" in k:
                    text_lines.append(f"{k}: {v:.2%}")
                else:
                    text_lines.append(f"{k}: {v:.4f}")
            else:
                text_lines.append(f"{k}: {v}")
        text = "\n".join(text_lines)
        ax1.text(0.98, 0.95, text, transform=ax1.transAxes,
                 fontsize=8, verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
                 fontfamily="monospace")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
