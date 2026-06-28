"""
因子评估模块

四步评估流程: IC 分析 → 回归分析 → 换手率分析 → 收益分析
"""
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .ic_analysis import compute_ic_analysis, ICReport
from .regression_analysis import compute_regression_analysis, RegressionReport
from .turnover_analysis import compute_turnover_analysis, TurnoverReport
from .return_analysis import compute_return_analysis, ReturnReport
from .plotting import (
    plot_ic_charts, plot_regression_charts,
    plot_turnover_charts, plot_return_charts,
)


def run_factor_eval(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    factor_name: str = "",
    horizons: List[int] = None,
    horizon: Optional[int] = None,
    rebalance_freq: str = "M",
    sector_map: pd.Series = None,
    benchmark_close: pd.Series = None,
    save_dir: Optional[str] = None,
    plot: bool = True,
) -> dict:
    """
    完整因子评估 pipeline。

    Args:
        factor: 预处理后因子面板 (index=dates, columns=symbols)
        close: 收盘价面板
        factor_name: 因子名称 (用于图表标题和文件名)
        horizons: IC 分析的周期列表
        horizon: 手动指定评估 horizon (None → IC 分析自动选择)
        rebalance_freq: 调仓频率
        sector_map: 行业映射 (用于换手率分析)
        benchmark_close: 基准价格 (用于超额收益)
        save_dir: 图表保存目录 (None → 不保存)
        plot: 是否生成图表

    Returns:
        dict with ic_report, regression_report, turnover_report, return_report
    """
    print(f"\n{'='*60}")
    print(f"  FACTOR EVALUATION: {factor_name or 'unnamed'}")
    print(f"{'='*60}")

    # Step 1: IC 分析
    ic_report = compute_ic_analysis(factor, close, horizons=horizons)

    # 确定 horizon
    eval_horizon = horizon if horizon is not None else ic_report.best_horizon
    print(f"  Using horizon: {eval_horizon}d")

    # Step 2: 回归分析
    reg_report = compute_regression_analysis(factor, close, horizon=eval_horizon)

    # Step 3: 换手率分析
    turnover_report = compute_turnover_analysis(
        factor, n_quantiles=5, sector_map=sector_map,
    )

    # Step 4: 收益分析
    return_report = compute_return_analysis(
        factor, close,
        horizon=eval_horizon,
        rebalance_freq=rebalance_freq,
        benchmark_close=benchmark_close,
    )

    # 图表
    if plot and save_dir:
        out = Path(save_dir) / factor_name if factor_name else Path(save_dir)
        plot_ic_charts(ic_report, out, factor_name)
        plot_regression_charts(reg_report, out, factor_name)
        plot_turnover_charts(turnover_report, out, factor_name)
        plot_return_charts(return_report, out, factor_name)
        print(f"  Charts saved to: {out}")

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE: {factor_name or 'unnamed'}")
    print(f"{'='*60}\n")

    return {
        "ic_report": ic_report,
        "regression_report": reg_report,
        "turnover_report": turnover_report,
        "return_report": return_report,
    }