"""
对比 z-score 与 0-1 标准化在等权因子打分选股中的回测表现

流程:
    1. compute_all_factors → prepare_weekly → 17 筛选因子
    2. 分别用 z-score / 0-1 标准化构建截面打分
    3. backtest_predictions 评估 L/S, Long-only, 分档收益

用法:
    cd OneBacktest
    python -m factor_research.compare_normalization
"""
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List

from .weekly_pipeline import compute_all_factors, prepare_weekly, backtest_predictions
from .data_loader import load_index_symbols

warnings.filterwarnings('ignore')

# 筛选后的 17 因子 + IC 方向 (来自 run_full_pipeline 结果)
SELECTED_17 = {
    'noon_shade':            +1,
    'lone_goose':            +1,
    'peak_climbing':         +1,
    'modified_amplitude_2':  +1,
    'RS_12M':                +1,
    'Range_52W':             +1,
    'intraday_rev_volflip':  +1,
    'moderate_risk':         +1,
    'PS':                    +1,
    'EV_EBITDA':             +1,
    'vol_battle_pos':        +1,
    'EPS_Score':             +1,
    'fuzzy_corr':            +1,
    'FCF_Growth':            -1,
    'ROE':                   -1,
    'ROIC':                  -1,
    'full_tidal':            -1,
}


# ═══════════════════════════════════════════════════════════════
# 两种标准化
# ═══════════════════════════════════════════════════════════════

def _mad_winsorize(row: pd.Series) -> pd.Series:
    """MAD winsorize: clip to median ± 3σ_MAD"""
    median = row.median()
    mad = (row - median).abs().median()
    if mad == 0:
        return row
    cutoff = 3.0 * 1.4826 * mad
    return row.clip(median - cutoff, median + cutoff)


def cs_zscore(wide: pd.DataFrame, directions: Dict[str, int]) -> pd.DataFrame:
    """逐行 MAD winsorize → z-score, IC 方向翻转"""
    result = wide.copy()
    for col in result.columns:
        direction = directions.get(col, +1)
        s = result[col]
        s = _mad_winsorize(s)
        mu, sd = s.mean(), s.std()
        if sd > 0:
            result[col] = direction * (s - mu) / sd
        else:
            result[col] = 0.0
    return result


def cs_minmax(wide: pd.DataFrame, directions: Dict[str, int]) -> pd.DataFrame:
    """逐行 MAD winsorize → 0-1 标准化, IC 方向翻转"""
    result = wide.copy()
    for col in result.columns:
        direction = directions.get(col, +1)
        s = result[col]
        s = _mad_winsorize(s)
        if direction < 0:
            s = -s
        lo, hi = s.min(), s.max()
        if hi > lo:
            result[col] = (s - lo) / (hi - lo)
        else:
            result[col] = 0.5
    return result


def build_weekly_scores(
    factors_w: Dict[str, pd.DataFrame],
    fwd_ret_w: pd.DataFrame,
    selected: List[str],
    norm_fn,
    directions: Dict[str, int],
) -> pd.DataFrame:
    """
    构建等权打分的周频预测面板 (date × symbol).
    每周截面独立标准化, 然后等权平均.
    只保留所有 17 因子都有值的 symbol.
    """
    base_idx = fwd_ret_w.index
    base_cols = fwd_ret_w.columns
    for name in selected:
        if name in factors_w:
            base_cols = base_cols.intersection(factors_w[name].columns)

    scores = {}
    for dt in base_idx:
        row_data = {}
        for name in selected:
            if name not in factors_w:
                continue
            panel = factors_w[name]
            if dt not in panel.index:
                continue
            row_data[name] = panel.loc[dt, base_cols]

        if len(row_data) < len(selected):
            continue

        df = pd.DataFrame(row_data)
        # 剔除任何因子缺失的 symbol
        df = df.dropna()
        if len(df) < 30:
            continue

        normed = norm_fn(df, directions)
        scores[dt] = normed.mean(axis=1)

    return pd.DataFrame(scores).T.sort_index()


def main():
    t_start = time.time()

    print('=' * 70)
    print('NORMALIZATION COMPARISON: z-score vs 0-1')
    print('=' * 70)

    # 1. 计算因子
    print('\n[1/4] Computing factors ...')
    symbols = load_index_symbols()
    all_factors = compute_all_factors(symbols)

    # 2. 对齐周频
    print('\n[2/4] Preparing weekly data ...')
    from .data_loader import load_price_panel
    panels_1d = load_price_panel(symbols, '2025-01-01', '2026-12-31')
    close = panels_1d['close']
    factors_w, fwd_ret_w = prepare_weekly(all_factors, close)

    selected = [f for f in SELECTED_17 if f in factors_w]
    print(f'Available factors: {len(selected)}/{len(SELECTED_17)}')
    missing = set(SELECTED_17) - set(selected)
    if missing:
        print(f'Missing: {missing}')

    # 3. 两种标准化打分
    print('\n[3/4] Building scores ...')
    pred_zscore = build_weekly_scores(
        factors_w, fwd_ret_w, selected, cs_zscore, SELECTED_17)
    pred_minmax = build_weekly_scores(
        factors_w, fwd_ret_w, selected, cs_minmax, SELECTED_17)

    print(f'  z-score: {pred_zscore.shape[0]} weeks × {pred_zscore.shape[1]} symbols')
    print(f'  0-1:     {pred_minmax.shape[0]} weeks × {pred_minmax.shape[1]} symbols')

    # 4. 回测
    print('\n[4/4] Backtesting ...')
    results = {}
    for label, pred_df in [('z-score', pred_zscore), ('0-1', pred_minmax)]:
        bt = backtest_predictions(pred_df, fwd_ret_w, n_quantiles=5, long_only_n=30)
        results[label] = bt

    # ── 输出 ──
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)

    for label in ['z-score', '0-1']:
        bt = results[label]
        ls = bt['ls_metrics']
        lo = bt['lo_metrics']
        qa = bt['quantile_annual']
        print(f'\n--- {label} ---')
        print(f'  L/S:  AnnRet={ls.get("annual_ret",0):+.1%}  '
              f'Sharpe={ls.get("sharpe",0):.2f}  '
              f'WinRate={ls.get("win_rate",0):.1%}  '
              f'MaxDD={ls.get("max_dd",0):.1%}  '
              f'N={ls.get("n_weeks",0)}wk')
        print(f'  LO30: AnnRet={lo.get("annual_ret",0):+.1%}  '
              f'Sharpe={lo.get("sharpe",0):.2f}  '
              f'WinRate={lo.get("win_rate",0):.1%}  '
              f'MaxDD={lo.get("max_dd",0):.1%}')
        q_str = '  '.join(f'{k}={v:+.1%}' for k, v in sorted(qa.items()))
        print(f'  Quantile annual: {q_str}')

    # ── IC 对比 ──
    print('\n--- Rank IC comparison ---')
    for label, pred_df in [('z-score', pred_zscore), ('0-1', pred_minmax)]:
        common = pred_df.index.intersection(fwd_ret_w.index)
        common_cols = pred_df.columns.intersection(fwd_ret_w.columns)
        ics = []
        for dt in common:
            p = pred_df.loc[dt, common_cols].dropna()
            r = fwd_ret_w.loc[dt].reindex(p.index).dropna()
            c = p.index.intersection(r.index)
            if len(c) >= 30:
                ic = p[c].corr(r[c], method='spearman')
                if not np.isnan(ic):
                    ics.append(ic)
        if ics:
            s = pd.Series(ics)
            print(f'  {label:8s}: meanIC={s.mean():+.4f}  '
                  f'stdIC={s.std():.4f}  '
                  f'ICIR={s.mean()/s.std():.2f}  '
                  f'N={len(s)}wk')

    print(f'\nTotal time: {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
