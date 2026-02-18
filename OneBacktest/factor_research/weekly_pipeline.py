"""
Weekly Alpha Pipeline
=====================
All-in-one: Compute factors → Screen → ML → Backtest → Live Signal

Usage:
    cd OneBacktest
    python -m factor_research.weekly_pipeline                 # 全流程
    python -m factor_research.weekly_pipeline --no-backtest   # 只出信号
    python -m factor_research.weekly_pipeline --n-symbols 50  # 快速测试
"""
import warnings
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .screening import screen_factors

_ROOT = Path(__file__).resolve().parent.parent.parent


# ═══════════════════════════════════════════════════════════════
# Step 1: Compute All Factors
# ═══════════════════════════════════════════════════════════════

def compute_all_factors(
    symbols: List[str],
    start: str = '2025-04-03',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    """计算所有因子, 返回 {name: daily_panel}"""
    from .data_loader import load_price_panel
    from .factors import (
        compute_technical_factors,
        compute_reversal_factors,
        compute_fundamental_factors,
        compute_microstructure_factors,
        compute_battle_factors,
        compute_surge_factors,
        compute_regression_factors,
        compute_fuzzy_factors,
        compute_rebuild_factors,
        compute_tidal_factors,
        compute_jump_factors,
    )

    all_factors = {}

    # ── 日频因子 (快速, 只需1d数据) ──
    print('=' * 60)
    print('DAILY-ONLY FACTORS')
    print('=' * 60)

    t0 = time.time()
    panels_1d = load_price_panel(symbols, '2024-01-01', end)
    close = panels_1d['close']
    open_p = panels_1d['open']
    high = panels_1d['high']
    low = panels_1d['low']
    print(f'  Loaded 1d data: {close.shape} ({time.time()-t0:.1f}s)')

    # Technical
    print('\n[technical] ...', flush=True)
    t1 = time.time()
    tech = compute_technical_factors(close, high, low)
    all_factors.update(tech)
    print(f'  +{len(tech)} factors ({time.time()-t1:.1f}s)')

    # Reversal
    print('[reversal] ...', flush=True)
    t1 = time.time()
    rev = compute_reversal_factors(close, open_p)
    all_factors.update(rev)
    print(f'  +{len(rev)} factors ({time.time()-t1:.1f}s)')

    # Fundamental
    print('[fundamental] ...', flush=True)
    t1 = time.time()
    try:
        fund = compute_fundamental_factors(symbols, close)
        all_factors.update(fund)
        print(f'  +{len(fund)} factors ({time.time()-t1:.1f}s)')
    except Exception as e:
        print(f'  SKIPPED: {e}')

    # ── 分钟频因子 (慢, 需1min数据) ──
    print('\n' + '=' * 60)
    print('INTRADAY FACTORS (1min data)')
    print('=' * 60)

    intraday_funcs = [
        ('microstructure', compute_microstructure_factors),
        ('battle', compute_battle_factors),
        ('surge', compute_surge_factors),
        ('regression', compute_regression_factors),
        ('fuzzy', compute_fuzzy_factors),
        ('rebuild', compute_rebuild_factors),
        ('tidal', compute_tidal_factors),
        ('jump', compute_jump_factors),
    ]

    for label, func in intraday_funcs:
        print(f'\n[{label}] ...', flush=True)
        t1 = time.time()
        try:
            result = func(symbols, start, end)
            all_factors.update(result)
            print(f'  +{len(result)} factors ({time.time()-t1:.1f}s)')
        except Exception as e:
            print(f'  FAILED: {e}')

    print(f'\n{"=" * 60}')
    print(f'Total: {len(all_factors)} factors')
    return all_factors


# ═══════════════════════════════════════════════════════════════
# Step 2: Weekly Alignment
# ═══════════════════════════════════════════════════════════════

def prepare_weekly(
    all_factors: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    对齐到周频 (W-FRI):
    - 因子: 取周五收盘值
    - 前瞻收益: 下周五/本周五 - 1
    """
    close_w = close.resample('W-FRI').last().dropna(how='all')
    fwd_ret_w = close_w.shift(-1) / close_w - 1

    factors_w = {}
    for name, panel in all_factors.items():
        w = panel.resample('W-FRI').last().dropna(how='all')
        # 仅保留有足够数据的因子
        if len(w) >= 5 and w.notna().any(axis=1).sum() >= 5:
            factors_w[name] = w

    print(f'Weekly factors: {len(factors_w)}/{len(all_factors)} passed')
    print(f'Weekly dates: {len(close_w)}, symbols: {close_w.shape[1]}')
    return factors_w, fwd_ret_w


# ═══════════════════════════════════════════════════════════════
# Step 3: Factor Screening  →  screening.py
# ═══════════════════════════════════════════════════════════════

# screen_factors imported from .screening at top of file


# ═══════════════════════════════════════════════════════════════
# Step 4: Build Feature Matrix
# ═══════════════════════════════════════════════════════════════

def build_panel_data(
    factors_w: Dict[str, pd.DataFrame],
    fwd_ret_w: pd.DataFrame,
    selected: List[str],
) -> pd.DataFrame:
    """
    构建面板数据: (week, symbol) → [factor_1, ..., factor_n, fwd_ret]
    使用 fwd_ret 的日期/股票为主索引, 允许因子缺失 (填0)

    关键: 逐周截面 z-score, 保证所有因子在同一尺度上
    """
    # Use forward return dates as primary index
    base_idx = fwd_ret_w.index
    base_cols = fwd_ret_w.columns

    # Intersect columns across factors that have them
    for name in selected:
        if name in factors_w:
            base_cols = base_cols.intersection(factors_w[name].columns)

    # ── 逐因子: 逐周截面 z-score (MAD winsorize + z-score) ──
    def _cs_zscore(wide: pd.DataFrame) -> pd.DataFrame:
        """每行独立: MAD winsorize → (x-mean)/std"""
        def _row(row):
            median = row.median()
            mad = (row - median).abs().median()
            if mad == 0:
                return row
            cutoff = 3.0 * 1.4826 * mad
            w = row.clip(median - cutoff, median + cutoff)
            mu = w.mean()
            sd = w.std()
            if sd == 0 or np.isnan(sd):
                return row * 0
            return (w - mu) / sd
        return wide.apply(_row, axis=1)

    normalized_factors = {}
    for name in selected:
        if name in factors_w:
            wide = factors_w[name].reindex(index=base_idx, columns=base_cols)
            normalized_factors[name] = _cs_zscore(wide)

    # Stack into long format
    ret_long = fwd_ret_w.reindex(index=base_idx, columns=base_cols).stack()
    ret_long.name = 'fwd_ret'

    records = {'fwd_ret': ret_long}
    for name in selected:
        if name in normalized_factors:
            records[name] = normalized_factors[name].stack()

    panel = pd.DataFrame(records)
    panel.index.names = ['date', 'symbol']

    # Drop rows with NaN target
    panel = panel.dropna(subset=['fwd_ret'])
    # Fill NaN features with 0 (z-scored 后 0 = 截面均值)
    panel[selected] = panel[selected].fillna(0)

    n_dates = panel.index.get_level_values('date').nunique()
    n_syms = panel.index.get_level_values('symbol').nunique()
    print(f'\nPanel data: {panel.shape[0]:,} obs, '
          f'{len(selected)} features, '
          f'{n_dates} weeks, {n_syms} symbols')

    return panel


# ═══════════════════════════════════════════════════════════════
# Step 5: Walk-Forward ML
# ═══════════════════════════════════════════════════════════════

def walk_forward_predict(
    panel: pd.DataFrame,
    feature_cols: List[str],
    n_pca: int = 0,
    min_train_weeks: int = 8,
) -> Dict[str, pd.DataFrame]:
    """
    Walk-forward 预测:
    - 每周用历史数据训练, 预测本周截面收益率
    - 模型: Ridge, RandomForest, (XGBoost)
    - 可选 PCA 降维

    Returns: {model_name: DataFrame(date × symbol → predicted_return)}
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    pca_obj = None
    if n_pca > 0:
        from sklearn.decomposition import PCA
        pca_obj = PCA(n_components=min(n_pca, len(feature_cols)))

    # Optional: try XGBoost / LightGBM
    xgb_cls = None
    try:
        from xgboost import XGBRegressor
        xgb_cls = XGBRegressor
    except ImportError:
        pass

    lgb_cls = None
    try:
        from lightgbm import LGBMRegressor
        lgb_cls = LGBMRegressor
    except ImportError:
        pass

    dates = sorted(panel.index.get_level_values('date').unique())
    n_dates = len(dates)

    models_config = {
        'ridge': lambda: Ridge(alpha=1.0),
        'rf': lambda: RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=42),
    }
    if xgb_cls:
        models_config['xgb'] = lambda: xgb_cls(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42)
    if lgb_cls:
        models_config['lgb'] = lambda: lgb_cls(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42)

    # Also include simple equal-weight z-score average
    predictions = {name: {} for name in list(models_config.keys()) + ['equal_weight']}
    if n_pca > 0:
        predictions['pca_ridge'] = {}

    print(f'\nWalk-forward: {n_dates} weeks, min_train={min_train_weeks}')
    print(f'Models: {list(predictions.keys())}')

    for t in range(min_train_weeks, n_dates):
        dt = dates[t]
        train_dates = dates[:t]
        test_date = dt

        # Train/test split
        train_mask = panel.index.get_level_values('date').isin(train_dates)
        test_mask = panel.index.get_level_values('date') == test_date

        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, 'fwd_ret'].values
        X_test = panel.loc[test_mask, feature_cols].values
        test_symbols = panel.loc[test_mask].index.get_level_values('symbol')

        if len(X_test) < 20 or len(X_train) < 100:
            continue

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Equal-weight z-score average (no ML)
        eq_pred = X_test_s.mean(axis=1)
        predictions['equal_weight'][dt] = pd.Series(eq_pred, index=test_symbols)

        # PCA + Ridge
        if n_pca > 0 and pca_obj is not None:
            pca_fit = pca_obj.fit(X_train_s)
            X_train_pca = pca_fit.transform(X_train_s)
            X_test_pca = pca_fit.transform(X_test_s)
            m = Ridge(alpha=1.0)
            m.fit(X_train_pca, y_train)
            pred = m.predict(X_test_pca)
            predictions['pca_ridge'][dt] = pd.Series(pred, index=test_symbols)

        # ML models
        for model_name, model_factory in models_config.items():
            m = model_factory()
            m.fit(X_train_s, y_train)
            pred = m.predict(X_test_s)
            predictions[model_name][dt] = pd.Series(pred, index=test_symbols)

        done = t - min_train_weeks + 1
        total = n_dates - min_train_weeks
        if done % 3 == 0 or done == total:
            print(f'  week {done}/{total}: '
                  f'train={len(X_train):,}, test={len(X_test):,}', flush=True)

    # Convert to DataFrames
    result = {}
    for model_name, preds in predictions.items():
        if len(preds) > 0:
            result[model_name] = pd.DataFrame(preds).T.sort_index()

    return result


# ═══════════════════════════════════════════════════════════════
# Step 6: Backtest Predictions
# ═══════════════════════════════════════════════════════════════

def backtest_predictions(
    pred_df: pd.DataFrame,
    fwd_ret_w: pd.DataFrame,
    n_quantiles: int = 5,
    long_only_n: int = 30,
) -> dict:
    """
    回测预测结果:
    - L/S: long top quintile, short bottom quintile
    - Long-only: equal-weight top N stocks

    Returns: dict with performance metrics and return series
    """
    common_idx = pred_df.index.intersection(fwd_ret_w.index)
    common_cols = pred_df.columns.intersection(fwd_ret_w.columns)

    pred = pred_df.loc[common_idx, common_cols]
    ret = fwd_ret_w.loc[common_idx, common_cols]

    ls_returns = []
    lo_returns = []
    q_returns = {i: [] for i in range(1, n_quantiles + 1)}

    for dt in common_idx:
        p = pred.loc[dt].dropna()
        r = ret.loc[dt].reindex(p.index).dropna()
        common = p.index.intersection(r.index)

        if len(common) < n_quantiles * 5:
            continue

        p = p[common]
        r = r[common]

        # Quintile sort
        ranks = p.rank(pct=True)
        for q in range(1, n_quantiles + 1):
            lo = (q - 1) / n_quantiles
            hi = q / n_quantiles
            mask = (ranks > lo) & (ranks <= hi) if q > 1 else (ranks <= hi)
            if mask.sum() > 0:
                q_returns[q].append(r[mask].mean())
            else:
                q_returns[q].append(0.0)

        # L/S: Q5 - Q1
        q1_mask = ranks <= 1 / n_quantiles
        q5_mask = ranks > (n_quantiles - 1) / n_quantiles
        if q1_mask.sum() > 0 and q5_mask.sum() > 0:
            ls_ret = r[q5_mask].mean() - r[q1_mask].mean()
            ls_returns.append(ls_ret)
        else:
            ls_returns.append(0.0)

        # Long-only top N
        top_n = p.nlargest(min(long_only_n, len(p)))
        lo_ret = r.reindex(top_n.index).mean()
        lo_returns.append(lo_ret)

    ls = pd.Series(ls_returns, index=common_idx[:len(ls_returns)])
    lo = pd.Series(lo_returns, index=common_idx[:len(lo_returns)])

    # Metrics
    def _metrics(s, freq=52):
        if len(s) == 0 or s.std() == 0:
            return {}
        cum = (1 + s).cumprod()
        dd = cum / cum.cummax() - 1
        return {
            'annual_ret': (1 + s.mean()) ** freq - 1,
            'annual_vol': s.std() * np.sqrt(freq),
            'sharpe': s.mean() / s.std() * np.sqrt(freq),
            'win_rate': (s > 0).mean(),
            'max_dd': dd.min(),
            'n_weeks': len(s),
        }

    q_ann = {}
    for q, rets in q_returns.items():
        sr = pd.Series(rets)
        q_ann[f'Q{q}'] = (1 + sr.mean()) ** 52 - 1 if len(sr) > 0 else 0

    return {
        'ls_series': ls,
        'lo_series': lo,
        'ls_metrics': _metrics(ls),
        'lo_metrics': _metrics(lo),
        'quantile_annual': q_ann,
        'cumulative_ls': (1 + ls).cumprod(),
        'cumulative_lo': (1 + lo).cumprod(),
    }


# ═══════════════════════════════════════════════════════════════
# Step 7: Live Signal
# ═══════════════════════════════════════════════════════════════

def generate_live_signal(
    all_factors: Dict[str, pd.DataFrame],
    selected: List[str],
    panel: pd.DataFrame,
    feature_cols: List[str],
    n_pca: int = 0,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    生成本周交易信号:
    - 用全部历史训练
    - 预测下周截面收益率
    - 输出 top N 股票
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    # 最新一周的因子值 — 找每个因子的最新日期
    latest_factors = {}
    for name in selected:
        if name in all_factors:
            panel_f = all_factors[name]
            if len(panel_f) > 0:
                latest_factors[name] = panel_f.iloc[-1]

    avail = list(latest_factors.keys())
    print(f'\nLive signal: {len(avail)}/{len(selected)} factors available')

    latest_df = pd.DataFrame(latest_factors)
    # Ensure all feature columns exist, fill missing with 0
    for name in feature_cols:
        if name not in latest_df.columns:
            latest_df[name] = 0.0
    latest_df = latest_df.dropna(thresh=max(len(avail) // 2, 1))
    latest_df = latest_df.fillna(0)

    if len(latest_df) < 20:
        print('Not enough symbols with factor data for live signal.')
        return pd.DataFrame()

    # ── 逐因子截面 z-score (与 build_panel_data 一致) ──
    for col in feature_cols:
        s = latest_df[col]
        median = s.median()
        mad = (s - median).abs().median()
        if mad > 0:
            cutoff = 3.0 * 1.4826 * mad
            s = s.clip(median - cutoff, median + cutoff)
        mu, sd = s.mean(), s.std()
        if sd > 0 and not np.isnan(sd):
            latest_df[col] = (s - mu) / sd
        else:
            latest_df[col] = 0.0

    # Train on all historical data
    X_train = panel[feature_cols].values
    y_train = panel['fwd_ret'].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(latest_df[feature_cols].values)

    # Predict with multiple models
    models = {
        'ridge': Ridge(alpha=1.0),
        'rf': RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=42),
    }

    predictions = {}

    # Equal-weight: 标准化后等权平均 (回测中表现最好的方法)
    eq_pred = X_test_s.mean(axis=1)
    predictions['equal_weight'] = pd.Series(eq_pred, index=latest_df.index)

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        predictions[name] = pd.Series(pred, index=latest_df.index)

    # Ensemble: average of all model predictions
    pred_df = pd.DataFrame(predictions)
    pred_df['ensemble'] = pred_df.mean(axis=1)
    # 按 equal_weight 排序 (回测最佳模型)
    pred_df = pred_df.sort_values('equal_weight', ascending=False)

    print(f'\n{"=" * 80}')
    print(f'LIVE SIGNAL — Top {top_n} stocks (predicted next-week return)')
    print(f'{"=" * 80}')
    print(f'{"Rank":<6s} {"Symbol":<10s} {"EqWeight":>10s} '
          f'{"Ridge":>10s} {"RF":>10s} {"Ensemble":>10s}')
    print('-' * 80)

    for i, (sym, row) in enumerate(pred_df.head(top_n).iterrows()):
        print(f'{i+1:<6d} {sym:<10s} {row["equal_weight"]:>+10.4f} '
              f'{row["ridge"]:>+10.4f} {row["rf"]:>+10.4f} '
              f'{row["ensemble"]:>+10.4f}')

    return pred_df


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    warnings.filterwarnings('ignore', 'Mean of empty slice')
    warnings.filterwarnings('ignore', category=FutureWarning)

    parser = argparse.ArgumentParser(description='Weekly Alpha Pipeline')
    parser.add_argument('--n-symbols', type=int, default=0)
    parser.add_argument('--n-pca', type=int, default=5,
                        help='PCA components (0=no PCA)')
    parser.add_argument('--min-train', type=int, default=8,
                        help='Min training weeks')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Top N stocks for live signal')
    parser.add_argument('--no-backtest', action='store_true',
                        help='Skip backtest, only generate live signal')
    args = parser.parse_args()

    from .data_loader import load_index_symbols, load_price_panel

    symbols = load_index_symbols()
    if args.n_symbols > 0:
        symbols = symbols[:args.n_symbols]

    # ── Step 1: Compute all factors ──
    t0 = time.time()
    all_factors = compute_all_factors(symbols)
    print(f'\nFactor computation: {time.time()-t0:.1f}s')

    # ── Step 2: Weekly alignment ──
    panels_1d = load_price_panel(symbols, '2025-01-01', '2026-12-31')
    close = panels_1d['close']
    factors_w, fwd_ret_w = prepare_weekly(all_factors, close)

    # ── Step 3: Screening ──
    selected, ic_df = screen_factors(factors_w, fwd_ret_w)

    if len(selected) < 2:
        print('Not enough factors passed screening. Aborting.')
        return

    # ── Step 4: Build panel ──
    panel = build_panel_data(factors_w, fwd_ret_w, selected)

    # ── Step 5+6: Walk-forward ML + Backtest ──
    if not args.no_backtest:
        predictions = walk_forward_predict(
            panel, selected, n_pca=args.n_pca, min_train_weeks=args.min_train)

        print('\n' + '=' * 90)
        print('BACKTEST RESULTS')
        print('=' * 90)
        header = (f'{"Model":<20s} {"L/S Ret":>8s} {"L/S Vol":>8s} '
                  f'{"Sharpe":>8s} {"WinRate":>8s} {"MaxDD":>8s} '
                  f'{"LO Ret":>8s} {"LO Sharpe":>10s}')
        print(header)
        print('-' * 90)

        for model_name, pred_df in predictions.items():
            bt = backtest_predictions(pred_df, fwd_ret_w)
            ls = bt['ls_metrics']
            lo = bt['lo_metrics']

            if not ls:
                continue

            print(f'{model_name:<20s} '
                  f'{ls["annual_ret"]:>7.1%} {ls["annual_vol"]:>8.1%} '
                  f'{ls["sharpe"]:>8.2f} {ls["win_rate"]:>8.1%} '
                  f'{ls["max_dd"]:>8.2%} '
                  f'{lo["annual_ret"]:>7.1%} {lo.get("sharpe", 0):>10.2f}')

            # Quantile returns
            q = bt['quantile_annual']
            q_str = '  '.join(f'{k}:{v:+.1%}' for k, v in q.items())
            print(f'  Quantiles: {q_str}')

    # ── Step 7: Live signal ──
    generate_live_signal(
        all_factors, selected, panel, selected,
        n_pca=args.n_pca, top_n=args.top_n)


if __name__ == '__main__':
    main()
