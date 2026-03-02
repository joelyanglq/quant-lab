"""
日频 ML 选股: 17 因子 × pooled 样本 × 重叠 5 日收益

核心设计:
    - 每个交易日 t: 特征 = 17 因子截面 z-score, 目标 = t+1~t+5 累计收益
    - pooled 训练: 所有 (symbol, date) 拉成一张长表
    - walk-forward: 用 t 之前的数据训练, 预测 t 期截面
    - purge gap: 训练集排除最近 5 天 (避免重叠收益泄漏)
    - 模型: Ridge / ElasticNet / LightGBM
    - 评估: 逐日截面 Rank IC + 分档年化收益

用法:
    cd OneBacktest
    python -m factor_research.daily_ml_predict
    python -m factor_research.daily_ml_predict --min-train-days 30
"""
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from .weekly_pipeline import compute_all_factors
from .data_loader import load_index_symbols, load_price_panel

warnings.filterwarnings('ignore')

# ── 17 筛选因子 + IC 方向 ──
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

HOLDING_DAYS = 5
PURGE_DAYS = 5


# ═══════════════════════════════════════════════════════════════
# 1. 构建日频面板
# ═══════════════════════════════════════════════════════════════

def build_daily_panel(
    all_factors: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    selected: List[str],
    directions: Dict[str, int],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建 pooled 日频面板.
    周频因子 (如 vol_battle_pos) 先 ffill 到日频.

    Returns:
        panel: DataFrame(index=(date, symbol), columns=[features..., fwd_ret_5d])
        feature_cols: List[str] of feature column names
    """
    # 5 日前瞻收益
    fwd_ret = close.shift(-HOLDING_DAYS) / close - 1

    available = [f for f in selected if f in all_factors]

    # 用收盘价日期做基准
    daily_dates = close.index
    base_cols = fwd_ret.columns
    for name in available:
        base_cols = base_cols.intersection(all_factors[name].columns)

    # 将所有因子对齐到日频:
    # 因子的日期可能与 close 不完全匹配 (周频因子、不同 tz 等)
    # 先合并到 close 日期上, 用 ffill 填充
    factors_daily = {}
    for name in available:
        raw = all_factors[name]
        # 合并因子日期和 close 日期, sort, ffill, 再取 close 日期
        combined_idx = raw.index.union(daily_dates).sort_values()
        merged = raw.reindex(index=combined_idx, columns=base_cols).ffill()
        factors_daily[name] = merged.reindex(index=daily_dates)

    # 公共日期: fwd_ret 有效 + 所有因子在该日至少有 50 个 symbol 有值
    fwd_valid = fwd_ret.dropna(how='all').index
    base_dates = []
    for dt in fwd_valid:
        n_valid = min(
            factors_daily[name].loc[dt].notna().sum()
            for name in available
            if dt in factors_daily[name].index
        ) if all(dt in factors_daily[name].index for name in available) else 0
        if n_valid >= 50:
            base_dates.append(dt)
    base_dates = pd.DatetimeIndex(base_dates)

    print(f'  Common dates: {len(base_dates)}, symbols: {len(base_cols)}')
    print(f'  Available factors: {len(available)}/{len(selected)}')

    # 逐日截面标准化 + IC 方向调整
    rows = []
    for dt in base_dates:
        # 收集该日截面
        data = {}
        for name in available:
            data[name] = factors_daily[name].loc[dt, base_cols]
        df = pd.DataFrame(data)

        # 加入前瞻收益
        df['fwd_ret_5d'] = fwd_ret.loc[dt, base_cols]

        # 剔除任何因子或收益缺失的 symbol
        df = df.dropna()
        if len(df) < 30:
            continue

        # 截面 MAD winsorize + z-score + IC 方向
        for col in available:
            direction = directions.get(col, +1)
            s = df[col]
            median = s.median()
            mad = (s - median).abs().median()
            if mad > 0:
                cutoff = 3.0 * 1.4826 * mad
                s = s.clip(median - cutoff, median + cutoff)
            mu, sd = s.mean(), s.std()
            if sd > 0:
                df[col] = direction * (s - mu) / sd
            else:
                df[col] = 0.0

        df['date'] = dt
        df['symbol'] = df.index
        rows.append(df)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.set_index(['date', 'symbol']).sort_index()

    n_dates = panel.index.get_level_values('date').nunique()
    n_syms = panel.index.get_level_values('symbol').nunique()
    print(f'  Panel: {len(panel):,} obs, {n_dates} days, {n_syms} symbols')

    return panel, available


# ═══════════════════════════════════════════════════════════════
# 2. Walk-Forward 预测 (日频, 带 purge)
# ═══════════════════════════════════════════════════════════════

def walk_forward_daily(
    panel: pd.DataFrame,
    feature_cols: List[str],
    min_train_days: int = 30,
    purge_days: int = PURGE_DAYS,
) -> Dict[str, pd.DataFrame]:
    """
    日频 walk-forward:
    - 预测日 t: 训练集 = 所有 <= t - purge_days 的数据
    - purge gap 避免重叠 fwd_ret 泄漏

    Returns: {model_name: DataFrame(date × symbol → predicted)}
    """
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import LinearSVR

    lgb_cls = None
    try:
        from lightgbm import LGBMRegressor
        lgb_cls = LGBMRegressor
    except ImportError:
        pass

    xgb_cls = None
    try:
        from xgboost import XGBRegressor
        xgb_cls = XGBRegressor
    except ImportError:
        pass

    dates = sorted(panel.index.get_level_values('date').unique())
    n_dates = len(dates)

    models_config = {
        'ridge': lambda: Ridge(alpha=1.0),
        'rf': lambda: RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=50,
            max_features='sqrt', n_jobs=-1, random_state=42),
        'gbr': lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=50, random_state=42),
        'svr': lambda: LinearSVR(C=0.1, max_iter=5000),
    }
    if lgb_cls:
        models_config['lgb'] = lambda: lgb_cls(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=50, verbose=-1, random_state=42)
    if xgb_cls:
        models_config['xgb'] = lambda: xgb_cls(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=50, verbosity=0, random_state=42)

    predictions = {name: {} for name in list(models_config.keys()) + ['equal_weight']}

    print(f'\n  Walk-forward: {n_dates} days, min_train={min_train_days}, '
          f'purge={purge_days}d')
    print(f'  Models: {list(predictions.keys())}')

    for t in range(min_train_days + purge_days, n_dates):
        test_date = dates[t]
        # 训练: 所有 <= dates[t - purge_days - 1]
        train_cutoff = dates[t - purge_days]
        train_dates = [d for d in dates if d < train_cutoff]

        if len(train_dates) < min_train_days:
            continue

        train_mask = panel.index.get_level_values('date').isin(train_dates)
        test_mask = panel.index.get_level_values('date') == test_date

        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, 'fwd_ret_5d'].values
        X_test = panel.loc[test_mask, feature_cols].values
        test_symbols = panel.loc[test_mask].index.get_level_values('symbol')

        if len(X_test) < 20 or len(X_train) < 200:
            continue

        # Standardize (pooled training set)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Equal-weight (no ML)
        eq_pred = X_test_s.mean(axis=1)
        predictions['equal_weight'][test_date] = pd.Series(
            eq_pred, index=test_symbols)

        # ML models
        for model_name, model_factory in models_config.items():
            m = model_factory()
            m.fit(X_train_s, y_train)
            pred = m.predict(X_test_s)
            predictions[model_name][test_date] = pd.Series(
                pred, index=test_symbols)

        done = t - min_train_days - purge_days + 1
        total = n_dates - min_train_days - purge_days
        if done % 10 == 0 or done == total:
            print(f'    day {done}/{total}: '
                  f'train={len(X_train):,}  test={len(X_test):,}',
                  flush=True)

    result = {}
    for model_name, preds in predictions.items():
        if preds:
            result[model_name] = pd.DataFrame(preds).T.sort_index()

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 评估: 逐日截面 Rank IC + 分档收益
# ═══════════════════════════════════════════════════════════════

def evaluate_daily(
    pred_dict: Dict[str, pd.DataFrame],
    close: pd.DataFrame,
    n_quantiles: int = 5,
    long_only_n: int = 30,
) -> Dict[str, dict]:
    """评估每个模型的日频预测 → 周度化收益指标"""
    fwd_ret = close.shift(-HOLDING_DAYS) / close - 1
    results = {}

    for model_name, pred_df in pred_dict.items():
        common_idx = pred_df.index.intersection(fwd_ret.dropna(how='all').index)
        common_cols = pred_df.columns.intersection(fwd_ret.columns)

        ics = []
        ls_returns = []
        lo_returns = []
        q_returns = {q: [] for q in range(1, n_quantiles + 1)}

        for dt in common_idx:
            p = pred_df.loc[dt, common_cols].dropna()
            r = fwd_ret.loc[dt].reindex(p.index).dropna()
            common = p.index.intersection(r.index)
            if len(common) < n_quantiles * 10:
                continue

            p, r = p[common], r[common]

            # Rank IC
            ic = p.corr(r, method='spearman')
            if not np.isnan(ic):
                ics.append(ic)

            # Quintile sort
            ranks = p.rank(pct=True)
            for q in range(1, n_quantiles + 1):
                lo_q = (q - 1) / n_quantiles
                hi_q = q / n_quantiles
                mask = (ranks > lo_q) & (ranks <= hi_q) if q > 1 else (ranks <= hi_q)
                if mask.sum() > 0:
                    q_returns[q].append(r[mask].mean())

            # L/S
            q1_mask = ranks <= 1 / n_quantiles
            q5_mask = ranks > (n_quantiles - 1) / n_quantiles
            if q1_mask.sum() > 0 and q5_mask.sum() > 0:
                ls_returns.append(r[q5_mask].mean() - r[q1_mask].mean())

            # Long-only top N
            top_n = p.nlargest(min(long_only_n, len(p)))
            lo_returns.append(r.reindex(top_n.index).mean())

        # 汇总 (注意: 日频重叠收益, annualize 用 252/5)
        freq = 252 / HOLDING_DAYS  # ~50.4 "periods" per year

        def _metrics(s_list):
            if not s_list:
                return {}
            s = pd.Series(s_list)
            cum = (1 + s).cumprod()
            dd = cum / cum.cummax() - 1
            return {
                'mean_ret': s.mean(),
                'annual_ret': (1 + s.mean()) ** freq - 1,
                'annual_vol': s.std() * np.sqrt(freq),
                'sharpe': s.mean() / s.std() * np.sqrt(freq) if s.std() > 0 else 0,
                'win_rate': (s > 0).mean(),
                'max_dd': dd.min(),
                'n_periods': len(s),
            }

        ic_series = pd.Series(ics) if ics else pd.Series(dtype=float)

        results[model_name] = {
            'ic_mean': ic_series.mean() if len(ic_series) > 0 else 0,
            'ic_std': ic_series.std() if len(ic_series) > 0 else 0,
            'icir': (ic_series.mean() / ic_series.std()
                     if len(ic_series) > 0 and ic_series.std() > 0 else 0),
            'n_ic': len(ic_series),
            'ls': _metrics(ls_returns),
            'lo': _metrics(lo_returns),
            'quantile_annual': {
                f'Q{q}': ((1 + pd.Series(rets).mean()) ** freq - 1
                          if rets else 0)
                for q, rets in q_returns.items()
            },
        }

    return results


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Daily ML prediction (17 factors, overlapping 5d returns)')
    parser.add_argument('--min-train-days', type=int, default=30,
                        help='Minimum training days before first prediction')
    parser.add_argument('--purge-days', type=int, default=PURGE_DAYS,
                        help='Purge gap days for overlapping return leakage')
    args = parser.parse_args()

    t_start = time.time()

    print('=' * 70)
    print('DAILY ML PREDICTION (17 factors, pooled, overlapping 5d returns)')
    print('=' * 70)

    # 1. 因子
    print('\n[1/4] Computing factors ...')
    symbols = load_index_symbols()
    all_factors = compute_all_factors(symbols)

    selected = [f for f in SELECTED_17 if f in all_factors]
    print(f'\nAvailable: {len(selected)}/{len(SELECTED_17)} factors')
    missing = set(SELECTED_17) - set(selected)
    if missing:
        print(f'Missing: {missing}')

    # 2. 日频面板
    print('\n[2/4] Building daily panel ...')
    panels_1d = load_price_panel(symbols, '2024-01-01', '2026-12-31')
    close = panels_1d['close']
    panel, feature_cols = build_daily_panel(
        all_factors, close, selected, SELECTED_17)

    # 3. Walk-forward
    print('\n[3/4] Walk-forward prediction ...')
    pred_dict = walk_forward_daily(
        panel, feature_cols,
        min_train_days=args.min_train_days,
        purge_days=args.purge_days)

    # 4. 评估
    print('\n[4/4] Evaluating ...')
    results = evaluate_daily(pred_dict, close)

    # ── 输出 ──
    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)

    for model_name, res in results.items():
        print(f'\n--- {model_name} ---')
        print(f'  Rank IC: mean={res["ic_mean"]:+.4f}  '
              f'std={res["ic_std"]:.4f}  '
              f'ICIR={res["icir"]:.2f}  '
              f'N={res["n_ic"]}d')

        ls = res['ls']
        if ls:
            print(f'  L/S:  AnnRet={ls["annual_ret"]:+.1%}  '
                  f'Sharpe={ls["sharpe"]:.2f}  '
                  f'WinRate={ls["win_rate"]:.1%}  '
                  f'MaxDD={ls["max_dd"]:.1%}  '
                  f'N={ls["n_periods"]}')

        lo = res['lo']
        if lo:
            print(f'  LO30: AnnRet={lo["annual_ret"]:+.1%}  '
                  f'Sharpe={lo["sharpe"]:.2f}  '
                  f'WinRate={lo["win_rate"]:.1%}  '
                  f'MaxDD={lo["max_dd"]:.1%}')

        qa = res['quantile_annual']
        q_str = '  '.join(f'{k}={v:+.1%}' for k, v in sorted(qa.items()))
        print(f'  Quantiles: {q_str}')

    # Feature importance (tree models)
    for imp_model in ['rf', 'gbr', 'lgb', 'xgb']:
        if imp_model not in pred_dict:
            continue
        print(f'\n--- {imp_model.upper()} Feature Importance ---')
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            X_all = panel[feature_cols].values
            y_all = panel['fwd_ret_5d'].values
            if imp_model == 'rf':
                m = RandomForestRegressor(
                    n_estimators=200, max_depth=5, min_samples_leaf=50,
                    max_features='sqrt', n_jobs=-1, random_state=42)
            elif imp_model == 'gbr':
                m = GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.03,
                    subsample=0.7, min_samples_leaf=50, random_state=42)
            else:
                continue  # lgb/xgb handle separately
            m.fit(X_all, y_all)
            imp = pd.Series(m.feature_importances_, index=feature_cols)
            imp = imp.sort_values(ascending=False)
            for name, val in imp.items():
                print(f'  {name:<28s} {val:>8.4f}')
        except Exception as e:
            print(f'  Error: {e}')

    print(f'\nTotal time: {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
