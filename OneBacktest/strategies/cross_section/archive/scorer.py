"""
统一打分接口

支持两种打分方式:
    1. 等权合成: z-score × IC direction → 等权均值
    2. ML walk-forward: 滚动窗口训练 → 截面预测

Usage:
    from strategies.cross_section.scorer import score_equal_weight, score_ml_walk_forward
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ranking import cross_sectional_zscore

# ═══════════════════════════════════════════════════════════════
# ML 模型工厂
# ═══════════════════════════════════════════════════════════════

_AVAILABLE_MODELS = {}

try:
    from sklearn.linear_model import Ridge
    _AVAILABLE_MODELS['ridge'] = lambda: Ridge(alpha=1.0)
except ImportError:
    pass

try:
    from sklearn.ensemble import RandomForestRegressor
    _AVAILABLE_MODELS['rf'] = lambda: RandomForestRegressor(
        n_estimators=200, max_depth=5, n_jobs=-1, random_state=42)
except ImportError:
    pass

try:
    from xgboost import XGBRegressor
    _AVAILABLE_MODELS['xgb'] = lambda: XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        n_jobs=-1, random_state=42, verbosity=0)
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    _AVAILABLE_MODELS['lgb'] = lambda: LGBMRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        n_jobs=-1, random_state=42, verbose=-1)
except ImportError:
    pass


def available_models() -> List[str]:
    """返回当前环境可用的 ML 模型列表."""
    return list(_AVAILABLE_MODELS.keys())


def _fit_model(method: str):
    """创建 sklearn-compatible 模型实例."""
    if method not in _AVAILABLE_MODELS:
        installed = available_models()
        raise ValueError(
            f"Model '{method}' not available. "
            f"Installed: {installed or ['none (install scikit-learn)']}")
    return _AVAILABLE_MODELS[method]()


# ═══════════════════════════════════════════════════════════════
# 等权合成打分
# ═══════════════════════════════════════════════════════════════

def score_equal_weight(
    factors: Dict[str, pd.DataFrame],
    directions: Dict[str, int],
) -> pd.DataFrame:
    """
    等权 z-score 合成打分.

    对每个因子:
        1. cross_sectional_zscore() (MAD winsorize + z-score)
        2. × direction (+1 / -1)
    合成: 等权均值, 缺失因子视为 0 (截面中性).

    Args:
        factors: {name: dates × symbols} 因子面板
        directions: {name: +1 or -1} IC 方向

    Returns:
        dates × symbols 合成分数面板
    """
    if not factors:
        return pd.DataFrame()

    z_panels = []
    for name, panel in factors.items():
        direction = directions.get(name, +1)
        z = cross_sectional_zscore(panel) * direction
        z_panels.append(z)

    # 对齐所有因子 (union of dates × symbols)
    combined = pd.concat(z_panels, axis=0)
    all_dates = combined.index.unique().sort_values()
    all_symbols = combined.columns.unique().sort_values()

    # 逐日合成
    score_data = {}
    for dt in all_dates:
        vals = []
        for z in z_panels:
            if dt in z.index:
                vals.append(z.loc[dt])
        if vals:
            stacked = pd.concat(vals, axis=1)
            # 缺失因子填 0, 取等权均值
            score_data[dt] = stacked.fillna(0).mean(axis=1)

    return pd.DataFrame(score_data).T.sort_index()


# ═══════════════════════════════════════════════════════════════
# ML Walk-Forward 打分
# ═══════════════════════════════════════════════════════════════

def _build_feature_matrix(
    factors: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    构建 (date, symbol) × [features..., fwd_ret] 面板.

    每个截面独立 z-score 标准化. NaN 填 0.

    Returns:
        (panel, feature_cols)
    """
    feature_cols = sorted(factors.keys())
    common_dates = fwd_ret.index
    for name in feature_cols:
        common_dates = common_dates.intersection(factors[name].index)
    common_dates = common_dates.sort_values()

    rows = []
    for dt in common_dates:
        ret_row = fwd_ret.loc[dt].dropna()
        if len(ret_row) < 10:
            continue

        row_data = {'fwd_ret': ret_row}
        for name in feature_cols:
            if dt in factors[name].index:
                f_row = factors[name].loc[dt]
                # 截面 z-score
                median = f_row.median()
                mad = (f_row - median).abs().median()
                if mad > 0:
                    cutoff = 3.0 * 1.4826 * mad
                    f_row = f_row.clip(median - cutoff, median + cutoff)
                mu, sd = f_row.mean(), f_row.std()
                if sd > 0:
                    f_row = (f_row - mu) / sd
                else:
                    f_row = f_row * 0
                row_data[name] = f_row
            else:
                row_data[name] = pd.Series(0.0, index=ret_row.index)

        frame = pd.DataFrame(row_data)
        frame = frame.reindex(ret_row.index)
        frame[feature_cols] = frame[feature_cols].fillna(0)
        frame = frame.dropna(subset=['fwd_ret'])
        if len(frame) < 10:
            continue
        frame['_date'] = dt
        rows.append(frame)

    if not rows:
        return pd.DataFrame(), feature_cols

    panel = pd.concat(rows)
    panel.index.name = 'symbol'
    return panel, feature_cols


def score_ml_walk_forward(
    factors: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    method: str = 'ridge',
    min_train_periods: int = 52,
    purge_periods: int = 1,
    n_pca: Optional[int] = None,
) -> pd.DataFrame:
    """
    Walk-forward ML 打分.

    对每个 rebalance 日 t:
        1. 训练集: 日期 < (t - purge_periods) 的全部截面数据
        2. fit(X_train, y_train)
        3. predict(X_t) → 当期截面预测分

    Args:
        factors: {name: dates × symbols} (已 resample 到目标频率)
        fwd_ret: dates × symbols 前瞻收益 (同频率)
        method: 'ridge' | 'rf' | 'xgb' | 'lgb'
        min_train_periods: 最少训练期数
        purge_periods: 训练集与预测日之间的间隔期数 (避免前视)
        n_pca: PCA 降维维度 (None = 不降维)

    Returns:
        rebalance_dates × symbols 预测分数面板
    """
    panel, feature_cols = _build_feature_matrix(factors, fwd_ret)
    if panel.empty or not feature_cols:
        return pd.DataFrame()

    dates = panel['_date'].unique()
    dates = np.sort(dates)

    pca_transformer = None
    if n_pca is not None and n_pca > 0:
        try:
            from sklearn.decomposition import PCA
            pca_transformer = PCA(n_components=min(n_pca, len(feature_cols)))
        except ImportError:
            warnings.warn("sklearn not installed, skipping PCA")
            n_pca = None

    predictions = {}
    for i, dt in enumerate(dates):
        if i < min_train_periods:
            continue

        # 训练集: 日期 < (dt - purge)
        train_end_idx = i - purge_periods
        if train_end_idx < min_train_periods:
            continue
        train_dates = dates[:train_end_idx]
        train_mask = panel['_date'].isin(train_dates)
        train_data = panel.loc[train_mask]

        X_train = train_data[feature_cols].values
        y_train = train_data['fwd_ret'].values

        if len(X_train) < 30:
            continue

        # 当期测试集
        test_mask = panel['_date'] == dt
        test_data = panel.loc[test_mask]
        X_test = test_data[feature_cols].values

        if len(X_test) < 5:
            continue

        # PCA
        if pca_transformer is not None:
            pca = type(pca_transformer)(
                n_components=pca_transformer.n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # 训练 + 预测
        try:
            model = _fit_model(method)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions[dt] = pd.Series(
                preds, index=test_data.index, name=dt)
        except Exception as e:
            warnings.warn(f"ML scoring failed at {dt}: {e}")
            continue

    if not predictions:
        return pd.DataFrame()

    result = pd.DataFrame(predictions).T
    result.index = pd.DatetimeIndex(result.index)
    return result.sort_index()


def score_latest_ml(
    factors: Dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    method: str = 'ridge',
    n_pca: Optional[int] = None,
) -> pd.Series:
    """
    用全部历史训练, 预测最新截面.

    Args:
        factors: {name: dates × symbols}
        fwd_ret: dates × symbols 前瞻收益 (用于训练, 最后一行可以是 NaN)
        method: ML 模型

    Returns:
        symbol → predicted score (Series)
    """
    panel, feature_cols = _build_feature_matrix(factors, fwd_ret)
    if panel.empty or not feature_cols:
        return pd.Series(dtype=float)

    dates = np.sort(panel['_date'].unique())
    if len(dates) < 2:
        return pd.Series(dtype=float)

    latest_date = dates[-1]

    # 训练集: 除最后一期外全部
    train_mask = panel['_date'] != latest_date
    train_data = panel.loc[train_mask].dropna(subset=['fwd_ret'])
    X_train = train_data[feature_cols].values
    y_train = train_data['fwd_ret'].values

    if len(X_train) < 30:
        return pd.Series(dtype=float)

    # 预测最新截面 (可能 fwd_ret 是 NaN)
    test_mask = panel['_date'] == latest_date
    test_data = panel.loc[test_mask]
    X_test = test_data[feature_cols].values

    pca_transformer = None
    if n_pca is not None and n_pca > 0:
        try:
            from sklearn.decomposition import PCA
            pca_transformer = PCA(
                n_components=min(n_pca, len(feature_cols)))
            X_train = pca_transformer.fit_transform(X_train)
            X_test = pca_transformer.transform(X_test)
        except ImportError:
            pass

    model = _fit_model(method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return pd.Series(preds, index=test_data.index, name='score')
