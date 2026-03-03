"""
Scorer 单元测试

使用合成数据验证等权打分和 ML walk-forward 打分.
"""
import numpy as np
import pandas as pd
import pytest

from strategies.cross_section.scorer import (
    score_equal_weight,
    score_ml_walk_forward,
    score_latest_ml,
    _build_feature_matrix,
    _fit_model,
    available_models,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

def _make_dates(n=100):
    return pd.bdate_range('2024-01-02', periods=n, freq='B')


def _make_symbols(n=30):
    return [f'S{i}' for i in range(1, n + 1)]


def _make_factor(dates, symbols, seed=42):
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal((len(dates), len(symbols)))
    mask = rng.random(vals.shape) < 0.03
    vals[mask] = np.nan
    return pd.DataFrame(vals, index=dates, columns=symbols)


def _make_close(dates, symbols, seed=123):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, (len(dates), len(symbols)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=symbols)


def _make_fwd_ret(dates, symbols, seed=789):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.05, (len(dates), len(symbols))),
        index=dates, columns=symbols,
    )


# ═══════════════════════════════════════════════════════════════
# Equal Weight Scoring
# ═══════════════════════════════════════════════════════════════

def test_equal_weight_basic():
    """基本等权打分: 输出形状正确, 每行截面均值 ~0."""
    dates = _make_dates(50)
    symbols = _make_symbols(30)
    factors = {
        'f1': _make_factor(dates, symbols, seed=1),
        'f2': _make_factor(dates, symbols, seed=2),
    }
    directions = {'f1': +1, 'f2': -1}
    score = score_equal_weight(factors, directions)

    assert score.shape[0] == len(dates)
    assert score.shape[1] == len(symbols)
    # z-score 截面均值应接近 0
    cs_mean = score.mean(axis=1).abs().mean()
    assert cs_mean < 0.3


def test_equal_weight_direction_flip():
    """验证方向翻转: 同一因子 direction=+1 和 -1 的分数相反."""
    dates = _make_dates(30)
    symbols = _make_symbols(20)
    factor = _make_factor(dates, symbols, seed=42)

    score_pos = score_equal_weight({'f': factor}, {'f': +1})
    score_neg = score_equal_weight({'f': factor}, {'f': -1})

    # 应该近似相反
    corr = score_pos.iloc[0].corr(score_neg.iloc[0])
    assert corr < -0.95


def test_equal_weight_single_factor():
    """单因子等权 = z-scored factor × direction."""
    dates = _make_dates(30)
    symbols = _make_symbols(20)
    factor = _make_factor(dates, symbols, seed=42)

    score = score_equal_weight({'f': factor}, {'f': +1})
    assert score.shape == factor.shape


def test_equal_weight_empty():
    """空因子返回空 DataFrame."""
    result = score_equal_weight({}, {})
    assert result.empty


def test_equal_weight_missing_values():
    """部分因子有 NaN 时, 缺失值视为 0, 其余因子正常参与打分."""
    dates = _make_dates(30)
    symbols = _make_symbols(20)
    f1 = _make_factor(dates, symbols, seed=1)
    f2 = _make_factor(dates[:20], symbols, seed=2)  # 只有前 20 天

    score = score_equal_weight(
        {'f1': f1, 'f2': f2}, {'f1': +1, 'f2': +1})

    # 所有日期都应有分数
    assert score.shape[0] == len(dates)
    # 后 10 天只有 f1, 不应全 NaN
    assert score.iloc[-1].notna().sum() > 0


# ═══════════════════════════════════════════════════════════════
# Feature Matrix Construction
# ═══════════════════════════════════════════════════════════════

def test_build_feature_matrix_shape():
    """特征矩阵形状: (total_obs) × [features + fwd_ret + _date]."""
    dates = _make_dates(50)
    symbols = _make_symbols(20)
    factors = {
        'f1': _make_factor(dates, symbols, seed=1),
        'f2': _make_factor(dates, symbols, seed=2),
    }
    fwd_ret = _make_fwd_ret(dates, symbols)

    panel, feature_cols = _build_feature_matrix(factors, fwd_ret)
    assert len(feature_cols) == 2
    assert 'f1' in feature_cols
    assert 'f2' in feature_cols
    assert 'fwd_ret' in panel.columns
    assert '_date' in panel.columns
    assert len(panel) > 0


def test_build_feature_matrix_no_nan_features():
    """特征列不应有 NaN (已填 0)."""
    dates = _make_dates(50)
    symbols = _make_symbols(20)
    factors = {'f1': _make_factor(dates, symbols, seed=1)}
    fwd_ret = _make_fwd_ret(dates, symbols)

    panel, feature_cols = _build_feature_matrix(factors, fwd_ret)
    assert panel[feature_cols].isna().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════
# ML Model Factory
# ═══════════════════════════════════════════════════════════════

def test_fit_model_ridge():
    """Ridge 模型应可创建."""
    models = available_models()
    if 'ridge' not in models:
        pytest.skip("sklearn not installed")
    model = _fit_model('ridge')
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


def test_fit_model_invalid():
    """无效模型名应抛出 ValueError."""
    with pytest.raises(ValueError, match="not available"):
        _fit_model('nonexistent_model')


# ═══════════════════════════════════════════════════════════════
# ML Walk-Forward Scoring
# ═══════════════════════════════════════════════════════════════

def test_ml_walk_forward_basic():
    """Walk-forward 应返回非空预测面板."""
    models = available_models()
    if 'ridge' not in models:
        pytest.skip("sklearn not installed")

    dates = _make_dates(200)
    symbols = _make_symbols(30)
    factors = {
        'f1': _make_factor(dates, symbols, seed=1),
        'f2': _make_factor(dates, symbols, seed=2),
    }
    fwd_ret = _make_fwd_ret(dates, symbols)

    result = score_ml_walk_forward(
        factors, fwd_ret,
        method='ridge',
        min_train_periods=50,
        purge_periods=1,
    )
    assert not result.empty
    assert result.shape[0] > 0
    assert result.shape[1] == len(symbols)


def test_ml_walk_forward_predictive():
    """当因子 = fwd_ret + noise 时, ML 预测应有正 IC."""
    models = available_models()
    if 'ridge' not in models:
        pytest.skip("sklearn not installed")

    dates = _make_dates(200)
    symbols = _make_symbols(30)
    rng = np.random.default_rng(42)
    fwd_ret = pd.DataFrame(
        rng.normal(0, 0.05, (len(dates), len(symbols))),
        index=dates, columns=symbols,
    )
    # Factor = fwd_ret + small noise → 应有预测力
    factor = fwd_ret + rng.normal(0, 0.01, fwd_ret.shape)
    factors = {'signal': pd.DataFrame(factor, index=dates, columns=symbols)}

    result = score_ml_walk_forward(
        factors, fwd_ret,
        method='ridge',
        min_train_periods=50,
        purge_periods=1,
    )
    assert not result.empty

    # 计算 IC
    common_dates = result.index.intersection(fwd_ret.index)
    ics = []
    for dt in common_dates:
        pred = result.loc[dt].dropna()
        actual = fwd_ret.loc[dt].reindex(pred.index).dropna()
        common = pred.index.intersection(actual.index)
        if len(common) >= 10:
            ic = pred[common].corr(actual[common])
            ics.append(ic)
    mean_ic = np.mean(ics) if ics else 0
    assert mean_ic > 0.1  # 强信号 → 高 IC


def test_ml_walk_forward_too_short():
    """数据太短 → 返回空."""
    models = available_models()
    if 'ridge' not in models:
        pytest.skip("sklearn not installed")

    dates = _make_dates(20)
    symbols = _make_symbols(10)
    factors = {'f1': _make_factor(dates, symbols)}
    fwd_ret = _make_fwd_ret(dates, symbols)

    result = score_ml_walk_forward(
        factors, fwd_ret,
        method='ridge',
        min_train_periods=50,
    )
    assert result.empty


# ═══════════════════════════════════════════════════════════════
# Latest ML Scoring
# ═══════════════════════════════════════════════════════════════

def test_score_latest_ml():
    """score_latest_ml 应返回最新截面预测."""
    models = available_models()
    if 'ridge' not in models:
        pytest.skip("sklearn not installed")

    dates = _make_dates(100)
    symbols = _make_symbols(30)
    factors = {
        'f1': _make_factor(dates, symbols, seed=1),
        'f2': _make_factor(dates, symbols, seed=2),
    }
    fwd_ret = _make_fwd_ret(dates, symbols)

    result = score_latest_ml(factors, fwd_ret, method='ridge')
    assert len(result) > 0
    assert result.dtype == np.float64 or result.dtype == np.float32
