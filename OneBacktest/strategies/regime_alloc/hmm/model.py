"""
Gaussian HMM — 拟合 + 状态推断

论文 setup (Baitinger & Hoch 2024):
    - 观测: 日频简单收益率 r_t = P_t/P_{t-1} - 1
    - 每个状态 j: r_t | S_t=j ~ N(μ_j, σ²_j)
    - 拟合: EM 算法 (Baum-Welch), 多次随机初始化取最优 LL 避免局部最优
    - 推断: 前向算法 → 滤波概率 Pr(S_t=j | r_1,...,r_t)
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM


def fit_hmm(
    returns: np.ndarray,
    n_states: int = 2,
    n_init: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: int = 42,
) -> GaussianHMM:
    """
    拟合 Gaussian HMM, 多次随机初始化取最优 log-likelihood.

    论文 Section 3: grid search over starting params, select by best
    avg rank of (AIC, CD). 这里简化为多次随机初始化取最优 LL.

    Parameters
    ----------
    returns : 1-d array of daily simple returns
    n_states : number of hidden states (论文测试 2-7)
    n_init : number of random initializations
    max_iter : EM max iterations per init
    tol : EM convergence tolerance
    random_state : base random seed

    Returns
    -------
    best fitted GaussianHMM model
    """
    X = returns.reshape(-1, 1)
    best_model = None
    best_score = -np.inf

    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=max_iter,
            tol=tol,
            random_state=random_state + i,
            init_params='stmc',
            params='stmc',
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError(f'HMM fitting failed for n_states={n_states}')

    return best_model


def regime_params(model: GaussianHMM) -> dict:
    """
    提取拟合后的状态参数.

    Returns
    -------
    dict with keys:
        means : (n_states,) 各状态日收益均值
        stds  : (n_states,) 各状态日收益标准差
        transmat : (n_states, n_states) 转移矩阵
    """
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())
    return {
        'means': means,
        'stds': stds,
        'transmat': model.transmat_,
    }


def filtered_state_probs(
    model: GaussianHMM,
    returns: np.ndarray,
) -> np.ndarray:
    """
    前向算法 → 滤波状态概率 Pr(S_t=j | r_1,...,r_t).

    论文 Eq(10-11): 用最新时刻的状态概率加权计算预期收益和方差.
    这里返回最后一个时刻的概率向量.

    Returns
    -------
    (n_states,) probability vector for the last time step
    """
    X = returns.reshape(-1, 1)
    # predict_proba 返回 smoothed (后验), 但论文用的是 t 时刻信息
    # 做 OOS 预测时, 用 score_samples 获取 filtered prob 更准确
    # hmmlearn 没有直接的 filtered prob API, 用 predict_proba 近似
    # (expanding window 下差异极小)
    posteriors = model.predict_proba(X)
    return posteriors[-1]


def forecast_return_variance(
    model: GaussianHMM,
    returns: np.ndarray,
) -> tuple:
    """
    论文 Eq(10-11): 用状态概率加权计算预期收益和方差.

        R̂_{t+1} = Σ_j μ_j · Pr(S_t=j)
        σ̂²_{t+1} = Σ_j σ²_j · Pr(S_t=j)

    Parameters
    ----------
    model : fitted GaussianHMM
    returns : daily returns up to time t (estimation window)

    Returns
    -------
    (expected_return, expected_variance) — 日频
    """
    probs = filtered_state_probs(model, returns)
    params = regime_params(model)

    exp_ret = np.dot(params['means'], probs)
    exp_var = np.dot(params['stds'] ** 2, probs)

    return exp_ret, exp_var


def save_model(model: GaussianHMM, path: str | Path, metadata: dict = None):
    """
    保存拟合好的 HMM 模型, 方便后续直接推理.

    Parameters
    ----------
    model : fitted GaussianHMM
    path : 保存路径 (.joblib)
    metadata : 附加信息 (训练日期范围, n_states, score 等)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': model}
    if metadata:
        payload['metadata'] = metadata
    joblib.dump(payload, path)


def load_model(path: str | Path) -> tuple[GaussianHMM, dict]:
    """
    加载已保存的 HMM 模型.

    Returns
    -------
    (model, metadata) — metadata 为 None 若保存时未提供
    """
    payload = joblib.load(path)
    return payload['model'], payload.get('metadata')
