# Label-noise transition matrix estimation via selective classification

# Implements Algorithm 1 (threshold selection) and Algorithm 2 (cost-sensitive risk minimization)
# Both accept any scikit-learn classifier or Pipeline and return T_hat with T_hat[i, j] = P(noisy label = i | clean label = j)

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def estimate_T_threshold_selection(X, y, classifier=None, n_plus=None, n_classes=None, random_state=None, n_jobs=None, plot=False):
    """Estimate T with Algorithm 1 (threshold selection).

    Parameters
    ----------
    X : array-like of shape (n, d)
    y : array-like of shape (n,), noisy labels in {0, ..., K-1}
    classifier : scikit-learn classifier or Pipeline with predict_proba or decision_function; a clone is fit for each class (one vs. rest). Recommendation: use a classifier that works well in the dataset. Default: LogisticRegression.
    n_plus : minimum number of accepted samples N+. Default: n / 200.
    n_classes : number of classes |Y|. Default: max(y) + 1.
    random_state : seed for the split into S1 and S2.
    n_jobs : number of classes fit in parallel (joblib).
    plot : if True, plot T_jj against the number of accepted samples (as in Appendix B).

    Returns
    -------
    T_hat : ndarray of shape (|Y|, |Y|) where T_hat[i, j] = P(noisy label = i | clean label = j)
    """
    X1, y1, X2, y2, K, n_plus = _split(X, y, n_plus, n_classes, random_state)
    classifier = LogisticRegression(max_iter=1000) if classifier is None else classifier
    k = np.arange(1, len(y2) + 1)

    def column(j):
        model = clone(classifier).fit(X1, y1 == j)
        order = np.argsort(-_scores(model, X2), kind="stable")
        T_jj = np.cumsum(y2[order] == j) / k  # T_jj when accepting the top-k scores
        valid = T_jj[n_plus - 1:]
        k_hat = len(y2) - np.argmax(valid[::-1])
        return np.bincount(y2[order[:k_hat]], minlength=K) / k_hat, T_jj, k_hat

    columns = Parallel(n_jobs=n_jobs)(delayed(column)(j) for j in range(K))
    T_hat = np.column_stack([col for col, _, _ in columns])

    if plot:
        def draw(j, ax):
            _, T_jj, k_hat = columns[j]
            ax.semilogx(k, T_jj)
            ax.plot(k_hat, T_jj[k_hat - 1], "o")
            ax.set_xlabel("Number of accepted samples k")
            ax.set_ylabel(r"$\hat T_{j,j}$")
        _plot_classes(K, draw)
    return T_hat


def estimate_T_cost_sensitive(X, y, classifier=None, eps=0.05, n_plus=None, n_classes=None, random_state=None, n_jobs=None, plot=False):
    """Estimate T with Algorithm 2 (cost-sensitive risk minimization).

    Parameters
    ----------
    X : array-like of shape (n, d)
    y : array-like of shape (n,), noisy labels in {0, ..., |Y|-1}
    classifier : scikit-learn classifier or Pipeline whose fit accepts sample_weight. A clone is fit for each class and cost c. Recommendation: use a classifier that works well in the dataset. Default: LogisticRegression.
    eps : grid spacing; the costs are c = eps, 2 eps, ..., 1 - eps.
    n_plus : minimum number of accepted samples N+. Default: n / 200.
    n_classes : number of classes |Y|. Default: max(y) + 1.
    random_state : seed for the split into S1 and S2.
    n_jobs : number of (class, cost) problems fit in parallel (joblib).
    plot : if True, plot T_jj and the number of accepted samples against c (Appendix B).

    Returns
    -------
    T_hat : ndarray of shape (|Y|, |Y|) where T_hat[i, j] = P(noisy label = i | clean label = j).
    """
    X1, y1, X2, y2, K, n_plus = _split(X, y, n_plus, n_classes, random_state)
    classifier = LogisticRegression(max_iter=1000) if classifier is None else classifier
    costs = eps * np.arange(1, round(1 / eps))

    def accepted(j, c):
        not_j = y1 != j
        X_aug = np.concatenate([X1, X1[not_j]])
        y_aug = np.r_[np.ones(len(y1), dtype=int), np.zeros(not_j.sum(), dtype=int)]
        w = np.r_[np.full(len(y1), c), np.ones(not_j.sum())]
        return _fit_weighted(classifier, X_aug, y_aug, w).predict(X2) == 1

    acc = Parallel(n_jobs=n_jobs)(delayed(accepted)(j, c) for j in range(K) for c in costs)
    acc = np.array(acc).reshape(K, len(costs), len(y2))  # acc[j, q] = accepted samples of S2
    counts = acc.sum(axis=2)
    with np.errstate(invalid="ignore"):
        T_jj = (acc & (y2 == np.arange(K)[:, None, None])).sum(axis=2) / counts

    T_hat = np.empty((K, K))
    q_hat = [None] * K
    for j in range(K):
        valid = counts[j] >= n_plus
        if not valid.any():
            warnings.warn(f"Class {j}: no cost c accepts N+ samples, filling column uniformly.")
            T_hat[:, j] = 1 / K
            continue
        q = q_hat[j] = np.argmax(np.where(valid, T_jj[j], -np.inf))
        T_hat[:, j] = np.bincount(y2[acc[j, q]], minlength=K) / counts[j, q]

    if plot:
        def draw(j, ax):
            ax.plot(costs, T_jj[j], "b")
            if q_hat[j] is not None:
                ax.plot(costs[q_hat[j]], T_jj[j, q_hat[j]], "o", color="C1")
            ax.set_xlabel("c")
            ax.set_ylabel(r"$\hat T_{j,j}$", color="b")
            ax2 = ax.twinx()
            ax2.plot(costs, counts[j], "r")
            ax2.axhline(n_plus, color="r", ls=":")
            ax2.set_ylabel("Accepted samples", color="r")
        _plot_classes(K, draw)
    return T_hat


def _split(X, y, n_plus, n_classes, random_state):
    """Split the samples into two halves S1 and S2 (m1 = m2 = n/2)."""
    X, y = np.asarray(X), np.asarray(y).astype(int)
    n = len(y)
    K = y.max() + 1 if n_classes is None else n_classes
    n_plus = max(1, int(n / 200 if n_plus is None else n_plus))
    idx = np.random.default_rng(random_state).permutation(n)
    s1, s2 = idx[:n // 2], idx[n // 2:]
    if len(s2) < n_plus:
        raise ValueError(f"N_plus={n_plus} is larger than the second split ({len(s2)} samples).")
    return X[s1], y[s1], X[s2], y[s2], K, n_plus


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _fit_weighted(classifier, X, y, w):
    model = clone(classifier)
    if isinstance(model, Pipeline):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def _plot_classes(K, draw):
    import matplotlib.pyplot as plt

    cols = min(K, 5)
    rows = -(-K // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.8 * rows), squeeze=False)
    for j, ax in enumerate(axes.flat):
        if j < K:
            draw(j, ax)
            ax.set_title(f"Class {j}")
        else:
            ax.axis("off")
    fig.tight_layout()
    plt.show()