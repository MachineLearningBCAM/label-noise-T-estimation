#Estimate the transition matrix of synthetic 0.2-uniform label noise on the digits dataset
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from transition_matrix import estimate_T_cost_sensitive, estimate_T_threshold_selection

X, y = load_digits(return_X_y=True)
K, p = 10, 0.2

# True transition matrix T[i, j] = P(noisy = i | clean = j) and noisy labels
T_true = (1 - p) * np.eye(K) + p / (K - 1) * (1 - np.eye(K))
rng = np.random.default_rng(0)
y_noisy = np.array([rng.choice(K, p=T_true[:, label]) for label in y])

classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
n_plus = 20  # the default n/200 is only about 8 on this small dataset

T_thr = estimate_T_threshold_selection(X, y_noisy, classifier, n_plus=n_plus, random_state=0)
T_cost = estimate_T_cost_sensitive(X, y_noisy, classifier, n_plus=n_plus, random_state=0, n_jobs=-1)

np.set_printoptions(precision=3, suppress=True, linewidth=120)
print(f"True transition matrix:\n{T_true}\n")
print(f"Algorithm 1 (threshold selection), MAE {np.abs(T_thr - T_true).mean():.4f}:\n{T_thr}\n")
print(f"Algorithm 2 (cost-sensitive), MAE {np.abs(T_cost - T_true).mean():.4f}:\n{T_cost}")

# Appendix B plots (off by default)
# estimate_T_threshold_selection(X, y_noisy, classifier, n_plus=n_plus, random_state=0, plot=True)
# estimate_T_cost_sensitive(X, y_noisy, classifier, n_plus=n_plus, random_state=0, n_jobs=-1, plot=True)
