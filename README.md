# Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification

**Estimate the label-noise transition matrix with any binary classifier!**

This repository provides Python and MATLAB implementations of *Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification* (NeurIPS 2026)

The transition matrix $\mathbf{T}$ has entries $T[i, j] = \mathbf{P}(\widetilde Y = i | Y = j)$. 
The provided algorithms estimate each column $j$ in parallel. They split the noisy samples into two halves, learn on the first half selection functions that accept instances of noisy class $j$, and choose on the second half the one that minimizes the false discovery rate while accepting at least $N_+$ samples. Column $j$ of $\mathbf{T}$ is then the distribution of the noisy labels among the accepted samples.

- Algorithm 1 (threshold selection): learns a binary score function for class $j$ versus the rest and accepts the instances whose score is above a threshold.
- Algorithm 2 (cost-sensitive risk minimization): learns binary classifiers ($j$ vs rest) that penalize accepting instances of other classes and assign a cost $c$ to rejecting, for every $c$ on a grid of spacing $\varepsilon$, and selects the best cost.

## Python implementation

The folder `python/` contains:

- `transition_matrix.py`: the functions `estimate_T_threshold_selection` (Algorithm 1) and `estimate_T_cost_sensitive` (Algorithm 2)
- `example.py`: runs both algorithms on the digits dataset with synthetic label noise and prints the true and estimated matrices.

### Installation

Copy `transition_matrix.py` into your project. 

The requirements are `numpy` and `scikit-learn`, plus `matplotlib` if you use `plot=True`.

### Usage

Both functions accept any scikit-learn classifier or `Pipeline`:

```python
from sklearn.ensemble import RandomForestClassifier
from transition_matrix import estimate_T_threshold_selection, estimate_T_cost_sensitive

# X: features of shape (n, d); y_noisy: noisy labels in {0, ..., K-1}
T1 = estimate_T_threshold_selection(X, y_noisy, RandomForestClassifier(), n_jobs=-1)
T2 = estimate_T_cost_sensitive(X, y_noisy, RandomForestClassifier(), eps=0.05, n_jobs=-1)
```

| Argument | Meaning | Default |
|---|---|---|
| `classifier` | scikit-learn classifier or `Pipeline` | `LogisticRegression` |
| `n_plus` | minimum number of accepted samples `N+` | `n / 200` |
| `eps` | cost grid spacing, `c = eps, 2 eps, ..., 1 - eps` (`estimate_T_cost_sensitive` only) | `0.05` |
| `n_classes` | number of classes `K` | `max(y) + 1` |
| `random_state` | seed for the split into two halves | `None` |
| `n_jobs` | number of binary problems fit in parallel | `None` |
| `plot` | show the plots of Appendix B, one panel per class | `False` |

Classifier requirements:
- `estimate_T_threshold_selection` needs `predict_proba` or `decision_function`.
- `estimate_T_cost_sensitive` needs `fit` to accept `sample_weight`.

## MATLAB implementation

The folder `matlab/` provides code needed to replicate the experiments in the paper:

| File | Content |
|---|---|
| `main_exp.m` | runs an experiment: loads a dataset, injects label noise, and estimates the transition matrix |
| `thr.m` | Algorithm 1 (threshold selection) |
| `cost.m` | Algorithm 2 (cost-sensitive risk minimization) |
| `anchor_based.m` | anchor-based method (baseline) |
| `oracle.m` | maximum likelihood oracle used in Figure 1 |
| `predict_scores.m` | helper that returns classifier scores |

It requires the Statistics and Machine Learning, Deep Learning, Optimization, and Parallel Computing toolboxes.

### Usage

```matlab
main_exp(DATASET, type, N, N_rep, seed, noise_type, noise_p, estimator, method, eps_grid, N_plus, results_path)
```

| Argument | Values |
|---|---|
| `DATASET` | `"letter"`, `"satellite"`, `"MNIST"`, `"CIFAR10"` |
| `type` | `"table"` saves to `results_path/table/`; any other name saves to `results_path/plot/` with that name as prefix |
| `N`, `N_rep`, `seed` | number of samples, number of repetitions, random seed |
| `noise_type`, `noise_p` | `"uniform"` or `"flip"` with noise ratio `noise_p`; `noise_p > 1` draws a different noise ratio for each column (Table 3) |
| `estimator` | `"thr"`, `"cost"`, `"anchor_based"`, `"oracle"` |
| `method` | `"logistic"`, `"fitcnet"`, `"randomforest"` |
| `eps_grid`, `N_plus` | grid spacing and minimum number of accepted samples; `[]` selects the defaults `0.05` and `N/200` |
| `results_path` | folder where the results are saved |

For example, Algorithm 1 with random forests on Letter under 0.2-uniform noise:

```matlab
main_exp("letter", "table", 20000, 5, 0, "uniform", 0.2, "thr", "randomforest", [], [], "results")
```

Each run saves the true and estimated matrices of every repetition (`T_true`, `all_T`), the mean absolute error (`all_errors`), and the running times (`tiempos`).

### Datasets

The folder `data/` contains the datasets used in the experiments, and `main_exp` reads them from there:

| File | Dataset |
|---|---|
| `letter.mat` | Letter (UCI): 20,000 samples, 26 classes, 16 features |
| `satellite.mat` | Satellite (UCI): 6,435 samples, 6 classes, 36 features |
| `mnist.mat` | MNIST: 70,000 samples, 10 classes, 784 pixels |
| `mnist_convnext_base.mat` | MNIST features from a pre-trained ConvNeXt, used by the oracle |
| `cifar10_convnext_base.mat` | CIFAR-10 features from a pre-trained ConvNeXt: 60,000 samples, 10 classes, 1,024 features |

The two ConvNeXt feature files are too large for GitHub. Download them from [here](https://drive.google.com/drive/folders/1FWQ7KMAfi0XLQuPLsOwmT5cga_l7ENEo?usp=sharing) and place them in the folder `data/`.

## Citation and license

If you use this work in research, please cite:

```bibtex
@inproceedings{dejuan2026,
	title     = {Estimation of the Label-Noise Transition Matrix with Performance Guarantees via Selective Classification},
	author    = {de Juan, Xabier and Mazuelas, Santiago and Zhu, Yilun and Scott, Clayton},
	booktitle = {Advances in Neural Information Processing Systems},
	year      = {2026}
}
```

This code is released under the MIT License.
