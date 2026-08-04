from typing import Dict, Optional, Sequence
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

import numpy as np

DEFAULT_ALPHAS = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)


def _split_iter(X, n_splits, random_state, folds):
    """Yield (train_idx, val_idx) pairs.

    If ``folds`` is given (a per-row fold id, e.g. the lineage-aware
    ``cv_folds.csv`` assignment) it defines the splits exactly; otherwise fall
    back to a random shuffled KFold.
    """
    if folds is not None:
        folds = np.asarray(folds)
        if len(folds) != len(X):
            raise ValueError(f'folds length {len(folds)} != n_samples {len(X)}')
        for f in sorted(np.unique(folds)):
            val_idx = np.where(folds == f)[0]
            train_idx = np.where(folds != f)[0]
            yield train_idx, val_idx
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        yield from kf.split(X)


def cross_validate_ridge(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    alphas=DEFAULT_ALPHAS,
    folds: Optional[Sequence[int]] = None,
) -> Dict:
    preds = np.zeros_like(y, dtype=float)
    fold_alphas = []
    for train_idx, val_idx in _split_iter(X, n_splits, random_state, folds):
        model = RidgeCV(alphas=alphas)
        model.fit(X[train_idx], y[train_idx])
        preds[val_idx] = model.predict(X[val_idx])
        fold_alphas.append(float(model.alpha_))

    return {
        'spearman': float(spearmanr(y, preds).correlation),
        'mae': float(mean_absolute_error(y, preds)),
        'r2': float(r2_score(y, preds)),
        'predictions': preds,
        'fold_alphas': fold_alphas,
    }
