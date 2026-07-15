"""
Voxel-wise encoding model: predicts fMRI responses from DNN activations
using ridge regression.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr


def train_and_evaluate_encoding_model(X, Y, alpha=10.0, n_folds=5, seed=42):
    """
    Trains a ridge regression encoding model with cross-validation and
    returns per-voxel prediction accuracy.

    Parameters
    ----------
    X : np.ndarray
        DNN activations, shape (num_videos, num_features).
    Y : np.ndarray
        fMRI responses, shape (num_videos, num_voxels).
    alpha : float
        Ridge regularization strength.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for fold splitting.

    Returns
    -------
    np.ndarray
        Per-voxel correlation (Pearson r) between predicted and actual
        fMRI responses, averaged across folds. Shape (num_voxels,).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    num_voxels = Y.shape[1]
    fold_scores = np.zeros((n_folds, num_voxels))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        model = Ridge(alpha=alpha)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        for v in range(num_voxels):
            r, _ = pearsonr(Y_test[:, v], Y_pred[:, v])
            fold_scores[fold_idx, v] = r if not np.isnan(r) else 0.0

    return fold_scores.mean(axis=0)


def region_mean_accuracy(voxel_scores):
    """
    Returns the mean prediction accuracy across all voxels in a region.

    Parameters
    ----------
    voxel_scores : np.ndarray
        Per-voxel correlation scores, shape (num_voxels,).

    Returns
    -------
    float
        Mean correlation across voxels.
    """
    return float(np.mean(voxel_scores))