"""
Voxel-wise encoding model: predicts fMRI responses from DNN activations
using ridge regression.

Updated after validation (see notebooks/04_validation.ipynb):
- Standardization is now always applied, since it measurably improved
  accuracy (0.130 -> 0.174 for fc7 -> FFA).
- Alpha is no longer a fixed hardcoded value. Instead, RidgeCV with
  alpha_per_target=True selects the best alpha per voxel automatically
  via efficient built-in cross-validation, removing the arbitrary
  alpha=100.0 choice that the validation sweep showed was not
  well-justified (fc7 vs fc8 ranking changed depending on alpha).
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


DEFAULT_ALPHAS = np.logspace(0, 6, 13)  # 1 to 1,000,000, log-spaced


def train_and_evaluate_encoding_model(X, Y, alphas=None, n_folds=5, seed=42):
    """
    Trains a ridge regression encoding model with nested cross-validation
    (outer loop for held-out evaluation, inner loop via RidgeCV for
    automatic per-voxel alpha selection) and returns per-voxel prediction
    accuracy.

    Parameters
    ----------
    X : np.ndarray
        DNN activations, shape (num_videos, num_features).
    Y : np.ndarray
        fMRI responses, shape (num_videos, num_voxels).
    alphas : array-like, optional
        Candidate alpha values for RidgeCV to search over. Defaults to
        a log-spaced range from 1 to 1,000,000.
    n_folds : int
        Number of outer cross-validation folds.
    seed : int
        Random seed for fold splitting.

    Returns
    -------
    np.ndarray
        Per-voxel correlation (Pearson r) between predicted and actual
        fMRI responses, averaged across outer folds. Shape (num_voxels,).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    num_voxels = Y.shape[1]
    fold_scores = np.zeros((n_folds, num_voxels))

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # alpha_per_target=True lets RidgeCV pick a separate best alpha
        # for each voxel, via efficient leave-one-out-style internal CV,
        # instead of one alpha shared across all voxels.
        model = RidgeCV(alphas=alphas, alpha_per_target=True)
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
