"""Small NumPy implementations for gradient descent and training-only ridge CV."""
import numpy as np


def gradient_descent(X, y, alpha=0.0, steps=(20, 200, 2000)):
    """GD from zero for (||Xw-y||² + alpha||w||²)/2.

    Use 0.9/L, where L=s_max²+alpha is computed from training X only.
    Return copies at the requested iteration counts (no test-based stopping).
    """
    largest = np.linalg.svd(X, compute_uv=False)[0]
    step_size = 0.9 / (largest**2 + alpha)
    w = np.zeros(X.shape[1])
    snapshots = {}
    for iteration in range(1, max(steps) + 1):
        w -= step_size * (X.T @ (X @ w - y) + alpha * w)
        if iteration in steps:
            snapshots[iteration] = w.copy()
    return snapshots, step_size


def gd_convergence_check(X, y, reference, alpha=0.0, max_steps=20000, rtol=1e-8):
    """Training-only diagnostic: compare actual GD iterates to a reference fit.

    A capped run is explicitly reported as not converged if it misses rtol.
    This diagnostic is separate from the early-stopping test-error curves.
    """
    largest = np.linalg.svd(X, compute_uv=False)[0]
    step_size = 0.9 / (largest**2 + alpha)
    w = np.zeros(X.shape[1])
    scale = max(np.linalg.norm(reference), np.finfo(float).eps)
    for iteration in range(1, max_steps + 1):
        w -= step_size * (X.T @ (X @ w - y) + alpha * w)
        if iteration % 100 == 0 and np.linalg.norm(w - reference) / scale < rtol:
            break
    difference = np.linalg.norm(w - reference)
    return {"steps": iteration, "difference_norm": float(difference),
            "relative_difference": float(difference / scale),
            "converged": bool(difference / scale < rtol), "step_size": float(step_size)}


def preprocess(train_X, other_X, train_y):
    """Fit scaling, constant-column mask and intercept on these training rows."""
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    keep = std > 0
    target_mean = train_y.mean()
    return ((train_X[:, keep] - mean[keep]) / std[keep],
            (other_X[:, keep] - mean[keep]) / std[keep],
            train_y - target_mean, target_mean)


def ridge_path(X, y, alphas):
    """Ridge fits using the explicit SVD factors, without estimator classes.

    Zero singular values get weight zero; no inverse or pseudoinverse is used.
    This is algebraically the same ridge objective as the existing solve().
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    projected = U.T @ y
    return [Vt.T @ ((s / (s*s + alpha)) * projected) for alpha in alphas]


def select_ridge_alpha(X_raw, y_log, alphas, folds):
    """Select alpha inside the outer training subset, with fold-local preprocessing.

    Candidate raw columns are fixed before validation. The validation/test
    targets never determine column filtering, means, scales, or fitted weights.
    """
    scores = np.zeros(len(alphas))
    for validation in folds:
        training = np.setdiff1d(np.arange(len(y_log)), validation)
        Xtr, Xval, ytr, intercept = preprocess(
            X_raw[training], X_raw[validation], y_log[training])
        for j, w in enumerate(ridge_path(Xtr, ytr, alphas)):
            scores[j] += np.sum((y_log[validation] - (Xval @ w + intercept))**2)
    scores /= len(y_log)
    # np.argmin breaks an exact tie by the first alpha in the fixed grid.
    return int(np.argmin(scores)), scores


def run_real_ridge_extension(X_pool, y_pool, X_test, y_test, seed,
                             dimensions, alphas, n_train=80, k=5):
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y_pool), size=n_train, replace=False)
    order = rng.permutation(X_pool.shape[1])
    raw_train = X_pool[indices][:, order]
    raw_test = X_test[:, order]
    y_train = np.log1p(y_pool[indices])
    test_log = np.log1p(y_test)
    folds = np.array_split(np.random.default_rng(10000 + seed).permutation(n_train), k)
    baseline = float(np.mean((test_log - y_train.mean())**2))
    test_errors, train_errors, cv_errors, selected, scores, usable = [], [], [], [], [], []
    for d in dimensions:
        # Raw feature prefixes avoid choosing candidate columns using fold-validation rows.
        chosen, validation_scores = select_ridge_alpha(raw_train[:, :d], y_train, alphas, folds)
        Xtr, Xte, ytr, intercept = preprocess(raw_train[:, :d], raw_test[:, :d], y_train)
        weights = ridge_path(Xtr, ytr, alphas)
        errors = [float(np.mean((test_log - (Xte @ w + intercept))**2)) for w in weights]
        test_errors.append(errors)
        train_errors.append([float(np.mean((ytr - Xtr @ w)**2)) for w in weights])
        cv_errors.append(errors[chosen])
        selected.append(float(alphas[chosen]))
        scores.append(validation_scores.tolist())
        usable.append(Xtr.shape[1])
    return {"seed": seed, "test_by_alpha": test_errors, "train_by_alpha": train_errors,
            "cv_test": cv_errors, "selected_alpha": selected, "cv_scores": scores,
            "usable_dimensions": usable, "baseline_test": baseline}
