# Oral-exam notes

## What is the question?

Keep the number of training observations fixed and add features. Does test error first fall, then rise near interpolation, then fall again? Interpolation means fitting the training labels, not necessarily predicting well.

## How do the solvers work?

**Least squares:** SVD writes X as U diag(s) V-transpose. Project the target onto the left singular vectors, divide by retained singular values, and map back with the right singular vectors. When there are multiple fits, this selects the smallest coefficient norm. Small singular values can amplify noise enormously.

**Ridge:** add alpha times the squared coefficient norm to the sum of squared errors. The original solver uses `solve(X.T @ X + alpha*I, X.T @ y)`, without forming an inverse. The CV extension reuses an SVD and multiplies projected targets by `s/(s*s+alpha)` for each alpha. This is the same objective. Alpha is for the sum of errors, not mean error.

## Why those numerical thresholds?

The `1e-10 * largest singular value` cutoff treats tiny directions as numerically negligible. It is a stated numerical convention, not a theorem or a tuned prediction parameter. Use the same cutoff for fitting and reporting rank. Different cutoffs may affect ill-conditioned real-world fits; I have not performed a full tolerance sensitivity study.

The `1e-8` relative-residual threshold means training residual norm is negligible compared with target norm. It is a practical definition of numerical interpolation, not exact symbolic equality. Centering real-world features limits rank to n−1, but centered targets can still be fitted. Counting parameters alone is insufficient.

## What does each experiment show?

**Synthetic:** independent Gaussian features, known noisy linear labels, and early coordinates with stronger signal. Prefixes first capture useful signal. Near d=n=150, nearly singular directions amplify noise. Beyond it, the minimum-norm fit shows a second descent. Its final error still exceeds the best earlier error. Ridge reduces the spike.

**Real world:** only 80 training rows, correlated/sparse predictors, and one later test date. All repeated runs interpolate, but the median peak/decline is not a clean full double-descent curve. The seed-42 example is especially unstable. These results do not isolate whether conditioning, sampling or temporal differences caused the failure.

**GD:** start from zero. Each update stays in the row space of X, so stable unregularized GD tends toward the minimum-norm fit. Fixed early stopping suppresses slowly learned small-singular-value directions: implicit regularization. At d=200 and 400, the coefficient difference becomes tiny; at d=150, 20,000 unregularized steps are insufficient. Ridge improves conditioning. Finite steps are not the same as fully converged least squares, and early stopping is not exactly ridge.

**Real ridge/CV:** preserve the original alpha=1 curves. The extension uses raw feature prefixes, five alpha values and five folds inside each 80-row subset. Refit scaling, constant-column filtering and target centering in every fitting fold. Select alpha from validation errors only, then refit all training rows. Never select using the official test set. At raw d=280, CV beats its baseline in only 1/20 runs; median errors are about 3.557 versus 1.147. This differs from the original effective d=149 comparison.

**Noise:** larger noise makes the seed-42 spike more pronounced and raises the noise floor. This is one seed, not a general robustness estimate.

## What must I not overclaim?

IQR is variation across subsets/orderings, not a confidence interval across test dates. A lower error than unstable least squares is not enough: compare with the training-mean baseline. Raw and log-target MSE are not numerically comparable. The design was exploratory, including the reduction from n=150 to 80. GD has a finite iteration cap; CV has a finite alpha grid. AI assistance is acknowledged, and I must check the course policy before submission.
