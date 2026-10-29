# Double Descent in Linear Models

Stela Mile — Statistical Methods for Machine Learning, Assignment 4, A.Y. 2025/26

## Overview

I study how training and test errors change when the number of features increases while the training sample size stays fixed. The project compares minimum-norm least squares and ridge regression on synthetic linear data and the real-world BlogFeedback dataset. It also examines gradient descent, early stopping, label noise, and training-only selection of ridge strength.

The algorithms are implemented from scratch using NumPy matrix operations. There are no scikit-learn estimators, `lstsq`, or `pinv` solvers.

## How to reproduce

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_experiments.py
```

On Windows, activate with `.venv\Scripts\activate`. Python 3.13 is the tested interpreter. NumPy and Matplotlib use compatible version ranges; `results/environment.json` records the actual versions used for the saved results. A separate fresh-environment check is recorded in `results/fresh_environment.json`. Ill-conditioned least-squares fits can differ substantially across numerical libraries. In the fresh environment, one individual near-threshold test MSE differed by about 40%; the original saved results are retained. Ridge/CV differences were negligible.

The runner executes both notebooks from top to bottom, saves their outputs, exports figures to `figures/`, and saves numerical arrays to `results/`. It sets the working directory automatically. To run one notebook only:

```sh
python src/run_experiments.py --notebook synthetic
python src/run_experiments.py --notebook real
```

For interactive work, install JupyterLab separately, open a notebook with `notebooks/` as its working directory, and use **Restart Kernel and Run All**. The notebooks import the new extension functions from `src/extensions.py`.

## Data

BlogFeedback: Krisztian Buza (2014), UCI Machine Learning Repository, [DOI: 10.24432/C58S3F](https://doi.org/10.24432/C58S3F), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The real-world notebook downloads the [official archive](https://archive.ics.uci.edu/static/public/304/blogfeedback.zip) if needed. A complete cached dataset permits offline execution.

- Training pool: 52,397 observations with 280 predictors.
- Each experiment samples 80 training observations without replacement.
- Fixed official test file: `blogData_test-2012.03.31.01_00.csv`, 214 observations.
- Target: `log1p(comment count)`; preprocessing uses training statistics only.
- The same later test date is reused in every repetition; other dates are not pooled.

## Experimental settings

**Original synthetic experiment:** 150 training and 1,000 test observations, 400 Gaussian features, normalized coefficients with magnitudes decaying as `exp(-0.02*j)`, and Gaussian label noise with standard deviation 0.5. Models use nested prefixes of fixed feature matrices. Thirty seeds (0–29) are paired across least squares and ridge strengths 0.01, 0.1, 1 and 10.

**Gradient descent:** the original seed-42 dataset and dimension sweep, zero initialization, 20/200/2,000 iterations, and step size `0.9/(s_max**2 + alpha)` for half the unnormalized squared-error objective. Separate training-only convergence checks compare coefficients with the closed form, up to 20,000 iterations. The checks report the measured norm difference and do not label a capped run as converged.

**Original real-world experiment:** seed 42 for illustration and seeds 0–19 for repetitions; ridge alpha=1. SVD fitting/rank cutoff is `1e-10` relative to the largest singular value. Interpolation requires relative training residual below `1e-8`. The repeated curves use effective dimensions available in all runs (1–149).

**Ridge/CV extension:** five strengths (0.01, 0.1, 1, 10, 100), five inner folds, and fixed raw feature prefixes 10, 20, 40, 60, 80, 100, 120, 160, 200, 240 and 280. Every fold fits its own constant-column mask, scaler and target mean. Alpha is selected from inner validation MSE separately at each fixed dimension. This extension counts raw columns before filtering, so its x-axis differs from the original effective-feature axis. At all 280 raw columns, the alpha=1 fit matches the original full-feature fit. No test values enter selection.

**Noise extension:** seed 42, standard deviations 0.1, 0.5 and 1; a single-seed diagnostic rather than a repeated noise study.

## Findings

- Synthetic least squares shows a clear interpolation spike and second descent. The largest model still does not beat the best pre-interpolation error in this setup.
- Ridge suppresses the spike on the same data. Early-stopped GD also suppresses slowly learned singular directions. Near interpolation, unregularized GD may converge extremely slowly.
- All 20 original real-world repetitions interpolate, but test performance is unstable. A median peak and decline do not amount to a clean full double-descent curve.
- At the original common effective dimension 149, alpha=1 ridge loses to its baseline in all 20 runs.
- At the CV extension's final raw dimension 280, CV ridge has median test MSE about 3.557 versus 1.147 for the baseline and beats its paired baseline in only 1/20 runs. Stronger regularization helps but does not establish reliable prediction.

IQR bands show variability across training subsets and feature orderings, not confidence intervals over independent test dates. Synthetic raw-label MSE and real-world log-target MSE are different scales.

## Files

- `notebooks/01_synthetic_double_descent.ipynb`: synthetic experiments, paired ridge comparison and GD.
- `notebooks/02_real_world_double_descent.ipynb`: original real-world experiments, noise and ridge/CV extension.
- `src/extensions.py`: GD, fold-local preprocessing and ridge cross-validation.
- `src/run_experiments.py`: runs notebooks and saves outputs and environment information.
- `figures/`: PNG/PDF exports referenced by the report.
- `results/`: original experiments, GD, CV, environment and verification records.
- `report/report.tex`: manually maintained LaTeX source using ordinary figure files.
- `report/report.pdf`: previous compiled version; not refreshed for this source revision.
- `NOTES_FOR_ORAL.md`: short explanations for preparing the oral exam.

The unused `start.py` import stub and the report-source generator have been removed. Edit `report.tex` directly; regenerating it is no longer part of the workflow.

## Report

From the repository root, when ready to rebuild:

```sh
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The report reads figures from `../figures/`. Re-run the experiments first if changing their settings, and update the written interpretation when results change. No additional generator or BibTeX step is required. The current source passed a ten-page TeX/XDV check with the external figures, without exporting a new report PDF. A final two-pass pdflatex build and visual PDF review remain for the PDF stage.

## References and use of AI tools

- Belkin, Hsu, Ma and Mandal (2019), [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://doi.org/10.1073/pnas.1903070116).
- Buza (2014), [BlogFeedback](https://doi.org/10.24432/C58S3F).
- Supplied course notes and Assignment 4 project brief.

AI tools (Codex) were used for code review, implementation revisions, experiment execution, numerical checks, plotting, and report drafting. This assistance is acknowledged separately from the course declaration in the report.
