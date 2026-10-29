# Double Descent in Linear Models

Stela Mile · Statistical Methods for Machine Learning · Assignment 4 · A.Y. 2025/26

This project compares minimum-norm least squares and ridge regression as the number of features grows with a fixed training sample size. Both algorithms are implemented using NumPy linear algebra, without estimator classes. Experiments use noisy synthetic linear data and the UCI BlogFeedback regression dataset.

## Reproduce the experiments

The verified environment is Python **3.13.7**, NumPy **2.4.6**, and Matplotlib **3.11.0**. The runner uses only the standard library plus the two pinned packages.

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/run_experiments.py
```

On Windows activate the environment with `.venv\Scripts\activate` instead. Python 3.13 is the tested interpreter; identical floating-point results on other Python/BLAS builds are not guaranteed, particularly for ill-conditioned real-world fits.

The command runs both notebooks in order, embeds fresh outputs, exports PNG and vector PDF figures to `figures/`, and writes numerical arrays to `results/synthetic.json` and `results/real.json`. `results/environment.json` records versions, platform, run times, and SHA-256 checksums for the two CSVs actually used. It stops with an exception on a failed cell and only replaces each executed notebook after all its cells succeed. If running one notebook only, the environment file describes that latest invocation.

To run only one experiment:

```sh
python src/run_experiments.py --notebook synthetic
python src/run_experiments.py --notebook real
```

The notebooks also run interactively: install JupyterLab separately if desired, open a notebook, select the environment above, and **Restart Kernel and Run All**. Use `notebooks/` as the working directory. Figure saving is built into the notebooks. The command-line runner handles the working directory automatically and does not require Jupyter.

## Data

BlogFeedback is attributed to Krisztian Buza (2014), UCI Machine Learning Repository, DOI [10.24432/C58S3F](https://doi.org/10.24432/C58S3F), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The real-world notebook downloads [the official archive](https://archive.ics.uci.edu/static/public/304/blogfeedback.zip) only when it is absent, and extracts it if the expected training CSV or complete 60-file test collection is missing. Internet access is required on the first run if local data are absent. A complete local dataset allows offline execution. Generated data are not needed in Git.

- Official training CSV: 52,397 rows and 280 predictors plus the target.
- Training subset: 80 observations, fixed seed 42 for illustration; seeds 0–19 for repetitions.
- Held-out test CSV: `blogData_test-2012.03.31.01_00.csv`, 214 observations.
- Training-only standardization and constant-column removal; target is `log1p(comment count)`.
- Other test dates are not pooled. The same later test set is reused in every repetition.

## Experimental settings

Synthetic data: 150 training observations, 1,000 independent test observations, 400 full features, Gaussian noise standard deviation 0.5, and normalized exponentially decaying coefficient magnitudes. Thirty paired seeds (0–29) are used for least squares and all ridge strengths (0.01, 0.1, 1, 10). A noise extension uses standard deviations 0.1, 0.5 and 1 with seed 42; this is a single-seed diagnostic, not a repeated robustness study.

Real-world data: each run fixes n=80 and varies the prefix of a random feature ordering after training-only preprocessing. SVD fitting and rank diagnostics use a relative cutoff of 1e-10; interpolation is declared when the relative training residual is below 1e-8. Ridge uses alpha=1 in the unnormalized squared-error objective. The repeated plots compare dimensions supported by every run (1–149 in the verified run), with the median, descriptive interquartile range, and a separately fitted training-mean baseline. No hyperparameter is selected using the test curves.

## Main findings

- Synthetic least squares exhibits a clear interpolation spike and second descent; the large model does not beat the best pre-interpolation model in this setup.
- Ridge suppresses the spike on exactly the same synthetic data.
- All 20 real-world repetitions reach interpolation, but test performance is unstable. The median curve falls after a large peak; the seed-42 illustration does not show a convincing second descent.
- At the final common real-world dimension, ridge loses to the training-mean baseline in all 20 runs. Improvement over unstable least squares is not sufficient evidence of useful prediction.

IQR bands describe variation across training subsets and feature orderings, not confidence intervals or uncertainty across independent test dates. Synthetic raw-target MSE and real-world log-target MSE have different scales and should not be compared numerically.

## Files

- `notebooks/01_synthetic_double_descent.ipynb`: synthetic least squares, paired ridge comparison, regularization sensitivity.
- `notebooks/02_real_world_double_descent.ipynb`: BlogFeedback, interpolation diagnostics, paired baselines, repeated experiments, noise extension.
- `src/run_experiments.py`: headless execution and output capture.
- `figures/`: exported PNG/PDF figures; `results/figure_manifest.json` links these to notebook cells.
- `results/`: machine-readable numerical evidence and environment record.
- `report/report.tex`: standalone LaTeX source with plots embedded as vector drawing commands derived from the numerical results; no external figure files or BibTeX step are needed.
- `report/report.pdf`: compiled report.

## Rebuild the report

With a TeX distribution providing `pdflatex`:

```sh
python report/build_source.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

The report's embedded plot coordinates and numerical summaries are a snapshot of the verified run. `build_source.py` reads the result JSON files to generate the plot coordinates. It checks key expected results and stops if the setup has materially changed; update the narrative and checks when changing the experiments rather than recompiling an outdated interpretation. The LaTeX source can also be opened in Codex's built-in LaTeX editor.

## Sources and authorship review

- Belkin, Hsu, Ma and Mandal (2019), [Reconciling modern machine-learning practice and the classical bias–variance trade-off](https://doi.org/10.1073/pnas.1903070116).
- Buza (2014), [BlogFeedback](https://doi.org/10.24432/C58S3F).
- Course project brief and course notes supplied by the student.

Codex assisted with code review, revisions, execution, visualizations, and report drafting. The report acknowledges this assistance and reproduces the course's required declaration as **text for the author to review**, not a signed certification. Review the report, verify all claims, and ensure the declaration and acknowledgment meet the course's rules before submission.

Publication to the public repository and submission of its URL are separate final steps. No submission or new publication is performed by this package.
