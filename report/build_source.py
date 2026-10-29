"""Regenerate standalone report.tex from the verified results JSON files."""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT/'results/synthetic.json').read_text())
R = json.loads((ROOT/'results/real.json').read_text())
E = json.loads((ROOT/'results/environment.json').read_text())

def fmt(v):
    v=float(v)
    return f'{v:,.3f}' if abs(v)<1e5 else f'{v:.3g}'

def plot(title, series, xmax, ymin, ymax, xticks, vline=None, bands=(), ylabel='Test MSE (log scale)'):
    # Native LaTeX picture commands: all coordinates embedded, no external files.
    left,bottom,width,height=48,34,392,145
    lo,hi=math.log10(ymin),math.log10(ymax)
    def px(x):return left+width*float(x)/xmax
    def py(y):return bottom+height*(math.log10(max(float(y),ymin))-lo)/(hi-lo)
    out=[r'\begin{center}\setlength{\unitlength}{1pt}',r'\begin{picture}(460,222)',
         rf'\put(48,211){{\small\bfseries {title}}}',rf'\put(0,190){{\scriptsize {ylabel}}}']
    for power in range(math.ceil(lo),math.floor(hi)+1):
        yy=py(10**power)
        out += [rf'{{\color{{grid}}\linethickness{{0.35pt}}\qbezier(48,{yy:.2f})(244,{yy:.2f})(440,{yy:.2f})}}',
                rf'\put(42,{yy-2:.2f}){{\makebox(0,0)[r]{{\scriptsize $10^{{{power}}}$}}}}']
    for x in xticks:
        xx=px(x)
        out += [rf'{{\color{{grid}}\linethickness{{0.35pt}}\qbezier({xx:.2f},34)({xx:.2f},106.5)({xx:.2f},179)}}',
                rf'\put({xx:.2f},22){{\makebox(0,0){{\scriptsize {x:g}}}}}']
    # Dense vertical strips encode the IQR with coordinates from percentiles.
    for xs,lower,upper,color in bands:
        for j,(x,a,b) in enumerate(zip(xs,lower,upper)):
            thick=width/xmax*(float(xs[min(j+1,len(xs)-1)])-float(xs[max(j-1,0)]))/2+0.15
            xx=px(x);ya=max(bottom,min(bottom+height,py(a)));yb=max(bottom,min(bottom+height,py(b)))
            out.append(rf'{{\color{{{color}}}\put({xx-thick/2:.2f},{ya:.2f}){{\rule{{{thick:.2f}pt}}{{{max(yb-ya,0.01):.2f}pt}}}}}}')
    legends=[]
    for k,(xs,ys,color,label) in enumerate(series):
        points=[(px(x),py(y)) for x,y in zip(xs,ys)]
        for (x1,y1),(x2,y2) in zip(points,points[1:]):
            if not (bottom-0.01<=y1<=bottom+height+0.01 and bottom-0.01<=y2<=bottom+height+0.01):continue
            out.append(rf'{{\color{{{color}}}\linethickness{{1.05pt}}\qbezier({x1:.2f},{y1:.2f})({(x1+x2)/2:.2f},{(y1+y2)/2:.2f})({x2:.2f},{y2:.2f})}}')
        lx=48+k*130
        legends += [rf'{{\color{{{color}}}\linethickness{{1.3pt}}\qbezier({lx},199)({lx+7},199)({lx+14},199)}}',
                rf'{{\color{{black}}\put({lx+19},196){{\scriptsize {label}}}}}']
    if vline is not None:
        xx=px(vline)
        for yy in range(bottom,bottom+height,7):out.append(rf'{{\color{{black}}\linethickness{{0.7pt}}\qbezier({xx:.2f},{yy})({xx:.2f},{yy+2})({xx:.2f},{yy+4})}}')
    out += legends
    out += [r'\linethickness{0.6pt}\put(48,34){\line(1,0){392}}\put(48,34){\line(0,1){145}}',
            r'\put(244,5){\makebox(0,0){\scriptsize Number of features ($d$)}}',r'\end{picture}\end{center}']
    return '%\n'.join(out)

D=np.array(S['D_VALUES']);mean=np.array(S['mean_test_errors']); ridge=np.array(S['mean_ridge_test']['1.0'])
DX=np.array(R['D_VALUES_REAL']);DR=np.array(R['D_VALUES_REPEAT']);M=np.array(R['median_real_ls']);MR=np.array(R['median_real_ridge'])
valid=len(DR)
all_ls=np.array([r['ls_test'][:valid] for r in R['real_repetition_results']])
all_ridge=np.array([r['ridge_test'][:valid] for r in R['real_repetition_results']])
q1,q3=np.percentile(all_ls,[25,75],axis=0);rq1,rq3=np.percentile(all_ridge,[25,75],axis=0)
peak=int(np.argmax(M));best=int(np.argmin(mean[D<150]));hit=[r['interpolation_dimension'] for r in R['real_repetition_results']]
fig_s=plot('Synthetic data: paired means across 30 seeds',[(D,mean,'bluecurve','Least squares'),(D,ridge,'orangecurve','Ridge, alpha=1')],400,.3,2000,[0,100,150,200,300,400],150)
fig_r=plot('BlogFeedback: seed 42',[(DX,R['real_test_errors'],'bluecurve','Least squares'),(DX,R['real_ridge_test_errors'],'orangecurve','Ridge, alpha=1'),([1,DX[-1]],[R['real_baseline_test_mse']]*2,'gray','Baseline')],160,1,1e8,[0,40,80,120,160],R['interpolation_dimension'])
fig_res=plot('BlogFeedback: training interpolation diagnostic',[(DX,np.maximum(R['real_relative_residuals'],1e-16),'bluecurve','Relative residual'),([1,DX[-1]],[1e-8]*2,'orangecurve','Tolerance')],160,1e-13,1,[0,40,80,120,160],R['interpolation_dimension'],ylabel='Relative training residual (log scale)')
fig_rep=plot('BlogFeedback: median and interquartile range',[(DR,M,'bluecurve','Least squares'),(DR,MR,'orangecurve','Ridge, alpha=1'),([1,DR[-1]],[R['median_real_baseline']]*2,'gray','Baseline')],150,1,2e5,[0,30,60,90,120,150],bands=[(DR,q1,q3,'blueband'),(DR,rq1,rq3,'orangeband')])
noise_series=[(np.array(R['D_VALUES_NOISE']),v,c,rf'$\sigma={k}$') for (k,v),c in zip(R['noise_results'].items(),['bluecurve','orangecurve','greencurve'])]
fig_noise=plot('Noise extension: synthetic data, seed 42',noise_series,400,.01,1000,[0,100,150,200,300,400],150)

preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=23mm,top=20mm,bottom=20mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,xcolor,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black,pdftitle={Double Descent in Linear Models},pdfauthor={Stela Mile}}
\definecolor{bluecurve}{RGB}{30,105,165}
\definecolor{orangecurve}{RGB}{224,110,25}
\definecolor{greencurve}{RGB}{28,128,82}
\definecolor{blueband}{RGB}{205,225,240}
\definecolor{orangeband}{RGB}{251,226,204}
\definecolor{grid}{RGB}{220,224,228}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.2}
\newcommand{\pageheading}[1]{\section*{#1}}
\begin{document}
'''
pages=[]
pages.append(r'''{\Large\bfseries Double Descent in Linear Models}\par
{\large Minimum-norm least squares and ridge regression}\par
\textbf{Stela Mile}\quad |\quad Statistical Methods for Machine Learning\par
Assignment 4\quad |\quad A.Y. 2025/26\quad |\quad 29 October 2026

\pageheading{1. Research question and main findings}
Classical accounts of generalization emphasize a trade-off: increasing model complexity can reduce approximation error while increasing estimation error. Beyond the capacity needed to fit the training set exactly, test error can decline again. This is the double-descent phenomenon discussed by Belkin et al.\ [1]. This project investigates that behavior in linear regression by holding the number of training observations fixed and varying the number of available features.

\textbf{Abstract.} Minimum-norm least squares and ridge regression are implemented with explicit NumPy linear algebra and evaluated on noisy synthetic linear data and a real-world BlogFeedback regression task. The synthetic experiment exhibits a pronounced interpolation spike followed by a second descent. Ridge substantially suppresses that spike. The real-world experiment reaches numerical interpolation, but its predictive performance is unstable. Across repeated subsets, a median peak and decline coexist with weak performance against a constant baseline. These findings support a distinction between identifying interpolation, observing a second descent, and achieving useful prediction.

\begin{center}
\begin{tabular}{p{0.23\linewidth}p{0.32\linewidth}p{0.32\linewidth}}
\toprule
 & Synthetic data & BlogFeedback \\
\midrule
Fixed training size & 150 & 80 \\
Test observations & 1,000 per run & 214, fixed later date \\
Repeated seeds & 0--29 & 0--19 \\
Observed interpolation & At $d=150$ & $d=80$--$93$ in repetitions \\
Main pattern & Clear double descent & Mixed evidence; high instability \\
Regularization & Suppresses the spike & Improves on unstable least squares \\
Practical caveat & Largest model is not best & Final ridge models lose to baseline \\
\bottomrule
\end{tabular}
\end{center}

\textbf{Scope.} The experiments are diagnostic sweeps, not a model-selection exercise. The test curves are used to study behavior, not to choose the dimension or regularization strength for deployment. No claim is made that these linear experiments reproduce the models or full empirical scope of the reference paper.

\textbf{Evidence.} Every numerical statement below is derived from the executed notebooks and accompanying JSON results. Linear and logarithmic views are retained in the repository; the report uses logarithmic error plots where large spikes would otherwise obscure the ridge and baseline curves.
''')
pages.append(r'''\pageheading{2. Models, loss, and numerical interpolation}
For a design matrix $X\in\mathbb{R}^{n\times d}$ and labels $y\in\mathbb{R}^n$, least squares minimizes the unnormalized sum of squared errors:
\[
\widehat w\in\arg\min_w\|Xw-y\|_2^2.
\]
When the minimizer is not unique, the implementation chooses the solution with smallest Euclidean norm. Given a reduced singular value decomposition $X=U\Sigma V^\top$, it computes
\[
\widehat w=V_K\Sigma_K^{-1}U_K^\top y,
\qquad K=\{j:s_j>10^{-10}s_1\}.
\]
The cutoff prevents division by numerically negligible singular values. This is a truncated numerical pseudoinverse; the numerical rank, and hence the fitted solution, depends on the stated tolerance. The real-world diagnostic table uses the same retained singular values as the solver, rather than a separate default rank threshold.

Ridge regression minimizes
\[
\|Xw-y\|_2^2+\alpha\|w\|_2^2,\qquad\alpha>0,
\]
and solves the linear system
\[
(X^\top X+\alpha I)\widehat w_\alpha=X^\top y.
\]
The code uses \texttt{numpy.linalg.solve}, not an explicit matrix inverse. The objective uses the sum of errors, so $\alpha$ is not the penalty coefficient of an objective normalized by $1/n$. Its spectral action replaces the factor $1/s_j$ by $s_j/(s_j^2+\alpha)$, limiting the amplification associated with small singular values. The implementation and normalization match the ridge formulation in the supplied course notes [3].

\textbf{Error metric.} Both train and test errors are means:
\[
\operatorname{MSE}(y,\widehat y)=\frac{1}{m}\sum_{i=1}^m(y_i-\widehat y_i)^2.
\]
Synthetic MSE is on the original label scale. BlogFeedback MSE is on $\log(1+y)$; centering does not change the error when the training mean is restored to predictions.

\textbf{Interpolation diagnostic.} The real-world implementation evaluates
\[
\rho_d=\frac{\|y-X_d\widehat w_d\|_2}
{\max(\|y\|_2,\epsilon_{\rm machine})}
\]
and records the first dimension with $\rho_d<10^{-8}$. Interpolation is therefore observed from the training residual, not inferred from a large parameter count or a test-error spike. Centering the real-world columns restricts their rank to at most $n-1$; a centered target can still be interpolated if it lies in their column space. Neither $d=n$ nor rank alone is a sufficient diagnostic in rank-deficient data.
''')
pages.append(r'''\pageheading{3. Datasets and evaluation protocol}
\textbf{Synthetic regression.} Each run draws independent standard Gaussian feature matrices with 150 training rows, 1,000 test rows, and 400 columns. Labels are generated once from all 400 features:
\[
y=X\beta+\varepsilon,\quad\varepsilon\sim\mathcal N(0,\sigma^2I),
\qquad \beta_j\propto s_j e^{-0.02j},\ j=0,\ldots,399,
\]
where $s_j\in\{-1,1\}$ are random signs and $\|\beta\|_2=1$. Thus the population signal variance is one. The main experiment uses $\sigma=0.5$, corresponding to noise variance 0.25.

The models take prefixes of the same matrices. Labels, noise, and coefficient vector do not change as $d$ varies within a run. Early features deliberately carry more signal, so adding features initially reduces omitted-variable error. Thirty seeds generate independent problems. Least squares and all four ridge strengths, $\alpha\in\{0.01,0.1,1,10\}$, use identical data within each seed. A dense set of dimensions near 150 resolves the interpolation spike; the sweep continues to 400.

\textbf{Real-world regression.} BlogFeedback [2] provides a training file with 52,397 observations and 280 predictors. The target is a future comment count. The experiment uses the final provided test file, dated 31 March 2012, containing 214 observations. This preserves the official separation of training and test files, while limiting the evaluation to one test date. The other 59 test files are not pooled.

For each run, 80 observations are sampled without replacement from the official training file. The target is transformed with $\log(1+y)$ because of its skew. Predictors are randomly ordered, standardized using only the selected training rows, and columns with zero training variance are removed. Targets are centered with the training mean; this is equivalent to fitting an unpenalized intercept. All preprocessing is applied unchanged to the held-out file.

\textbf{Why a small fixed sample?} The aim is to cross interpolation within the available feature set, rather than to train the most accurate predictor. An earlier $n=150$ setup remained rank deficient and did not interpolate. Reducing the fixed size to 80 made interpolation accessible. This was an exploratory design revision, not an independent confirmatory experiment. Seed 42 is retained for illustration; seeds 0--19 provide repeated diagnostics without replacing unfavorable runs.

Every usable dimension is evaluated in each real-world run. Unsupported dimensions are stored as missing values, not silently truncated. Aggregate plots use only dimensions available in all 20 runs, namely 1--149 in the verified execution. Ridge uses $\alpha=1$. The baseline predicts the mean of that run's training log-targets and never uses the test mean.
''')
pages.append(r'''\pageheading{4. Synthetic results: a clear interpolation spike}
'''+fig_s+r'''
{\small\textit{Figure 1.} Mean errors on the same 30 synthetic problems. The dashed line marks $d=n=150$. The log scale preserves the ridge curve without hiding the least-squares peak.}

\begin{center}\begin{tabular}{rrr}\toprule
$d$ & Mean least-squares MSE & Mean ridge MSE ($\alpha=1$)\\\midrule
'''+''.join(f'{d} & {fmt(mean[list(D).index(d)])} & {fmt(ridge[list(D).index(d)])} \\\\\n' for d in [10,60,150,200,400])+r'''\bottomrule\end{tabular}\end{center}
The least-squares curve initially falls, rises toward interpolation, and declines again as additional features are introduced. The mean test error at $d=150$ is about 1,477.95, compared with about 1.026 at $d=400$. Training error reaches numerical zero in the interpolating regime. This is a clear empirical instance of double descent under the specified data-generating process.

However, the best observed pre-interpolation mean is about 0.591 at $d=60$, below the final error. A second descent does not imply that the largest model is the best model. Once the informative early features have been included, adding further coordinates also changes the bias of the minimum-norm solution.

Ridge with $\alpha=1$ lowers mean error at $d=150$ to about 1.725. At $d=400$, ridge and least squares are nearly identical, about 1.024 and 1.026. Its largest benefit here is near the ill-conditioned interpolation transition. The four-strength sensitivity plot in the notebook shows that stronger regularization suppresses the spike more aggressively; those test curves are descriptive, not a tuning procedure.

Means near the interpolation threshold can be sensitive to extreme realizations. The notebook therefore also displays interquartile ranges. An empirical mean need not lie inside the middle-50\% band; the band is not a confidence interval for that mean.
''')
pages.append(r'''\pageheading{5. Real-world interpolation and an unstable illustration}
'''+fig_res+r'''
{\small\textit{Figure 2.} The relative training residual falls below $10^{-8}$ at $d=133$ in the seed-42 run. The dashed vertical line marks that observed event.}
'''+fig_r+r'''
{\small\textit{Figure 3.} Test errors for the same seed-42 run. The baseline is fitted only to the training labels; all errors use the log-target scale.}

The seed-42 subset retains 155 nonconstant predictors. Its final training MSE is approximately $4.12\times10^{-23}$, confirming numerical interpolation. Nevertheless, final test MSE is about $4.31\times10^7$ for least squares and 13.37 for ridge, versus 1.14 for the baseline. This run does not demonstrate a convincing second descent. It is retained because replacing it with a favorable seed would conceal important instability. Tiny training residuals alone do not establish good generalization.
''')
pages.append(r'''\pageheading{6. Real-world repetitions and baseline comparison}
'''+fig_rep+r'''
{\small\textit{Figure 4.} Medians with descriptive 25th--75th percentile bands. All curves use the same 20 runs and common dimensions. The gray line is the median of separately fitted baseline errors.}

All 20 repetitions reach interpolation, with first interpolating dimensions ranging from 80 to 93. Their feature sets differ because both the training subset and feature ordering change. The aggregate median curve has a substantial peak around this region, followed by a decline. It does not display a clean initial descending segment comparable to the synthetic curve.

\begin{center}\begin{tabular}{lr}\toprule
Quantity & Verified value\\\midrule
Median least-squares peak, $d=82$ & 6,063.209\\
Median least squares at common $d=149$ & 99.277\\
Median ridge at common $d=149$ & 7.903\\
Median training-mean baseline & 1.147\\
Ridge wins versus paired baseline at $d=149$ & 0 / 20\\
\bottomrule\end{tabular}\end{center}

The median least-squares decline from the peak is substantial, but its final error remains far above its low-dimensional error. Ridge reduces the instability without recovering baseline-level performance at the final common dimension. The paired comparison is stronger than simply comparing the two medians: every final ridge fit loses to the baseline fitted from its own training labels. This statement is specific to $d=149$, not to every dimension.

The IQR documents large variability. Separate mean curves in the notebook remain strongly affected by extreme least-squares fits; aggregating by the mean and by the median answers different questions. Because test observations are fixed across repetitions, these bands describe sensitivity to training subsets and feature ordering. They do not quantify uncertainty over future dates or independent test samples.
''')
pages.append(r'''\pageheading{7. Noise extension and comparison across datasets}
'''+fig_noise+r'''
{\small\textit{Figure 5.} The optional synthetic noise extension uses the same seed, feature matrices, coefficient vector and underlying standardized noise draws at each noise level. The displayed errors include test-label noise.}

With seed 42 fixed, the experiment repeats the synthetic sweep at $\sigma\in\{0.1,0.5,1\}$. At $d=150$, test MSE is approximately 7.71, 57.02 and 200.16 respectively. In this diagnostic, larger label noise makes the instability near interpolation more pronounced. The noise variance also changes the irreducible error floor, so the curve's vertical displacement cannot be interpreted as estimation error alone.

This is a single-seed extension. It supports a within-run comparison, but does not establish the same effect size across random datasets. Its seed-42 peak should not be compared numerically with the 30-run mean peak from Figure 1 as if both were the same statistic.

\begin{center}\begin{tabular}{p{0.25\linewidth}p{0.30\linewidth}p{0.30\linewidth}}\toprule
Aspect & Synthetic & Real world\\\midrule
Feature structure & Independent Gaussian & Correlated, sparse, ill-conditioned\\
Signal ordering & Early coordinates designed to matter & Random ordering after filtering\\
Test protocol & Independent same-process samples & One later official test date\\
Threshold & Clear at $d=n$ & Measured residual; varies by run\\
Outcome & Clear double descent & Peak and decline, mixed full-curve evidence\\
\bottomrule\end{tabular}\end{center}

The same solvers can behave very differently under different feature geometry and evaluation protocols. Raw synthetic MSE and log-target BlogFeedback MSE are not on comparable numerical scales. The useful comparison concerns curve shape, interpolation, instability, and response to regularization, rather than which dataset has a smaller MSE.
''')
pages.append(r'''\pageheading{8. Limitations, reproducibility, and conclusions}
\textbf{Numerical sensitivity.} The real-world least-squares fits depend on very small retained singular values. The shared $10^{-10}$ SVD cutoff makes rank reporting consistent with fitting, but it does not eliminate ill-conditioning. Floating-point libraries can produce slightly different results near the cutoff. Threshold and error magnitudes should therefore be interpreted with the stated tolerances, not as exact symbolic quantities.

\textbf{Exploratory protocol.} The reduction from 150 to 80 training observations was made after inspecting rank and residual behavior. The final results are an exploratory analysis of the revised setup; they are not an untouched-test estimate of a selected prediction procedure. No independent validation protocol is implemented to select $d$ or $\alpha$. The analysis does not establish that any chosen model would be optimal for future data.

\textbf{Data limitations.} The real-world study uses only a small subset of the available training data and a single held-out date. Correlation, sparse features, and temporal differences may contribute to the observed errors, but the experiment does not isolate their causal contributions. A multi-date evaluation, repeated noise extension, or comparison with a better-conditioned real-world dataset would provide additional evidence. None is claimed here.

\textbf{Reproducible workflow.} The repository provides an executable runner, pinned numerical dependencies, both executed notebooks, PNG/PDF exports, numerical JSON files, and data checksums. From the repository root, run:
\begin{verbatim}
python -m pip install -r requirements.txt
python src/run_experiments.py
\end{verbatim}
The verified runtime is Python 3.13.7, NumPy 2.4.6 and Matplotlib 3.11.0. The first run downloads the official dataset only if it is absent; a complete cached dataset permits offline execution. The runner sets the notebook working directory automatically and does not require Jupyter. Detailed environment and runtime information are stored in \texttt{results/environment.json}.

The standalone LaTeX source embeds vector plot coordinates derived from the same JSON evidence, so it has no external image dependency. The supplied source builder regenerates these coordinates after an experiment rerun. Standard \texttt{pdflatex} commands are documented in the README; no BibTeX pass is needed.

\textbf{Conclusions.} Three distinctions are central. First, sufficient parameter count must be separated from observed numerical interpolation. Second, a post-peak decline must be separated from a clean full double-descent curve. Third, improvement over unstable least squares must be separated from performance that beats a meaningful baseline. The synthetic results satisfy the first two strongly. The real-world results verify interpolation but provide mixed evidence for the full curve, and their final ridge fits fail the baseline comparison. Reporting these negative findings is essential to an accurate interpretation of the experiment.
''')
pages.append(r'''\pageheading{9. Sources and authorship}
\begin{enumerate}
\item M. Belkin, D. Hsu, S. Ma and S. Mandal (2019). Reconciling modern machine-learning practice and the classical bias--variance trade-off. \textit{Proceedings of the National Academy of Sciences}, 116(32), 15849--15854. \url{https://doi.org/10.1073/pnas.1903070116}.
\item K. Buza (2014). \textit{BlogFeedback} [Dataset]. UCI Machine Learning Repository. \url{https://doi.org/10.24432/C58S3F}. Dataset licensed under Creative Commons Attribution 4.0. The project uses the official archive and the attribution above.
\item N. Cesa-Bianchi. \textit{Machine learning course notes}, supplied course PDF. Relevant material: linear and ridge regression (PDF p. 40) and double descent (PDF p. 68).
\item \textit{Statistical Methods for Machine Learning: Experimental Projects, A.Y. 2025/26}, supplied project brief. Assignment 4 and general coding/report requirements (PDF pp. 1--3, 8--9, 16).
\end{enumerate}

\textbf{Assistance acknowledgment.} Codex assisted with code review, revisions, experiment execution, visualization, and drafting this report. Numerical results were checked against the executed code. The student must review the implementation, evidence, references, and all report text, and ensure that the use and acknowledgment of assistance comply with the course's rules.

\textbf{Required declaration: author review pending.} The following is the declaration reproduced from the project brief. It is included for the student to review; this draft does not sign or certify it on the student's behalf. Before submission, the student must determine whether the declaration is accurate, whether all assistance is adequately acknowledged, and whether any course-specific disclosure or approval is required.

\begin{quote}\small
I declare that this material, which I now submit for assessment, is entirely my own work and has not been taken from the work of others, save and to the extent that such work has been cited and acknowledged within the text of my work. I understand that plagiarism, collusion, and copying are grave and serious offences in the university and accept the penalties that would be imposed should I engage in plagiarism, collusion or copying. This assignment, or any part of it, has not been previously submitted by me or any other person for assessment on this or any other course of study.
\end{quote}

\textbf{Repository review.} The local package contains the report source, compiled PDF, reproducibility instructions, exported figures, numerical evidence, and executed code. Publishing the code/report and submitting the repository link are subsequent actions, as required by the project brief. They are not performed or certified by this report.
''')
tex=preamble+'\n\\newpage\n'.join(pages)+'\n\\end{document}\n'
# Basic provenance assertions protect hard-coded prose summaries from stale results.
assert R['interpolation_dimension']==133 and min(hit)==80 and max(hit)==93
assert int(DR[-1])==149 and int(DR[peak])==82
assert round(float(mean[best]),3)==.591
assert np.sum(all_ridge[:,-1]<np.array(R['all_real_baseline']))==0
(ROOT/'report/report.tex').write_text(tex)
print('Wrote standalone LaTeX source with',len(pages),'planned pages.')
