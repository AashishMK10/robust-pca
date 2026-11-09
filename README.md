# Robust Low-Rank Matrix Recovery Under Sparse Corruptions

This Pattern Recognition and Machine Learning project compares truncated singular value decomposition (SVD) with principal component pursuit (PCP) for recovering a low-rank matrix from sparse, high-magnitude errors. The study measures reconstruction accuracy, corruption detection, and optimization cost. A paired extension measures the effect of dense Gaussian noise.

## Reproduce the study

Use Python 3.11 or newer. From the project directory, create an environment and install the dependencies.

**Windows PowerShell**

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe run_all.py
    .\.venv\Scripts\python.exe -m pytest -q

**macOS or Linux**

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -r requirements.txt
    python run_all.py
    python -m pytest -q

The entry point is **`python run_all.py`** in any environment with the requirements installed. It runs 25 main trials and 15 noise trials with the default settings, writes the CSV and NPZ data, exports four figures as 320-DPI PNGs and PDFs, generates the LaTeX results macros, and compiles the report when a supported engine is on `PATH`. Use `--skip-report` to generate the data, figures, and LaTeX macros without compiling. The `--seed`, `--trials`, `--size`, and `--rank` options change the study configuration. Each run writes to the project’s result and figure paths. Same-seed runs with the same software and numerical backend reproduce the numerical data apart from wall-clock timings.

### Compile the report manually

Install a LaTeX distribution with IEEEtran, BibTeX, and the packages used in `report/main.tex`, or install Tectonic with its TeX bundle available. Run these commands from the project root.

    cd report
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

With Tectonic, run `tectonic main.tex` from the same directory. A first build may fetch TeX resources. The compiled paper is `report/main.pdf`. Edit the author, affiliation, location, and email placeholders in `report/main.tex` before submission. The run regenerates `report/generated_results.tex` from the CSVs. Use the same run for the figures, macros, and compiled PDF.

## Model and evaluation

The signal is $L_0=AB^\top/\sqrt r$, where $A$ and $B$ have independent standard normal entries. We observe $M=L_0+S_0$, with exactly $\operatorname{round}(pmn)$ uniformly sampled corrupted entries of independent amplitude $+10$ or $-10$. Defaults are $m=n=100$, $r=5$, seed 42, and five independent trials each at 0%, 2%, 5%, 10%, and 20% corruption. The two methods share the same observation within each trial. NumPy `SeedSequence` spawns separate streams for the main study and extension, then separate streams for the individual trials.

- **Truncated SVD** keeps the top $r$ singular components of the raw matrix. It minimizes the Frobenius residual to $M$ under the rank bound. Mean-centered PCA would subtract column means first and add them back to reconstruct the data. The uncentered baseline matches the raw matrix recovery objective.
- **Principal component pursuit** minimizes $\|L\|_*+\lambda\|S\|_1$ subject to $L+S=M$, where $\lambda=1/\sqrt{\max(m,n)}$. The scaled ADMM solver alternates singular-value shrinkage and entrywise soft thresholding, then updates the multiplier on $M-L-S$. Its fixed penalty is $0.3\sqrt{mn}/\|M\|_F$ for nonzero input. The solver requires normalized primal feasibility and dual stationarity residuals at most $10^{-7}$ and caps runs at 5,000 iterations. Capped runs are flagged and included in summaries.
- **Scores** include $\|\hat L-L_0\|_F/\|L_0\|_F$, runtime, and support metrics using $|\hat S_{ij}|>1$. The cutoff is specific to the synthetic error magnitude. Recall and F1 are undefined on empty true support, precision is undefined with zero predictions, and relative sparse error is undefined for zero sparse truth. CSVs record undefined values as `nan`. Means exclude undefined scores. Standard deviations are sample standard deviations.
- **Noise extension** adds $\sigma Z$ at 10% corruption, with $\sigma\in\{0,0.1,0.5\}$. Within each extension trial, all three noise levels share $L_0$, $S_0$, and the standard normal draw $Z$. The same equality-constrained solver runs at every level.

## Measured findings

The following relative reconstruction errors are means from the included default run.

| Corruption | Truncated SVD | PCP |
|---:|---:|---:|
| 0% | $1.82\times10^{-15}$ | $1.78\times10^{-8}$ |
| 2% | 0.498 | $3.63\times10^{-8}$ |
| 5% | 0.866 | $3.73\times10^{-8}$ |
| 10% | 1.373 | $6.11\times10^{-8}$ |
| 20% | 1.941 | $1.24\times10^{-7}$ |

PCP had lower error in all 20 trials with gross corruptions. It found all 18,500 corrupted entries with zero false positives and zero false negatives at the fixed threshold. The clean trials also produced zero false alarms. Both residual criteria passed in all 25 main trials, taking 2–40 iterations. Truncated SVD achieved near machine precision in the clean control and used one decomposition per trial.

At dense-noise standard deviations of 0.1 and 0.5, mean PCP reconstruction errors were 0.0752 and 0.359, respectively. At 0.5, mean support F1 was 0.914, with 946 false positives across five trials. All 15 extension fits met the residual criteria. Equality-constrained PCP can attain numerical feasibility by allocating dense noise to both fitted components. Stable PCP instead allows a bounded observation residual.

## Files and figures

| Path | Contents |
|---|---|
| `run_all.py`, `requirements.txt` | Reproduction entry point and dependencies |
| `src/` | Data generator, two solvers, metrics, experiments, plotting, and report build |
| `tests/` | Unit tests for generation, solvers, and metrics |
| `results/metrics.csv`, `results/summary.csv` | Trial-level density results and condition aggregates |
| `results/noise_metrics.csv`, `results/noise_summary.csv` | Trial-level dense-noise results and aggregates |
| `results/convergence.csv` | Both normalized residuals at each iteration, with experiment and trial identifiers |
| `results/datasets.npz` | Aligned stacks of `l0`, `s0`, and `observed`, keyed by `experiment`, `density`, `sigma`, and `trial` |
| `results/representative_trial.npz` | Truth, observation, estimates, and both masks for trial 1 at 10% corruption |
| `results/metadata.json` | Configuration and measured software and processor details |
| `report/` | IEEEtran source, BibTeX, generated data macros, and compiled paper |

Each figure is in `figures/` with both a `.png` and `.pdf` extension.

| Figure | Contents |
|---|---|
| `reconstruction_error` | Density sweep with mean relative error and sample standard deviation |
| `matrix_recovery` | Clean matrix, observation, sparse components, both reconstructions, and signed errors |
| `corruption_detection` | True and predicted masks, false positives, and false negatives |
| `convergence` | Normalized primal and dual residuals against iteration for the 10% trial |

## Scope

These results apply to small synthetic Gaussian-factor matrices and uniformly scattered errors of one magnitude. The comparison gives SVD the true rank. In real data, coherent signal subspaces, structured corruption, or dense noise can undermine the PCP separation. The detector’s threshold and the solver’s regularization and numerical tolerances affect the observed trade-offs. Repeated full SVDs increase PCP's computation time, especially as matrix dimensions grow. Five trials per condition support descriptive comparisons.
