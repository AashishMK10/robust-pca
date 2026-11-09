"""Reproduce the experiments, figures, and IEEE research report."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Run the study end to end from any working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5, help="trials per condition")
    parser.add_argument("--size", type=int, default=100, help="square matrix dimension")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--skip-report", action="store_true", help="generate TeX data without compiling the PDF")
    args = parser.parse_args()
    if args.seed < 0 or args.trials < 1 or not 1 <= args.rank <= args.size:
        parser.error("require seed >= 0, trials >= 1, and 1 <= rank <= size")

    # Small dense SVDs are measured with a single BLAS thread for both methods.
    for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[variable] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))

    import numpy as np

    from src.experiments import REPRESENTATIVE_DENSITY, Settings, run_experiments
    from src.plotting import FIGURE_NAMES, make_figures
    from src.reporting import compile_report, read_csv, write_report_data

    results, figures, report = (ROOT / name for name in ("results", "figures", "report"))
    settings = Settings(size=args.size, rank=args.rank, seed=args.seed, trials=args.trials)
    summary, _ = run_experiments(settings, results)
    meta = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    history = [
        row for row in read_csv(results / "convergence.csv")
        if row["experiment"] == "density" and row["trial"] == "1"
        and float(row["density"]) == REPRESENTATIVE_DENSITY
    ]
    with np.load(results / "representative_trial.npz", allow_pickle=False) as components:
        make_figures(summary, dict(components), history, meta["pcp_tolerance"], figures)
    write_report_data(results, report)

    outputs = [
        results / name for name in (
            "metrics.csv", "summary.csv", "convergence.csv", "representative_trial.npz",
            "datasets.npz", "noise_metrics.csv", "noise_summary.csv", "metadata.json",
        )
    ]
    outputs.extend(figures / f"{name}.{extension}" for name in FIGURE_NAMES for extension in ("png", "pdf"))
    outputs.append(report / "generated_results.tex")
    if not args.skip_report and compile_report(report):
        outputs.append(report / "main.pdf")
    missing = [str(path) for path in outputs if not path.is_file() or not path.stat().st_size]
    if missing:
        raise FileNotFoundError("Missing or empty generated files: " + ", ".join(missing))
    print(f"Validated {len(outputs)} generated files under {ROOT}", flush=True)


if __name__ == "__main__":
    main()
