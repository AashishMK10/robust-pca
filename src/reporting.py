"""Bind saved results into LaTeX and compile the report when possible."""

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a results CSV without inferring or discarding missing values."""
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _number(value: float | str, digits: int = 3) -> str:
    value = float(value)
    if math.isnan(value):
        return r"\text{--}"
    if 0 < abs(value) < 0.001:
        mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
        return rf"{mantissa}\times10^{{{int(exponent)}}}"
    return f"{value:.{digits}f}"


def _tex_text(value: str) -> str:
    return value.translate(str.maketrans({
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }))


def write_report_data(results: Path, report: Path) -> None:
    """Generate table rows and prose macros directly from persisted results."""
    meta = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    summary = read_csv(results / "summary.csv")
    noise_summary = read_csv(results / "noise_summary.csv")
    trials = read_csv(results / "metrics.csv")
    clean = summary[0]
    endpoint = summary[-1]
    noisy = noise_summary[-1]
    corrupt = [row for row in trials if int(row["tp"]) + int(row["fn"]) > 0]
    representative = next(row for row in trials if float(row["density"]) == 0.10 and row["trial"] == "1")
    gap = [float(row["pca_error_mean"]) - float(row["pcp_error_mean"]) for row in summary]
    true_detections = sum(int(row["tp"]) for row in corrupt)
    true_corruptions = sum(int(row["tp"]) + int(row["fn"]) for row in corrupt)
    false_detections = sum(int(row["fp"]) for row in corrupt)
    detections = true_detections + false_detections

    macros = {
        "MatrixSize": meta["size"], "TrueRank": meta["rank"], "Seed": meta["seed"],
        "Trials": meta["trials"], "Magnitude": f'{meta["corruption_magnitude"]:g}',
        "Threshold": f'{meta["support_threshold"]:g}',
        "LambdaValue": _number(meta["pcp_regularization"]),
        "PenaltyScale": f'{meta["pcp_penalty_scale"]:g}',
        "StopTolerance": _number(meta["pcp_tolerance"], 1),
        "IterationCap": meta["pcp_max_iterations"],
        "MainTrialCount": len(trials), "CorruptTrialCount": len(corrupt),
        "CorruptWins": sum(float(row["pcp_error"]) < float(row["pca_error"]) for row in corrupt),
        "GapTrend": "increases at every density step"
        if all(a < b for a, b in zip(gap, gap[1:])) else "varies across the density sweep",
        "MainConverged": sum(int(row["converged_count"]) for row in summary),
        "NoiseConverged": sum(int(row["converged_count"]) for row in noise_summary),
        "NoiseTrials": sum(int(row["trials"]) for row in noise_summary),
        "CleanPca": _number(clean["pca_error_mean"]),
        "CleanPcp": _number(clean["pcp_error_mean"]),
        "EndPca": _number(endpoint["pca_error_mean"]),
        "EndPcp": _number(endpoint["pcp_error_mean"]),
        "MainMinIterations": min(int(row["iterations"]) for row in trials),
        "MainMaxIterations": max(int(row["iterations"]) for row in trials),
        "MeanPcaMs": _number(1000 * mean(float(row["pca_seconds"]) for row in trials), 2),
        "MeanPcpMs": _number(1000 * mean(float(row["pcp_seconds"]) for row in trials), 2),
        "CleanFalsePos": clean["false_positives"],
        "CorruptTotalPos": true_corruptions,
        "CorruptFalsePos": false_detections,
        "CorruptFalseNeg": true_corruptions - true_detections,
        "PooledPrecision": _number(true_detections / detections if detections else math.nan),
        "PooledRecall": _number(true_detections / true_corruptions if true_corruptions else math.nan),
        "RepPca": _number(representative["pca_error"]),
        "RepPcp": _number(representative["pcp_error"]),
        "RepRank": representative["pcp_rank"],
        "RepIterations": representative["iterations"],
        "RepStatus": "converged" if representative["converged"] == "True" else "reached the iteration cap",
        "RepPrimal": _number(representative["relative_primal"]),
        "RepDual": _number(representative["relative_dual"]),
        "RepTp": representative["tp"], "RepFp": representative["fp"], "RepFn": representative["fn"],
        "NoisePcp": _number(noisy["pcp_error_mean"]),
        "NoiseFalsePos": noisy["false_positives"],
        "NoiseFalseNeg": noisy["false_negatives"],
        "NoiseMildIterations": _number(noise_summary[1]["iterations_mean"], 1),
        "NoiseHighIterations": _number(noisy["iterations_mean"], 1),
        "Environment": _tex_text(
            f'{meta["system"]} on an {meta["processor"]} CPU. '
            f'Python {meta["python"]}, NumPy {meta["numpy"]}, SciPy {meta["scipy"]}, '
            f'and Matplotlib {meta["matplotlib"]}.'
        ),
    }
    lines = ["% Generated by run_all.py from results/. Edits will be overwritten."]
    lines.extend(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items())

    errors, diagnostics, noise = [], [], []
    for row in summary:
        percent = rf'{100 * float(row["density"]):g}\%'
        entries = []
        for method in ("pca", "pcp"):
            avg = _number(row[f"{method}_error_mean"])
            sd = _number(row[f"{method}_error_sd"])
            cell = f"{avg} \\pm {sd}" if row[f"{method}_error_sd"] != "nan" else avg
            entries.append("$" + cell + "$")
        errors.append(" & ".join([percent, *entries]))
        iterations = f'{float(row["iterations_mean"]):.1f} [{row["iterations_min"]}, {row["iterations_max"]}]'
        diagnostics.append(" & ".join([
            percent, "$" + _number(row["f1_mean"]) + "$", f'{row["converged_count"]}/{row["trials"]}',
            iterations, f'{1000 * float(row["pcp_seconds_mean"]):.1f}',
        ]))
    for row in noise_summary:
        noise.append(" & ".join([
            f'{float(row["sigma"]):.1f}', "$" + _number(row["pca_error_mean"]) + "$",
            "$" + _number(row["pcp_error_mean"]) + "$",
            f'{float(row["pcp_rank_mean"]):.1f}', "$" + _number(row["f1_mean"]) + "$",
        ]))
    for name, rows in (("ErrorRows", errors), ("DiagnosticRows", diagnostics), ("NoiseRows", noise)):
        lines.append(rf"\newcommand{{\{name}}}{{%")
        lines.extend(row + r" \\" for row in rows)
        lines.append("}")
    (report / "generated_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_report(report: Path) -> bool:
    """Compile IEEEtran with a supported engine on PATH and retain the final PDF."""
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    tectonic = shutil.which("tectonic")
    build = report / ".build"
    engine_options = ["-interaction=nonstopmode", "-halt-on-error", "-output-directory=.build"]
    if latexmk:
        commands = [[latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=.build", "main.tex"]]
    elif pdflatex and bibtex:
        latex = [pdflatex, *engine_options, "main.tex"]
        commands = [latex, [bibtex, ".build/main"], latex, latex]
    elif tectonic:
        commands = [[tectonic, "--keep-logs", "--outdir", ".build", "main.tex"]]
    else:
        print("WARNING: Install latexmk, pdfLaTeX with BibTeX, or Tectonic to compile the report.", flush=True)
        return False

    build.mkdir(exist_ok=True)
    for command in commands:
        subprocess.run(command, cwd=report, check=True)
    log = (build / "main.log").read_text(encoding="utf-8", errors="replace").lower()
    if any(message in log for message in ("undefined references", "undefined citations", "multiply defined")):
        raise RuntimeError(f"Report has unresolved or ambiguous cross-references. See {build / 'main.log'}")
    shutil.copy2(build / "main.pdf", report / "main.pdf")
    return True
