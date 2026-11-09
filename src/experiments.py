"""Run paired PCA/PCP evaluations and persist the measured data."""

import csv
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import matplotlib
import numpy as np
import scipy

from .classical_pca import truncated_svd
from .data_generation import SyntheticTrial, generate_trial
from .metrics import relative_frobenius, support_metrics
from .robust_pca import DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE, PENALTY_SCALE
from .robust_pca import principal_component_pursuit

DENSITIES = (0.0, 0.02, 0.05, 0.10, 0.20)
NOISE_LEVELS = (0.0, 0.1, 0.5)
REPRESENTATIVE_DENSITY = 0.10
MAGNITUDE = 10.0
THRESHOLD = 1.0


@dataclass(frozen=True)
class Settings:
    """Configuration shared by the density sweep and the noise extension."""

    size: int = 100
    rank: int = 5
    seed: int = 42
    trials: int = 5


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty sequence of uniform records with an explicit header."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(
    data: SyntheticTrial,
    settings: Settings,
    experiment: str,
    density: float,
    sigma: float,
    trial: int,
) -> tuple[dict, list[dict], dict]:
    key = {
        "experiment": experiment, "density": density, "sigma": sigma, "trial": trial,
    }
    start = perf_counter()
    pca = truncated_svd(data.observed, settings.rank)
    pca_seconds = perf_counter() - start
    start = perf_counter()
    fit = principal_component_pursuit(data.observed)
    pcp_seconds = perf_counter() - start

    actual = data.sparse != 0
    predicted = np.abs(fit.sparse) > THRESHOLD
    scores = support_metrics(actual, predicted)
    final = fit.history[-1]
    metrics = key | {
        "realized_density": float(actual.mean()),
        "pca_error": relative_frobenius(pca, data.low_rank),
        "pcp_error": relative_frobenius(fit.low_rank, data.low_rank),
        "sparse_error": relative_frobenius(fit.sparse, data.sparse) if actual.any() else np.nan,
        "pca_seconds": pca_seconds,
        "pcp_seconds": pcp_seconds,
        "pcp_rank": fit.rank,
        "penalty": fit.penalty,
        "regularization": fit.regularization,
        "converged": fit.converged,
        "iterations": final.iteration,
        "relative_primal": final.relative_primal,
        "relative_dual": final.relative_dual,
    } | asdict(scores)
    if not fit.converged:
        print(
            f"WARNING: {experiment}, p={density:g}, sigma={sigma:g}, trial {trial} "
            f"reached {final.iteration} iterations "
            f"(primal={final.relative_primal:.2e}, dual={final.relative_dual:.2e})",
            flush=True,
        )
    history = [key | asdict(step) for step in fit.history]
    components = {
        "l0": data.low_rank, "s0": data.sparse, "observed": data.observed,
        "l_pca": pca, "l_pcp": fit.low_rank, "s_pcp": fit.sparse,
        "actual": actual, "predicted": predicted,
    }
    return metrics, history, components


def summarize(rows: list[dict], condition: str) -> list[dict]:
    """Summarize all runs, including any last iterates that hit the cap."""
    output = []
    for level in sorted({row[condition] for row in rows}):
        group = [row for row in rows if row[condition] == level]
        record = {condition: level, "trials": len(group)}
        for method in ("pca", "pcp"):
            errors = [row[f"{method}_error"] for row in group]
            record[f"{method}_error_mean"] = mean(errors)
            record[f"{method}_error_sd"] = stdev(errors) if len(errors) > 1 else np.nan
            record[f"{method}_error_median"] = median(errors)
            record[f"{method}_seconds_mean"] = mean(row[f"{method}_seconds"] for row in group)
        for metric in ("precision", "recall", "f1", "fpr", "sparse_error"):
            defined = [row[metric] for row in group if np.isfinite(row[metric])]
            record[f"{metric}_mean"] = mean(defined) if defined else np.nan
        record |= {
            "pcp_wins": sum(row["pcp_error"] < row["pca_error"] for row in group),
            "converged_count": sum(row["converged"] for row in group),
            "convergence_rate": mean(float(row["converged"]) for row in group),
            "iterations_mean": mean(row["iterations"] for row in group),
            "iterations_min": min(row["iterations"] for row in group),
            "iterations_max": max(row["iterations"] for row in group),
            "pcp_rank_mean": mean(row["pcp_rank"] for row in group),
            "false_positives": sum(row["fp"] for row in group),
            "false_negatives": sum(row["fn"] for row in group),
        }
        output.append(record)
    return output


def _processor_name() -> str:
    if platform.system() == "Windows":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def run_experiments(settings: Settings, destination: Path) -> tuple[list[dict], list[dict]]:
    """Run the full study and write trial data, metadata, and summaries."""
    destination.mkdir(parents=True, exist_ok=True)
    main_seed, noise_seed = np.random.SeedSequence(settings.seed).spawn(2)
    rows = []
    history = []
    datasets = {key: [] for key in ("l0", "s0", "observed")}
    representative = {}

    def record(data: SyntheticTrial, experiment: str, density: float, sigma: float, trial: int) -> None:
        metrics, steps, components = _evaluate(data, settings, experiment, density, sigma, trial)
        rows.append(metrics)
        history.extend(steps)
        for key in datasets:
            datasets[key].append(components[key])
        if experiment == "density" and density == REPRESENTATIVE_DENSITY and trial == 1:
            representative.update(components, density=density, trial=trial)

    shape_and_rank = (settings.size, settings.size, settings.rank)
    for density, level_seed in zip(DENSITIES, main_seed.spawn(len(DENSITIES)), strict=True):
        for trial, child in enumerate(level_seed.spawn(settings.trials), 1):
            data = generate_trial(np.random.default_rng(child), *shape_and_rank, density, MAGNITUDE)
            record(data, "density", density, 0.0, trial)
        print(f"Density sweep: completed p={density:.0%} ({settings.trials} trials)", flush=True)

    for trial, child in enumerate(noise_seed.spawn(settings.trials), 1):
        rng = np.random.default_rng(child)
        density = REPRESENTATIVE_DENSITY
        data = generate_trial(rng, *shape_and_rank, density, MAGNITUDE)
        noise = rng.standard_normal(data.observed.shape)
        for sigma in NOISE_LEVELS:
            noisy = SyntheticTrial(data.low_rank, data.sparse, data.observed + sigma * noise)
            record(noisy, "noise", density, sigma, trial)
        print(f"Dense noise extension: completed paired trial {trial}/{settings.trials}", flush=True)

    main = [row for row in rows if row["experiment"] == "density"]
    noise = [row for row in rows if row["experiment"] == "noise"]
    summary = summarize(main, "density")
    noise_summary = summarize(noise, "sigma")
    for name, records in (
        ("metrics", main), ("summary", summary), ("convergence", history),
        ("noise_metrics", noise), ("noise_summary", noise_summary),
    ):
        write_csv(destination / f"{name}.csv", records)

    np.savez_compressed(destination / "representative_trial.npz", **representative)
    np.savez_compressed(
        destination / "datasets.npz",
        **{key: np.stack(matrices) for key, matrices in datasets.items()},
        **{key: np.array([row[key] for row in rows]) for key in ("experiment", "density", "sigma", "trial")},
    )
    metadata = asdict(settings) | {
        "densities": DENSITIES, "noise_levels": NOISE_LEVELS,
        "noise_density": REPRESENTATIVE_DENSITY,
        "corruption_magnitude": MAGNITUDE, "support_threshold": THRESHOLD,
        "pcp_regularization": 1 / np.sqrt(settings.size), "pcp_penalty_scale": PENALTY_SCALE,
        "pcp_tolerance": DEFAULT_TOLERANCE, "pcp_max_iterations": DEFAULT_MAX_ITERATIONS,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__, "system": f"{platform.system()} {platform.release()}",
        "processor": _processor_name(), "blas_threads_requested": 1,
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary, noise_summary
