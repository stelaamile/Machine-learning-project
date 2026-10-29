"""Run both notebooks from top to bottom and save their outputs.

The script displays no windows. It saves figures, executed notebooks, result
arrays and a short environment record. Run it from the repository root with
python src/run_experiments.py; Jupyter is not required.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def run_notebook(path: Path, kind: str):
    notebook = json.loads(path.read_text())
    env = {"__name__": "__main__"}
    original_show = plt.show
    current_outputs = []

    def capture_show(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = io.BytesIO()
            plt.figure(number).savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            current_outputs.append({
                "output_type": "display_data", "metadata": {},
                "data": {"image/png": base64.b64encode(buffer.getvalue()).decode(),
                         "text/plain": ["<Matplotlib figure>"]},
            })

    plt.show = capture_show
    previous_cwd = Path.cwd()
    start = time.perf_counter()
    try:
        os.chdir(path.parent)
        count = 0
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            current_outputs = []
            stdout, stderr = io.StringIO(), io.StringIO()
            source = "".join(cell["source"])
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell-{index + 1}", "exec"), env)
            if stdout.getvalue():
                current_outputs.insert(0, {"output_type": "stream", "name": "stdout",
                                           "text": stdout.getvalue().splitlines(keepends=True)})
            if stderr.getvalue():
                current_outputs.append({"output_type": "stream", "name": "stderr",
                                        "text": stderr.getvalue().splitlines(keepends=True)})
            cell["outputs"] = current_outputs
            cell["execution_count"] = count
            print(f"{path.name}: executed cell {index + 1}", flush=True)
    finally:
        plt.show = original_show
        plt.close("all")
        os.chdir(previous_cwd)
    # Do not overwrite the saved notebook until every cell has succeeded.
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    if kind == "synthetic":
        keys = ["N_TRAIN", "N_TEST", "D_MAX", "NOISE_STD", "D_VALUES", "N_REPETITIONS",
                "RIDGE_ALPHAS", "all_train_errors", "all_test_errors", "all_ridge_train",
                "all_ridge_test", "mean_test_errors", "mean_ridge_test"]
    else:
        keys = ["N_TRAIN_REAL", "D_EFFECTIVE", "D_VALUES_REAL", "real_train_errors",
                "real_test_errors", "real_ridge_train_errors", "real_ridge_test_errors",
                "real_relative_residuals", "interpolation_dimension", "real_baseline_test_mse",
                "D_VALUES_REPEAT", "D_VALUES_REPEAT_FULL", "real_repetition_results",
                "all_real_baseline", "mean_real_baseline", "median_real_baseline",
                "median_real_ls", "median_real_ridge", "mean_real_ls", "mean_real_ridge",
                "N_REAL_REPETITIONS", "D_VALUES_NOISE", "NOISE_LEVELS", "noise_results"]
    extension_key = "gd_results" if kind == "synthetic" else "ridge_cv_results"
    extension_file = "gradient_descent.json" if kind == "synthetic" else "ridge_cv.json"
    (ROOT / "results" / extension_file).write_text(
        json.dumps(serializable(env[extension_key]), indent=2, allow_nan=False) + "\n")
    elapsed = time.perf_counter() - start
    result = {key: env[key] for key in keys}
    result["runtime_seconds"] = elapsed
    (ROOT / "results" / f"{kind}.json").write_text(
        json.dumps(serializable(result), indent=2, allow_nan=False) + "\n")
    print(f"Finished {kind} in {elapsed:.1f} seconds", flush=True)
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=["all", "synthetic", "real"], default="all")
    args = parser.parse_args()
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    targets = {"synthetic": "01_synthetic_double_descent.ipynb",
               "real": "02_real_world_double_descent.ipynb"}
    timings = {}
    for kind, name in targets.items():
        if args.notebook in ("all", kind):
            timings[kind] = run_notebook(ROOT / "notebooks" / name, kind)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "timings_seconds": timings,
        "command": "python src/run_experiments.py " + " ".join(sys.argv[1:]),
    }
    (ROOT / "results" / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
