"""Run the project's plain-Python notebook cells and persist reproducible outputs.

Usage: python src/run_experiments.py [--notebook all|synthetic|real]
No notebook execution library or IPython extension is required.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
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
    elapsed = time.perf_counter() - start
    result = {key: env[key] for key in keys}
    result["runtime_seconds"] = elapsed
    result["notebook_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
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
    manifest = []
    for notebook_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        document = json.loads(notebook_path.read_text())
        for index, cell in enumerate(document["cells"], start=1):
            for stem in re.findall(r'save_and_show\("([^"\n]+)"\)', "".join(cell["source"])):
                exports = sorted((ROOT / "figures").glob(stem + ".*"))
                exports += sorted((ROOT / "figures").glob(stem + "_[0-9]*.*"))
                manifest.append({"notebook": notebook_path.name, "cell": index,
                                 "stem": stem, "exports": [str(p.relative_to(ROOT)) for p in exports]})
    (ROOT / "results" / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Hash only the actual training and held-out test data used, not other dates.
    data_dir = ROOT / "data" / "blogfeedback"
    checksums = {}
    for name in ["blogData_train.csv", "blogData_test-2012.03.31.01_00.csv"]:
        path = data_dir / name
        if path.exists():
            checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__, "platform": platform.platform(),
        "timings_seconds": timings, "dataset_sha256": checksums,
        "command": "python src/run_experiments.py " + " ".join(sys.argv[1:]),
    }
    (ROOT / "results" / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
