import argparse
import os
import sys
from pathlib import Path
from typing import Callable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    v_measure_score
)

REGRESSION_METRICS: dict[str, Callable] = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}

CLASSIFICATION_METRICS: dict[str, Callable] = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

CLUSTERING_METRICS: dict[str, Callable] = {
    "silhouette": silhouette_score,
    "v_measure": v_measure_score,
}


def _auto_discover_paths(dataset_name: str, dataset_source: str) -> dict[str, str]:
    base = Path(__file__).resolve().parent.parent / "datasets"

    if dataset_source == "auto":
        openml_path = base / "openml" / dataset_name
        synthetic_path = base / "synthetic" / dataset_name
        # Check for actual data files, not just directory existence
        openml_has_data = openml_path.is_dir() and (openml_path / "test_y.csv").exists()
        synthetic_has_data = synthetic_path.is_dir() and (synthetic_path / "test_y.csv").exists()
        
        if openml_has_data:
            dataset_source = "openml"
        elif synthetic_has_data:
            dataset_source = "synthetic"
        else:
            print(f"[-] Error: Dataset '{dataset_name}' not found in openml/ or synthetic/")
            sys.exit(1)

    dset_dir = base / dataset_source / dataset_name
    preds_dir = dset_dir / "predictions"

    # Prefer .npy (no header issues), fallback to CSV
    def prefer_npy(path: Path) -> str:
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            return str(npy_path)
        return str(path)

    return {
        "ground_truth": prefer_npy(dset_dir / "test_y.csv"),
        "cpp_predictions": prefer_npy(preds_dir / "cpp-predictions.csv"),
        "python_predictions": prefer_npy(preds_dir / "python-predictions.csv"),
    }


def _validate_path(path: str, label: str) -> None:
    if not os.path.isfile(path):
        print(f"[-] Error: {label} file not found: {path}")
        sys.exit(1)


def _load(path: str) -> np.ndarray:
    """Load from .npy (preferred) or .csv with skip_header=1."""
    if path.endswith(".npy"):
        data = np.load(path)
    else:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 0:
        data = np.array([data])
    return data.reshape((-1, 1))


def _run_side_by_side(
    metrics: dict[str, Callable],
    test_y: np.ndarray,
    cpp_preds: np.ndarray,
    python_preds: np.ndarray | None,
    spec_metric: str | None,
) -> None:
    metric_names = [spec_metric] if spec_metric else list(metrics.keys())

    header = f"{'Metric':<20} {'Python (sklearn)':<20} {'C++ (custom)':<20} {'Delta':<20}"
    print(header)
    print("-" * len(header))

    for name in metric_names:
        if name not in metrics:
            print(f"[-] Error: Unknown metric '{name}'")
            sys.exit(1)

        cpp_val = float(metrics[name](test_y, cpp_preds))

        if python_preds is not None:
            py_val = float(metrics[name](test_y, python_preds))
            delta = cpp_val - py_val
            print(f"{name:<20} {py_val:<20.6f} {cpp_val:<20.6f} {delta:<+20.6f}")
        else:
            print(f"{name:<20} {'N/A':<20} {cpp_val:<20.6f} {'N/A':<20}")


def _plot_side_by_side(
    metrics: dict[str, Callable],
    test_y: np.ndarray,
    cpp_preds: np.ndarray,
    python_preds: np.ndarray | None,
    spec_metric: str | None,
    metric_type: str,
    save_dir: str,
) -> None:
    metric_names = [spec_metric] if spec_metric else list(metrics.keys())

    cpp_values = []
    py_values = []
    for name in metric_names:
        cpp_values.append(float(metrics[name](test_y, cpp_preds)))
        if python_preds is not None:
            py_values.append(float(metrics[name](test_y, python_preds)))
        else:
            py_values.append(None)

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(metric_names) * 2.5), 5))
    bars_py = ax.bar(x - width / 2, [v if v is not None else 0 for v in py_values], width,
                     label="Python (sklearn)", color="#4C72B0")
    bars_cpp = ax.bar(x + width / 2, cpp_values, width,
                      label="C++ (custom)", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title(f"{metric_type.title()} Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    def _annotate_bars(bars, values):
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    _annotate_bars(bars_py, py_values)
    _annotate_bars(bars_cpp, cpp_values)

    save_path = os.path.join(save_dir, f"{metric_type}-metrics-comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Plot saved to: {save_path}")


def main() -> None:
    argp = argparse.ArgumentParser(description="Side-by-side metric comparison of sklearn vs C++ models")
    argp.add_argument("--metric-type", required=True, dest="metric_type")
    argp.add_argument("--dataset-name", dest="dataset_name", default=None)
    argp.add_argument("--dataset-source", dest="dataset_source",
                      choices=["openml", "synthetic", "auto"], default="auto")
    argp.add_argument("--ground-truth", dest="ground_truth", default=None)
    argp.add_argument("--cpp-predictions", dest="cpp_predictions", default=None)
    argp.add_argument("--python-predictions", dest="python_predictions", default=None)
    argp.add_argument("--run-all-metrics", action="store_true", dest="all_metrics")
    argp.add_argument("--run-spec-metric", action="store_true", dest="spec_metrics")
    argp.add_argument("--spec-metric", dest="spec_metric", default=None)
    argp.add_argument("--plot", action="store_true", dest="generate_plot")

    args = argp.parse_args()

    if args.metric_type == "regression":
        metrics = REGRESSION_METRICS
    elif args.metric_type == "classification":
        metrics = CLASSIFICATION_METRICS
    elif args.metric_type == "clustering":
        metrics = CLUSTERING_METRICS
    else:
        print(f"[-] Error: Unknown metric type '{args.metric_type}'")
        sys.exit(1)

    if args.dataset_name:
        paths = _auto_discover_paths(args.dataset_name, args.dataset_source)
        gt_path = paths["ground_truth"]
        cpp_path = paths["cpp_predictions"]
        py_path = paths["python_predictions"]
    else:
        gt_path = args.ground_truth
        cpp_path = args.cpp_predictions
        py_path = args.python_predictions

    if not gt_path:
        print("[-] Error: No ground truth path. Provide --dataset-name or --ground-truth")
        sys.exit(1)

    _validate_path(gt_path, "Ground truth")
    _validate_path(cpp_path, "C++ predictions")

    test_y = _load(gt_path)
    cpp_preds = _load(cpp_path)
    python_preds = _load(py_path) if py_path and os.path.isfile(py_path) else None

    spec_metric = args.spec_metric if args.spec_metrics else None
    _run_side_by_side(metrics, test_y, cpp_preds, python_preds, spec_metric)

    if args.generate_plot:
        if args.dataset_name:
            save_dir = os.path.dirname(cpp_path)
        else:
            save_dir = "."
        _plot_side_by_side(metrics, test_y, cpp_preds, python_preds, spec_metric,
                           args.metric_type, save_dir)


if __name__ == "__main__":
    main()
