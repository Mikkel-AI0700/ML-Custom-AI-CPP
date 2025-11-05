import argparse
from typing import Callable
import numpy as np
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

def _determine_run_type (
    will_run_all: bool = False,
    will_run_spec: bool = False,
    spec_metric: str,
    metrics_references_dict: dict[str, Callable],
    test_y: np.ndarray,
    predictions: np.ndarray
):
    if will_run_all:
        _run_all_metrics(metrics_references_dict, test_y, predictions)

    if will_run_spec:
        _run_individual(metrics_references_dict, spec_metric, test_y, predictions)

def _run_all_metrics (
    metrics_references_dict: dict[str, Callable],
    test_y: np.ndarray,
    predictions: np.ndarray
):
    for (metric_name, metric_reference) in metrics_references_dict.keys():
        print(f"[+] {metric_name} -> {metric_reference(test_y, predictions)}")

def _run_individual (
    metrics_references_dict: dict[str, Callable],
    spec_metric: str,
    test_y: np.ndarray,
    predictions: np.ndarray
):
    try:
        if spec_metric not in metrics_references_dict.keys():
            raise ValueError("[-] Error: User selected metric doesn't exist")

        print(f"{spec_metric} -> {metrics_references_dict.get(spec_metric)(test_y, predictions)}")
    except ValueError as non_existent_metric:
        print(non_existent_metric)
        exit(1)

def main ():
    regressor_metrics = {
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "r2": r2_score
    }

    classification_metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "fi": f1_score
    }

    clustering_metrics = {
        "silhouette": silhouette_score,
        "vs": v_measure_score
    }

    argp = argparse.ArgumentHelper(description="Evaluating the metrics of C++ ML models")
    argp.add_argument("--metric-type", required=True, action="store_true", dest="metric_type")
    argp.add_argument("--run-all-metrics", required=False, action="store_true", dest="all_metrics")
    argp.add_argument("--run-spec-metric", required=False, action="store_true", dest="spec_metrics")
    argp.add_argument("--spec-metric", required=False, dest="spec_metric")
    argp.add_argument("--train-data", required=True, dest="train_y")
    argp.add_argument("--predictions", required=True, dest="predictions")

    try:
        args = argp.parse_args()

        if not any(args.metric_type == metric for metric in ["regression", "classification", "clustering"]):
            raise ValueError("[-] Error: Incorrect selected metric type")

        if args.metric_type == "regression":
            _determine_run_type(
                args.all_metrics,
                args.spec_metrics,
                args.spec_metric,
                regressor_metrics,
                np.asarray(args.train_y),
                np.asarray(args.predictions)
            )

        if args.metric_type == "classification":
            _determine_run_type(
                args.all_metrics,
                args.spec_metrics,
                args.spec_metric,
                classification_metrics,
                np.asarray(args.train_y),
                np.asarray(args.predictions)
            )

        if args.metric_type == "clustering":
            _determine_run_type(
                args.all_metrics,
                args.spec_metrics,
                args.spec_metric,
                clustering_metrics,
                np.asarray(args.train_y),
                np.asarray(args.predictions)
            )
    except ValueError as incorrect_argument_error:
        print(incorrect_argument_error)
        exit(1)

