import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main() -> None:
    argp = argparse.ArgumentParser(description="Fetch ML datasets from OpenML")
    argp.add_argument("--dataset-type", required=True, dest="dset_type",
                      choices=["regression", "classification", "clustering"])
    argp.add_argument("--dataset-name", required=True, dest="dset_name")

    parsed = argp.parse_args()

    base_path = Path(__file__).resolve().parent.parent

    dset_type = parsed.dset_type
    dset_name = parsed.dset_name

    target_dir = base_path / "datasets" / "openml" / dset_name

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[-] Error: Could not create directory '{target_dir}': {exc}")
        sys.exit(1)

    try:
        print(f"[+] Fetching OpenML dataset '{dset_name}' ...")
        bunch = fetch_openml(name=dset_name, as_frame=True)
    except Exception as exc:
        print(f"[-] Error: Failed to fetch OpenML dataset '{dset_name}': {exc}")
        sys.exit(1)

    try:
        X = bunch.data
        y = bunch.target
    except AttributeError as exc:
        print(f"[-] Error: OpenML Bunch object missing expected attributes: {exc}")
        sys.exit(1)

    # Convert to numpy arrays
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()

    numerical_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)

    features_path = target_dir / "features.txt"
    try:
        with open(features_path, "w") as f:
            f.write(f"# Numerical features ({len(numerical_cols)}):\n")
            for col in numerical_cols:
                f.write(f"{col}\n")
            f.write(f"\n# Categorical features ({len(categorical_cols)}):\n")
            for col in categorical_cols:
                f.write(f"{col}\n")
    except OSError as exc:
        print(f"[-] Error: Could not write features file '{features_path}': {exc}")
        sys.exit(1)

    stratify = y_arr if dset_type == "classification" else None
    try:
        train_x, test_x, train_y, test_y = train_test_split(
            X_arr,
            y_arr,
            train_size=0.8,
            test_size=0.2,
            shuffle=True,
            random_state=42,
            stratify=stratify
        )
    except Exception as exc:
        print(f"[-] Error: Train/test split failed: {exc}")
        sys.exit(1)

    train_x_path = target_dir / "train_x.csv"
    test_x_path = target_dir / "test_x.csv"
    train_y_path = target_dir / "train_y.csv"
    test_y_path = target_dir / "test_y.csv"

    generated_datasets = [train_x, test_x, train_y, test_y]
    dataset_paths = [train_x_path, test_x_path, train_y_path, test_y_path]

    try:
        for dset, dset_path in zip(generated_datasets, dataset_paths):
            # dset is numpy array; convert to DataFrame for mixed-type saving
            pd.DataFrame(dset).to_csv(dset_path, index=False, header=False)
    except OSError as exc:
        print(f"[-] Error: Could not write CSV files to '{target_dir}': {exc}")
        sys.exit(1)

if __name__ == "__main__":
    main()
