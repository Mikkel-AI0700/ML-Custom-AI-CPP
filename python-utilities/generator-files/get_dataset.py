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

    if dset_type == "regression":
        target_dir = base_path / "test-data" / "regression" / dset_name
    elif dset_type == "classification":
        target_dir = base_path / "test-data" / "classification" / dset_name
    elif dset_type == "clustering":
        target_dir = base_path / "test-data" / "clustering" / dset_name
    else:
        print(f"[-] Error: Unknown dataset type '{dset_type}'")
        sys.exit(1)

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

    df = pd.concat([X, y.to_frame(name="target")], axis=1)

    numerical_cols = list(df.select_dtypes(include=["number"]).columns)
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    features_path = target_dir / f"{dset_name}-features.txt"
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

    stratify = y if dset_type == "classification" else None
    try:
        train_df, test_df = train_test_split(
            df, train_size=0.8, test_size=0.2, shuffle=True,
            random_state=42, stratify=stratify
        )
    except Exception as exc:
        print(f"[-] Error: Train/test split failed: {exc}")
        sys.exit(1)

    train_path = target_dir / f"{dset_name}-train.csv"
    test_path = target_dir / f"{dset_name}-test.csv"

    try:
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    except OSError as exc:
        print(f"[-] Error: Could not write CSV files to '{target_dir}': {exc}")
        sys.exit(1)

    print(f"[+] Dataset '{dset_name}' saved to '{target_dir}'")
    print(f"    Train: {train_path} ({len(train_df)} rows)")
    print(f"    Test:  {test_path} ({len(test_df)} rows)")
    print(f"    Features: {features_path}")


if __name__ == "__main__":
    main()
