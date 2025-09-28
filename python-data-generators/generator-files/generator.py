#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import (
    make_regression,
    make_classification,
    make_blobs
)

def _read_generator_configuration (json_configuration_path: Path):
    return json.load(json_configuration_path)

def _write_to_file (filename: str, generated_dataset: pd.DataFrame):
    generated_dataset.to_csv(filename)

def create_regression (dataset_filename: str, regressor_config: Path):
    regression_config_dict = _return_generator_configuration(regressor_config)
    regressor_dataset = make_regression(**regression_config_dict)
    _write_to_file(dataset_filename, pd.DataFrame(regressor_dataset))

def create_classification (dataset_filename: str, classification_config: Path):
    classification_config_dict = _return_generator_configuration(classification_config)
    classification_dataset = make_classification(**classification_config_dict)
    _write_to_file(dataset_filename, pd.DataFrame(classification_dataset))

def create_clustering (dataset_filename: str, clustering_config: Path):
    clustering_config_dict = _return_generator_configuration(clustering_config)
    clustering_dataset = make_blobs(**clustering_config_dict)
    _write_to_file(dataset_filename, pd.DataFrame(clustering_dataset))

def main ():
    regressor_json_path = Path("python-data-generators/json-config-files/regressor/")
    classification_json_path = Path("python-data-generators/json-config-files/classification/")
    clustering_json_path = Path("python-data-generators/json-config-files/clustering/")

    argp = argparse.ArgumentParser(description="ML dataset generators")
    argp.add_argument("--dataset_type", required=True, dest="dset_type")
    argp.add_argument("--use-default", required=False, action="store_true")
    argp.add_argument("--key-value-change", required=False, dest="key_value_parameter")
    argp.add_argument("--dataset-filename", required=False, dset="dataset_filename")

    parsed_arguments = argp.parse_args()

    try:
        if parsed_arguments.dset_type == "regression":
            create_regression(parsed_args.dataset_filename, regressor_json_path)
        elif parsed_arguments.dset_type == "classification":
            create_classification(parsed_args.dataset_filename, classification_json_path)
        elif parsed_arguments.dset_type == "clustering":
            create_clustering(parsed_args.dset_type, clustering_json_path)
        else:
            raise ValueError(f"[-] Error: Incorrect dataset type -> {parsed_argument.dset_type}")
    except ValueError as incorrect_argument_error:
        print(incorrect_argument_error)
        exit(1)

