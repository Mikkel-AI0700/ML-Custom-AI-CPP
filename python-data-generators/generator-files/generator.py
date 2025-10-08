import os
import re
import json
import argparse
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_regression,
    make_classification,
    make_blobs
)

def _read_generator_configuration (json_configuration_path: Path):
    return json.load(json_configuration_path)

def _write_to_file (filepath: Path, generated_dataset: pd.DataFrame):
    for file in os.listdir(filepath):
        if re.findall("(train|test)_(x|y)_dataset.csv", file):
            os.remove(file)

    train_x, test_x, train_y, test_y = train_test_split(
        generated_dataset.iloc[:, :-1],
        generated_dataset.iloc[:, -1],
        train_size=0.8,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    for dataset in [train_x, test_x, train_y, test_y]:
        dataset = pd.DataFrame(dataset)
        dataset.to_csv(filepath)

def _change_configuration (config_key_value: dict[str, Any], generator_configuration: dict[str, Any]):
    try:
        if len(config_key_value) == 0:
            return
        else:
            config_key, config_value = config_key_value.items()

        if config_key in generator_configuration.keys():
            generator_configuration.update({config_key: config_value})
        else:
            raise ValueError("[-] Error: Non existent parameter name detected")
    except ValueError as non_existent_config_key:
        print(non_existent_config_key)
        exit(1)

def create_regression (dataset_filename: Path, regressor_config: Path, key_value_config: dict[str, Any]):
    regressor_configuration = _read_generator_configuration(regressor_config)
    _change_configuration(key_value_config, regressor_configuration)
    regressor_dataset = make_regression(**regressor_config)
    _write_to_file(dataset_filename, pd.DataFrame(regressor_dataset))

def create_classification (dataset_filename: Path, classification_config: Path, key_value_config: dict[str, Any]):
    classification_configuration = _read_generator_configuration(classification_config)
    _change_configuration(key_value_config, classification_configuration)
    classification_dataset = make_classification(**classification_configuration)
    _write_to_file(dataset_filename, pd.DataFrame(classification_dataset))

def create_clustering (dataset_filename: Path, clustering_config: Path, key_value_config: dict[str, Any]):
    clustering_configuration = _read_generator_configuration(clustering_config)
    _change_configuration(key_value_config, clustering_configuration)
    clustering_dataset = make_blobs(**clustering_configuration)
    _write_to_file(dataset_filename, pd.DataFrame(clustering_dataset))

def main ():
    regressor_json_path = Path("python-data-generators/json-config-files/regressor/")
    classification_json_path = Path("python-data-generators/json-config-files/classification/")
    clustering_json_path = Path("python-data-generators/json-config-files/clustering/")

    argp = argparse.ArgumentParser(description="ML dataset generators")
    argp.add_argument("--dataset-type", required=True, dest="dset_type")
    argp.add_argument("--use-default", required=False, action="store_true")
    argp.add_argument("--key-value-change", required=False, dest="key_value_parameter")
    argp.add_argument("--dataset-filename", required=False, dest="dataset_filename")

    parsed_arguments = argp.parse_args()

    try:
        if parsed_arguments.dset_type == "regression":
            create_regression(
                parsed_arguments.dataset_filename, 
                regressor_json_path, 
                parsed_arguments.key_value_parameter
            )
        elif parsed_arguments.dset_type == "classification":
            create_classification(
                parsed_arguments.dataset_filename, 
                classification_json_path,
                parsed_arguments.key_value_change
            )
        elif parsed_arguments.dset_type == "clustering":
            create_clustering(
                parsed_arguments.dset_type, 
                clustering_json_path,
                parsed_arguments.key_value_change
            )
        else:
            raise ValueError(f"[-] Error: Incorrect dataset type -> {parsed_arguments.dset_type}")
    except ValueError as incorrect_argument_error:
        print(incorrect_argument_error)
        exit(1)

main()

