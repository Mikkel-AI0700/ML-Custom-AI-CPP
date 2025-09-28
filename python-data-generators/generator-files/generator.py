#!/usr/bin/env python3

import os
import json
import argparse
from sklearn.datasets import (
    make_regression,
    make_classification,
    make_multilabel_classification,
    make_blobs
)

def _read_generator_configuration (filepath: str):
    pass

def create_regression ():
    pass

def create_classification ():
    pass

def create_clustering ():
    pass

def main ():
    argp = argparse.ArgumentParser(description="ML dataset generators")

    argp.add_argument("--dataset_type", required=True, dest="dset_type")
    argp.add_argument("--use-default", required=False, action="store_true")
    argp.add_argument("--key-value-change", required=False, dest="key_value_parameter")

