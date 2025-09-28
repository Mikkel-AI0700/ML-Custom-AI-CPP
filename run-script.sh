#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-data-generator/generator-files/generator.py"
python_generator_configuration=""
python_test_files_path="$(pwd)/python-data-generator/test-data"

# Machine learning algorithms file paths
linreg_source_path="$(pwd)/linear-regression/source-codes/lin-reg-main.cpp"

# Activate Python's random generator

function generate_and_store () {
    local learning_type=""
}

function main () {
    local model_argument=""
    local data_generation_type=""
    local test_data_name=""

    while getopts "m:d:f:" option_flag; do
        case "${option_flag}" in
            m) model_argument="${OPTARG}" ;;
            d) data_generation_type="${OPTARG}" ;;
            f) test_data_name="${OPTARG}" ;;
        esac
    done

    # Check if user wants to generate test data
    if [[ "${data_generation_type}" == "regression" ]]; then
        :
    elif [[ "${data_generation_type}" == "classification" ]]; then
        :
    elif [[ "${data_generation_type}" == "clustering" ]]; then
        :
    else
        :
    fi

    # Activate the machine learning models
    if [[]]; then
        :
    fi
}

