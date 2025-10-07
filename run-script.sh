#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-data-generator/generator-files/generator.py"

# Python's generator configuration paths
regressor_generator_config_path="$(pwd)/python-data-generators/json-config-files/regressor"
classification_generator_config_path="$(pwd)/python-data-generators/json-config-files/classification"
clustering_generator_config_path="$(pwd)/python-data-generators/json-config-files/clustering"

# Machine learning algorithms file paths
linreg_source_path="$(pwd)/linear-regression/source-codes/lin-reg-main.cpp"

function generate_datasets () {
    local key_value_change=""
    local dataset_type=""
    local test_data_name=""
    local skip_data_generation=0

    if [[ ! ${skip_data_generation} -eq 1 ]]; then
        ./"${python_generator_path}" \
            --dataset-type "${data_generation_type}" \
            --key-value-change "${key_value_change}" \
            --dataset-filename "${test_data_name}"
        return
    else
        return
    fi 
}

function activate_machine_learning_models () {
    local algorithm_type="$1"

    if [[ "${algorithm_type}" == "linreg" ]]; then
        g++ "${linreg-source-path}" -o "linear-regression/source-codes/compiled-${algorithm_type}"
        ./"linear-regression/source-codes/compiled-${algorithm_type}"
    fi
}

function main () {
    local key_value_change=""
    local data_generation_type=""
    local test_data_name=""
    local skip_data_generation=0

    while getopts "sk:d:f:m" option_flag; do
        case "${option_flag}" in
            m) key_value_change="${OPTARG}" ;;
            d) data_generation_type="${OPTARG}" ;;
            f) test_data_name="${OPTARG}" ;;
            s) skip_data_generation=1 ;;
        esac
    done

    generate_datasets "${key_value_change}" "${data_generaton_type}" "${test_data_name}" "${skip_data_generation}"
    activate_machine_learning_models "${data_generation_type}"
}

