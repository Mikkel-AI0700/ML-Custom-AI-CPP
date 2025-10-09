#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-data-generators/generator-files/generator.py"

# Python's generator configuration paths
regressor_generator_config_path="$(pwd)/python-data-generators/json-config-files/regressor"
classification_generator_config_path="$(pwd)/python-data-generators/json-config-files/classification"
clustering_generator_config_path="$(pwd)/python-data-generators/json-config-files/clustering"

# Machine learning algorithms file paths
linreg_source_path="$(pwd)/linear-regression/source-codes/lin-reg-main.cpp"

function generate_datasets () {
    local dataset_type="$1"
    local filename="$2"
    local key_value_change="$3"
    local skip_data_generation=$4

    if [[ ! ${skip_data_generation} -eq 1 ]]; then
        echo "[+] Running the generator: ${python_generator_path}"
        python3 "${python_generator_path}" \
            --dataset-type "${dataset_type}" \
            --dataset-filename "${filename}" \
            --key-value-change "${key_value_change}"
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
    local dataset_type=""
    local filename=""
    local key_value_change=""
    local skip_dataset_generation=0

    while getopts "sk:d:f:m" option_flag; do
        case "${option_flag}" in
            d) dataset_type="${OPTARG}" ;;
            f) filename="${OPTARG}" ;;
            k) key_value_change="${OPTARG}" ;;
            s) skip_dataset_generation="${OPTARG}" ;;
        esac
    done

    generate_datasets "${dataset_type}" "${filename}" "${key_value_change}" "${skip_dataset_generation}"
    activate_machine_learning_models "${data_generation_type}"
}

main
