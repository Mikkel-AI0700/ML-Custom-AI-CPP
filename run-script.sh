#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-data-generators/generator-files/generator.py"
python_tld="$(pwd)/python-data-generators"

# Machine learning algorithms file paths
linreg_source_path="$(pwd)/linear-regression/source-codes/lin-reg-main.cpp"

# Directory to place the compiled ML algorithms
compiled_algorithms_path="$(pwd)/compiled-algorithms"

function generate_datasets () {
    local dataset_type="$1"
    local filename="$2"
    local key_value_change="$3"
    local skip_data_generation=$4 

    if [[ ! -z "${PYTHONPATH}" && ! -z "${VIRTUAL_ENV}" ]]; then
        echo -n "[+] Python TLD set: $(echo ${PYTHONPATH})\n [+] Venv set: $(echo ${VIRTUAL_ENV})"
    else
        export PYTHONPATH=${python_tld} && echo "[*] TLD set: ${PYTHONPATH}"
        source python-data-generators/generator-venv/bin/activate && echo "[*] VENV set: ${VIRTUAL_ENV}"
    fi

    if [[ -n "${dataset_type}" && ${skip_data_generation} -eq 1 ]]; then
        echo "[-] Error: Dataset type is set but skip dataset generation flag is also set"
        exit 1
    fi

    if [[ ! ${skip_data_generation} -eq 1 ]]; then
        echo "[+] Running the generator: ${python_generator_path}"
        python3 "${python_generator_path}" \
            --dataset-type "${dataset_type}" \
            --key-value-change "${key_value_change}"
    else
        echo "[*] Training and testing dataset generation skipped. Proceeding to compilation"
        return
    fi
}

function activate_machine_learning_models () {
    local algorithm_type="$1"
    declare -A ml_algorithms=(
        ["linreg"]="${linreg_source_path}"
    )

    for ml_key in "${!ml_algorithms[@]}"; do
        ml_value="${ml_algorithms[${ml_key}]}"

        if [[ -e "${compiled_algorithms_path}/compiled-${algorithm_type}" ]]; then
            echo "[-] A compiled version of ${algorithm_type} exists! Removing..."
            rm -fr "${compiled_algorithms_path}/compiled-${algorithm_type}"
        fi

        if [[ "${ml_key}" == "${algorithm_type}" ]]; then
            g++ "${ml_value}" "$(pwd)/base/base.cpp" -o "${compiled_algorithms_path}/compiled-${algorithm_type}" \
                -Iinclude \
                -larmadillo \
                -lblas \
                -llapack \
                -fdiagnostics-color=always \
                -fdiagnostics-show-line-numbers \
                -fdiagnostics-show-caret \
                -O3
            echo "[+] Successfully generated compiled-${algorithm_type}"
        fi
    done
}

function main () {
    local algorithm_type=""
    local dataset_type=""
    local filename=""
    local key_value_change=""
    local skip_dataset_generation=0

    while getopts "m:d:f:s" option_flag; do
        case "${option_flag}" in
            m) algorithm_type="${OPTARG}" ;;
            d) dataset_type="${OPTARG}" ;;
            k) key_value_change="${OPTARG}" ;;
            s) skip_dataset_generation=1 ;;
        esac
    done

    if [[  -n "${algorithm_type}" ]]; then
        echo "[+] Passing on script arguments to generator.py"
        generate_datasets "${dataset_type}" "${filename}" "${key_value_change}" $skip_dataset_generation
        activate_machine_learning_models "${algorithm_type}"
    else
        echo "[-] Error: Machine learning algorithm type is not set! Aborting."
        exit 1
    fi
}

main "$@"
