#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-utilities/generator-files/generator.py"
python_model_metric_checker="$(pwd)/python-utilities/performance-metric-checkers/model_metric_checker.py"
python_tld="$(pwd)/python-utilities"

# Machine learning algorithms file paths
linreg_source_path="$(pwd)/linear-regression/source-codes/lin-reg-main.cpp"
logreg_source_path="$(pwd)/logistic-regression/source-codes/log-reg-main.cpp"

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
        source python-utilities/generator-venv/bin/activate && echo "[*] VENV set: ${VIRTUAL_ENV}"
    fi

    if [[ ! ${skip_data_generation} -eq 1 ]]; then
        echo "[+] Running the generator: ${python_generator_path}"
        python3 "${python_generator_path}" \
            --dataset-type "${dataset_type}" \
            --key-value-change "${key_value_change}"
    else
        echo "[*] Training and testing dataset generation skipped. You may now proceed to compilation"
        return
    fi
}

function evaluate_machine_learning_predictions () {
    local true_y_path="$1"
    local predictions_path="$2"
    local metric_type="$3"
    local spec_metric="$4"
    local specs_metrics="$5"
    local all_metrics="$6"

    if [[ -z "${PYTHONPATH}" && -z "${VIRTUAL_ENV}" ]]; then
        echo "[-] Error: Cannot run script when both PYTHONPATH and VIRTUAL_ENV is not set"
        exit 1
    fi

    if [[ -z "${true_y_path}" || -z "${predictions_path}" ]]; then
        echo "[-] Error: Both ground truths and predictions path cannot be empty"
        exit 1
    fi

    if [[ "${metric_type}" != "regression" || "${metric_type}" != "classification" || "${metric_type}" != "clustering" ]]; then
        echo "[-] Error: User passed metric type is non-existent"
        exit 1
    fi

    # Condition to check if user will run specific metrics
    if [[ ${specs_metrics} -eq 1 && ! -z "${spec_metric}" ]]; then
        python3 "${python_model_metric_checker}" \
            --metric-type "${metric_type}" \
            --run-spec-metrics \
            --spec-metric "${spec_metric}" \
            --true-data "$(pwd)/${true_y_path}" \
            --predictions "$(pwd)/${predictions_path}"
    else
        echo "[-] Error: Must provide a specific metric relative to the chosen metric"
    fi

    # Condition to check if user will run all metrics according to chosen learning type
    if [[ ${all_metrics} -eq 1 ]]; then
        python3 "${python_model_metric_checker}" \
            --metric-type "${metric_type}" \
            --run-all-metrics \
            --true-data "$$(pwd)/{true_y_path}" \
            --predictions-path "$(pwd)/${predictions_path}"
    fi
}

function activate_machine_learning_models () {
    local algorithm_type="$1"
    declare -A ml_algorithms=(
        ["linreg"]="${linreg_source_path}"
        ["logreg"]="${logreg_source_path}"
    )

    for ml_key in "${!ml_algorithms[@]}"; do
        ml_value="${ml_algorithms[${ml_key}]}"

        if [[ -e "${compiled_algorithms_path}/compiled-${algorithm_type}" ]]; then
            echo "[-] A compiled version of ${algorithm_type} exists! Removing..."
            rm -fr "${compiled_algorithms_path}/compiled-${algorithm_type}"
        fi

        if [[ "${ml_key}" == "${algorithm_type}" ]]; then
            g++ "${ml_value}" -o "${compiled_algorithms_path}/compiled-${algorithm_type}" \
                -Iheader-source \
                -larmadillo \
                -lblas \
                -llapack \
                -fdiagnostics-color=always \
                -fdiagnostics-show-line-numbers \
                -fdiagnostics-show-caret \
                -Wall \
                -Wextra \
                -O3
            echo "[+] Successfully generated compiled-${algorithm_type}"
        fi
    done
}

function display_help () {
    echo -e "[-G <Activate GENERATE_DATASETS FLAG>] \n[-C <Activate the METRIC_CHECK flag>] \n[-O <Activate the COMPILE_ML flag>]"

    # For the GENERATE_COMPILE Flag
    echo -e "[-S <Activates the SPEC_METRIC flag, to be used with METRIC_CHECK>] \n[-A <Activates the ALL_METRICS flag, to be used with METRIC_CHECK>]"
    echo -e "[-a <Provide the C++ ML algorithm to compile>] \n[-d <Provide the type of dataset to make [regression, classification, clustering]>]"
    echo -e "[-k <Provide the key-value-change to change in the generator configurations>] \n[-s <Skips the dataset generation>]"
    
    # For the METRIC_CHECK Flag
    echo -e "[-m <Provide the metric type for the learning type>]"
    echo -e "[-S <Activate specific metrics>] \n[-A <Activate all the metrics>] \n[-c <Provide the specific metric>]"
    echo -e "[-T <Provide the ground truths path>] \n[-P <Provide the model predictions path>]"
}

function main () {
    # Flags to check to either generate/compile or to check metrics
    local GENERATE_DATASETS=0
    local METRIC_CHECK=0
    local COMPILE_ML=0

    # Generate/Compile options
    local algorithm_type=""
    local dataset_type=""
    local filename=""
    local key_value_change=""
    local skip_dataset_generation=0

    # Metric checkers options
    local metric_type=""
    local true_y_path=""
    local predictions_path=""
    local speci_metric=""
    local SPEC_METRIC=0
    local ALL_METRICS=0

    local options="a:d:f:t:p:mSATPcksGCOh"

    while getopts "${options}" option_flag; do
        case "${option_flag}" in
            a) algorithm_type="${OPTARG}" ;;
            d) dataset_type="${OPTARG}" ;;
            k) key_value_change="${OPTARG}" ;;
            s) skip_dataset_generation=1 ;;
            m) metric_type="${OPTARG}" ;;
            S) SPEC_METRIC=1 ;;
            A) ALL_METRICS=1 ;;
            T) true_y_path="${OPTARG}" ;;
            P) predictions_path="${OPTARG}" ;;
            c) speci_metric="${OPTARG}" ;;
            O) COMPILE_ML=1 ;;
            G) GENERATE_DATASETS=1 ;;
            C) METRIC_CHECK=1 ;;
            h) display_help ;;
            *) display_help ;;
        esac
    done

    if [[ ${GENERATE_DATASETS} -eq 1 ]]; then
        echo "[+] Passing on script arguments to generator.py"
        generate_datasets "${dataset_type}" "${filename}" "${key_value_change}" ${skip_dataset_generation}
    elif [[ ${COMPILE_ML} -eq 1 ]]; then
        echo "[+] Passing on scripts to compile the machine learning algorithm"
        activate_machine_learning_models "${algorithm_type}"
    elif [[ ${METRIC_CHECK} -eq 1 ]]; then
        echo "[+] Passing on the script arguments to model_metric_checker.py"
        evaluate_machine_learning_predictions \
            "${true_y_path}" \
            "${predictions_path}" \
            "${metric_type}" \
            "${speci_metric}" \
            ${SPEC_METRIC} \
            ${ALL_METRICS}
    else
        echo "[-] Error: Either generate/compile or metric_check flag must be set"
        exit 1
    fi
}

main "$@"
