#!/bin/bash

# Python's data generator variables
python_generator_path="$(pwd)/python-utilities/generator-files/generator.py"
python_get_dataset_path="$(pwd)/python-utilities/generator-files/get_dataset.py"
python_model_metric_checker="$(pwd)/python-utilities/performance-metric-checkers/model_metric_checker.py"
python_tld="$(pwd)/python-utilities"

# CMake build directory
cmake_build_directory="$(pwd)/build"

function generate_datasets () {
    local dataset_type_name="$1"
    local openml_dataset_name="$2"
    local generator_config_change="$3"
    local SKIP_DATA_GENERATION="$4"

    if [[ -n "${PYTHONPATH:-}" && -n "${VIRTUAL_ENV:-}" ]]; then
        printf "[+] Python TLD set: %s\n[+] Venv set: %s\n" "${PYTHONPATH}" "${VIRTUAL_ENV}"
    else
        export PYTHONPATH=${python_tld} && echo "[*] TLD set: ${PYTHONPATH}"
        source python-utilities/generator-venv/bin/activate && echo "[*] VENV set: ${VIRTUAL_ENV}"
    fi

    if [[ "${SKIP_DATA_GENERATION}" != "true" ]]; then
        if [[ -n "${openml_dataset_name}" ]]; then
            local dataset_name="${openml_dataset_name}"
            local model_test_dir="${python_tld}/python-sklearn-test/openml/${dataset_name}"
            local predictions_dir="${python_tld}/datasets/openml/${dataset_name}/predictions"

            mkdir -p "${model_test_dir}" "${predictions_dir}"
            echo "[+] Fetching OpenML dataset: ${openml_dataset_name}"
            python3 "${python_get_dataset_path}" \
                --dataset-type "${dataset_type_name}" \
                --dataset-name "${openml_dataset_name}"
        else
            local dataset_name="${dataset_type_name}"
            local model_test_dir="${python_tld}/python-sklearn-test/synthetic/${dataset_name}"
            local predictions_dir="${python_tld}/datasets/synthetic/${dataset_name}/predictions"

            mkdir -p "${model_test_dir}" "${predictions_dir}"
            echo "[+] Running the generator: ${python_generator_path}"
            python3 "${python_generator_path}" \
                --dataset-type "${dataset_type_name}" \
                --key-value-change "${generator_config_change}"
        fi
    else
        echo "[*] Training and testing dataset generation skipped. You may now proceed to compilation"
        return
    fi
}

function evaluate_machine_learning_predictions () {
    local metric_type_name="$1"
    local dataset_name="$2"
    local specific_metric_name="$3"
    local RUN_SPECIFIC_METRICS="$4"
    local RUN_ALL_METRICS="$5"
    local true_data_override="$6"
    local predictions_override="$7"

    if [[ -z "${PYTHONPATH:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
        echo "[-] Error: Cannot run script when both PYTHONPATH and VIRTUAL_ENV is not set"
        exit 1
    fi

    case "${metric_type_name}" in
        regression|classification|clustering) ;;
        *)
            echo "[-] Error: User passed metric type is non-existent"
            exit 1
            ;;
    esac

    local true_data_path=""
    local cpp_preds=""
    local python_preds=""

    if [[ -n "${dataset_name}" ]]; then
        local source=""
        if [[ -d "${python_tld}/datasets/openml/${dataset_name}" ]]; then
            source="openml"
        elif [[ -d "${python_tld}/datasets/synthetic/${dataset_name}" ]]; then
            source="synthetic"
        else
            echo "[-] Error: Dataset '${dataset_name}' not found in openml/ or synthetic/"
            exit 1
        fi
        local base="${python_tld}/datasets/${source}/${dataset_name}"
        true_data_path="${base}/test_y.csv"
        cpp_preds="${base}/predictions/cpp-predictions.csv"
        python_preds="${base}/predictions/python-predictions.csv"
    else
        true_data_path="${true_data_override}"
        cpp_preds="${predictions_override}"
    fi

    if [[ -z "${true_data_path}" ]]; then
        echo "[-] Error: Ground truth path is empty. Provide -n <dataset-name> or -T <path>"
        exit 1
    fi

    if [[ "${true_data_path}" != /* ]]; then
        true_data_path="$(pwd)/${true_data_path}"
    fi
    if [[ -n "${cpp_preds}" && "${cpp_preds}" != /* ]]; then
        cpp_preds="$(pwd)/${cpp_preds}"
    fi
    if [[ -n "${python_preds}" && "${python_preds}" != /* ]]; then
        python_preds="$(pwd)/${python_preds}"
    fi

    local py_args=(
        --metric-type "${metric_type_name}"
        --ground-truth "${true_data_path}"
    )
    if [[ -n "${cpp_preds}" ]]; then
        py_args+=( --cpp-predictions "${cpp_preds}" )
    fi
    if [[ -n "${python_preds}" ]]; then
        py_args+=( --python-predictions "${python_preds}" )
    fi

    if [[ "${RUN_SPECIFIC_METRICS}" == "true" ]]; then
        if [[ -z "${specific_metric_name}" ]]; then
            echo "[-] Error: Must provide a specific metric name with -c when using -S"
            exit 1
        fi
        py_args+=( --run-spec-metric --spec-metric "${specific_metric_name}" )
    elif [[ "${RUN_ALL_METRICS}" == "true" ]]; then
        py_args+=( --run-all-metrics )
    else
        echo "[-] Error: Must select either -S (specific metric) or -A (all metrics)"
        exit 1
    fi

    python3 "${python_model_metric_checker}" "${py_args[@]}"
}

function build_machine_learning_models () {
    local build_tests="$1"

    mkdir -p "${cmake_build_directory}"

    local cmake_args="-S $(pwd) -B ${cmake_build_directory} -DCMAKE_BUILD_TYPE=Release"
    if [[ "${build_tests}" == "true" ]]; then
        cmake_args+=" -DBUILD_TESTS=ON"
    fi

    echo "[+] Configuring CMake build directory: ${cmake_build_directory}"
    if ! cmake ${cmake_args}; then
        echo "[-] Error: CMake configure failed"
        exit 1
    fi

    echo "[+] Building CMake project"
    if ! cmake --build "${cmake_build_directory}" --config Release; then
        echo "[-] Error: CMake build failed"
        exit 1
    fi

    if [[ "${build_tests}" == "true" ]]; then
        echo "[+] Running unit tests"
        if ! ctest --test-dir "${cmake_build_directory}" --output-on-failure; then
            echo "[-] Error: Some tests failed"
            exit 1
        fi
        echo "[+] All tests passed successfully"
    fi
}

function display_help () {
    echo "Usage:"
    echo "  ./run-script.sh -G -d <dataset_type> [-n <dataset_name>] [-k <key_value_change>] [-s]"
    echo "  ./run-script.sh -C -m <metric_type> -n <dataset_name> (-S -c <metric_name> | -A)"
    echo "  ./run-script.sh -O [-t]"
    echo
    echo "Modes:"
    echo "  -G  Generate datasets (generator.py or get_dataset.py)"
    echo "  -C  Evaluate ML predictions (model_metric_checker.py)"
    echo "  -O  Build ML models via CMake (build/)"
    echo "  -t  Enable unit tests (use with -O; sets BUILD_TESTS=ON)"
    echo "  -h  Display help"
    echo
    echo "Dataset generation options (-G):"
    echo "  -d  Dataset type: regression | classification | clustering"
    echo "  -n  OpenML dataset name (e.g. adult, iris) – fetches from OpenML via get_dataset.py"
    echo "  -k  Config override for generator (passed to --key-value-change)"
    echo "  -s  Skip dataset generation"
    echo
    echo "Prediction evaluation options (-C):"
    echo "  -m  Metric type: regression | classification | clustering"
    echo "  -n  Dataset name (auto-discovers paths from datasets/openml|synthetic/<name>/)"
    echo "  -T  Ground truth CSV path (override; requires -P)"
    echo "  -P  C++ predictions CSV path (override; requires -T)"
    echo "  -S  Run a specific metric (requires -c)"
    echo "  -c  Specific metric name (eg: mse, accuracy)"
    echo "  -A  Run all metrics"
}

function main () {
    # Mode flags
    local GENERATE_DATASETS=false
    local EVALUATE_MACHINE_LEARNING_PREDICTIONS=false
    local BUILD_MACHINE_LEARNING_MODELS=false
    local BUILD_WITH_TESTS=false

    # Dataset generation args
    local dataset_type_name=""
    local openml_dataset_name=""
    local generator_config_change=""
    local SKIP_DATA_GENERATION=false

    # Prediction evaluation args
    local metric_type_name=""
    local true_data_path=""
    local model_predictions_path=""
    local specific_metric_name=""
    local RUN_SPECIFIC_METRICS=false
    local RUN_ALL_METRICS=false

    local options="GCOhd:n:k:sm:T:P:c:SAt"

    while getopts "${options}" option_flag; do
        case "${option_flag}" in
            d) dataset_type_name="${OPTARG}" ;;
            n) openml_dataset_name="${OPTARG}" ;;
            k) generator_config_change="${OPTARG}" ;;
            s) SKIP_DATA_GENERATION=true ;;

            m) metric_type_name="${OPTARG}" ;;
            T) true_data_path="${OPTARG}" ;;
            P) model_predictions_path="${OPTARG}" ;;
            S) RUN_SPECIFIC_METRICS=true ;;
            A) RUN_ALL_METRICS=true ;;
            c) specific_metric_name="${OPTARG}" ;;

            O) BUILD_MACHINE_LEARNING_MODELS=true ;;
            t) BUILD_WITH_TESTS=true ;;
            G) GENERATE_DATASETS=true ;;
            C) EVALUATE_MACHINE_LEARNING_PREDICTIONS=true ;;
            h) display_help; exit 0 ;;
            *) display_help; exit 1 ;;
        esac
    done

    local selected_modes=0
    $GENERATE_DATASETS && ((selected_modes++))
    $EVALUATE_MACHINE_LEARNING_PREDICTIONS && ((selected_modes++))
    $BUILD_MACHINE_LEARNING_MODELS && ((selected_modes++))

    if [[ ${selected_modes} -ne 1 ]]; then
        echo "[-] Error: Must select exactly one mode (-G, -C, or -O)"
        display_help
        exit 1
    fi

    if $GENERATE_DATASETS; then
        echo "[+] Passing on script arguments to generator"
        generate_datasets "${dataset_type_name}" "${openml_dataset_name}" "${generator_config_change}" "${SKIP_DATA_GENERATION}"
        exit 0
    fi

    if $BUILD_MACHINE_LEARNING_MODELS; then
        echo "[+] Building machine learning models via CMake"
        build_machine_learning_models "${BUILD_WITH_TESTS}"
        exit 0
    fi

    if $EVALUATE_MACHINE_LEARNING_PREDICTIONS; then
        echo "[+] Passing on the script arguments to model_metric_checker.py"
        evaluate_machine_learning_predictions \
            "${metric_type_name}" \
            "${openml_dataset_name}" \
            "${specific_metric_name}" \
            "${RUN_SPECIFIC_METRICS}" \
            "${RUN_ALL_METRICS}" \
            "${true_data_path}" \
            "${model_predictions_path}"
        exit 0
    fi
}

main "$@"
