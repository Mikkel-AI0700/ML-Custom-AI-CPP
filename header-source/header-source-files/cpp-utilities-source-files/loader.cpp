#include <iostream>
#include <filesystem>
#include <armadillo>
#include "headers/cpp-utilities/loader.hpp"

void DatasetOperations::construct_datasets () {
    datasets_vector.emplace_back(arma::mat{});
    datasets_vector.emplace_back(arma::colvec{});
    datasets_vector.emplace_back(arma::mat{});
    datasets_vector.emplace_back(arma::colvec{});
}

void DatasetOperations::load_datasets (
    std::filesystem::path train_x,
    std::filesystem::path train_y,
    std::filesystem::path test_x,
    std::filesystem::path test_y
) {
    TrainTestData temp_filepaths = {train_x, train_y, test_x, test_y};
    try {
        for (int index = 0; index < temp_filepaths.size(); index++) {
            if (!std::filesystem::exists(temp_filepaths[index])) {
                throw new std::filesystem::filesystem_error("[-] Error: Filepath does not exist");
            }
            if (std::filesystem::is_directory(temp_filepaths[index])) {
                throw new std::filesystem::filesystem_error("[-] Error: Filesystem path leads to a directory");
            }
            if (!std::filesystem::file_size(temp_filepaths[index]) > 0) {
                throw new std::filesystem::filesystem_error("[-] Error: File isn't a directory and exists but doesn't have content");
            }
        }

        for (int index = 0; index < temp_filepaths.size(); index++) {
            if (std::holds_alternative<arma::mat>(datasets_vector[index])) {
                std::get<arma::mat>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
            } else {
                std::get<arma::colvec>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
            }
        }
    } catch (const std::filesystem::filesystem_error& dataset_path_error) {
        std::cerr << dataset_path_error.what() << std::endl;
        exit(1);
    }
}

void DatasetOperations::save_datasets (std::string save_type, arma::colvec model_predictions) {
    if (save_type == "regression") {
        model_predictions.save(SavePaths.regression_save_path, arma::csv_ascii);
    } else if (save_type == ) {
        model_predictions.save(SavePaths.classification_save_path, arma::csv_ascii);
    } else {
        model_predictions.save(SavePaths.clustering_save_path, arma::csv_ascii);
    }
}

