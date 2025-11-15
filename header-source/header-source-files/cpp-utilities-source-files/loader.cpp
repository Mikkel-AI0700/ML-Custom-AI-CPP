#include <iostream>
#include <filesystem>
#include <format>
#include <armadillo>
#include "complex-datatypes/complex_datatypes.hpp"
#include "cpp-utilities/loader.hpp"

struct SavePaths save_paths;

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
    DatasetsFilepaths temp_filepaths = {train_x, train_y, test_x, test_y};
    try {
        for (auto& dataset_path : temp_filepaths) {
            if (!std::filesystem::exists(dataset_path)) {
                throw std::filesystem::filesystem_error(
                    "[-] Error: File/Filepath doesn't exist",
                    dataset_path,
                    std::make_error_code(std::errc::no_such_file_or_directory)
                );
            }
            if (!std::filesystem::is_regular_file(dataset_path)) {
                throw std::filesystem::filesystem_error(
                    "[-] Error: Filepath isn't a file and/or leads to a directory",
                    dataset_path,
                    std::make_error_code(std::errc::is_a_directory)
                );
            }
            if (std::filesystem::file_size(dataset_path) == 0) {
                throw std::runtime_error("[-] Error: Provided path isn't a directory and is a file but empty");
            }
        }

        for (int index = 0; index < temp_filepaths.size(); index++) {
            if (std::holds_alternative<arma::mat>(datasets_vector[index])) {
                std::get<arma::mat>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
                std::cout << "[+] Loaded a arma::mat dataset" << std::endl;
            } else {
                std::get<arma::colvec>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
                std::cout << "[+] Loaded a arma::colvec dataset" << std::endl;
            }
        }
    } catch (const std::filesystem::filesystem_error& dataset_path_error) {
        std::cerr << dataset_path_error.what() << std::endl;
        exit(1);
    }
}

void DatasetOperations::save_dataset (std::string save_mode, std::string model_filename, arma::colvec model_predictions) {
    if (save_mode == "regression") {
        model_predictions.save(
            std::format(save_paths.regression_save_path.string(), model_filename),
            arma::csv_ascii
        );
    } else if (save_mode == "classification") {
        model_predictions.save(
            std::format(save_paths.classification_save_path.string()),
            arma::csv_ascii
        );
    } else {
        model_predictions.save(
            std::format(save_paths.clustering_save_path.string()),
            arma::csv_ascii
        );
    }
}
