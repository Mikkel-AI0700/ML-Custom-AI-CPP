#include <iostream>
#include <filesystem>
#include <armadillo>
#include "dataset_types.hpp"
#include "loader.hpp"

using arma::vec;
using arma::mat;
using std::filesystem::path;
using std::runtime_error;
using std::filesystem::filesystem_error;

void DatasetOperations::construct_datasets () {
    datasets_vector.emplace_back(mat{});
    datasets_vector.emplace_back(vec{});
    datasets_vector.emplace_back(mat{});
    datasets_vector.emplace_back(vec{});
}

void DatasetOperations::load_datasets (
    path train_x,
    path train_y,
    path test_x,
    path test_y
) {
    DatasetsFilepaths temp_filepaths = {
        train_x, 
        train_y, 
        test_x, 
        test_y
    };

    try {
        for (auto& dataset_path : temp_filepaths) {
            if (!std::filesystem::exists(dataset_path)) {
                throw filesystem_error(
                    "[-] Error: File/Filepath doesn't exist",
                    dataset_path,
                    std::make_error_code(std::errc::no_such_file_or_directory)
                );
            }
            if (!std::filesystem::is_regular_file(dataset_path)) {
                throw filesystem_error(
                    "[-] Error: Filepath isn't a file and/or leads to a directory",
                    dataset_path,
                    std::make_error_code(std::errc::is_a_directory)
                );
            }
            if (std::filesystem::file_size(dataset_path) == 0) {
                throw runtime_error(
                    "[-] Error: Provided path isn't a directory and is a file but empty"
                );
            }
        }

        for (int index = 0; index < temp_filepaths.size(); index++) {
            if (std::holds_alternative<mat>(datasets_vector[index])) {
                std::get<mat>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
                std::cout << "[+] Loaded a mat dataset" << std::endl;
            } else {
                std::get<vec>(datasets_vector[index]).load(temp_filepaths[index], arma::csv_ascii);
                std::cout << "[+] Loaded a vec dataset" << std::endl;
            }
        }
    } catch (const filesystem_error& dataset_path_error) {
        std::cerr << dataset_path_error.what() << std::endl;
        exit(1);
    }
}

void DatasetOperations::save_dataset (path save_path, vec model_predictions) {
    model_predictions.save(save_path, arma::csv_ascii);
    std::cout << "[+] Saved predictions to " << save_path << std::endl;
}
