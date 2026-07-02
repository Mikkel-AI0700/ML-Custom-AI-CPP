#pragma once
#include <filesystem>
#include <armadillo>
#include "dataset_types.hpp"

class DatasetOperations {
    public:
        TrainTestData datasets_vector;

        void construct_datasets ();
        void load_datasets (
            std::filesystem::path train_x,
            std::filesystem::path train_y,
            std::filesystem::path test_x,
            std::filesystem::path test_y
        );
        void save_dataset (
            std::filesystem::path save_path,
            arma::vec model_predictions
        );
};
