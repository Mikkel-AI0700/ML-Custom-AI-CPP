#pragma once
#include <filesystem>
#include <armadillo>
#include "loader.hpp"

struct SavePaths {
    std::filesystem::path regression_save_path = "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/model-predictions/regression/";
    std::filesystem::path classification_save_path = "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/model-predictions/classification/";
    std::filesystem::path clustering_save_path = "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-utilities/model-predictions/clustering/";
};

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
            std::string save_type,
            std::string model_filename,
            arma::colvec model_predictions
        );
};
