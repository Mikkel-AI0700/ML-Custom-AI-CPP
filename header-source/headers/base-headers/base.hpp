#pragma once
#include <string>
#include <filesystem>
#include <armadillo>
#include "model_types.hpp"

class BaseEstimator {
    public:
        virtual ~BaseEstimator() = default;

        virtual HashMapParameters get_parameters (
            HashMapParameters hyperparameter_hashmap
        );
        virtual void set_parameters (
            HashMapParameters original_hyperparameters, 
            HashMapParameters user_hyperparameters
        );

        virtual void fit (const arma::mat& train_x, const arma::vec& train_y);
        virtual PredictionResult predict (InputFeatures& test_x);
        virtual ProbabilityResult predict_proba (InputFeatures& test_x);
        virtual ProbabilityResult predict_proba_log (InputFeatures& test_x);

        virtual void save_model (std::string model_save_name, std::filesystem::path save_path);
        virtual void load_model (std::filesystem::path load_path);
};
