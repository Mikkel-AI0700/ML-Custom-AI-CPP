#pragma once
#include <string>
#include <filesystem>
#include <armadillo>
#include "model_types.hpp"

class BaseEstimator {
    public:
        virtual ~BaseEstimator() = default;

        virtual HashMapParameters get_parameters (HashMapParameters hyperparameter_hashmap);
        virtual void set_parameters (HashMapParameters original_hyperparameters, HashMapParameters user_hyperparameters);

        virtual void fit (arma::mat& train_x, arma::vec& train_y);
        virtual std::variant<int, arma::vec> predict (std::variant<arma::vec, arma::mat>& test_x);
        virtual std::variant<arma::vec, arma::mat> predict_proba (std::variant<arma::vec, arma::mat>& test_x);
        virtual std::variant<arma::vec, arma::mat> predict_proba_log (std::variant<arma::vec, arma::mat>& test_x);

        virtual void save_model (std::string model_save_name, std::filesystem::path save_path);
        virtual void load_model (std::filesystem::path load_path);
};
