#ifndef BASE_ESTIMATOR_HPP
#define BASE_ESTIMATOR_HPP

#include <any>
#include <variant>
#include <tuple>
#include <map>
#include <string>
#include <filesystem>
#include <armadillo>
#include "../complex-datatypes/complex.hpp"

class BaseEstimator {
    public:
        ~BaseEstimator() = default;

        virtual HashMapParameters get_parameters (HashMapParameters hyperparameter_hashmap);
        virtual void set_parameters (HashMapParameters original_hyperparameters, HashMapParameters user_hyperparameters);

        virtual void fit (arma::mat train_x, arma::rowvec train_y);
        virtual arma::rowvec predict (arma::mat test_x);
        virtual arma::rowvec predict_proba (arma::mat test_x);
        virtual arma::rowvec predict_proba_log (arma::mat test_x);

        virtual void save_model (std::string model_save_name, std::filesystem::path save_path);
        virtual void load_model (std::filesystem::path load_path);
};

#endif

