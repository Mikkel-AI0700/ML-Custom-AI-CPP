#ifndef BASE_ESTIMATOR_HPP
#define BASE_ESTIMATOR_HPP

#include <any>
#include <variant>
#include <tuple>
#include <map>
#include <string>
#include <filesystem>
#include <armadillo>

using namespace arma;

class BaseEstimator {
    public:
        BaseEstimator::~BaseEstimator() = default;

        virtual std::map<std::string, std::variant<int, float, std::string>> get_parameters ();
        virtual void set_parameters (
            std::map<std::string, std::variant<int, float, std::string>> original_hyperparameters,
            std::map<std::string, std::variant<int, float, std::string>> user_hyperparameters
        );

        virtual void fit (mat train_x, rowvec train_y);
        virtual arma::rowvec predict (mat test_x);
        virtual arma::rowvec predict_proba (mat test_x);
        virtual arma::rowvec predict_proba_log (mat test_x);

        virtual void save_model (
            std::string model_save_name,
            std::filesystem::path save_path
        );
        virtual void load_model (
            std::filesystem::path load_path
        );
};

#endif

