#include <map>
#include <string>
#include <tuple>
#include <algorithm>
#include <armadillo>
#include <filesystem>
#include "base.hpp"
#include "model_types.hpp"

HashMapParameters BaseEstimator::get_parameters (HashMapParameters hyperparameter_hashmap) {
    return hyperparameter_hashmap;
}

void BaseEstimator::set_parameters (
    HashMapParameters user_hyperparameters,
    HashMapParameters original_hyperparameters
) {
    try {
        for (const auto& [key, value] : user_hyperparameters) {
            if (original_hyperparameters.find(key) == original_hyperparameters.end()) {
                throw std::invalid_argument("Key to change did not appear in the original hyperparameters");
            } else {
                original_hyperparameters[key] = value;
            }
        }
    } catch (const std::invalid_argument& non_existent_hyperparameter) {
        std::cout << "[-] Error: " << non_existent_hyperparameter.what() << std::endl;
    }
}

// nopsled as child classess will override definition
void BaseEstimator::fit (arma::mat& train_x, arma::colvec& train_y) {
    ;
}

// nopsled as child classess will override definition
arma::colvec BaseEstimator::predict (arma::mat& test_x) {
    ;
}

// nopsled as child classess will override definition
arma::rowvec BaseEstimator::predict_proba (arma::mat& test_x) {
    ;
}

// nopsled as child classess will override definition
arma::rowvec BaseEstimator::predict_proba_log (arma::mat& test_x) {
    ;
}

// nopsled as child classess will override definition
void BaseEstimator::save_model (
    const std::string model_save_file,
    const std::filesystem::path save_path
) {
    ;
}

// nopsled as child classess will override definition
void BaseEstimator::load_model (std::filesystem::path load_path) {
    ;
}
