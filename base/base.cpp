#include <any>
#include <variant>
#include <map>
#include <string>
#include <filesystem>
#include <armadillo>
#include "../include/class-inheritors/base.hpp"
#include "../include/complex-datatypes/complex.hpp"

HashMapParameters BaseEstimator::get_parameters (HashMapParameters hyperparameter_hashmap) {
    return hyperparameter_hashmap;
}

void BaseEstimator::set_parameters (
    HashMapParameters user_hyperparameters,
    HashMapParameters original_hyperparameters
) {

}

void BaseEstimator::save_model (
    const std::string model_save_file,
    const std::filesystem::path save_path
) {

}

void BaseEstimator::load_model (std::filesystem::path load_path) {

}
