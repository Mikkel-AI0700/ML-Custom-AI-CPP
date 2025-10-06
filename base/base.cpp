#include <any>
#include <variant>
#include <map>
#include <string>
#include <filesystem>
#include <armadillo>
#include "../include/class-inheritors/base.hpp"

std::map<std::string, std::variant<int, float, std::string>> BaseEstimator::get_parameters () {

}

void BaseEstimator::set_parameters (
    std::map<std::string, std::variant<int, float, std::string>> user_hyperparameters,
    std::map<std::string, std::variant<int, float, std::string>> original_hyperparameters
) {

}

void BaseEstimator::save_model (
    const std::string model_save_file,
    const std::filesystem::path save_path
) {

}

void BaseEstimator::load_model (std::filesystem::path load_path) {

}
