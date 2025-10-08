#include <map>
#include <string>
#include <tuple>
#include <algorithm>
#include <armadillo>
#include <filesystem>
#include "../include/class-inheritors/base.hpp"
#include "../include/complex-datatypes/complex.hpp"

HashMapParameters BaseEstimator::get_parameters (HashMapParameters hyperparameter_hashmap) {
    return hyperparameter_hashmap;
}

void BaseEstimator::set_parameters (
    HashMapParameters user_hyperparams,
    HashMapParameters orig_hyperparams
) {
    try {
        for (const auto& [key, value] : user_hyperparams) {
            if (orig_hyperparams.find(key) == orig_hyperparams.end()) {
                throw "Key to change did not appear in the original hyperparameters";
            } else {
                orig_hyperparams[key] = value;
            }
        }
    } catch (std::string non_existent_key) {
        std::cout << "[-] Error: " << non_existent_key << std::endl;
    }
}

void BaseEstimator::save_model (
    const std::string model_save_file,
    const std::filesystem::path save_path
) {

}

void BaseEstimator::load_model (std::filesystem::path load_path) {

}
