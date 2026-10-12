#pragma once
#include <map>
#include <string>
#include <vector>
#include <variant>
#include <armadillo>

// Hyperparameters
using HyperparameterValue = std::variant<int, float, std::string, bool>;
using HashMapParameters = std::map<std::string, HyperparameterValue>;

// Feature / sample subsampling types
using MaxFeatures = std::variant<int, float, std::string>;
using MaxSamples = std::variant<int, float>;
using SIDual = std::variant<std::string, int>;

// Model inference / I/O types
using InputFeatures = std::variant<arma::vec, arma::mat>;
using PredictionResult = std::variant<int, arma::vec>;
using ProbabilityResult = std::variant<arma::vec, arma::mat>;

// Feature value representations
using IntegerCategoricalList = std::variant<
    std::vector<int>,
    std::vector<float>
>;
