#pragma once
#include <vector>
#include <variant>
#include <armadillo>
#include <filesystem>

using DatasetMatrix = std::variant<arma::mat, arma::rowvec, arma::colvec>;
using TrainTestData = std::vector<DatasetMatrix>;
using DatasetsFilepaths = std::vector<std::filesystem::path>;
