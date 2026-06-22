#pragma once
#include <vector>
#include <variant>
#include <armadillo>
#include <filesystem>

using TrainTestData = std::vector<
    std::variant<arma::mat, arma::rowvec, arma::colvec>
>;

using DatasetsFilepaths = std::vector<std::filesystem::path>;
