#ifndef LINEAR_REGRESSION_COMPLEX_HPP
#define LINEAR_REGRESSION_COMPLEX_HPP

#include <vector>
#include <variant>
#include <filesystem>
#include <armadillo>

using TrainTestData = std::vector<std::variant<arma::mat, arma::rowvec, arma::colvec>>;
using DatasetsFilepaths = std::vector<std::filesystem::path>;

#endif