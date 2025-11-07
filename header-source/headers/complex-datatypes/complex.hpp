#ifndef COMPLEX_DATATYPES_HPP
#define COMPLEX_DATATYPES_HPP

#include <vector>
#include <variant>
#include <map>
#include <armadillo>

using HashMapParameters = std::map<std::string, std::variant<int, float, std::string, bool>>;
using TrainTestData = std::vector<std::variant<arma::mat, arma::rowvec, arma::colvec>>;
using DatasetsFilepaths = std::vector<std::filesystem::path>;

#endif
