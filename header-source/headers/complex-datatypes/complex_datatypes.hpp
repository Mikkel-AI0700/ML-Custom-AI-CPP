#ifndef COMPLEX_DATATYPES_HPP
#define COMPLEX_DATATYPES_HPP

#include <vector>
#include <variant>
#include <map>
#include <armadillo>
#include <filesystem>

using HashMapParameters = std::map<
    std::string, 
    std::variant<int, float, std::string, bool>
>;

using TrainTestData = std::vector<
    std::variant<arma::mat, arma::rowvec, arma::colvec>
>;

using DatasetsFilepaths = std::vector<std::filesystem::path>;

using SIDual = std::variant<std::string, int>;

using IntegerCategoricalList = std::variant<
    std::vector<int>,
    std::vector<std::string>
>;

using YieldedInformation = std::vector<
    std::variant<
        std::string,
        int,
        std::variant<int, float, std::string>,
        arma::mat,
        arma::vec,
        arma::mat,
        arma::vec
    >
>;

#endif
