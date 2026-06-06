#pragma once
#include <vector>
#include <armadillo>

struct UniqueFunctionReturns {
    std::vector<int> labels;
    std::vector<int> label_counts;
};

UniqueFunctionReturns unique (arma::mat& Y, bool return_counts);
