#pragma once
#include <vector>
#include <variant>
#include <armadillo>

struct UniqueFunctionReturns {
    std::vector<int> labels;
    std::vector<int> label_counts;
};

UniqueFunctionReturns unique (vec& Y, bool return_counts);
