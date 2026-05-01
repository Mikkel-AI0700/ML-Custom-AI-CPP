#ifndef LIN_ALG_OPERATIONS
#define LIN_ALG_OPERATIONS

#include <vector>
#include <armadillo>

struct UniqueFunctionReturns {
    std::vector<int> labels;
    std::vector<int> label_counts;
};

UniqueFunctionReturns* unique (arma::vec& Y, bool return_counts);

#endif