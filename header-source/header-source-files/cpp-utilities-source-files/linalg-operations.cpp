#include <vector>
#include <armadillo>
#include "linalg-operations.hpp"

UniqueFunctionReturns unique (arma::mat& Y, bool return_counts) {
    std::vector<int> discovered_labels;
    std::vector<int> discovered_label_counts;
    UniqueFunctionReturns u_func_ret;

    for (int index = 0; index < Y.n_elem; index++) {
        if (index == 0) {
            continue;
        }

        if (Y[index] != Y[index - 1]) {
            discovered_labels.push_back(Y[index]);
        }
    }
    u_func_ret.labels = discovered_labels;

    if (return_counts) {
        for (int outer_index = 0; outer_index < discovered_labels.size(); outer_index++) {
            int label_counter = 0;
            for (int inner_index = 0; inner_index < Y.n_elem; inner_index++) {
                if (Y[outer_index] == discovered_labels[inner_index]) {
                    label_counter++;
                }
            }
            discovered_label_counts.push_back(label_counter);
        }
        u_func_ret.label_counts = discovered_label_counts;
    }
}
