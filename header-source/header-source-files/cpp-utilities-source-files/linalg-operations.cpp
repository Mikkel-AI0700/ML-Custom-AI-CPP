#include <vector>
#include <variant>
#include <armadillo>
#include "linalg-operations.hpp"

using std::vector;
using std::variant;
using arma::vec;
using arma::ivec;

UniqueFunctionReturns unique (vec& Y, bool return_counts) {
    std::vector<int> discovered_labels;
    std::vector<int> discovered_label_counts;
    UniqueFunctionReturns u_func_ret;

    vec temp_subview = arma::sort(Y);
    ivec converted_subview = arma::conv_to<ivec>::from(temp_subview);

    for (int index = 1; index < converted_subview.n_rows; index++) {
        if (index == 1) {
            discovered_labels.push_back(converted_subview[0]);
        }

        if (converted_subview[index] != converted_subview[index - 1]) {
            discovered_labels.push_back(converted_subview[index]);
        }
    }

    u_func_ret.labels = discovered_labels;

    if (return_counts) {
        for (int outer_index = 0; outer_index < discovered_labels.size(); outer_index++) {
            int label_counter = 0;
            for (int inner_index = 0; inner_index < converted_subview.n_rows; inner_index++) {
                if (converted_subview[inner_index] == discovered_labels[outer_index]) {
                    label_counter++;
                }
            }
            discovered_label_counts.push_back(label_counter);
        }
        u_func_ret.label_counts = discovered_label_counts;
    }

    return u_func_ret;
}
