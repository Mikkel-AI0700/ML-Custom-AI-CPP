#include <iostream>
#include <cmath>
#include <map>
#include <algorithm>
#include <generator>
#include <vector>
#include <variant>
#include <cstdlib>
#include <armadillo>
#include "loader.hpp"
#include "tree-header.hpp"
#include "linalg-operations.hpp"

arma::vec DecisionTreeClassifier::compute_class_probability (arma::vec& Y) {
    std::vector<int> labels;
    std::vector<int> label_counts;
    UniqueFunctionReturns unq_ret = unique(Y, true);
    
    if (unique_classes.empty()) {
        unique_classes = unq_ret.labels;
        prob_vec = arma::zeros(unique_classes.size());

        for (int i = 0; i < unq_ret.labels.size(); i++) {
            classes_to_index.insert({i, unq_ret.labels[i]});
        }
    } else {
        prob_vec = arma::zeros(unq_ret.labels.size());
    }

    for (int i = 0; i < unq_ret.labels.size(); i++) {
        if (classes_to_index.contains(unq_ret.labels[i])) {
            int index = classes_to_index.at(unq_ret.labels[i]);
            prob_vec.at(classes_to_index.at(index)) = unq_ret.label_counts[i] / Y.n_elem;
        }
    }

    return prob_vec;
}

float DecisionTreeClassifier::compute_impurity (arma::vec& Y) {
    arma::vec prob_vec = DecisionTreeClassifier::compute_class_probability(Y);
    float imp_val = 1 - arma::sum(arma::square(prob_vec));
    return imp_val;
}

float DecisionTreeClassifier::compute_entropy (arma::vec& Y) {
    arma::vec prob_vec = DecisionTreeClassifier::compute_class_probability(Y);
    arma::uvec pos_idx = arma::find(prob_vec > 0.0);
    arma::vec pos_prob = prob_vec.elem(pos_idx);
    float ent_val = -(arma::sum(pos_prob * arma::log2(pos_prob)));
    return ent_val;
}

float DecisionTreeClassifier::compute_information_gain (
    arma::vec& Y, 
    arma::mat& left_subset,
    arma::mat& right_subset
) {

}

float DecisionTreeClassifier::determine_impurity_metric (arma::vec& Y) {
    if (split_metric == "gini") {
        return DecisionTreeClassifier::compute_impurity(Y);
    } else if (split_metric == "entropy") {
        return DecisionTreeClassifier::compute_entropy(Y);
    } else {
        //return DecisionTreeClassifier::compute_log_loss();
    }
}

std::vector<int> DecisionTreeClassifier::determine_feature_split_metric (std::vector<int> feature_list) {
    try {
        auto string_ptr = std::get_if<std::string>(&max_features);
        auto integer_ptr = std::get_if<int>(&max_features);
        std::vector<int> selected_features;
        int math_length;
        int generated_number;

        if (!string_ptr || !integer_ptr) {
            return;
        }

        if (std::get<std::string>(max_features) == "sqrt") {
            math_length = abs(sqrt(feature_list.size()));
        } else if (std::get<std::string>(max_features) == "log2") {
            math_length = abs(log2(feature_list.size()));
        } else {
            selected_features = feature_list;
            return selected_features;
        }

        for (int index = 0; index < math_length; index++) {
            generated_number = rand() % math_length;
            auto math_mem_end = std::find(
                selected_features.begin(),
                selected_features.end(),
                generated_number
            );
            if (math_mem_end != selected_features.end()) {
                selected_features.push_back(generated_number);
            } else {
                continue;
            }
        }

        return selected_features;
    } catch (const std::bad_variant_access& error) {
        std::cout << "Error: " << error.what() << std::endl;
    }
}

std::generator<GeneratorVariables*> DecisionTreeClassifier::split_yield (
    SplitYieldParameters *yield_params,
    GeneratorVariables *gen_var
) {
    for (int index = 0; index < yield_params->num_cols.size(); index++) {
        arma::vec column_subview = yield_params->x_feat_mat.col(
            yield_params->num_cols.size()
        );

        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        arma::vec labels = arma::conv_to<arma::vec>::from(subview_unq.labels);
        arma::vec midpoints = (
            labels.subvec(0, labels.n_elem - 2) + 
            labels.subvec(1, labels.n_elem - 1)
        ) / 2.0;

        for (int inner_index = 0; inner_index < midpoints.n_elem; inner_index++) {
            arma::uvec left_idx = arma::find(
                yield_params->x_feat_mat > midpoints[inner_index]
            );
            arma::uvec right_idx = arma::find(
                yield_params->x_feat_mat <= midpoints[inner_index]
            );

            gen_var->split_kind = "numeric";
            gen_var->split_idx = yield_params->num_cols[index];
            gen_var->num_thresh = midpoints[inner_index];
            gen_var->left_x_mat = yield_params->x_feat_mat.rows(left_idx);
            gen_var->left_y_vec = yield_params->y_target_vec.rows(left_idx);
            gen_var->right_x_mat = yield_params->x_feat_mat.rows(right_idx);
            gen_var->right_y_vec = yield_params->y_target_vec.rows(right_idx);

            co_yield gen_var;
        }
    }

    for (int index = 0; index < yield_params->cat_cols.size(); index++) {
        arma::vec column_subview = yield_params->x_feat_mat.col(
            yield_params->cat_cols[index]
        );
        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        
        for (int inner_index = 0; inner_index < subview_unq.labels.size(); inner_index++) {
            arma::uvec left_idx = arma::find(
                yield_params->x_feat_mat == subview_unq.labels[inner_index]
            );
            arma::uvec right_idx = arma::find(
                yield_params->x_feat_mat != subview_unq.labels[inner_index]
            );

            gen_var->split_kind = "categorical";
            gen_var->split_idx = yield_params->cat_cols[index];
            gen_var->cat_thresh = subview_unq.labels[inner_index];
            gen_var->left_x_mat = yield_params->x_feat_mat.rows(left_idx);
            gen_var->left_y_vec = yield_params->y_target_vec.rows(left_idx);
            gen_var->right_x_mat = yield_params->x_feat_mat.rows(right_idx);
            gen_var->right_y_vec = yield_params->y_target_vec.rows(right_idx);

            co_yield gen_var;
        }
    }
}

std::variant<LeafNode, DecisionNode> DecisionTreeClassifier::create_node (
    CreateNodeParameters& node_parameter
) {
    
}

std::variant<LeafNode, DecisionNode> DecisionTreeClassifier::build_decision_tree (
    arma::mat& X,
    arma::vec& Y,
    int recursive_max_depth
) {

}

int main () {

}

 
