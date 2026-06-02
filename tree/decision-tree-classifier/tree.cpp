#include <map>
#include <memory>
#include <vector>
#include <cstdlib>
#include <variant>
#include <algorithm>
#include <optional>
#include <generator>
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

float DecisionTreeClassifier::compute_log_loss (arma::vec& Y, arma::vec& prob_vec) {

}

float DecisionTreeClassifier::compute_information_gain (
    arma::vec& Y, 
    arma::vec& left_subset,
    arma::vec& right_subset
) {
    float main_data_imp = DecisionTreeClassifier::determine_impurity_metric(Y);
    float left_subset_imp = DecisionTreeClassifier::determine_impurity_metric(left_subset);
    float right_subset_imp = DecisionTreeClassifier::determine_impurity_metric(right_subset);

    float left_subset_weighted = left_subset.n_rows / Y.n_rows * left_subset_imp;
    float right_subset_weighted = right_subset.n_rows / Y.n_rows * right_subset_imp;

    return main_data_imp - (left_subset_weighted + right_subset_weighted);
}

float DecisionTreeClassifier::determine_impurity_metric (vec& Y) {
    if (split_metric == "gini") {
        return DecisionTreeClassifier::compute_impurity(Y);
    } else if (split_metric == "entropy") {
        return DecisionTreeClassifier::compute_entropy(Y);
    } else {
        //return DecisionTreeClassifier::compute_log_loss();
    }
}

vector<int> DecisionTreeClassifier::determine_feature_split_metric (vector<int> feature_list) {
    try {
        vector<int> selected_features;
        int math_length;
        int generated_number;

        if (!string_ptr || !integer_ptr) {
            return selected_features;
        }

            if (std::get<string>(max_feature) == "log2") {
                math_length = static_cast<int>(log2(feature_list.size()));
            }
        } else {
            return feature_list;
        }

        // Manual random feature selection
        while (selected_features.size() < math_length) {
            generated_number = rand() % feature_list.size();
            auto math_mem_end = std::find(
                selected_features.begin(),
                selected_features.end(),
                generated_number
            );
            if (math_mem_end == selected_features.end()) {
                selected_features.push_back(
                    feature_list[generated_number]
                );
            } else {
                continue;
            }
        }

        return selected_features;
    } catch (const std::bad_variant_access& error) {
        std::cout << "Error: " << error.what();
    }
}

std::generator<GeneratorVariables&> DecisionTreeClassifier::split_yield (
    SplitYieldParameters &yield_params,
    GeneratorVariables &gen_var
) {
    for (int index = 0; index < yield_params.num_cols.size(); index++) {
        arma::vec column_subview = yield_params.x_feat_mat.col(
            yield_params.num_cols.size()
        );

        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        arma::vec labels = arma::conv_to<arma::vec>::from(subview_unq.labels);
        arma::vec midpoints = (
            labels.subvec(0, labels.n_elem - 2) + 
            labels.subvec(1, labels.n_elem - 1)
        ) / 2.0;

        for (int inner_index = 0; inner_index < midpoints.n_elem; inner_index++) {
            arma::uvec left_idx = arma::find(
                yield_params.x_feat_mat > midpoints[inner_index]
            );
            arma::uvec right_idx = arma::find(
                yield_params.x_feat_mat <= midpoints[inner_index]
            );

            gen_var.split_kind = "numeric";
            gen_var.split_idx = yield_params.num_cols[index];
            gen_var.num_thresh = midpoints[inner_index];
            gen_var.left_x_mat = yield_params.x_feat_mat.rows(left_idx);
            gen_var.left_y_vec = yield_params.y_target_vec.rows(left_idx);
            gen_var.right_x_mat = yield_params.x_feat_mat.rows(right_idx);
            gen_var.right_y_vec = yield_params.y_target_vec.rows(right_idx);

            co_yield gen_var;
        }
    }

    for (int index = 0; index < yield_params.cat_cols.size(); index++) {
        arma::vec column_subview = yield_params.x_feat_mat.col(
            yield_params.cat_cols[index]
        );
        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        
        for (int inner_index = 0; inner_index < subview_unq.labels.size(); inner_index++) {
            arma::uvec left_idx = arma::find(
                yield_params.x_feat_mat == subview_unq.labels[inner_index]
            );
            arma::uvec right_idx = arma::find(
                yield_params.x_feat_mat != subview_unq.labels[inner_index]
            );

            gen_var.split_kind = "categorical";
            gen_var.split_idx = yield_params.cat_cols[index];
            gen_var.cat_thresh = subview_unq.labels[inner_index];
            gen_var.left_x_mat = yield_params.x_feat_mat.rows(left_idx);
            gen_var.left_y_vec = yield_params.y_target_vec.rows(left_idx);
            gen_var.right_x_mat = yield_params.x_feat_mat.rows(right_idx);
            gen_var.right_y_vec = yield_params.y_target_vec.rows(right_idx);

            co_yield gen_var;
        }
    }
}

std::unique_ptr<Node> DecisionTreeClassifier::create_node (
    CreateNodeParameters& node_parameter
) {
    auto node = std::make_unique<Node>();

    if (node_parameter.make_decision_node) {
        auto num_feat_ptr = node_parameter.num_feats;
        auto cat_feat_ptr = node_parameter.cat_feats;

        if (!num_feat_ptr) {
            node->num_condition = node_parameter.num_feats;
        }

        if (!cat_feat_ptr) {
            node->cat_condition = node_parameter.cat_feats;
        }

        node->is_decision_node = true;
        node->split_index = node_parameter.split_idx;
    }

    if (node_parameter.make_leaf_node) {
        node->is_leaf_node = true;
        node->computed_probabilities = node_parameter.class_probs;
    }

    return node;
}

std::unique_ptr<Node> DecisionTreeClassifier::build_decision_tree (
    arma::mat& X,
    arma::vec& Y,
    int recursive_max_depth
) {
    CreateNodeParameters node_param;
    RecursiveTreeBuilder rec_build;
    SplitYieldParameters splt_yld;
    GeneratorVariables gen_var;

    if (recursive_max_depth == max_depth) {
        std::cout << "[*] Stopping training. Max depth hit" << std::endl;
        auto node = DecisionTreeClassifier::create_node(node_param);
        return node;
    }

    if (X.n_rows <= min_samples_split) {
        std::cout << "[*] Stopping training. Min samples split hit" << std::endl;
        auto node = DecisionTreeClassifier::create_node(node_param);
        return node;
    }

    for (GeneratorVariables& gen_var : DecisionTreeClassifier::split_yield(splt_yld, gen_var)) {
        float crt_inf_gain = DecisionTreeClassifier::compute_information_gain(
            Y,
            gen_var.left_y_vec,
            gen_var.right_y_vec
        );

        if (gen_var.split_kind == "numeric") {
            if (crt_inf_gain > rec_build.best_gain) {
                rec_build.best_gain = crt_inf_gain;
                rec_build.best_idx = gen_var.split_idx;
                rec_build.num_cond = gen_var.num_thresh;

                // Left and Right decision matrices
                rec_build.left_x_mat = gen_var.left_x_mat;
                rec_build.right_x_mat = gen_var.right_x_mat;

                // Left and Right decision split vectors
                rec_build.left_y_vec = gen_var.left_y_vec;
                rec_build.right_y_vec = gen_var.right_y_vec;
            }
        } else {
            continue;
        }

        if (gen_var.split_kind == "categorical") {
            if (crt_inf_gain > rec_build.best_gain) {
                rec_build.best_gain = crt_inf_gain;
                rec_build.best_idx = gen_var.split_idx;
                rec_build.cat_cond = gen_var.cat_thresh;

                // Left and Right decision matrices
                rec_build.left_x_mat = gen_var.left_x_mat;
                rec_build.right_x_mat = gen_var.right_x_mat;

                // Left and Right decision vectors
                rec_build.left_y_vec = gen_var.left_y_vec;
                rec_build.right_y_vec = gen_var.right_y_vec;
            }
        } else {
            continue;
        }
    }

    if (rec_build.best_gain < min_information_gain) {
        std::cout << "[*] Stopping training! Minimum Information Gain hit.";
        auto leaf_node = DecisionTreeClassifier::create_node(node_param);
        return leaf_node;
    }

    if (rec_build.left_y_vec.n_rows != 0 || rec_build.right_y_vec.n_rows != 0) {
        if (rec_build.left_y_vec.n_rows < min_samples_leaf || 
            rec_build.right_y_vec.n_rows < min_samples_leaf
        ) {
            std::cout << "[*] Stopping training! Minimum samples per leaf hit.";
            auto leaf_node = DecisionTreeClassifier::create_node(node_param);
            return leaf_node;
        }
    }

    auto decision_node = DecisionTreeClassifier::create_node(node_param);

    // Left branch of the tree
    decision_node->left_branch = DecisionTreeClassifier::build_decision_tree(
        rec_build.left_x_mat,
        rec_build.left_y_vec,
        ++recursive_max_depth
    );

    // Right branch of the tree
    decision_node->right_branch = DecisionTreeClassifier::build_decision_tree(
        rec_build.right_x_mat,
        rec_build.right_y_vec,
        ++recursive_max_depth
    );

    return decision_node;
}

float DecisionTreeClassifieer::traverse_tree_prediction (
    arma::vec element, 
    std::unique_ptr<Node> node
) {
    if (node->is_leaf_node) {
        // Logic here
    }

    if (node->num_condition.has_value()) {
        if (element.at(node->split_index) > node->num_condition) {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->left_branch
            );
        } else {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->right_branch
            );
        }
    }

    if (node->cat_condition.has_value()) {
        float float_cat_cond = std::get<float>(node->cat_condition);
        if (element.at(node->split_index) == node->float_cat_cond) {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->left_branch
            );
        } else {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->right_branch
            );
        }
    }

    // Big TODO for you Sebastien: Figure out the mess
    // you made with the Node struct and figure out how the tree
    // knows if it's going left or right.
}

int main () {

}

 
