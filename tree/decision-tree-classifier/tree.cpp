#include <map>
#include <memory>
#include <vector>
#include <cstdlib>
#include <variant>
#include <optional>
#include <algorithm>
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
        probability_vector = arma::zeros(unique_classes.size());

        for (int i = 0; i < unq_ret.labels.size(); i++) {
            classes_to_index.insert({i, unq_ret.labels[i]});
        }
    } else {
        probability_vector = arma::zeros(unq_ret.labels.size());
    }

    for (int i = 0; i < unq_ret.labels.size(); i++) {
        if (classes_to_index.contains(unq_ret.labels[i])) {
            int index = classes_to_index.at(unq_ret.labels[i]);
            probability_vector.at(classes_to_index.at(index)) = unq_ret.label_counts[i] / Y.n_elem;
        }
    }

    return prob_vec;
}

float DecisionTreeClassifier::compute_impurity (vec& Y) {
    vec prob_vec = DecisionTreeClassifier::compute_class_probability(Y);
    float imp_val = 1 - sum(arma::square(prob_vec));
    return imp_val;
}

float DecisionTreeClassifier::compute_entropy (vec& Y) {
    vec prob_vec = DecisionTreeClassifier::compute_class_probability(Y);
    uvec pos_idx = arma::find(prob_vec > 0.0);
    vec pos_prob = prob_vec.elem(pos_idx);
    float ent_val = -(arma::sum(pos_prob % arma::log2(pos_prob)));
    return ent_val;
}

float DecisionTreeClassifier::compute_log_loss (vec& Y, vec& prob_vec) {

}

float DecisionTreeClassifier::compute_information_gain (
    vec& Y,
    vec& left_subset,
    vec& right_subset
) {
    float main_data_imp = DecisionTreeClassifier::determine_impurity_metric(Y);
    float left_subset_imp = DecisionTreeClassifier::determine_impurity_metric(left_subset);
    float right_subset_imp = DecisionTreeClassifier::determine_impurity_metric(right_subset);

    float left_subset_weighted = (static_cast<float>(left_subset.n_rows) / static_cast<float>(Y.n_rows)) * left_subset_imp;
    float right_subset_weighted = (static_cast<float>(right_subset.n_rows) / static_cast<float>(Y.n_rows)) * right_subset_imp;

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

        if (std::holds_alternative<string>(max_feature)) {
            if (std::get<string>(max_feature) == "sqrt") {
                math_length = static_cast<int>(sqrt(feature_list.size()));
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
    } catch (const bad_variant_access& error) {
        cout << "Error: " << error.what();
        return {};
    }
}

generator<GeneratorVariables&> DecisionTreeClassifier::split_yield (
    SplitYieldParameters &yield_params,
    GeneratorVariables &gen_var
) {
    SplitYieldParameters *yield_parameter = yield_parameter;
    GeneratorVariables *gen_var = generator_variables;
    
    for (int index = 0; index < yield_parameter->num_feats.size(); index++) {
        arma::vec column_subview = yield_parameter->X.col(
            yield_parameter->num_feats.size()
        );

        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        arma::vec labels = arma::conv_to<arma::vec>::from(subview_unq.labels);
        arma::vec midpoints = (
            labels.subvec(0, labels.n_elem - 2) + 
            labels.subvec(1, labels.n_elem - 1)
        ) / 2.0;

        for (int inner_index = 0; inner_index < midpoints.n_elem; inner_index++) {
            arma::uvec left_satisfying_indices = arma::find(
                yield_parameters->X > midpoints[inner_index]
            );
            arma::uvec right_satisfying_indices = arma::find(
                yield_parameters->X <= midpoints[inner_index]
            );

            gen_var->split_type = "numeric";
            gen_var->split_index = yield_parameter->num_feats[index];
            gen_var->best_temporary_numeric_threshold = midpoints[inner_index];
            gen_var->left_x_subset = yield_parameter->X.rows(left_satisfying_indices);
            gen_var->left_y_subset = yield_parameter->Y.rows(left_satisfying_indices);
            gen_var->right_x_subset = yield_parameter->X.rows(right_satisfying_indices);
            gen_var->right_y_subset = yield_parameter->Y.rows(right_satisfying_indices);            
        }

        co_yield gen_var;
    }

    for (int index = 0; index < yield_parameter->cat_feats.size(); index++) {

    for (int index = 0; index < categorical_features.size(); index++) {
        vec column_subview = yield_params.x_feat_mat.col(
            categorical_features[index]
        );
        UniqueFunctionReturns subview_unq = unique(column_subview, false);

        for (int inner_index = 0; inner_index < subview_unq.labels.size(); inner_index++) {
            uvec left_idx = arma::find(
                yield_params.x_feat_mat.col(categorical_features[index]) == subview_unq.labels[inner_index]
            );
            uvec right_idx = arma::find(
                yield_params.x_feat_mat.col(categorical_features[index]) != subview_unq.labels[inner_index]
            );

            gen_var.split_kind = "categorical";
            gen_var.split_idx = categorical_features[index];
            gen_var.cat_thresh = subview_unq.labels[inner_index];
            gen_var.left_x_mat = yield_params.x_feat_mat.rows(left_idx);
            gen_var.left_y_vec = yield_params.y_target_vec.rows(left_idx);
            gen_var.right_x_mat = yield_params.x_feat_mat.rows(right_idx);
            gen_var.right_y_vec = yield_params.y_target_vec.rows(right_idx);

            co_yield gen_var;
        }
    }
}

unique_ptr<Node> DecisionTreeClassifier::create_node (
    BestCandidateSplit& best_candidate_var,
    bool create_decision_node,
    bool create_leaf_node
) {
    auto node = std::make_unique<Node>();

    if (create_decision_node) {
        if (best_candidate_var.num_cond.has_value()) {
            node->num_condition = best_candidate_var.num_cond;
        }

        if (best_candidate_var.cat_cond.has_value()) {
            node->cat_condition = best_candidate_var.cat_cond;
        }

        node->is_decision_node = true;
        node->split_index = best_candidate_var.best_idx;
    }

    if (create_leaf_node) {
        node->is_leaf_node = true;
        node->computed_probabilities = best_candidate_var.computed_probs;
    }

    return node;
}

unique_ptr<Node> DecisionTreeClassifier::build_decision_tree (
    mat& X,
    vec& Y,
    int recursive_max_depth
) {
    BestCandidateSplit best_candidate_var;
    SplitYieldParameters splt_yld;
    GeneratorVariables gen_var;

    // Explicitly assigning the X and Y matrix and vector
    splt_yld.x_feat_mat = X;
    splt_yld.y_target_vec = Y;

    if (recursive_max_depth == max_depth) {
        cout << "[*] Stopping training. Max depth hit" << std::endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return node;
    }

    if (X.n_rows <= min_samples_split) {
        cout << "[*] Stopping training. Min samples split hit" << std::endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return node;
    }

    for (GeneratorVariables& gen_var : DecisionTreeClassifier::split_yield(splt_yld, gen_var)) {
        float crt_inf_gain = DecisionTreeClassifier::compute_information_gain(
            Y,
            gen_var.left_y_vec,
            gen_var.right_y_vec
        );

        if (gen_var.split_kind == "numeric") {
            if (crt_inf_gain > best_candidate_var.best_gain) {
                best_candidate_var.best_gain = crt_inf_gain;
                best_candidate_var.best_idx = gen_var.split_idx;
                best_candidate_var.num_cond = gen_var.num_thresh;

                // Left and Right decision matrices
                best_candidate_var.left_x_mat = gen_var.left_x_mat;
                best_candidate_var.right_x_mat = gen_var.right_x_mat;

                // Left and Right decision split vectors
                best_candidate_var.left_y_vec = gen_var.left_y_vec;
                best_candidate_var.right_y_vec = gen_var.right_y_vec;
            }
        }

        if (gen_var.split_kind == "categorical") {
            if (crt_inf_gain > best_candidate_var.best_gain) {
                best_candidate_var.best_gain = crt_inf_gain;
                best_candidate_var.best_idx = gen_var.split_idx;
                best_candidate_var.cat_cond = gen_var.cat_thresh;

                // Left and Right decision matrices
                best_candidate_var.left_x_mat = gen_var.left_x_mat;
                best_candidate_var.right_x_mat = gen_var.right_x_mat;

                // Left and Right decision vectors
                best_candidate_var.left_y_vec = gen_var.left_y_vec;
                best_candidate_var.right_y_vec = gen_var.right_y_vec;
            }
        }
    }

    if (best_candidate_var.best_gain < min_information_gain) {
        cout << "[*] Stopping training! Minimum Information Gain hit.";
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto leaf_node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return leaf_node;
    }

    if (best_candidate_var.left_y_vec.n_rows == 0 || best_candidate_var.right_y_vec.n_rows == 0) {
        if (best_candidate_var.left_y_vec.n_rows < min_samples_leaf ||
            best_candidate_var.right_y_vec.n_rows < min_samples_leaf
        ) {
            cout << "[*] Stopping training! Minimum samples per leaf hit.";
            best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
            auto leaf_node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
            return leaf_node;
        }
    }

    auto decision_node = DecisionTreeClassifier::create_node(
        best_candidate_var,
        true,
        false
    );

    decision_node->left_branch = DecisionTreeClassifier::build_decision_tree(
        best_candidate_var.left_x_mat,
        best_candidate_var.left_y_vec,
        recursive_max_depth + 1
    );

    decision_node->right_branch = DecisionTreeClassifier::build_decision_tree(
        best_candidate_var.right_x_mat,
        best_candidate_var.right_y_vec,
        recursive_max_depth + 1
    );

    return decision_node;
}

int DecisionTreeClassifier::traverse_tree_prediction (
    mat element,
    const unique_ptr<Node>& node
) {
    if (node->is_leaf_node) {
        vector<double> computed_probabilities(
            node->computed_probabilities.begin(),
            node->computed_probabilities.end()
        );

        auto max_elem_ptr = std::max_element(
            computed_probabilities.begin(),
            computed_probabilities.end()
        );

        return std::distance(
            computed_probabilities.begin(),
            max_elem_ptr
        );
    }

    if (node->num_condition.has_value()) {
        float extracted_num_cond = node->num_condition.value();
        if (element.at(node->split_index) > extracted_num_cond) {
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
        float extracted_cat_cond = node->cat_condition.value();
        if (element.at(node->split_index) == extracted_cat_cond) {
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

    throw std::runtime_error("[-] Error: Unexpected corrupted node encountered.");
}

void DecisionTreeClassifier::fit (mat& X, vec& Y) {
    int recursive_max_depth = 1;
    root_node = DecisionTreeClassifier::build_decision_tree(
        X,
        Y,
        recursive_max_depth
    );
}

vec DecisionTreeClassifier::predict (mat& X) {
    vector<float> predictions;

    if (unique_classes.empty()) {
        cout << "[*] Error: DecisionTreeClassifier has not been fitted yet";
    }

    for (int index = 0; index < X.n_rows; index++) {
        mat row = X.row(index);
        predictions.push_back(
            DecisionTreeClassifier::traverse_tree_prediction(
                row,
                root_node
            )
        );
    }

    return arma::conv_to<vec>::from(predictions);
}

int main () {

}

 
