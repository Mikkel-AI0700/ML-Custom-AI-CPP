#include <map>
#include <cmath>
#include <memory>
#include <vector>
#include <random>
#include <cstdlib>
#include <variant>
#include <optional>
#include <algorithm>
#include <generator>
#include <armadillo>
#include "loader.hpp"
#include "tree-header.hpp"
#include "linalg-operations.hpp"

// C++ STL library
using std::endl;
using std::cout;
using std::cerr;
using std::string;
using std::vector;
using std::variant;
using std::optional;
using std::generator;
using std::unique_ptr;
using std::out_of_range;
using std::runtime_error;
using std::invalid_argument;
using std::bad_variant_access;

// Armadillo library
using arma::mat;
using arma::vec;
using arma::ivec;
using arma::uvec;

DecisionTreeClassifier::DecisionTreeClassifier(
    string                              split_metric,
    int                                 max_depth,
    variant<int, float, string>         max_feature,
    int                                 max_leaf_nodes,
    int                                 min_samples_leaf,
    int                                 min_samples_split,
    float                               min_information_gain,
    vector<int>                         numerical_features,
    vector<int>                         categorical_features
):
    split_metric(split_metric),
    max_depth(max_depth),
    max_feature(max_feature),
    max_leaf_nodes(max_leaf_nodes),
    min_samples_leaf(min_samples_leaf),
    min_samples_split(min_samples_split),
    min_information_gain(min_information_gain),
    numerical_features(numerical_features),
    categorical_features(categorical_features)
{}

vec DecisionTreeClassifier::compute_class_probability (vec& Y) {
    vector<int> labels;
    vector<int> label_counts;
    vec prob_vec;
    UniqueFunctionReturns unq_ret = unique(Y, true);

    if (unique_classes.empty()) {
        unique_classes = unq_ret.labels;
        prob_vec = arma::zeros(unique_classes.size());

        for (int i = 0; i < unq_ret.labels.size(); i++) {
            classes_to_index.insert({unq_ret.labels[i], i});
        }

        return prob_vec;
    } else {
        prob_vec = arma::zeros(unique_classes.size());

        for (int i = 0; i < unq_ret.labels.size(); i++) {
            float cvtd_label_count = static_cast<float>(unq_ret.label_counts[i]);
            float cvtd_y_rows = static_cast<float>(Y.n_rows);

            if (classes_to_index.contains(unq_ret.labels[i])) {
                int index = classes_to_index.at(unq_ret.labels[i]);
                prob_vec.at(index) = cvtd_label_count / cvtd_y_rows;
            }
        }
        return prob_vec;
    }

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

float DecisionTreeClassifier::compute_information_gain (vec& Y, vec& left, vec& right) {
    if (left.n_rows == 0 || right.n_rows == 0) {
        return 0.0f;
    }

    /* Information Gain = parent_impurity - weighted_sum(child_impurities) */
    const float n_parent = static_cast<float>(Y.n_rows);
    const float weight_left = static_cast<float>(left.n_rows) / n_parent;
    const float weight_right = static_cast<float>(right.n_rows) / n_parent;

    return determine_impurity_metric(Y)
        - (weight_left * determine_impurity_metric(left)
         + weight_right * determine_impurity_metric(right));
}

float DecisionTreeClassifier::determine_impurity_metric (vec& Y) {
    if (split_metric == "gini") {
        return DecisionTreeClassifier::compute_impurity(Y);
    } else if (split_metric == "entropy") {
        return DecisionTreeClassifier::compute_entropy(Y);
    } else {
        throw invalid_argument("[-] Error: Invalid argument detected.");
    }
}

vector<int> DecisionTreeClassifier::determine_feature_split_metric (vector<int> feature_list) {
    try {
        vector<int> selected_features;
        int math_length;
        int generated_number;

        // New refactored code
        std::visit([this, &math_length, &feature_list](auto&& member) {
            using Type = std::decay_t<decltype(member)>;
            
            if constexpr (std::is_same_v<Type, int>) {
                math_length = member;
            } else if constexpr (std::is_same_v<Type, float>) {
                math_length = static_cast<int>(member * feature_list.size());
            } else if constexpr (std::is_same_v<Type, string>) {
                const string& tmp_feat_type = member;
                vector<string> valid_hyperparam = {"sqrt", "log2", "none"};
                
                bool is_max_feat_valid = std::any_of(
                    valid_hyperparam.begin(),
                    valid_hyperparam.end(),
                    [&tmp_feat_type](const string& hyperparam_value) {
                        if (tmp_feat_type == hyperparam_value) {
                            return true;
                        } else {
                            return false;
                        }
                    }
                );

                if (!is_max_feat_valid) {
                    throw invalid_argument("User-supplied argument invalid. Choices: sqrt/log2/none");
                }

                if (tmp_feat_type == "sqrt") {
                    math_length = static_cast<int>(sqrt(feature_list.size()));
                } else if (tmp_feat_type == "log2") {
                    math_length = static_cast<int>(log2(feature_list.size()));
                } else {
                    math_length = feature_list.size();
                }
            } else {
                throw bad_variant_access();
            }

            if (math_length > feature_list.size()) {
                throw out_of_range("Subsampling amount is greater than features expected");
            }
        }, max_feature);

        // Guard to prevent infinite while-loop recursion
        math_length = std::min(math_length, static_cast<int>(feature_list.size()));

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
        cerr << "Error: " << error.what();
        return feature_list;
    } catch (const runtime_error& error) {
        cerr << "Error: " << error.what();
        return feature_list;
    } catch (const out_of_range& error) {
        cerr << "Error: " << error.what();
        return feature_list;
    } catch (const invalid_argument& error) {
        cerr << "Error: " << error.what();
        return feature_list;
    }
}

generator<GeneratorVariables> DecisionTreeClassifier::split_yield (
    SplitYieldParameters &yield_params
) {
    GeneratorVariables gen_var;

    for (int index = 0; index < yield_params.subsampled_numerical_features.size(); index++) {
        vec column_subview = yield_params.x_feat_mat.col(
            yield_params.subsampled_numerical_features[index]
        );
        vec midpoints;

        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        vec labels = arma::conv_to<vec>::from(subview_unq.labels);

        // Small guard to check if feature still
        // still has any unique values to make a threshold midpoint
        if (subview_unq.labels.size() >= 2) {
            midpoints = (
                labels.subvec(0, labels.n_elem - 2) +
                labels.subvec(1, labels.n_elem - 1)
            ) / 2.0;
        } else {
            continue;
        }

        for (int inner_index = 0; inner_index < midpoints.n_rows; inner_index++) {
            gen_var.left_subset_indices = arma::find(
                column_subview > midpoints[inner_index]
            );
            gen_var.right_subset_indices = arma::find(
                column_subview <= midpoints[inner_index]
            );

            gen_var.split_kind = "numeric";
            gen_var.split_idx = yield_params.subsampled_numerical_features[index];
            gen_var.num_cond = midpoints[inner_index];

            co_yield gen_var;
        }
    }

    for (int index = 0; index < yield_params.subsampled_categorical_features.size(); index++) {
        vec column_subview = yield_params.x_feat_mat.col(
            yield_params.subsampled_categorical_features[index]
        );
        ivec temp_int_subview = arma::conv_to<ivec>::from(column_subview);
        UniqueFunctionReturns subview_unq = unique(column_subview, false);

        if (subview_unq.labels.size() >= 2) {
            for (int inner_index = 0; inner_index < subview_unq.labels.size(); inner_index++) {
                gen_var.left_subset_indices = arma::find(
                    temp_int_subview == subview_unq.labels[inner_index]
                );
                gen_var.right_subset_indices = arma::find(
                    temp_int_subview != subview_unq.labels[inner_index]
                );

                gen_var.split_kind = "categorical";
                gen_var.split_idx = yield_params.subsampled_categorical_features[index];
                gen_var.cat_cond = subview_unq.labels[inner_index];

                co_yield gen_var;
            }
        } else {
            continue;
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
        leaf_node_count++;
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

    vector<int> temp_num_feat = DecisionTreeClassifier::determine_feature_split_metric(
        numerical_features
    );
    vector<int> temp_cat_feat = DecisionTreeClassifier::determine_feature_split_metric(
        categorical_features
    );

    // Explicitly assigning matrix/vector and subsampled features
    splt_yld.x_feat_mat = X;
    splt_yld.y_target_vec = Y;
    splt_yld.subsampled_numerical_features = temp_num_feat;
    splt_yld.subsampled_categorical_features = temp_cat_feat;

    if (recursive_max_depth >= max_depth) {
        cout << "[*] Stopping training. Max depth hit" << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return node;
    }

    if (X.n_rows < min_samples_split) {
        cout << "[*] Stopping training. Min samples split hit" << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return node;
    }

    for (const auto& gen_var : DecisionTreeClassifier::split_yield(splt_yld)) {
        vec left_vec_subview = Y.rows(gen_var.left_subset_indices);
        vec right_vec_subview = Y.rows(gen_var.right_subset_indices);

        float crt_inf_gain = DecisionTreeClassifier::compute_information_gain(
            Y,
            left_vec_subview,
            right_vec_subview
        );

        if (gen_var.split_kind == "numeric") {
            if (crt_inf_gain > best_candidate_var.best_gain) {
                best_candidate_var.best_gain = crt_inf_gain;
                best_candidate_var.best_idx = gen_var.split_idx;
                best_candidate_var.num_cond = gen_var.num_cond;

                // Left and Right decision matrices
                best_candidate_var.left_x_mat = X.rows(gen_var.left_subset_indices);
                best_candidate_var.right_x_mat = X.rows(gen_var.right_subset_indices);

                // Left and Right decision split vectors
                best_candidate_var.left_y_vec = left_vec_subview;
                best_candidate_var.right_y_vec = right_vec_subview;
                best_candidate_var.cat_cond = std::nullopt;
            }
        }

        if (gen_var.split_kind == "categorical") {
            if (crt_inf_gain > best_candidate_var.best_gain) {
                best_candidate_var.best_gain = crt_inf_gain;
                best_candidate_var.best_idx = gen_var.split_idx;
                best_candidate_var.cat_cond = gen_var.cat_cond;

                // Left and Right decision matrices
                best_candidate_var.left_x_mat = X.rows(gen_var.left_subset_indices);
                best_candidate_var.right_x_mat = X.rows(gen_var.right_subset_indices);

                // Left and Right decision split vectors
                best_candidate_var.left_y_vec = left_vec_subview;
                best_candidate_var.right_y_vec = right_vec_subview;
                best_candidate_var.num_cond = std::nullopt;
            }
        }
    }

    // Ambiguous hyperparameter check
    // Checks if the subsampled features yielded any information
    if (!best_candidate_var.num_cond.has_value() && 
        !best_candidate_var.cat_cond.has_value()
    ) {
        cout << "[*] Stopping training! No numerical and categorical feats found." << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto leaf_node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return leaf_node;
    }

    if (best_candidate_var.best_gain < min_information_gain) {
        cout << "[*] Stopping training! Minimum Information Gain hit." << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto leaf_node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return leaf_node;
    }

    if (best_candidate_var.left_y_vec.n_rows < min_samples_leaf ||
        best_candidate_var.right_y_vec.n_rows < min_samples_leaf
    ) {
        cout << "[*] Stopping training! Minimum samples split hit." << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto leaf_node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return leaf_node;
    }

    // If you're going to ask why this hyperparameter condition check is like this
    // Go study Decision Tree algorithms сука блять, cyka blyat :)
    if (leaf_node_count >= max_leaf_nodes) {
        cout << "[*] Stopping training. Maximum leaf nodes reached" << endl;
        best_candidate_var.computed_probs = DecisionTreeClassifier::compute_class_probability(Y);
        auto node = DecisionTreeClassifier::create_node(best_candidate_var, false, true);
        return node;
    } else {
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

}

variant<int, vec> DecisionTreeClassifier::traverse_tree_prediction (
    const mat& element,
    const unique_ptr<Node>& node,
    bool standard_traverse,
    bool probability_traverse
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

        if (standard_traverse) {
            return unique_classes[
                std::distance(computed_probabilities.begin(), max_elem_ptr)
            ];
        }
        if (probability_traverse) {
            return node->computed_probabilities;
        }
    }

    if (node->num_condition.has_value()) {
        float extracted_num_cond = node->num_condition.value();
        if (element.at(node->split_index) > extracted_num_cond) {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->left_branch,
                standard_traverse,
                probability_traverse
            );
        } else {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->right_branch,
                standard_traverse,
                probability_traverse
            );
        }
    }

    if (node->cat_condition.has_value()) {
        float extracted_cat_cond = node->cat_condition.value();
        if (element.at(node->split_index) == extracted_cat_cond) {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->left_branch,
                standard_traverse,
                probability_traverse
            );
        } else {
            return DecisionTreeClassifier::traverse_tree_prediction(
                element,
                node->right_branch,
                standard_traverse,
                probability_traverse
            );
        }
    }

    throw std::runtime_error("[-] Error: Unexpected corrupted node encountered.");
}

void DecisionTreeClassifier::fit (mat& X, vec& Y) {
    int recursive_max_depth = 1;

    // To prevent overlap of old and new labels
    // If user decided to be stupid and fits multiple times
    unique_classes.clear();
    classes_to_index.clear();

    // Initialize unique_classes
    DecisionTreeClassifier::compute_class_probability(Y);

    root_node = DecisionTreeClassifier::build_decision_tree(
        X,
        Y,
        recursive_max_depth
    );
}

variant<int, vec> DecisionTreeClassifier::predict (
    variant<vec, mat>& X
) {
    try {
        if (unique_classes.empty()) {
            throw runtime_error("Model has not been fitted yet.");
        }

        if (std::holds_alternative<vec>(X)) {
            variant<int, vec> pred = DecisionTreeClassifier::traverse_tree_prediction(
                std::get<vec>(X), root_node, true, false
            );
            return std::get<int>(pred);
        } else if (std::holds_alternative<mat>(X)) {
            vector<int> predictions;
            for (arma::uword index = 0; index < std::get<mat>(X).n_rows; index++) {
                variant<int, vec> prediction = DecisionTreeClassifier::traverse_tree_prediction(
                    std::get<mat>(X).row(index), root_node, true, false
                );
                predictions.emplace_back(std::get<int>(prediction));
            }
            return arma::conv_to<vec>::from(predictions);
        } else {
            throw invalid_argument("Argument is neither arma::vec nor arma::mat");
        }
    } catch (const runtime_error& error) {
        cerr << error.what() << endl;
        exit(1);
    } catch (const invalid_argument& error) {
        cerr << error.what() << endl;
        exit(1);
    } catch (const out_of_range& error) {
        cerr << error.what() << endl;
        exit(1);
    }
}

variant<vec, mat> DecisionTreeClassifier::predict_proba (
    variant<vec, mat>& X
) {
    try {
        if (unique_classes.empty()) {
            throw runtime_error("[-] Error: Model has not been fitted yet");
        }

        if (std::holds_alternative<vec>(X)) {
            variant<int, vec> prediction = DecisionTreeClassifier::traverse_tree_prediction(
                std::get<vec>(X), root_node, false, true
            );
            return std::get<vec>(prediction);
        } else if (std::holds_alternative<mat>(X)) {
            mat batch_matrix;
            vector<vec> probability_distributions;

            for (arma::uword index = 0; index < std::get<mat>(X).n_rows; index++) {
                variant<int, vec> prediction = DecisionTreeClassifier::traverse_tree_prediction(
                    std::get<mat>(X).row(index), root_node, false, true
                );
                probability_distributions.emplace_back(std::get<vec>(prediction).t());
            }
            
            for (int index = 0; index < probability_distributions.size(); index++) {
                batch_matrix = arma::join_cols(
                    batch_matrix, 
                    probability_distributions[index]
                );
            }

            return batch_matrix;
        } else {
            throw invalid_argument("[-] Error: Argument is neither arma::vec or arma::mat");
        }
    } catch (const runtime_error& error) {
        cerr << error.what() << endl;
        exit(1);
    } catch (const invalid_argument& error) {
        cerr << error.what() << endl;
        exit(1);
    } catch (const out_of_range& error) {
        cerr << error.what() << endl;
        exit(1);
    }
}

#ifndef SKIP_MAIN
int main () {
    const string dataset_path = "python-utilities/datasets/openml/adult";

    vector<int> num_feats = {};
    vector<int> cat_feats = {};

    for (int i = 0; i <= 5; i++) {
        cout << "Appended feature index " << i << endl;
        num_feats.push_back(i);
    }

    for (int i = 6; i <= 21; i++) {
        cout << "Appended feature index " << i << endl;
        cat_feats.push_back(i);
    }

    DatasetOperations dset_ops;
    DecisionTreeClassifier tree(
        "gini", // Split metric
        10, // Max depth
        "none", // Max feature
        10, // Max leaf nodes
        20, // Min samples leaf
        20, // Min samples split
        0.0001, // Minimum information gain
        num_feats, // Numerical features
        cat_feats // Categorical features
    );

    dset_ops.construct_datasets();
    dset_ops.load_datasets(
        dataset_path + "/train_x.csv",
        dataset_path + "/train_y.csv",
        dataset_path + "/test_x.csv",
        dataset_path + "/test_y.csv"
    );

    mat train_x = std::get<mat>(dset_ops.datasets_vector.at(0));
    vec train_y = std::get<vec>(dset_ops.datasets_vector.at(1));
    mat test_x = std::get<mat>(dset_ops.datasets_vector.at(2));

    tree.fit(train_x, train_y);

    //vec predictions = tree.predict(test_x);

    //dset_ops.save_dataset(
    //    dataset_path + "/predictions/cpp-predictions.csv",
    //    predictions
    //);
}
#endif
