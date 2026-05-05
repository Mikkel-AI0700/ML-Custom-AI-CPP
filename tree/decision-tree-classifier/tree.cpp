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
    UniqueFunctionReturns* unq_ret = unique(Y, true);
    
    if (unique_classes.empty()) {
        unique_classes = unq_ret->labels;
        probability_vector = arma::zeros(unique_classes.size());

        for (int i = 0; i < unq_ret->labels.size(); i++) {
            classes_to_index.insert({i, unq_ret->labels[i]});
        }
    } else {
        probability_vector = arma::zeros(unq_ret->labels.size());
    }

    for (int i = 0; i < unq_ret->labels.size(); i++) {
        if (classes_to_index.contains(unq_ret->labels[i])) {
            int index = classes_to_index.at(unq_ret->labels[i]);
            probability_vector.at(classes_to_index.at(index)) = unq_ret->label_counts[i] / Y.n_elem;
        }
    }

    return probability_vector;
}

float DecisionTreeClassifier::compute_impurity (arma::vec& Y) {
    arma::vec probability_vector = DecisionTreeClassifier::compute_class_probability(Y);
    float computed_impurity = 1 - arma::sum(arma::square(probability_vector));
    return computed_impurity;
}

float DecisionTreeClassifier::compute_entropy (arma::vec& Y) {
    arma::vec probability_vector = DecisionTreeClassifier::compute_class_probability(Y);
    arma::uvec probability_vector_above_zero = arma::find(probability_vector > 0.0);
    arma::vec probability_vector = probability_vector.elem(probability_vector_above_zero);
    float computed_entropy = -(arma::sum(probability_vector * arma::log2(probability_vector)));
    return computed_entropy;
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
    SplitYieldParameters *yield_parameters,
    GeneratorVariables *generator_variables
) {
    SplitYieldParameters *yield_parameter = yield_parameter;
    GeneratorVariables *gen_var = generator_variables;
    
    for (int index = 0; index < yield_parameter->num_feats.size(); index++) {
        arma::vec column_subview = yield_parameter->X.col(yield_parameter->num_feats.size());

        UniqueFunctionReturns subview_unq = unique(column_subview, false);
        arma::vec labels = arma::conv_to<arma::vec>::from(subview_unq.labels);
        arma::vec midpoints = (
            labels.subvec(0, labels.n_elem - 2) + labels.subvec(1, labels.n_elem - 1)
        ) / 2.0;

        for (int inner_index = 0; inner_index < midpoints.n_elem; inner_index++) {
            arma::uvec left_satisfying_indices = arma::find(
                yield_parameters->X > midpoints[inner_index] // Unfinished code
            );
            arma::uvec right_satisfying_indices = arma::find(
                yield_parameters->X <= midpoints[inner_index]
            );

            gen_var->split_type = "numeric";
            gen_var->split_index = yield_parameter->num_feats[index];
            gen_var->best_temporary_numeric_threshold = midpoints.at(inner_index);
            gen_var->best_temporary_categorical_threshold = nullptr;
        }
    }
    for (int index = 0; index < yield_parameter->cat_feats.size(); index++) {

    }
}

int main () {

}

 
