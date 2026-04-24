#include <iostream>
#include <map>
#include <vector>
#include <variant>
#include <armadillo>
#include "loader.hpp"
#include "tree-header.hpp"

arma::vec DecisionTreeClassifier::compute_class_probability (arma::vec& Y) {
    std::vector<int> labels;
    std::vector<int> label_counts;
    for (int index = 0; index < Y.n_elem; index++) {
        if (index == 0) {
            continue;
        }

        if (Y[index] != Y[index - 1]) {
            labels.push_back(Y[index]);
        }
    }

    for (int i = 0; i < labels.size(); i++) {
        int label_counter = 0;
        for (int j = 0; j < Y.n_elem; j++) {
            if (Y[j] == labels[i]) {
                label_counter++;
            }
        }
        label_counts.push_back(label_counter);
    }

    if (unique_classes.empty()) {
        unique_classes = labels;
        probability_vector = arma::zeros(unique_classes.size());

        for (int i = 0; i < labels.size(); i++) {
            classes_to_index.insert({i, labels[i]});
        }
    } else {
        probability_vector = arma::zeros(labels.size());
    }

    for (int i = 0; i < labels.size(); i++) {
        if (classes_to_index.contains(labels[i])) {
            probability_vector.at(classes_to_index.at(labels[i])) = label_counts[i] / Y.n_elem;
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

int main () {

}

 