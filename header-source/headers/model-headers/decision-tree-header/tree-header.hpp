#pragma once

#include <vector>
#include <memory>
#include <cstdlib>
#include <variant>
#include <optional>
#include <generator>
#include <armadillo>
#include "base.hpp"
#include "classifier_mixin.hpp"
#include "complex_datatypes.hpp"

struct CreateNodeParameters {
    int                         split_index;
    std::variant<int, float>    numerical_feats;
    std::variant<int, float>    categorical_feats;
    float                       information_gain;
    std::vector<float>          computed_class_probabilities;
    bool                        create_decision_node;
    bool                        create_leaf_node;
};

struct RecursiveTreeBuilder {
    float                       best_information_gain;
    int                         best_index;
    std::variant<int, float>    best_num_split_condition;
    int                         best_cat_split_condition;
    arma::mat&                  best_left_x_subset;
    arma::vec&                  best_left_y_subset;
    arma::mat&                  best_right_x_subset;
    arma::vec&                  best_right_y_subset;
};

struct SplitYieldParameters {
    arma::mat                                  x_feat_mat;
    arma::vec                                  y_target_vec;
};

struct GeneratorVariables {
    arma::mat                                  left_x_mat;
    arma::mat                                  right_x_mat;
    arma::vec                                  left_y_vec;
    arma::vec                                  right_y_vec;
    std::string                                split_kind;
    std::optional<float>                       num_thresh;
    std::optional<float>                       cat_thresh;
    int                                        split_idx;
};

struct Node {
    // Leaf Node probabilities
    arma::vec                                  computed_probabilities;

    // Decision Node Data
    int                                        split_index;
    std::optional<float>                       num_condition;
    std::optional<float>                       cat_condition;

    // Boolean flags for checking if Leaf or Decision
    bool                                       is_leaf_node = false;
    bool                                       is_decision_node = false;

    // Unique pointers to the left and right branches
    std::unique_ptr<Node>                      left_branch;
    std::unique_ptr<Node>                      right_branch;
};

class DecisionTreeClassifier: public BaseEstimator, public ClassifierMixin {
    public:
        std::string split_metric;
        int max_depth;
        SIDual max_feature;
        int max_leaf_nodes;
        int min_samples_leaf;
        int min_samples_split;
        float min_information_gain;
        std::vector<int> numerical_features;
        std::vector<int> categorical_features;

        DecisionTreeClassifier(
            std::string split_metric = "gini",
            int max_depth = 10,
            SIDual max_feature = "sqrt",
            int max_leaf_nodes = 10,
            int min_samples_leaf = 50,
            int min_samples_split = 30,
            float min_information_gain = 1e-5f,
            std::vector<int> numerical_features = {},
            std::vector<int> categorical_features = {}
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

        void fit (arma::mat& X, arma::vec& Y) override;
        arma::vec predict (arma::mat& X) override;

    private:
        std::unique_ptr<Node> root_node;
        std::vector<int> unique_classes;
        arma::vec prob_vec;
        std::map<int, int> classes_to_index;
        arma::vec compute_class_probability (arma::vec& Y);

        // Math functions to compute impurity and randomness
        float compute_impurity (arma::vec& Y);
        float compute_entropy (arma::vec& Y);
        float compute_log_loss (
            arma::vec& pred_y,
            arma::vec& pred_y_prob
        );
        float compute_information_gain (
            arma::vec& Y,
            arma::vec& left_subset,
            arma::vec& right_subset
        );

        // Functions that build the tree's node
        float determine_impurity_metric (arma::vec& Y);
        std::vector<int> determine_feature_split_metric (
            std::vector<int> feature_list
        );
        std::generator<GeneratorVariables&> split_yield (
            SplitYieldParameters& yield_parameters,
            GeneratorVariables& generator_parameters
        );
};
