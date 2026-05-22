#include <vector>
#include <variant>
#include <generator>
#include <armadillo>
#include <cstdlib>
#include "base.hpp"
#include "classifier_mixin.hpp"
#include "complex_datatypes.hpp"

struct CreateNodeParameters {
    std::variant<int, float>    num_feats;
    std::variant<int, float>    cat_feats;
    std::vector<float>          class_probs;
    float                       info_gain;
    int                         split_idx;
    bool                        make_decision;
    bool                        make_leaf;
};

struct RecursiveTreeBuilder {
    arma::mat                   left_x_mat;
    arma::mat                   right_x_mat;
    arma::vec                   left_y_vec;
    arma::vec                   right_y_vec;
    std::variant<int, float>    num_cond;
    float                       best_gain;
    int                         best_idx;
    int                         cat_cond;
};

struct SplitYieldParameters {
    arma::mat               x_feat_mat;
    arma::vec               y_target_vec;
    std::vector<int>        num_cols;
    std::vector<int>        cat_cols;
};

struct GeneratorVariables {
    arma::mat                   left_x_mat;
    arma::mat                   right_x_mat;
    arma::vec                   left_y_vec;
    arma::vec                   right_y_vec;
    std::string                 split_kind;
    std::variant<int, double>   num_thresh;
    int                         cat_thresh;
    int                         split_idx;
};

class LeafNode {
    public:
        LeafNode (std::vector<float> computed_probabilities):
            computed_probabilities(computed_probabilities)
        {}

    private:
        bool is_leaf_node = true;
        bool is_decision_node = false;
        std::vector<float> computed_probabilities;
};

class DecisionNode {
    public:
        DecisionNode (
            int split_index,
            std::variant<int, float> num_condition,
            std::variant<int, std::string> cat_condition
        ):
            split_index(split_index),
            num_condition(num_condition),
            cat_condition(cat_condition)
        {}

    private:
        bool is_leaf_node = false;
        bool is_decision_node = true;
        int split_index;
        std::variant<int, float> num_condition;
        std::variant<int, std::string> cat_condition;
};

class DecisionTreeClassifier: public BaseEstimator, public ClassifierMixin {
    public:
        std::string split_metric;
        int max_depth;
        SIDual max_features;
        int max_leaf_nodes;
        int min_samples_leaf;
        int min_samples_split;
        float min_information_gain;
        IntegerCategoricalList numerical_features;
        IntegerCategoricalList categorical_features;

        DecisionTreeClassifier(
            std::string split_metric = "gini",
            int max_depth = 10,
            SIDual max_feature,
            int max_leaf_nodes = 10,
            int min_samples_leaf = 50,
            int min_samples_split = 30,
            float min_information_gain = 1e-5f,
            IntegerCategoricalList numerical_features,
            IntegerCategoricalList categorical_features
        ): 
            split_metric(split_metric),
            max_depth(max_depth),
            max_features(max_features),
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
        std::vector<int> unique_classes;
        arma::vec prob_vec;
        std::map<int, int> classes_to_index;
        arma::vec compute_class_probability (arma::vec& Y);

        // Math functions to compute impurity and randomness
        float compute_impurity (arma::vec& Y);
        float compute_entropy (arma::vec& Y);
        float compute_log_loss (arma::vec& pred_y, arma::vec& pred_y_prob);
        float compute_information_gain (
            arma::vec& Y, 
            arma::mat& left_subset, 
            arma::mat& right_subset
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
        std::variant<LeafNode, DecisionNode> create_node (
            CreateNodeParameters& node_parameters
        );
        std::variant<LeafNode, DecisionNode> build_decision_tree (
            arma::mat& X, 
            arma::vec& Y, 
            int recusive_depth
        );
};
