#include <vector>
#include <variant>
#include <generator>
#include <armadillo>
#include <cstdlib>
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
    arma::mat&              X;
    arma::vec&              Y;
    std::vector<int>        num_feats;
    std::vector<int>        cat_feats;
};

struct GeneratorVariables {
    std::string                 split_type;
    int                         split_index;
    std::variant<int, float>    best_temporary_numeric_threshold;
    int                         best_temporary_categorical_threshold;
    arma::mat&                  left_x_subset;
    arma::vec&                  left_y_subset;
    arma::mat&                  right_x_subset;
    arma::vec&                  right_y_subset;
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
            std::variant<int, float> split_feature_num_condition,
            std::variant<int, std::string> split_feature_cat_condition
        ):
            split_index(split_index),
            split_feature_num_condition(split_feature_num_condition),
            split_feature_cat_condition(split_feature_cat_condition)
        {}

    private:
        bool is_leaf_node = false;
        bool is_decision_node = true;
        int split_index;
        std::variant<int, float> split_feature_num_condition;
        std::variant<int, std::string> split_feature_cat_condition;
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
        arma::vec probability_vector;
        std::map<int, int> classes_to_index;
        arma::vec compute_class_probability (arma::vec& Y);
        float compute_impurity (arma::vec& Y);
        float compute_entropy (arma::vec& Y);
        float compute_log_loss (arma::vec& pred_y, arma::vec& pred_y_prob);
        float compute_information_gain (arma::vec& Y, arma::mat& left_subset, arma::mat& right_subset);
        float determine_impurity_metric (arma::vec& Y);
        std::vector<int> determine_feature_split_metric (std::vector<int> feature_list);
        std::generator<GeneratorVariables*> split_yield (
            SplitYieldParameters* yield_parameters,
            GeneratorVariables* generator_parameters
        );
        std::variant<LeafNode, DecisionNode> create_node (CreateNodeParameters& node_parameters);
        std::variant<LeafNode, DecisionNode> build_decision_tree (
            arma::mat& X, 
            arma::vec& Y, 
            int recusive_depth
        );
};
