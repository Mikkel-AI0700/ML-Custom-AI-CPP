#include <gtest/gtest.h>

#include <armadillo>
#include <vector>
#include <memory>
#include <cstdlib>
#include <stdexcept>
#include <cmath>

#define private public
#define protected public
#include "tree-header.hpp"
#include "linalg-operations.hpp"
#undef private
#undef protected

using arma::mat;
using arma::vec;
using std::vector;
using std::unique_ptr;

TEST(ComputeClassProbability, AllLabelsSame) {
    DecisionTreeClassifier dtc;
    vec Y = {1.0, 1.0, 1.0};
    vec prob = dtc.compute_class_probability(Y);

    ASSERT_EQ(prob.n_elem, 1);
    EXPECT_NEAR(prob(0), 1.0, 1e-5);
    EXPECT_EQ(dtc.unique_classes.size(), 1);
    EXPECT_EQ(dtc.classes_to_index.size(), 1);
}

TEST(ComputeClassProbability, BalancedTwoClasses) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 1.0, 0.0, 1.0};
    vec prob = dtc.compute_class_probability(Y);

    ASSERT_EQ(prob.n_elem, 2);
    EXPECT_NEAR(prob(0), 0.5, 1e-5);
    EXPECT_NEAR(prob(1), 0.5, 1e-5);
}

TEST(ComputeClassProbability, ThreeClassesWeighted) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    vec prob = dtc.compute_class_probability(Y);

    ASSERT_EQ(prob.n_elem, 3);
    EXPECT_NEAR(prob(0), 0.2, 1e-5);
    EXPECT_NEAR(prob(1), 0.3, 1e-5);
    EXPECT_NEAR(prob(2), 0.5, 1e-5);
}

TEST(ComputeClassProbability, EmptyLabels) {
    DecisionTreeClassifier dtc;
    vec Y;
    vec prob = dtc.compute_class_probability(Y);

    EXPECT_EQ(prob.n_elem, 0);
}

TEST(ComputeImpurity, PureLabels) {
    DecisionTreeClassifier dtc;
    vec Y = {1.0, 1.0, 1.0};
    float imp = dtc.compute_impurity(Y);
    EXPECT_NEAR(imp, 0.0f, 1e-5);
}

TEST(ComputeImpurity, BalancedTwoClasses) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 1.0, 0.0, 1.0};
    float imp = dtc.compute_impurity(Y);
    EXPECT_NEAR(imp, 0.5f, 1e-5);
}

TEST(ComputeImpurity, ThreeClassesWeighted) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    float imp = dtc.compute_impurity(Y);
    float expected = 1.0f - (0.04f + 0.09f + 0.25f);
    EXPECT_NEAR(imp, expected, 1e-5);
}

TEST(ComputeImpurity, SingleSample) {
    DecisionTreeClassifier dtc;
    vec Y = {5.0};
    float imp = dtc.compute_impurity(Y);
    EXPECT_NEAR(imp, 0.0f, 1e-5);
}

TEST(ComputeEntropy, PureLabels) {
    DecisionTreeClassifier dtc;
    vec Y = {1.0, 1.0, 1.0};
    float ent = dtc.compute_entropy(Y);
    EXPECT_NEAR(ent, 0.0f, 1e-5);
}

TEST(ComputeEntropy, BalancedTwoClasses) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 1.0, 0.0, 1.0};
    float ent = dtc.compute_entropy(Y);
    EXPECT_NEAR(ent, 1.0f, 1e-5);
}

TEST(ComputeEntropy, ThreeClassesWeighted) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    float ent = dtc.compute_entropy(Y);
    float expected = -(0.2f * std::log2(0.2f) + 0.3f * std::log2(0.3f) + 0.5f * std::log2(0.5f));
    EXPECT_NEAR(ent, expected, 1e-3);
}

TEST(ComputeEntropy, ZeroProbabilityIgnored) {
    DecisionTreeClassifier dtc;
    vec Y = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float ent = dtc.compute_entropy(Y);

    dtc.unique_classes.clear();
    dtc.classes_to_index.clear();

    vec Y2 = {0.0, 0.0, 0.0};
    float ent2 = dtc.compute_entropy(Y2);
    EXPECT_NEAR(ent2, 0.0f, 1e-5);

    float expected = -(0.5f * std::log2(0.5f) + 0.5f * std::log2(0.5f));
    EXPECT_NEAR(ent, expected, 1e-3);
}

TEST(ComputeInformationGain, PerfectSplit) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "gini";

    vec Y = {0.0, 0.0, 1.0, 1.0};
    vec left_subset = {0.0, 0.0};
    vec right_subset = {1.0, 1.0};

    float gain = dtc.compute_information_gain(Y, left_subset, right_subset);

    dtc.unique_classes.clear();
    dtc.classes_to_index.clear();
    float parent_imp = dtc.compute_impurity(Y);
    EXPECT_NEAR(gain, parent_imp, 1e-5);
}

TEST(ComputeInformationGain, NoGain) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "gini";

    vec Y = {0.0, 0.0, 1.0, 1.0};
    vec left_subset = {0.0, 1.0};
    vec right_subset = {0.0, 1.0};

    float gain = dtc.compute_information_gain(Y, left_subset, right_subset);
    EXPECT_NEAR(gain, 0.0f, 1e-5);
}

TEST(ComputeInformationGain, EmptySubset) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "gini";

    vec Y = {0.0, 1.0};
    vec empty_vec;
    vec right_subset = {0.0, 1.0};

    float gain = dtc.compute_information_gain(Y, empty_vec, right_subset);
    EXPECT_NEAR(gain, 0.0f, 1e-5);
}

TEST(DetermineImpurityMetric, Gini) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "gini";
    vec Y = {0.0, 1.0, 0.0, 1.0};

    float metric = dtc.determine_impurity_metric(Y);

    dtc.unique_classes.clear();
    dtc.classes_to_index.clear();
    float expected = dtc.compute_impurity(Y);
    EXPECT_NEAR(metric, expected, 1e-5);
}

TEST(DetermineImpurityMetric, Entropy) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "entropy";
    vec Y = {0.0, 1.0, 0.0, 1.0};

    float metric = dtc.determine_impurity_metric(Y);

    dtc.unique_classes.clear();
    dtc.classes_to_index.clear();
    float expected = dtc.compute_entropy(Y);
    EXPECT_NEAR(metric, expected, 1e-5);
}

TEST(DetermineImpurityMetric, InvalidMetric) {
    DecisionTreeClassifier dtc;
    dtc.split_metric = "invalid";
    vec Y = {0.0, 1.0};

    EXPECT_THROW(dtc.determine_impurity_metric(Y), std::invalid_argument);
}

TEST(DetermineFeatureSplitMetric, Sqrt) {
    DecisionTreeClassifier dtc;
    dtc.max_feature = std::string("sqrt");
    srand(12345);

    vector<int> feature_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int> selected = dtc.determine_feature_split_metric(feature_list);

    int expected_size = static_cast<int>(std::sqrt(feature_list.size()));
    ASSERT_EQ(selected.size(), static_cast<size_t>(expected_size));

    for (auto& feat : selected) {
        bool found = std::find(feature_list.begin(), feature_list.end(), feat) != feature_list.end();
        EXPECT_TRUE(found);
    }

    std::sort(selected.begin(), selected.end());
    auto dup = std::adjacent_find(selected.begin(), selected.end());
    EXPECT_EQ(dup, selected.end());
}

TEST(DetermineFeatureSplitMetric, Log2) {
    DecisionTreeClassifier dtc;
    dtc.max_feature = std::string("log2");
    srand(12345);

    vector<int> feature_list = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int> selected = dtc.determine_feature_split_metric(feature_list);

    int expected_size = static_cast<int>(std::log2(feature_list.size()));
    ASSERT_EQ(selected.size(), static_cast<size_t>(expected_size));
}

TEST(DetermineFeatureSplitMetric, EmptyFeatureList) {
    DecisionTreeClassifier dtc;
    dtc.max_feature = std::string("sqrt");
    srand(12345);

    vector<int> feature_list;
    vector<int> selected = dtc.determine_feature_split_metric(feature_list);

    EXPECT_EQ(selected.size(), 0);
}

TEST(SplitYield, NumericFeature) {
    DecisionTreeClassifier dtc;
    dtc.numerical_features = {0};
    dtc.categorical_features = {};

    SplitYieldParameters params;
    params.x_feat_mat.set_size(5, 1);
    params.x_feat_mat.col(0) = vec({1.0, 2.0, 3.0, 4.0, 5.0});
    params.y_target_vec = vec({0.0, 0.0, 0.0, 0.0, 0.0});

    int count = 0;
    for (const auto& gen_var : dtc.split_yield(params)) {
        EXPECT_EQ(gen_var.split_kind, "numeric");
        EXPECT_EQ(gen_var.split_idx, 0);
        EXPECT_TRUE(gen_var.num_cond.has_value());
        EXPECT_FALSE(gen_var.cat_cond.has_value());
        count++;
    }
    EXPECT_EQ(count, 4);
}

TEST(SplitYield, CategoricalFeature) {
    DecisionTreeClassifier dtc;
    dtc.numerical_features = {};
    dtc.categorical_features = {0};

    SplitYieldParameters params;
    params.x_feat_mat.set_size(5, 1);
    params.x_feat_mat.col(0) = vec({0.0, 0.0, 1.0, 1.0, 2.0});
    params.y_target_vec = vec({0.0, 0.0, 0.0, 0.0, 0.0});

    int count = 0;
    for (const auto& gen_var : dtc.split_yield(params)) {
        EXPECT_EQ(gen_var.split_kind, "categorical");
        EXPECT_EQ(gen_var.split_idx, 0);
        EXPECT_TRUE(gen_var.cat_cond.has_value());
        EXPECT_FALSE(gen_var.num_cond.has_value());
        count++;
    }
    EXPECT_EQ(count, 3);
}

TEST(SplitYield, AllSameValuesNumeric) {
    DecisionTreeClassifier dtc;
    dtc.numerical_features = {0};
    dtc.categorical_features = {};

    SplitYieldParameters params;
    params.x_feat_mat.set_size(3, 1);
    params.x_feat_mat.col(0) = vec({5.0, 5.0, 5.0});
    params.y_target_vec = vec({0.0, 0.0, 0.0});

    int count = 0;
    for (const auto& gen_var : dtc.split_yield(params)) {
        count++;
    }
    EXPECT_EQ(count, 0);
}

TEST(CreateNode, DecisionNodeNumeric) {
    DecisionTreeClassifier dtc;
    BestCandidateSplit best;
    best.num_cond = 3.5f;
    best.best_idx = 2;

    auto node = dtc.create_node(best, true, false);

    EXPECT_TRUE(node->is_decision_node);
    EXPECT_FALSE(node->is_leaf_node);
    EXPECT_EQ(node->split_index, 2);
    EXPECT_TRUE(node->num_condition.has_value());
    EXPECT_NEAR(node->num_condition.value(), 3.5f, 1e-5);
    EXPECT_FALSE(node->cat_condition.has_value());
}

TEST(CreateNode, DecisionNodeCategorical) {
    DecisionTreeClassifier dtc;
    BestCandidateSplit best;
    best.cat_cond = 1.0f;
    best.best_idx = 3;

    auto node = dtc.create_node(best, true, false);

    EXPECT_TRUE(node->is_decision_node);
    EXPECT_FALSE(node->is_leaf_node);
    EXPECT_EQ(node->split_index, 3);
    EXPECT_TRUE(node->cat_condition.has_value());
    EXPECT_NEAR(node->cat_condition.value(), 1.0f, 1e-5);
    EXPECT_FALSE(node->num_condition.has_value());
}

TEST(CreateNode, LeafNode) {
    DecisionTreeClassifier dtc;
    BestCandidateSplit best;
    best.computed_probs = vec({0.3, 0.7});

    auto node = dtc.create_node(best, false, true);

    EXPECT_TRUE(node->is_leaf_node);
    EXPECT_FALSE(node->is_decision_node);
    ASSERT_EQ(node->computed_probabilities.n_elem, 2);
    EXPECT_NEAR(node->computed_probabilities(0), 0.3, 1e-5);
    EXPECT_NEAR(node->computed_probabilities(1), 0.7, 1e-5);
}

TEST(CreateNode, BothFlagsFalse) {
    DecisionTreeClassifier dtc;
    BestCandidateSplit best;

    auto node = dtc.create_node(best, false, false);

    EXPECT_FALSE(node->is_leaf_node);
    EXPECT_FALSE(node->is_decision_node);
}

TEST(BuildDecisionTree, MaxDepthHit) {
    DecisionTreeClassifier dtc;
    dtc.max_depth = 0;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0};

    mat X(4, 1);
    X.col(0) = vec({1.0, 2.0, 3.0, 4.0});
    vec Y = {0.0, 1.0, 0.0, 1.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_leaf_node);
    EXPECT_FALSE(node->is_decision_node);
}

TEST(BuildDecisionTree, MinSamplesSplit) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 10;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0};

    mat X(3, 1);
    X.col(0) = vec({1.0, 2.0, 3.0});
    vec Y = {0.0, 1.0, 0.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_leaf_node);
}

TEST(BuildDecisionTree, MinInformationGain) {
    DecisionTreeClassifier dtc;
    dtc.min_information_gain = 1.0f;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0};

    mat X(4, 1);
    X.col(0) = vec({1.0, 2.0, 3.0, 4.0});
    vec Y = {0.0, 1.0, 0.0, 1.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_leaf_node);
}

TEST(BuildDecisionTree, MinSamplesLeaf) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 2;
    dtc.min_samples_leaf = 5;
    dtc.min_information_gain = 1e-5f;
    dtc.numerical_features = {0};

    mat X(4, 1);
    X.col(0) = vec({1.0, 2.0, 3.0, 4.0});
    vec Y = {0.0, 1.0, 0.0, 1.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_leaf_node);
}

TEST(BuildDecisionTree, AllFeaturesConstant) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.min_information_gain = 1e-5f;
    dtc.numerical_features = {0};

    mat X(4, 1);
    X.col(0) = vec({5.0, 5.0, 5.0, 5.0});
    vec Y = {0.0, 1.0, 0.0, 1.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_leaf_node);
}

TEST(BuildDecisionTree, ValidSplit) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.min_information_gain = 1e-5f;
    dtc.numerical_features = {0};

    mat X(4, 1);
    X.col(0) = vec({1.0, 2.0, 20.0, 25.0});
    vec Y = {0.0, 0.0, 1.0, 1.0};

    auto node = dtc.build_decision_tree(X, Y, 0);

    EXPECT_TRUE(node->is_decision_node);
    EXPECT_TRUE(node->left_branch != nullptr);
    EXPECT_TRUE(node->right_branch != nullptr);
}

TEST(TraverseTreePrediction, LeafNode) {
    DecisionTreeClassifier dtc;

    auto leaf = std::make_unique<Node>();
    leaf->is_leaf_node = true;
    leaf->computed_probabilities = vec({0.1, 0.2, 0.7});

    mat element(1, 1);
    element.fill(0);
    int result = dtc.traverse_tree_prediction(element, leaf);
    EXPECT_EQ(result, 2);
}

TEST(TraverseTreePrediction, NumericConditionGreater) {
    DecisionTreeClassifier dtc;

    auto decision = std::make_unique<Node>();
    decision->split_index = 0;
    decision->num_condition = 10.0f;

    auto left_leaf = std::make_unique<Node>();
    left_leaf->is_leaf_node = true;
    left_leaf->computed_probabilities = vec({1.0, 0.0});

    auto right_leaf = std::make_unique<Node>();
    right_leaf->is_leaf_node = true;
    right_leaf->computed_probabilities = vec({0.0, 1.0});

    decision->left_branch = std::move(left_leaf);
    decision->right_branch = std::move(right_leaf);

    mat element(1, 1);
    element(0, 0) = 15.0;
    int result = dtc.traverse_tree_prediction(element, decision);
    EXPECT_EQ(result, 0);

    mat element2(1, 1);
    element2(0, 0) = 5.0;
    int result2 = dtc.traverse_tree_prediction(element2, decision);
    EXPECT_EQ(result2, 1);
}

TEST(TraverseTreePrediction, CategoricalCondition) {
    DecisionTreeClassifier dtc;

    auto decision = std::make_unique<Node>();
    decision->split_index = 0;
    decision->cat_condition = 1.0f;

    auto left_leaf = std::make_unique<Node>();
    left_leaf->is_leaf_node = true;
    left_leaf->computed_probabilities = vec({1.0, 0.0});

    auto right_leaf = std::make_unique<Node>();
    right_leaf->is_leaf_node = true;
    right_leaf->computed_probabilities = vec({0.0, 1.0});

    decision->left_branch = std::move(left_leaf);
    decision->right_branch = std::move(right_leaf);

    mat element(1, 1);
    element(0, 0) = 1.0;
    int result = dtc.traverse_tree_prediction(element, decision);
    EXPECT_EQ(result, 0);

    mat element2(1, 1);
    element2(0, 0) = 2.0;
    int result2 = dtc.traverse_tree_prediction(element2, decision);
    EXPECT_EQ(result2, 1);
}

TEST(TraverseTreePrediction, CorruptNode) {
    DecisionTreeClassifier dtc;

    auto corrupt = std::make_unique<Node>();
    corrupt->num_condition = std::nullopt;
    corrupt->cat_condition = std::nullopt;

    mat element(1, 1);
    element.fill(0);
    EXPECT_THROW(dtc.traverse_tree_prediction(element, corrupt), std::runtime_error);
}

TEST(Fit, TrivialDataset) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0, 1};

    mat X(4, 2);
    X.col(0) = vec({1.0, 2.0, 3.0, 4.0});
    X.col(1) = vec({5.0, 6.0, 7.0, 8.0});
    vec Y = {0.0, 0.0, 1.0, 1.0};

    dtc.fit(X, Y);

    EXPECT_TRUE(dtc.root_node != nullptr);
    EXPECT_FALSE(dtc.unique_classes.empty());
}

TEST(Fit, EmptyX) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;

    mat X;
    vec Y = {0.0, 1.0};

    EXPECT_NO_THROW(dtc.fit(X, Y));
}

TEST(Predict, OverfitOnTraining) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0, 1};

    mat X(4, 2);
    X.col(0) = vec({1.0, 2.0, 3.0, 4.0});
    X.col(1) = vec({5.0, 6.0, 7.0, 8.0});
    vec Y = {0.0, 0.0, 1.0, 1.0};

    dtc.fit(X, Y);
    vec preds = dtc.predict(X);

    ASSERT_EQ(preds.n_rows, 4);
}

TEST(Predict, UnfittedModel) {
    DecisionTreeClassifier dtc;
    mat X(2, 1);
    X.fill(0);

    EXPECT_THROW(dtc.predict(X), std::runtime_error);
}

TEST(Predict, ReturnsCorrectLength) {
    DecisionTreeClassifier dtc;
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;
    dtc.numerical_features = {0};

    mat X(3, 1);
    X.col(0) = vec({1.0, 2.0, 3.0});
    vec Y = {0.0, 1.0, 0.0};

    dtc.fit(X, Y);
    vec preds = dtc.predict(X);

    EXPECT_EQ(preds.n_rows, 3);
}

TEST(EdgeCases, SingleClassInY) {
    DecisionTreeClassifier dtc;
    dtc.numerical_features = {0};
    dtc.min_samples_split = 1;
    dtc.min_samples_leaf = 1;

    mat X(3, 1);
    X.col(0) = vec({1.0, 2.0, 3.0});
    vec Y = {0.0, 0.0, 0.0};

    dtc.fit(X, Y);

    EXPECT_TRUE(dtc.root_node != nullptr);

    vec preds = dtc.predict(X);
    EXPECT_EQ(preds.n_rows, 3);
    for (uint i = 0; i < preds.n_rows; i++) {
        EXPECT_NEAR(preds(i), 0.0, 1e-5);
    }
}

TEST(EdgeCases, LargeFeatureCountSubsampling) {
    DecisionTreeClassifier dtc;
    dtc.max_feature = std::string("sqrt");
    srand(12345);

    vector<int> feature_list(100);
    for (int i = 0; i < 100; i++) {
        feature_list[i] = i;
    }

    vector<int> selected = dtc.determine_feature_split_metric(feature_list);

    int expected_size = static_cast<int>(std::sqrt(100));
    EXPECT_EQ(selected.size(), static_cast<size_t>(expected_size));
    EXPECT_GT(selected.size(), 0);
}
