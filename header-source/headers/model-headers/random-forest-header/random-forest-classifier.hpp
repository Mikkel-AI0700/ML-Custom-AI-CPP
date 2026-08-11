#pragma once

#include <vector>
#include <memory>
#include <random>
#include <optional>
#include <generator>
#include <armadillo>
#include "base.hpp"
#include "classifier_mixin.hpp"
#include "dataset_types.hpp"
#include "model_types.hpp"
#include "tree-header.hpp"

struct BootstrappedDataset {
    arma::mat bootstrapped_X;
    arma::vec bootstrapped_Y;
};

struct BootstrapIndices {
    arma::uvec bootstrapped_rows;
    arma::uvec bootstrapped_cols;
};

class RandomForestClassifier: public BaseEstimator, public ClassifierMixin {
    public:
        // ── RandomForest-specific hyperparameters ──
        int                                                        n_estimators;
        bool                                                       bootstrap;
        bool                                                       oob_score;
        std::optional<std::variant<int, float>>                    max_samples;
        int                                                        random_state;
        int                                                        verbose;

        // ── DecisionTree pass-through hyperparameters ──
        std::string                                                criterion;
        int                                                        max_depth;
        std::optional<std::variant<int, std::string>>              max_features;
        int                                                        max_leaf_nodes;
        int                                                        min_samples_leaf;
        int                                                        min_samples_split;
        float                                                      min_information_gain;
        std::vector<int>                                           numerical_features;
        std::vector<int>                                           categorical_features;

        // ── Learned state ──
        std::vector<int>                                           unique_classes;
        float                                                      oob_score_;

        RandomForestClassifier(
            int                                                    n_estimators = 100,
            std::string                                            criterion = "gini",
            int                                                    max_depth = 10,
            std::optional<std::variant<int, std::string>>          max_features = "sqrt",
            int                                                    max_leaf_nodes = 10,
            int                                                    min_samples_leaf = 20,
            int                                                    min_samples_split = 30,
            float                                                  min_information_gain = 1e-5f,
            bool                                                   bootstrap = true,
            bool                                                   oob_score = false,
            std::optional<std::variant<int, float>>                max_samples = 1000,
            int                                                    random_state = 42,
            int                                                    verbose = 0,
            std::vector<int>                                       numerical_features = {},
            std::vector<int>                                       categorical_features = {}
        );

        void fit (arma::mat& X, arma::vec& Y) override;
        arma::vec predict (arma::mat& X) override;
        arma::vec predict_proba (arma::mat& X) override;

    private:
        std::vector<std::unique_ptr<DecisionTreeClassifier>> trees;
        std::vector<BootstrappedDataset>                     generated_bsd;
        std::mt19937                                         rng;

        BootstrapIndices create_bootstrap_indices (
            const int subsampled_max_samples,
            const int dataset_row_count,
            const int dataset_column_count,
            BootstrapIndices& bsd_indices,
            bool skip_column_generation
        );
        std::generator<BootstrappedDataset> bootstrap_dataset (
            const arma::mat& X,
            const arma::vec& Y,
            BootstrappedDataset& bsd,
            BootstrapIndices& bsd_indices,
            int estimator_counter
        );
        arma::mat build_forest (
            const arma::mat& X,
            const arma::vec& Y,
            arma::vec& out_y
        );
        int majority_vote (const std::vector<int>& tree_preds);
        arma::vec average_probabilities (
            const std::vector<arma::vec>& all_probs
        );
};
