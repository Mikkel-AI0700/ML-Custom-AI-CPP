#pragma once

#include <iostream>
#include <armadillo>
#include "base.hpp"
#include "classifier_mixin.hpp"
#include "loader.hpp"
#include "dataset_types.hpp"
#include "model_types.hpp"

class LogisticRegression: public BaseEstimator, public ClassifierMixin {
    public:
        arma::vec weights;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;
        HashMapParameters parameter_constraints;

        LogisticRegression(int epochs, float learning_rate, bool fit_intercept);
        void fit (arma::mat& train_x, arma::vec& train_y) override;
        arma::vec predict (arma::mat& test_x) override;

    private:
        void initialize_weights_bias (arma::mat& train_x);
        arma::vec sigmoid (arma::vec& logits);
        arma::vec compute_weights_gradients (arma::mat& train_x, arma::vec& train_y, arma::vec& predictions);
        double compute_bias_gradients (arma::vec& train_y, arma::vec& predictions);
        void update_weights (arma::vec& computed_weight_gradients);
        void update_bias (double computed_bias_gradient);
};
