#include <iostream>
#include "base-headers/base.hpp"
#include "base-headers/classifier_mixin.hpp"
#include "cpp-utilities/loader.hpp"
#include "complex-datatypes/complex_datatypes.hpp"

class LogisticRegression: public BaseEstimator, public ClassifierMixin {
    public:
        arma::vec weights;
        int epochs;
        float learning_rate;
        bool fit_intercept;
        HashMapParameters parameter_constraints;

        LogisticRegression(int epochs, float learning_rate, bool fit_intercept);
        void fit (arma::mat& train_x, arma::vec& train_y) override;
        arma::vec predict (arma::mat& test_x) override;

    private:
        void initialize_weights_bias (arma::mat& train_x);
        arma::vec compute_weights_gradients (arma::mat& train_x, arma::vec& train_y, arma::vec& predictions);
        double compute_bias_gradients (arma::vec& train_y, arma::vec& predictions);
        void update_weights (arma::vec& computed_weight_gradients);
        void update_bias (float computed_bias_gradient);
};

LogisticRegression::LogisticRegression (int epochs, float learning_rate, bool fit_intercept) {
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    this->fit_intercept = fit_intercept;
    parameter_constraints = {
        {"epochs", epochs},
        {"learning_rate", learning_rate},
        {"fit_intercept", fit_intercept}
    };
};

void LogisticRegression::initialize_weights_bias (arma::mat& train_x) {

}

arma::vec LogisticRegression::compute_weights_gradients (arma::mat& train_x, arma::vec& train_y, arma::vec& predictions) {

}

double LogisticRegression::compute_bias_gradients (arma::vec& train_y, arma::vec& predictions) {

}

void LogisticRegression::update_weights (arma::vec& computed_weights_gradients) {

}

void LogisticRegression::update_bias (float computed_weights_gradients) {

}

void LogisticRegression::fit (arma::mat& train_x, arma::vec& train_y) {

}

arma::vec LogisticRegression::predict (arma::mat& test_x) {

}

int main () {

}
