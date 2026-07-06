#include <iostream>
#include <armadillo>
#include "base-headers/base.hpp"
#include "base-headers/classifier_mixin.hpp"
#include "cpp-utilities/loader.hpp"
#include "complex-datatypes/dataset_types.hpp"
#include "complex-datatypes/model_types.hpp"

using std::string;

using arma::vec;
using arma::mat;

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

void LogisticRegression::initialize_weights_bias (mat& train_x) {
    weights = arma::zeros<vec>(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

vec LogisticRegression::sigmoid (vec& logits) {
    vec sigmoid_predictions = 1.0 / (1.0 + arma::exp(-logits));
    return sigmoid_predictions;
}

vec LogisticRegression::compute_weights_gradients (mat& train_x, vec& train_y, vec& predictions) {
    vec computed_gradients = (1.0 / static_cast<double>(train_x.n_rows)) * (train_x.t() * (predictions - train_y));
    return computed_gradients;
}

double LogisticRegression::compute_bias_gradients (vec& train_y, vec& predictions) {
    double computed_bias = (1.0 / static_cast<double>(train_y.n_rows)) * arma::accu(predictions - train_y);
    return computed_bias;
}

void LogisticRegression::update_weights (vec& computed_weights_gradients) {
    weights = weights - learning_rate * computed_weights_gradients;
}

void LogisticRegression::update_bias (double computed_bias_gradient) {
    bias = bias - learning_rate * computed_bias_gradient;
}

void LogisticRegression::fit (mat& train_x, vec& train_y) {
    LogisticRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1) << std::endl;
        std::cout << "[+] Weights: " << weights.t() << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        vec predictions = (train_x * weights) + bias;
        vec sigmoided = LogisticRegression::sigmoid(predictions);

        vec computed_weights = LogisticRegression::compute_weights_gradients(train_x, train_y, sigmoided);
        double computed_bias = LogisticRegression::compute_bias_gradients(train_y, sigmoided);
        LogisticRegression::update_weights(computed_weights);
        LogisticRegression::update_bias(computed_bias);
    }
}

vec LogisticRegression::predict (mat& test_x) {
    vec logit_predictions = (test_x * weights) + bias;
    vec sigmoid_predictions = LogisticRegression::sigmoid(logit_predictions);
    return sigmoid_predictions;
}

int main () {
    const string dataset_path = "python-utilities/datasets/synthetic/classification";

    DatasetOperations dset_oper;
    LogisticRegression logreg_instance(100, 0.01f, true);

    dset_oper.construct_datasets();
    dset_oper.load_datasets(
        dataset_path + "/train_x.csv",
        dataset_path + "/train_y.csv",
        dataset_path + "/test_x.csv",
        dataset_path + "/test_y.csv"
    );

    logreg_instance.fit(
        std::get<mat>(dset_oper.datasets_vector.at(0)),
        std::get<vec>(dset_oper.datasets_vector.at(1))
    );

    vec logreg_preds = logreg_instance.predict(
        std::get<mat>(dset_oper.datasets_vector.at(2))
    );

    dset_oper.save_dataset(
        dataset_path + "/predictions/cpp-predictions.csv",
        logreg_preds
    );
}
