#include <string>
#include <format>
#include <iostream>
#include <exception>
#include <armadillo>
#include <filesystem>
#include <variant>
#include "base.hpp"
#include "classifier_mixin.hpp"
#include "loader.hpp"
#include "dataset_types.hpp"
#include "model_types.hpp"

class LinearRegression: public BaseEstimator, public ClassifierMixin {
    public:
        arma::vec weights;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;
        HashMapParameters parameter_constrains;

        LinearRegression(int epochs, float learning_rate, bool fit_intercept);
        void fit (arma::mat& train_x, arma::vec& train_y) override;
        arma::vec predict (arma::mat& test_x) override;

    private:
        void initialize_weights_bias (arma::mat& train_x);
        arma::vec compute_weights_gradients (arma::mat& train_x, arma::vec& train_y, arma::vec& predictions);
        double compute_bias_gradients (arma::vec& train_y, arma::vec& predictions);
        void update_weights (arma::vec& computed_weight_gradients);
        void update_bias (float computed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    this->bias = 0.0;
    this->fit_intercept = fit_intercept;
    parameter_constrains = {
        {"epochs", epochs},
        {"learning_rate", learning_rate},
        {"fit_intercept", fit_intercept}
    };
};

void LinearRegression::initialize_weights_bias (arma::mat& train_x) {
    weights = arma::zeros<arma::vec>(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

arma::vec LinearRegression::compute_weights_gradients (
    arma::mat& train_x,
    arma::vec& train_y,
    arma::vec& predictions
) {
    arma::vec weights_gradient = 1.0 / train_x.n_rows * (train_x.t() * (predictions - train_y));
    return weights_gradient;
}

double LinearRegression::compute_bias_gradients (arma::vec& train_y, arma::vec& predictions) {
    double bias_gradient = 1.0 / train_y.n_rows * sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (arma::vec& computed_weight_gradients) {
    weights = weights - learning_rate * computed_weight_gradients;
}

void LinearRegression::update_bias (float computed_bias_gradient) {
    bias = bias - learning_rate * computed_bias_gradient;
}

void LinearRegression::fit (arma::mat& train_x, arma::vec& train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1) << std::endl;
        std::cout << "[+] Weights: " << weights.t() << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        // Main predictions logic
        arma::vec predictions = (train_x * weights) + bias;

        // Weights and bias computing and updating
        arma::vec weight_gradient = compute_weights_gradients(train_x, train_y, predictions);
        double bias_gradient = compute_bias_gradients(train_y, predictions);
        update_weights(weight_gradient);
        update_bias(bias_gradient);
    }
}

arma::vec LinearRegression::predict (arma::mat& test_x) {
    arma::vec predictions = (test_x * weights) + bias;
    return predictions;
}

// Purpose of main function is to provide a way to test the model
int main (int argc, char* argv[]) {
    if (argc < 8) {
        throw std::range_error("[-] Error: Argument count must be 8");
    }

    DatasetOperations dset_op;
    LinearRegression linreg_instance(
        std::stoi(argv[1]),
        std::stof(argv[2]),
        true
    );

    dset_op.construct_datasets();
    dset_op.load_datasets(
        std::string(argv[3]),
        std::string(argv[4]),
        std::string(argv[5]),
        std::string(argv[6])
    );

    linreg_instance.fit(
        std::get<arma::mat>(dset_op.datasets_vector.at(0)),
        std::get<arma::vec>(dset_op.datasets_vector.at(1))
    );

    arma::vec predictions = linreg_instance.predict(
        std::get<arma::mat>(dset_op.datasets_vector.at(2))
    );

    dset_op.save_dataset(
        "regression",
        std::string(argv[7]),
        predictions
    ); 
}

