#include <string>
#include <format>
#include <iostream>
#include <exception>
#include <armadillo>
#include <filesystem>
#include <variant>
#include "linear-regression.hpp"

using arma::vec;
using arma::mat;

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

void LinearRegression::initialize_weights_bias (mat& train_x) {
    weights = arma::zeros(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

vec LinearRegression::compute_weights_gradients (
    mat& train_x,
    vec& train_y,
    vec& predictions
) {
    vec weights_gradient = 1.0 / train_x.n_rows * (train_x.t() * (predictions - train_y));
    return weights_gradient;
}

double LinearRegression::compute_bias_gradients (vec& train_y, vec& predictions) {
    double bias_gradient = 1.0 / train_y.n_rows * arma::sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (vec& computed_weight_gradients) {
    weights = weights - learning_rate * computed_weight_gradients;
}

void LinearRegression::update_bias (float computed_bias_gradient) {
    bias = bias - learning_rate * computed_bias_gradient;
}

void LinearRegression::fit (mat& train_x, vec& train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1) << std::endl;
        std::cout << "[+] Weights: " << weights.t() << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        // Main predictions logic
        vec predictions = (train_x * weights) + bias;

        // Weights and bias computing and updating
        vec pred_vec = predictions;
        vec weight_gradient = compute_weights_gradients(train_x, train_y, pred_vec);
        double bias_gradient = compute_bias_gradients(train_y, pred_vec);
        vec wg_vec = weight_gradient;
        update_weights(wg_vec);
        update_bias(bias_gradient);
    }
}

vec LinearRegression::predict (mat& test_x) {
    vec predictions = (test_x * weights) + bias;
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

    mat train_x = std::get<mat>(dset_op.datasets_vector.at(0));
    vec train_y = std::get<vec>(dset_op.datasets_vector.at(1));
    mat test_x = std::get<mat>(dset_op.datasets_vector.at(2));
    linreg_instance.fit(train_x, train_y);

    vec predictions = linreg_instance.predict(test_x);

    dset_op.save_dataset(
        std::filesystem::path(argv[7]),
        predictions
    ); 
}

