#include <string>
#include <format>
#include <iostream>
#include <exception>
#include <armadillo>
#include <filesystem>
#include <variant>
#include "base-headers/base.hpp"
#include "base-headers/classifier_mixin.hpp"
#include "cpp-utilities/loader.hpp"
#include "complex-datatypes/complex_datatypes.hpp"

class LinearRegression: public BaseEstimator, public ClassifierMixin {
    public:
        arma::colvec weights;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;
        HashMapParameters parameter_constrains;

        LinearRegression(int epochs, float learning_rate, bool fit_intercept);
        void fit (arma::mat& train_x, arma::colvec& train_y) override;
        arma::colvec predict (arma::mat& test_x) override;

    private:
        void initialize_weights_bias (arma::mat& train_x);
        arma::colvec compute_weights_gradients (arma::mat& train_x, arma::colvec& train_y, arma::colvec& predictions);
        double compute_bias_gradients (arma::colvec& train_y, arma::colvec& predictions);
        void update_weights (arma::colvec& computed_weight_gradients);
        void update_bias (float copmputed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    this->fit_intercept = fit_intercept;
    parameter_constrains = {
        {"epochs", epochs},
        {"learning_rate", learning_rate},
        {"fit_intercept", fit_intercept}
    };
};

void LinearRegression::initialize_weights_bias (arma::mat& train_x) {
    weights = arma::zeros<arma::colvec>(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

arma::colvec LinearRegression::compute_weights_gradients (arma::mat& train_x, arma::colvec& train_y, arma::colvec& predictions) {
    arma::colvec weights_gradient = 1.0 / train_x.n_rows * (train_x.t() * (predictions - train_y));
    return weights_gradient;
}

double LinearRegression::compute_bias_gradients (arma::colvec& train_y, arma::colvec& predictions) {
    double bias_gradient = 1.0 / train_y.n_rows * sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (arma::colvec& weights_gradient) {
    weights = weights - learning_rate * weights_gradient;
}

void LinearRegression::update_bias (float bias_gradient) {
    bias = bias - learning_rate * bias_gradient;
}

void LinearRegression::fit (arma::mat& train_x, arma::colvec& train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1) << std::endl;
        std::cout << "[+] Weights: " << weights.t() << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        // Main predictions logic
        arma::colvec predictions = (train_x * weights) + bias;

        // Weights and bias computing and updating
        arma::colvec weight_gradient = LinearRegression::compute_weights_gradients(train_x, train_y, predictions);
        double bias_gradient = LinearRegression::compute_bias_gradients(train_y, predictions);
        LinearRegression::update_weights(weight_gradient);
        LinearRegression::update_bias(bias_gradient);
    }
}

arma::colvec LinearRegression::predict (arma::mat& test_x) {
    arma::colvec predictions = (test_x * weights) + bias;
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
        std::get<arma::colvec>(dset_op.datasets_vector.at(1))
    );

    arma::colvec predictions = linreg_instance.predict(
        std::get<arma::mat>(dset_op.datasets_vector.at(2))
    );

    std::string regression_save_type = "regression";
    dset_op.save_dataset(
        regression_save_type,
        std::string(argv[7]),
        predictions
    ); 
}

