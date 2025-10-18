#include <iostream>
#include <cctype>
#include <string>
#include <exception>
#include <format>
#include <filesystem>
#include <armadillo>
#include "../../include/class-inheritors/base.hpp"
#include "../../include/class-inheritors/classifier_mixin.hpp"

using namespace arma;

class LinearRegression: public BaseEstimator, public ClassifierMixin {
    public:
        arma::rowvec weights;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;
        HashMapParameters parameter_constrains;

        LinearRegression(int epochs, float learning_rate, bool fit_intercept);

        void fit (arma::mat train_x, arma::rowvec train_y) override;
        arma::rowvec predict (arma::mat test_x) override;

    private:
        void initialize_weights_bias (arma::mat train_x);
        arma::rowvec compute_weights_gradients (arma::mat train_x, arma::rowvec train_y, arma::rowvec predictions);
        double compute_bias_gradients (arma::rowvec train_y, arma::rowvec predictions);
        void update_weights (arma::rowvec computed_weight_gradients);
        void update_bias (float copmputed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    weights;
    bias;
    epochs = epochs;
    learning_rate = learning_rate;
    fit_intercept = fit_intercept;
    parameter_constrains = {
        {"epochs", epochs},
        {"learning_rate", learning_rate},
        {"fit_intercept", fit_intercept}
    };
};

void LinearRegression::initialize_weights_bias (arma::mat train_x) {
    weights = weights.zeros(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

arma::rowvec LinearRegression::compute_weights_gradients (arma::mat train_x, arma::rowvec train_y, arma::rowvec predictions) {
    arma::rowvec weights_gradient = 1.0 / train_x.n_rows * (train_x * (predictions - train_y));
    return weights_gradient;
}

double LinearRegression::compute_bias_gradients (arma::rowvec train_y, arma::rowvec predictions) {
    double bias_gradient = 1 / train_y.n_rows * sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (arma::rowvec weights_gradient) {
    weights = weights - learning_rate * weights_gradient;
}

void LinearRegression::update_bias (float bias_gradient) {
    bias = bias - learning_rate * bias_gradient;
}

void LinearRegression::fit (arma::mat train_x, arma::rowvec train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 1; index < epochs; index++) {
        std::cout << "[+] Epoch: " << index << std::endl;
        std::cout << "[+] Weights: " << weights << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        // Main predictions logic
        arma::rowvec predictions = (train_x * weights.t()) + bias;

        // Weights and bias computing and updating
        arma::rowvec weight_gradient = LinearRegression::compute_weights_gradients(train_x, train_y, predictions);
        double bias_gradient = LinearRegression::compute_bias_gradients(train_y, predictions);
        LinearRegression::update_weights(weight_gradient);
        LinearRegression::update_bias(bias_gradient);
    }
}

arma::rowvec LinearRegression::predict (arma::mat test_x) {
    return (test_x * weights) + bias;
}

int main (int argc, char* argv[]) {
    arma::mat train_x;
    arma::colvec train_y;
    arma::mat test_x;
    arma::colvec test_y;
    
    std::vector<std::variant<arma::mat, arma::colvec>> datasets = {
        train_x,
        train_y,
        test_x,
        test_y
    };
    
    std::vector<std::filesystem::path> data_path = {
        "python-data-generators/test-data/train_x.csv",
        "python-data-generators/test-data/train_y.csv",
        "python-data-generators/test-data/test_x.csv",
        "python-data-generators/test-data/test_y.csv"
    };
    
    try { 
        int epoch_script_argument = std::stoi(std::string(argv[1]));
        float learning_rate_script_argument = std::stof(std::string(argv[2]));

        if (argc < 2) {
            throw std::length_error("[-] Error: Argument provided is less than two!. Aborting");
        }

        LinearRegression linreg_instance(
            epoch_script_argument,
            learning_rate_script_argument,
            true
        );

        for (int index = 0; index < data_path.size(); index++) {
            if (std::holds_alternative<arma::mat>(datasets[index])) {
                std::get<arma::mat>(datasets[index]).load(data_path[index]);
            } else {
                std::get<arma::colvec>(datasets[index]).load(data_path[index]);
            }
        }

        linreg_instance.fit(
            std::get<arma::mat>(datasets.at(0)), 
            std::get<arma::colvec>(datasets.at(1))
        );
    } catch (const std::invalid_argument& invalid_script_argument) {
        std::cerr << "[-] Error: Invalid argument catch triggered" << std::endl;
        std::cerr << invalid_script_argument.what() << std::endl;
    }
    catch (const std::length_error& invalid_argument_length) {
        std::cerr << "[-] Error: Invalid argument length catch triggered" << std::endl;
        std::cerr << invalid_argument_length.what() << std::endl;
    }
}

