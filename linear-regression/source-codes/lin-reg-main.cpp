#include <iostream>
#include <filesystem>
#include <string>
#include <format>
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

        LinearRegression(int epochs, float learning_rate, bool fit_intercept);
        void fit (arma::mat train_x, arma::rowvec train_y);
        arma::rowvec predict (arma::mat test_x);

    private:
        void initialize_weights_bias (arma::mat train_x);
        arma::rowvec compute_weights_gradients (arma::mat train_x, arma::rowvec train_y, arma::rowvec predictions);
        double compute_bias_gradients (arma::rowvec train_y, arma::rowvec predictions);
        void update_weights (arma::rowvec computed_weight_gradients);
        void update_bias (float copmputed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    this->weights;
    this->bias;
    this->epochs = epochs;
    this->learning_rate = learning_rate;
    this->fit_intercept = fit_intercept;
};

void LinearRegression::initialize_weights_bias (mat train_x) {
    this->weights = this->weights.zeros(train_x.n_cols);

    if (this->fit_intercept) {
        this->bias = 0.0;
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
    this->weights = this->weights - this->learning_rate * weights_gradient;
}

void LinearRegression::update_bias (float bias_gradient) {
    this->bias = this->bias - this->learning_rate * bias_gradient;
}

void LinearRegression::fit (arma::mat train_x, arma::rowvec train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < this->epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1);
        std::cout << "[+] Weights: " << this->weights;
        std::cout << "[+] Bias: " << this->bias;

        // Main predictions logic
        arma::rowvec predictions = (train_x * this->weights.t()) + this->bias;

        // Weights and bias computing and updating
        arma::rowvec weight_gradient = LinearRegression::compute_weights_gradients(train_x, train_y, predictions);
        double bias_gradient = LinearRegression::compute_bias_gradients(train_y, predictions);
        LinearRegression::update_weights(weight_gradient);
        LinearRegression::update_bias(bias_gradient);
    }
}

arma::rowvec LinearRegression::predict (arma::mat test_x) {
    return (test_x * this->weights) + this->bias;
}

int main (int argc, char *argv[]) {
    LinearRegression linreg_instance(2000, 0.0001, true);
    arma::mat train_x;
    arma::colvec train_y;
    arma::mat test_x;
    arma::colvec test_y;

    std::map<std::string, std::variant<int, float, bool>> model_hyperparams = {
        {"epochs", linreg_instance.epochs},
        {"learning_rate", linreg_instance.learning_rate},
        {"fit_intercept", linreg_instance.fit_intercept}
    };

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
        if (argc < 3) {
            throw "[-] Insufficient amount of arguments";
        }

        for (int index = 0; index < datasets.size(); index++) {
            if (std::holds_alternative<arma::mat>(datasets[index])) {
                std::get<arma::mat>(datasets[index]).load(data_path[index]);
            } else {
                std::get<arma::colvec>(datasets[index]).load(data_path[index]);
            }
        }

        linreg_instance.fit(std::get<arma::mat>(datasets.at(0)), std::get<arma::colvec>(datasets.at(1)));
    } catch (std::string incorrect_insufficient_argument) {
        std::cout << incorrect_insufficient_argument << std::endl;
    }
}

