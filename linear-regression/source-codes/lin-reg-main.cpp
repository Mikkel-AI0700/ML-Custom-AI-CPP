#include <iostream>
#include <map>
#include <string>
#include <format>
#include <armadillo>

using namespace arma;

class LinearRegression {
    public:
        rowvec weights;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;

        void fit (mat train_x, rowvec train_y);
        rowvec predict (mat test_x);

    private:
        void initialize_weights_bias (mat train_x);
        double compute_weights_gradients (mat train_x, rowvec train_y, rowvec predictions);
        double compute_bias_gradients (rowvec train_y, rowvec predictions);
        void update_weights (rowvec computed_weight_gradients);
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

double LinearRegression::compute_weights_gradients (mat train_x, rowvec train_y, rowvec predictions) {
    double weights_gradient = 1 / train_x.n_rows() * dot(train_x.t(), (predictions - train_y));
    return weights_gradient;
}

dobule LinearRegression::compute_bias_gradients (rowvec train_y, rowvec predictions) {
    double bias_gradient = 1 / train_y.n_rows() * sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (double weights_gradient) {
    this->weights = this->weights - this->learning_rate - weights_gradient;
}

void LinearRegression::update_bias (double bias_gradient) {
    this->bias = this->bias - this->learning_rate - bias_gradient;
}

void LinearRegression::fit (mat train_x, rowvec train_y) {
    LinearRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < this->epochs; index++) {
        std::cout << std::format("Epoch: {} | Weights: {} | Bias: {}", epoch, this->weights, this->bias);

        rowvec predictions = dot(train_x, this->weights) * this->bias;
        double weight_gradient = LinearRegression::compute_weights_gradients(train_x, train_y, predictions);
        double bias_gradient = LinearRegression::compute_bias_gradients(train_y, predictions);
        LinearRegression::update_weights(weight_gradient);
        LinearRegression::update_bias(bias_gradient);
    }
}

void LinearRegression::predict (mat test_x) {
    return dot(test_x, this->weights) * this->bias;
}

int main () {

}

