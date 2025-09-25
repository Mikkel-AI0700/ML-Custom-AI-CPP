#include <iostream>
#include <map>
#include <string>
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
        void compute_weights_gradients (mat train_x, rowvec train_y, rowvec predictions);
        void compute_bias_gradients (rowvec train_y, rowvec predictions);
        void update_weights (rowvec computed_weight_gradients);
        void update_bias (float copmputed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    if (epochs == 0) {
        epochs = 25;
    }
    if (learning_rate <= 0.0001) {
        learning_rate = 0.0001;
    }
    
    fit_intercept = fit_intercept;
    Row<float> weights;
    float bias; 
};

void LinearRegression::compute_weights_gradients (mat train_x, rowvec train_y, rowvec predictions) {
    double weights_gradient = 1 / train_x.n_rows() * dot(train_x.t(), (predictions - train_y));
    return weights_gradient;
}

void LinearRegression::compute_bias_gradients (rowvec train_y, rowvec predictions) {
    double bias_gradient = 1 / train_y.n_rows() * sum((predictions - train_y));
    return bias_gradient;
}

void LinearRegression::update_weights (double weights_gradient) {

}

void LinearRegression::update_bias (double bias_gradient) {

}

void LinearRegression::fit (mat train_x, rowvec train_y) {

}

void LinearRegression::predict (mat test_x) {

}

int main () {

}

