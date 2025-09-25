#include <iostream>
#include <map>
#include <string>
#include <armadillo>

enum Errors = {
    OutOfBoundsException
};

class LinearRegression {
    public:
        Row<float>;
        float bias;
        int epochs;
        float learning_rate;
        bool fit_intercept;

        void fit (Mat<float> train_x, Row<float> train_y);
        Row<float> predict (Mat<float> test_x);

    private:
        void compute_weights_gradients (Mat<float> train_x, Row<float> train_y, Row<float> predictions);
        void compute_bias_gradients (Row<float> train_y, Row<float> predictions);
        void update_weights (Row<float> computed_weight_gradients);
        void update_bias (float copmputed_bias_gradient);
};

LinearRegression::LinearRegression (int epochs, float learning_rate, bool fit_intercept) {
    if (epochs <= 0) {
        printf(Errors::OutOfBoundsException);
        epochs = 25;
    }
    if (learning_rate <= 0.0001) {
        printf(Errors::OutOfBoundsException);
        learning_rate = 0.0001;
    }
    fit_intercept = fit_intercept;
};

void fit (Mat<float> train_x, Row<float> train_y) {

}

void predict (Row<float> train_y, Row<float> predictions) {

}

int main () {

}

