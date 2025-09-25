#include <iostream>
#include <map>
#include <string>
#include <armadillo>

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

int main () {

}

