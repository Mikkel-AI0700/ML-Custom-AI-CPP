#include <iostream>
#include <armadillo>
#include "base-headers/base.hpp"
#include "base-headers/classifier_mixin.hpp"
#include "cpp-utilities/loader.hpp"
#include "complex-datatypes/complex_datatypes.hpp"

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

void LogisticRegression::initialize_weights_bias (arma::mat& train_x) {
    weights = arma::zeros<arma::vec>(train_x.n_cols);

    if (fit_intercept) {
        bias = 0.0;
    }
}

arma::vec LogisticRegression::sigmoid (arma::vec& logits) {
    arma::vec sigmoid_predictions = 1.0 / (1.0 + arma::exp(-logits));
    return sigmoid_predictions;
}

arma::vec LogisticRegression::compute_weights_gradients (arma::mat& train_x, arma::vec& train_y, arma::vec& predictions) {
    arma::vec computed_gradients = (1.0 / static_cast<double>(train_x.n_rows)) * (train_x.t() * (predictions - train_y));
    return computed_gradients;
}

double LogisticRegression::compute_bias_gradients (arma::vec& train_y, arma::vec& predictions) {
    double computed_bias = (1.0 / static_cast<double>(train_y.n_rows)) * arma::accu(predictions - train_y);
    return computed_bias;
}

void LogisticRegression::update_weights (arma::vec& computed_weights_gradients) {
    weights = weights - learning_rate * computed_weights_gradients;
}

void LogisticRegression::update_bias (double computed_bias_gradient) {
    bias = bias - learning_rate * computed_bias_gradient;
}

void LogisticRegression::fit (arma::mat& train_x, arma::vec& train_y) {
    LogisticRegression::initialize_weights_bias(train_x);

    for (int index = 0; index < epochs; index++) {
        std::cout << "[+] Epoch: " << (index + 1) << std::endl;
        std::cout << "[+] Weights: " << weights.t() << std::endl;
        std::cout << "[+] Bias: " << bias << std::endl;

        arma::vec predictions = (train_x * weights) + bias;
        arma::vec sigmoided = LogisticRegression::sigmoid(predictions);

        arma::vec computed_weights = LogisticRegression::compute_weights_gradients(train_x, train_y, sigmoided);
        double computed_bias = LogisticRegression::compute_bias_gradients(train_y, sigmoided);
        LogisticRegression::update_weights(computed_weights);
        LogisticRegression::update_bias(computed_bias);
    }
}

arma::vec LogisticRegression::predict (arma::mat& test_x) {
    arma::vec logit_predictions = (test_x * weights) + bias;
    arma::vec sigmoid_predictions = LogisticRegression::sigmoid(logit_predictions);
    return sigmoid_predictions;
}

int main (int argc, char* argv[]) {
    if (argc < 8) {
        throw std::runtime_error("[-] Error: Argument count must be exactly 8");
    }

    DatasetOperations dset_oper;
    LogisticRegression logreg_instance (std::stoi(argv[1]), std::stof(argv[2]), true);

    dset_oper.construct_datasets();
    dset_oper.load_datasets(
        std::filesystem::path(argv[3]),
        std::filesystem::path(argv[4]),
        std::filesystem::path(argv[5]),
        std::filesystem::path(argv[6])
    );

    logreg_instance.fit(
        std::get<arma::mat>(dset_oper.datasets_vector.at(0)),
        std::get<arma::vec>(dset_oper.datasets_vector.at(1))
    );

    arma::vec logreg_preds = logreg_instance.predict(
        std::get<arma::mat>(dset_oper.datasets_vector.at(2))
    );

    dset_oper.save_dataset(
        "classification", 
        std::string(argv[7]),
        logreg_preds
    );
    
}
