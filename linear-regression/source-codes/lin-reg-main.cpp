#include <iostream>
#include <cctype>
#include <string>
#include <exception>
#include <format>
#include <filesystem>
#include <armadillo>
#include "base-headers/base.hpp"
#include "base-headers/classifier_mixin.hpp"
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
        std::cout << "[+] Epoch: " << index << std::endl;
        std::cout << "[+] Weights: " << weights << std::endl;
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
    TrainTestData datasets = {};
    std::vector<std::string> dataset_metadata = {"mat", "colvec", "mat", "colvec"};

    for (const auto& dset_metadata : dataset_metadata) {
        if (dset_metadata == "mat") {
            datasets.emplace_back(arma::mat{});
        } else {
            datasets.emplace_back(arma::colvec{});
        }
    }

    DatasetsFilepaths data_path = {
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-data-generators/test-data/regression-data/train_x.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-data-generators/test-data/regression-data/train_y.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-data-generators/test-data/regression-data/test_x.csv",
        "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-data-generators/test-data/regression-data/test_y.csv"
    };
    
    try {
        // Designed to throw std::length_error if user has not provided three arguments
        if (argc < 3) {
            throw std::length_error("[-] Error: Argument provided is less than two!. Aborting");
        }

        // Designed to throw std::invalid_argument if user has provided a incorrect argument
        // Such as a character or a string
        LinearRegression linreg_instance(
            std::stoi(std::string(argv[1])),
            std::stof(std::string(argv[2])),
            true
        );

        // Loop to load in the datasets
        for (int index = 0; index < datasets.size(); index++) {
            if (std::holds_alternative<arma::mat>(datasets[index])) {
                std::get<arma::mat>(datasets[index]).load(data_path[index], arma::csv_ascii);
                std::cout << "[+] From C++, loading the training/testing matrices datset" << std::endl;
            } else {
                std::get<arma::colvec>(datasets[index]).load(data_path[index], arma::csv_ascii);
                std::cout << "[+] From C++, loading the training/testing column vectors" << std::endl;
            }
        }

        linreg_instance.fit(
            std::get<arma::mat>(datasets.at(0)),
            std::get<arma::colvec>(datasets.at(1))
        );

        arma::colvec model_predictions = linreg_instance.predict(
            std::get<arma::mat>(datasets.at(2))
        );

        model_predictions.save(
            "/home/mikkel/Desktop/ai-projects/machine-learning/custom-ai-cpp/python-data-generators/model-predictions/regressor/test_data_one.csv",
            arma::csv_ascii
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

