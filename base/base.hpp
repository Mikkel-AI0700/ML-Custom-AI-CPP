#ifndef BASE_ESTIMATOR_HPP
#define BASE_ESTIMATOR_HPP

#include <any>
#include <map>
#include <string>
#include <armadillo>

using namespace arma;

class BaseEstimator {
    public:
        virtual std::map<std::string, any> get_parameters ();
        virtual void set_parameters (
            std::map<std::string, any> original_hyperparameters,
            std::map<std::string, any> user_hyperparameters
        );

        virtual void fit (mat train_x, rowvec train_y);
        virtual void predict (mat test_x);
        virtual void predict_proba (mat test_x);
        virtual void predict_proba_log (mat test_x);

        virtual void save_model (
            std::string model_save_name,
            std::filesystem::path save_path
        );
        virtual void load_model (
            std::filesystem::path load_path
        );
};

#endif

