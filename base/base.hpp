#ifndef BASE_ESTIMATOR_HPP
#define BASE_ESTIMATOR_HPP

#include <any>
#include <map>
#include <string>

class BaseEstimator {
    public:
        virtual std::map<std::string, any> get_parameters ();
        virtual void set_parameters ();

        virtual void fit ();
        virtual void predict ();
        virtual void predict_proba ();
        virtual void predict_proba_log ();

        virtual void save_model ();
        virtual void load_model ();
};

#endif

