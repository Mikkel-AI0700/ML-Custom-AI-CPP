#ifndef CLASSIFIER_MIXIN_HPP
#define CLASSIFIER_MIXIN_HPP

#include <any>
#include <map>
#include <string>
#include <armadillo>

class ClassifierMixin {
    float score (arma::vec& predictions, arma::vec& test_y);
};

#endif

