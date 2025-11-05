#ifndef CLASSIFIER_MIXIN_HPP
#define CLASSIFIER_MIXIN_HPP

#include <any>
#include <map>
#include <string>
#include <armadillo>

using namespace arma;

class ClassifierMixin {
    float score (rowvec predictions, rowvec test_y);
};

#endif

