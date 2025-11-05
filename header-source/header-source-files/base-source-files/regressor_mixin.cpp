#ifndef REGRESSOR_MIXIN_HPP
#define REGRESSOR_MIXIN_HPP

#include <any>
#include <format>
#include <armadillo>

using namespace arma;

class RegressorMixin {
    float score (rowvec predictions, rowvec test_x);
};

#endif

