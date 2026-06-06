#pragma once
#include <any>
#include <format>
#include <armadillo>

using namespace arma;

class RegressorMixin {
    float score (rowvec predictions, rowvec test_x);
};
