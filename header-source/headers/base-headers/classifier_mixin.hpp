#pragma once
#include <any>
#include <map>
#include <string>
#include <armadillo>

class ClassifierMixin {
    float score (arma::vec& predictions, arma::vec& test_y);
};
