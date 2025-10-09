#include <armadillo>
#include<vector>

class DatasetValidator {
    public:
        virtual bool check_dataset_existence(std::vector<arma::mat>& datasets_vector);
        virtual bool check_shapes(const arma::mat& X, const arma::colvec& Y);
};
