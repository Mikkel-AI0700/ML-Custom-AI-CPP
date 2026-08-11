#include <map>
#include <random>
#include <memory>
#include <vector>
#include <variant>
#include <random>
#include <optional>
#include <generator>
#include <algorithm>
#include <armadillo>
#include "loader.hpp"
#include "tree-header.hpp"
#include "random-forest-classifier.hpp"
#include "linalg-operations.hpp"

// Standard C++ STL
using std::endl;
using std::cout;
using std::string;
using std::vector;
using std::unique_ptr;
using std::optional;
using std::generator;
using std::map;
using std::variant;
using std::find;
using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;
using std::invalid_argument;

// Armadillo
using arma::mat;
using arma::vec;
using arma::uvec;
using arma::uword;
using arma::regspace;

RandomForestClassifier::RandomForestClassifier(
    int                               n_estimators,
    string                            criterion,
    int                               max_depth,
    optional<variant<int, string>>    max_features,
    int                               max_leaf_nodes,
    int                               min_samples_leaf,
    int                               min_samples_split,
    float                             min_information_gain,
    bool                              bootstrap,
    bool                              oob_score,
    optional<variant<int, float>>     max_samples,
    int                               random_state,
    int                               verbose,
    vector<int>                       numerical_features,
    vector<int>                       categorical_features
):
    n_estimators(n_estimators),
    criterion(criterion),
    max_depth(max_depth),
    max_features(max_features),
    max_leaf_nodes(max_leaf_nodes),
    min_samples_leaf(min_samples_leaf),
    min_samples_split(min_samples_split),
    min_information_gain(min_information_gain),
    bootstrap(bootstrap),
    oob_score(oob_score),
    max_samples(max_samples),
    random_state(random_state),
    verbose(verbose),
    numerical_features(numerical_features),
    categorical_features(categorical_features),
    rng(random_state),
    oob_score_(0.0f)
{}

BootstrapIndices RandomForestClassifier::create_bootstrap_indices (
    const int subsampled_max_samples,
    const int dataset_row_count,
    const int dataset_column_count,
    BootstrapIndices& bsd_indices,
    bool skip_column_generation
) {
    uvec temp_bsd_samples(subsampled_max_samples);
    uvec temp_bsd_features;
    uniform_int_distribution<int> uid_sample_generator(0, dataset_row_count);

    for (uword uw_int = 0; uw_int < subsampled_max_samples; ++uw_int) {
        temp_bsd_samples[uw_int] = uid_sample_generator(rng);
    }
    bsd_indices.bootstrapped_rows = temp_bsd_samples;
    
    if (skip_column_generation) {
        return bsd_indices;
    } else {
        temp_bsd_features = regspace<uvec>(0, dataset_column_count);
        bsd_indices.bootstrapped_cols = temp_bsd_features;

        return bsd_indices;
    }
}   

generator<BootstrappedDataset> RandomForestClassifier::bootstrap_dataset (
    const mat& X,
    const vec& Y,
    BootstrappedDataset& bsd,
    BootstrapIndices& bsd_indices,
    int estimator_count
) {
    
    if (bootstrap) {
        int cvtd_bootstrapped_samples;

        try {
            // Processing of max samples to bootstrap
            if (max_samples.has_value() && std::holds_alternative<int>(max_samples.value())) {
                cvtd_bootstrapped_samples = std::get<int>(max_samples.value());
            } else {
                cvtd_bootstrapped_samples = std::get<float>(max_samples.value()) * X.n_rows;
            }

            if (estimator_count == 0) {
                bsd_indices = RandomForestClassifier::create_bootstrap_indices(
                    cvtd_bootstrapped_samples,
                    X.n_rows,
                    X.n_cols,
                    bsd_indices,
                    false
                );
            } else {
                bsd_indices = RandomForestClassifier::create_bootstrap_indices(
                    cvtd_bootstrapped_samples,
                    X.n_rows,
                    X.n_cols,
                    bsd_indices,
                    true
                );
            }

            // Potential improvement: Find a way to remove the estimator
            // count parameter

            // Potential improvement: Find a way to remove the boolean
            // parameter and remove the condition

            // Potential improvement: For the X and Y parameter, pass the row count.

            bsd.bootstrapped_X = X.submat(
                bsd_indices.bootstrapped_rows, bsd_indices.bootstrapped_cols
            );
            bsd.bootstrapped_Y = Y.rows(
                bsd_indices.bootstrapped_rows
            );

            co_yield bsd;
        } catch (invalid_argument& incorrect_argument_selection) {
            cout << incorrect_argument_selection.what() << endl;
            co_return;
        }
    } else {
        cout << "[*] Bootstrapping disabled." << endl;
        co_return;
    }
}

mat RandomForestClassifier::build_forest (
    const mat& X,
    const vec& Y,
    vec& out_y
) {
    random_device rd;
    BootstrappedDataset bsd;
    BootstrapIndices bsd_indices;
    int est_cnt = 0;

    for (int est_cnt = 0; est_cnt < n_estimators; ++est_cnt) {
        auto dt_instance = std::make_unique<DecisionTreeClassifier>(
            criterion,
            max_depth,
            "sqrt",
            max_leaf_nodes,
            min_samples_leaf,
            min_samples_split,
            min_information_gain,
            numerical_features,
            categorical_features
        );

        cout << "[+] Tree no." << est_cnt << "created" << endl;
        trees.emplace_back(dt_instance);
    }
}

// ──────────────────────────────────────────────
// fit  —  TODO: implement RF training
// ──────────────────────────────────────────────

void RandomForestClassifier::fit (mat& X, vec& Y) {
    // Bootstrap the dataset here
    
    RandomForestClassifier::bootstrap_dataset();

    // Return it from here
}

// ──────────────────────────────────────────────
// predict  —  TODO: implement majority-vote inference
// ──────────────────────────────────────────────

vec RandomForestClassifier::predict (mat& X) {
    // TODO:
    //   For each row in X:
    //     Collect predictions from all trees
    //     result[i] = majority_vote(tree_preds)
    //   Return result as column vector

    return vec();
}

// ──────────────────────────────────────────────
// predict_proba  —  TODO: implement probability averaging
// ──────────────────────────────────────────────

vec RandomForestClassifier::predict_proba (mat& X) {
    // TODO:
    //   For each tree, collect probability rowvectors
    //   Average them element-wise across all trees
    //   Return averaged probability rowvector

    return vec();
}

// ──────────────────────────────────────────────
// majority_vote  —  TODO: implement voting logic
// ──────────────────────────────────────────────

int RandomForestClassifier::majority_vote (const vector<int>& tree_preds) {
    // TODO:
    //   Count frequency of each class in tree_preds
    //   Return the class with the highest count
    //   (Tie-break: return the smaller class index)

    return -1;
}

// ──────────────────────────────────────────────
// average_probabilities  —  TODO: implement averaging
// ──────────────────────────────────────────────

vec RandomForestClassifier::average_probabilities (
    const vector<vec>& all_probs
) {
    // TODO:
    //   Sum all rowvecs element-wise, divide by n_estimators
    //   Return the averaged probability vector

    return vec();
}

// ──────────────────────────────────────────────
// main  —  fully functional driver
// ──────────────────────────────────────────────

#ifndef SKIP_MAIN
int main () {
    const string dataset_path = "python-utilities/data/classification/adult";

    // Feature type indices (adult dataset: cols 0-5 numerical, 6-21 categorical)
    vector<int> num_feats = {};
    vector<int> cat_feats = {};

    for (int i = 0; i <= 5; i++) {
        cout << "Appended feature index " << i << endl;
        num_feats.push_back(i);
    }

    for (int i = 6; i <= 21; i++) {
        cout << "Appended feature index " << i << endl;
        cat_feats.push_back(i);
    }

    RandomForestClassifier rf(
        100,             // n_estimators
        "gini",          // criterion
        10,              // max_depth
        "sqrt",          // max_features
        10,              // max_leaf_nodes
        20,              // min_samples_leaf
        20,              // min_samples_split
        0.0001f,         // min_information_gain
        true,            // bootstrap
        false,           // oob_score
        10000,    // max_samples
        42,              // random_state
        0,               // verbose
        num_feats,       // numerical_features
        cat_feats        // categorical_features
    );

    DatasetOperations dset_ops;
    dset_ops.construct_datasets();
    dset_ops.load_datasets(
        dataset_path + "/train_x.csv",
        dataset_path + "/train_y.csv",
        dataset_path + "/test_x.csv",
        dataset_path + "/test_y.csv"
    );

    mat train_x = std::get<mat>(dset_ops.datasets_vector.at(0));
    vec train_y = std::get<vec>(dset_ops.datasets_vector.at(1));
    mat test_x  = std::get<mat>(dset_ops.datasets_vector.at(2));

    rf.fit(train_x, train_y);

    vec predictions = rf.predict(test_x);

    dset_ops.save_dataset(
        dataset_path + "/predictions/cpp-predictions.csv",
        predictions
    );

    cout << "RandomForestClassifier predictions saved to "
         << dataset_path << "/predictions/cpp-predictions.csv" << endl;
}
#endif
