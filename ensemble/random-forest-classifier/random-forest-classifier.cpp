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
    optional<float>                   max_samples,
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

variant<mat, vec> RandomForestClassifier::create_bootstrap_dataset (
    const std::variant<mat, vec>& dataset,
    const int subsampled_max_samples,
    const vector<int> subsampled_features,
    const mt19937& mt_generator
) {
    // Need to get non-contiguous row and column indices for matrix
    // Numerical features get different sets of rows and columns
    // Same goes for categorical features
    // The entire function needs to be feature, sample, and dataset agnostic

    // Use the already bootstrapped features as function parameter
    // Now the return type is the problem
}   

vector<int> RandomForestClassifier::create_bootstrap_features (
    const int bootstrapped_feature_count,
    vector<int> feature_vector,
    const mt19937& mt_generator
) {
    vector<int> temp_bsd_feats;
    uniform_int_distribution<int> uid_feat_generator(1, feature_vector.size());

    while (temp_bsd_feats.size() != bootstrapped_feature_count) {
        int bsd_generated_number = uid_feat_generator(mt_generator);

        auto bsd_num_location = find(
            temp_bsd_feats.begin(),
            temp_bsd_feats.end(),
            bsd_generated_number
        );

        if (bsd_num_location == temp_bsd_feats.end()) {
            temp_bsd_feats.push_back(bsd_generated_number);
        } else {
            continue;
        }
    }

    return temp_bsd_feats;
}

generator<BootstrappedDataset> RandomForestClassifier::bootstrap_dataset (
    const mat& X,
    vector<int> feature_vector,
    BootstrappedDataset& bsd
) {
    
    if (bootstrap) {
        random_device rd;
        mt19937 mt_generator(rd());
        int cvtd_bootstrapped_feats;
        int cvtd_bootstrapped_samples;
        vector<int> bootstrapped_features;

        try {
            // Processing of max samples to bootstrap
            if (max_samples.has_value() &&
                std::holds_alternative<int>(max_samples.value())
            ) {
                cvtd_bootstrapped_samples = std::get<int>(max_samples.value());
            } else {
                cvtd_bootstrapped_samples = std::get<float>(max_samples.value()) * X.n_rows;
            }

            // Processing of max features to bootstrap
            if (max_features.has_value()) {
                if (std::holds_alternative<int>(max_features.value())) {
                    cvtd_bootstrapped_feats = std::get<int>(max_features.value());
                } else if (std::holds_alternative<float>(max_features.value())) {
                    cvtd_bootstrapped_feats = static_cast<int>(
                        std::get<float>(max_features.value())
                    );
                } else if (std::get<string>(max_features.value()) == "sqrt") {
                    cvtd_bootstrapped_feats = static_cast<int>(
                        sqrt(feature_vector.size())
                    );
                } else if (std::get<string>(max_features.value()) == "log2") {
                    cvtd_bootstrapped_feats = static_cast<int>(
                        log2(feature_vector.size())
                    );
                } else {
                    throw invalid_argument("[-] Incorrect argument selection. Exiting!");
                }
            }

            bsd.bootstrapped_numerical_features = RandomForestClassifier::create_bootstrap_features(
                cvtd_bootstrapped_feats,
                numerical_features,
                mt_generator
            );
            bsd.bootstrapped_categorical_features = RandomForestClassifier::create_bootstrap_features(
                cvtd_bootstrapped_feats,
                categorical_features,
                mt_generator
            );
            variant<mat, vec> tmp_bs_X = RandomForestClassifier::create_bootstrap_dataset(
                X,
                cvtd_bootstrapped_samples.
                bsd.boot
            );
            // And here another problem arises, numerical and categorical features collide
            // and increase code length

            if () {

            }

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

// ──────────────────────────────────────────────
// bootstrap_sample  —  TODO: implement bootstrapping
// ──────────────────────────────────────────────

mat RandomForestClassifier::build_forest (
    const mat& X,
    const vec& Y,
    vec& out_y
) {
    random_device rd;
    mt19937 mt_generator_seeded(rd());

    for (int index = 0; index < n_estimators; index++) {

    }
}

// ──────────────────────────────────────────────
// fit  —  TODO: implement RF training
// ──────────────────────────────────────────────

void RandomForestClassifier::fit (const mat& X, const vec& Y) {
    
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
        std::nullopt,    // max_samples
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
