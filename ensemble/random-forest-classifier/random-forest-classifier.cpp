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
using std::bad_variant_access;

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

void RandomForestClassifier::create_sub_rng_seed (
    const int dataset_row_count,
    mt19937& mt_generator
) {
    uniform_int_distribution<int> uid_child_rng_generator(0, 1000);
    for (int est_cnt = 0; est_cnt < n_estimators; ++est_cnt) {
        rng_per_tree.emplace_back(uid_child_rng_generator(mt_generator));
    }
}

void RandomForestClassifier::build_forest () {
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
        trees.emplace_back(std::move(dt_instance));
    }
}

int RandomForestClassifier::subsample_max_row_count (
    const int dataset_row_count
) {
    int cvtd_subsampled_max_features;

    try {
        if (max_samples.has_value()) {
            if (std::holds_alternative<int>(max_samples.value())) {
                cvtd_subsampled_max_features = std::get<int>(max_samples.value());
            } else if (std::holds_alternative<float>(max_samples.value())) {
                cvtd_subsampled_max_features = std::get<float>(max_samples.value()) * dataset_row_count;
            } else {
                throw bad_variant_access();
            }

            return cvtd_subsampled_max_features;
        } else {
            return 0;
        }
    } catch (bad_variant_access& invalid_user_argument_accessed) {
        cout << "[-] Error: " << invalid_user_argument_accessed.what() << endl;
        return 0;
    }
}

uvec RandomForestClassifier::create_indices (
    int row_count,
    int col_count,
    int subsampled_row_count,
    mt19937 sub_seeded_rng,
    bool create_row_indices,
    bool create_column_indices
) {
    if (create_row_indices) {
        uvec selected_indices = arma::regspace<uvec>(0, subsampled_row_count - 1);
        uniform_int_distribution<int> uid_generator(0, row_count - 1);

        for (arma::uword row_cnt = 0; row_cnt < subsampled_row_count; ++row_cnt) {
            selected_indices[row_cnt] = uid_generator(sub_seeded_rng);
        }

        return selected_indices;
    }

    if (create_column_indices) {
        uvec generated_col_indices = arma::regspace<uvec>(0, col_count - 1);
        return generated_col_indices;
    }
}

void RandomForestClassifier::fit (mat& X, vec& Y) {
    rng.seed(random_state);
    uniform_int_distribution<int> uid_samples_generator(0, X.n_rows);

    int subsampled_sample_count = RandomForestClassifier::subsample_max_row_count(X.n_rows);
    RandomForestClassifier::create_sub_rng_seed(X.n_rows, rng);
    RandomForestClassifier::build_forest();

    for (int est_idx = 0; est_idx < n_estimators; ++est_idx) {
        rng_child.seed(rng_per_tree[est_idx]);

        uvec bsd_row_indices = RandomForestClassifier::create_indices(
            X.n_rows,
            X.n_cols,
            subsampled_sample_count,
            rng_child,
            true,
            false
        );
        uvec bsd_col_indices = RandomForestClassifier::create_indices(
            X.n_rows,
            X.n_cols,
            subsampled_sample_count,
            rng_child,
            false,
            true
        );

        mat selected_X = X.submat(bsd_row_indices, bsd_col_indices);
        vec selected_y = Y.rows(bsd_row_indices);

        trees[est_idx]->fit(selected_X, selected_y);
    }
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
