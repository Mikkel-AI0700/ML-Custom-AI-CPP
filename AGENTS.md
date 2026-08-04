# AGENTS.md

# ML-CUSTOM-AI-CPP

## Project Overview

From-scratch reimplementation of scikit-learn's core supervised learning
algorithms in C++23, using Armadillo for linear algebra. Each model is
benchmarked against its scikit-learn counterpart via Python evaluation
scripts.

- **Motivation**: Understand ML internals by building from scratch;
  deepen C++ skills; establish a benchmark baseline against Python.
- **Sibling repo**: [ML-Custom-AI](https://github.com/anomalyco/ML-Custom-AI)
  — pure Python reference implementation.
- **Languages**: C++ (model logic), Python (evaluation, sklearn tests),
  Bash (`run-script.sh` automation).

## Quick start
- Build: `./run-script.sh -O` (or `cmake -S . -B build && cmake --build build`)
- Full workflow: build -> run -> evaluate via `run-script.sh`

## Architecture

### C++ class hierarchy
```
BaseEstimator (virtual: fit, predict, predict_proba, save/load)
  ├── ClassifierMixin (score: accuracy)
  ├── RegressorMixin   (score: R²)
  └── Model classes inherit BaseEstimator + one mixin
```

All models inherit `BaseEstimator` and parameterize via `HashMapParameters`
(`std::map<std::string, std::variant<int, float, std::string, bool>>`).

### Build system
- CMake >=3.31.6, C++23, **requires Armadillo** (`find_package(Armadillo REQUIRED)`)
- A static library `CPP_SCIKIT_LEARN_LIBRARIES` aggregates shared code
  (base classes, loader, linalg ops); each model is a separate executable
  with its own embedded `main()`.
- The linker picks one `main()` per translation unit — only the first
  model source listed in `src/CMakeLists.txt` actually runs.
- Header search: all subdirs under `header-source/headers/`.

### Conventions
- No test framework (embedded `main()` functions in each model cpp)
- No linter, formatter, or typechecker configured
- CSVs loaded/saved with `arma::csv_ascii`
- Datasets are preprocessed, stored under `data/<task>/<dataset_name>/`

### Key types
- `TrainTestData` = `std::vector<std::variant<arma::mat, arma::colvec, arma::rowvec>>`
  — must `std::get<>` on every access, even when the type is already known
- `HashMapParameters` = `std::map<std::string, variant<int, float, string, bool>>`
- `SIDual` = `variant<std::string, int>` (used by tree for `max_features`)

## Implemented Models

### 1. LinearRegression
| Property | Detail |
|---|---|
| Inheritance | `BaseEstimator` + `ClassifierMixin` (note: header at `linear-regression.hpp:16` says `ClassifierMixin` — likely a bug; should be `RegressorMixin`) |
| Algorithm | Gradient descent (full-batch) |
| Parameters | `epochs` (int), `learning_rate` (float), `fit_intercept` (bool) |
| Key methods | `fit()` → GD loop; `predict()` → `Xw + b` |
| Current dataset | `"python-utilities/datasets/synthetic/regression"` (hardcoded in `main()`, line 84) |
| Source | `linear-regression/source-codes/linear-regression.cpp` |

### 2. LogisticRegression
| Property | Detail |
|---|---|
| Inheritance | `BaseEstimator` + `ClassifierMixin` |
| Algorithm | Gradient descent with sigmoid activation |
| Parameters | `epochs` (int), `learning_rate` (float), `fit_intercept` (bool) |
| Key methods | `fit()` → GD loop over sigmoid(logits); `predict()` → sigmoid(`Xw + b`) |
| Current dataset | `"python-utilities/datasets/synthetic/classification"` (hardcoded in `main()`, line 103) |
| Source | `logistic-regression/source-codes/logistic-regression.cpp` |

### 3. DecisionTreeClassifier
| Property | Detail |
|---|---|
| Inheritance | `BaseEstimator` + `ClassifierMixin` |
| Algorithm | Recursive binary decision tree with configurable split criteria |
| Parameters | `split_metric` (gini/entropy), `max_depth`, `max_features` (`SIDual`), `max_leaf_nodes`, `min_samples_leaf`, `min_samples_split`, `min_information_gain` |
| Key methods | `build_decision_tree()` (recursive), C++23 `std::generator` for split enumeration, `traverse_tree_prediction()` for inference |
| Current dataset | `"python-utilities/datasets/openml/adult"` (hardcoded in `main()`, line 493) |
| Source | `tree/decision-tree-classifier/tree.cpp` + `tree-header.hpp` |
| Notes | Tree is built depth-first; `max_leaf_nodes` uses a post-split `>=` guard (see [Decision Tree notes](#decision-tree--max_leaf_nodes_enforcement) below) |

## Data Pipeline

### Workflow (C++ → Python)
1. C++ executable reads preprocessed CSVs (hardcoded paths in `main()`)
2. Model runs training + inference, writes predictions CSV
3. `./run-script.sh -C -m <type> -T <truth> -P <predictions> (-S -c <metric> | -A)`

### Python environment
- Venv at `python-utilities/generator-venv/` exists on the `benchmark-tests`
  branch but **not on master** — create it first if missing
- Dependencies: `scikit-learn pandas numpy`
- `run-script.sh` auto-sets `PYTHONPATH` and activates venv

### Project layout
```
.
├── AGENTS.md                          # This file
├── CMakeLists.txt                     # Root CMake (C++23, Armadillo, optional GTest)
├── README.md
├── run-script.sh                      # Build/run/evaluate automation
├── .gitignore
├── header-source/
│   ├── headers/
│   │   ├── base-headers/              # BaseEstimator, ClassifierMixin, RegressorMixin
│   │   ├── complex-datatypes/         # HashMapParameters, TrainTestData, SIDual
│   │   ├── cpp-utilities/             # loader.hpp, linalg-operations.hpp
│   │   ├── model-headers/             # Per-model headers (decision-tree, lr, logr)
│   │   ├── metrics/                   # (empty)
│   │   ├── armadillo/                 # Bundled Armadillo tarball
│   │   └── validators/                # (empty)
│   └── header-source-files/
│       ├── base-source-files/         # .cpp for base classes + mixins
│       └── cpp-utilities-source-files/ # loader.cpp, linalg-operations.cpp
├── src/
│   └── CMakeLists.txt                 # Static lib + 3 model executables
├── linear-regression/
│   └── source-codes/linear-regression.cpp
├── logistic-regression/
│   └── source-codes/logistic-regression.cpp
├── tree/
│   └── decision-tree-classifier/tree.cpp
├── tests/                             # GTest unit tests (optional, -DBUILD_TESTS=ON)
│   ├── CMakeLists.txt
│   └── decision_tree_classifier_test.cpp
├── validators/
│   └── dataset_validator.cpp
├── performance-metrics/
│   └── decision-tree/                 # Generated bar-chart PNGs
├── python-utilities/                  # Python data, evaluation, test scripts
│   ├── __init__.py
│   ├── data/
│   │   ├── regression/
│   │   │   └── <dataset_name>/
│   │   │       ├── features.txt
│   │   │       ├── train_x.csv / train_y.csv
│   │   │       └── test_x.csv  / test_y.csv
│   │   └── classification/
│   │       ├── adult/                 # Binary classification (OpenML)
│   │       │   ├── features.txt
│   │       │   ├── train_x.csv / train_y.csv
│   │       │   ├── test_x.csv  / test_y.csv
│   │       │   └── predictions/       # Orphaned predictions (legacy)
│   │       └── <dataset_name>/
│   │           ├── features.txt
│   │           ├── train_x.csv / train_y.csv
│   │           └── test_x.csv  / test_y.csv
│   ├── evaluation/                    # Metric evaluation scripts
│   │   └── model_metric_checker.py
│   ├── tests/                         # Python sklearn test scripts
│   │   ├── DecisionTreeClassifier/
│   │   │   └── dt-test.py
│   │   ├── LinearRegression/
│   │   └── LogisticRegression/
│   ├── predictions/                   # Standalone per-algorithm predictions
│   │   ├── LinearRegression/
│   │   ├── LogisticRegression/
│   │   └── DecisionTreeClassifier/
│   └── generator-venv/                # Python venv (gitignored)
├── build/                             # CMake build dir (gitignored)
├── errors/                            # (empty)
└── metrics/                           # (empty)
```
- **Dataset path** is set via a `const std::string dataset_path` in each model's `main()`.
- Switch datasets by changing the `dataset_path` variable in the C++ source.
- Predictions CSV paths should mirror `predictions/<AlgorithmName>/<dataset_name>/cpp-predictions.csv`.

### Evaluation
- `evaluation/model_metric_checker.py` compares C++ predictions against
  sklearn ground truth and computes metrics (MSE, MAE, R² for regression;
  accuracy, precision, recall, F1 for classification; silhouette, V-measure
  for clustering).
- `tests/<Algorithm>/` contains Python scripts that train sklearn's
  implementation on the same dataset and save predictions, enabling
  side-by-side comparison.

## Recent changes (July 2026)

The `python-utilities/` directory was restructured from a dual-origin
layout (synthetic + OpenML) with inline generation scripts to a flat,
preprocessed-only layout.

### Deleted
- `generator-files/` — synthetic generator (`generator.py`) and OpenML
  fetcher (`get_dataset.py`)
- `json-config-files/` — generator parameter configs
- `datasets/synthetic/` — regression and classification data
- `datasets/openml/regression/` — empty
- `.npy` binary duplicates from the adult dataset

### Moved / Renamed
- `datasets/openml/adult/` → `data/classification/adult/`
- `performance-metric-checkers/` → `evaluation/`
- `python-sklearn-test/` → `tests/` (flattened into per-algorithm subdirs)
- `python-sklearn-test/openml/adult/dt-test.py` → `tests/DecisionTreeClassifier/dt-test.py`

### Created
- `data/regression/`, `data/classification/adult/` — preprocessed dataset homes
- `predictions/` — standalone top-level directory with per-algorithm subdirs
- `tests/LinearRegression/`, `tests/LogisticRegression/` — ready for future tests

### C++ paths — NOT YET UPDATED
All three model `main()` functions still use the old hardcoded paths
(e.g. `"python-utilities/datasets/synthetic/regression"`). These must be
updated to reflect the new layout before the models will run correctly.

## Decision Tree — max_leaf_nodes enforcement

### Implementation (Issue #25 — Merged)

Enforced `max_leaf_nodes` in `DecisionTreeClassifier` using a shared
counter (`leaf_node_count`, class member incremented in `create_node`)
and a `>=` guard placed post-split, pre-decision-node within
`build_decision_tree`.

### Control flow (in order)

**Pre-split (entry):**
1. `recursive_max_depth == max_depth` — stops before splitting at limit
2. `X.n_rows < min_samples_split` — too few samples to split

**Split loop** — enumerates all candidate feature splits.

**Post-split (after best split found):**
3. No features found — both `num_cond` and `cat_cond` empty
4. `best_gain < min_information_gain` — split not informative enough
5. `min_samples_leaf` — proposed children too small
6. `leaf_node_count >= max_leaf_nodes` — global leaf budget exhausted
   → returns leaf
   `else` → creates decision node + recurses into both children

### What was tried

- **Entry-level `==` guard** — original scaffold. Only fired at exact
  count; once a sibling subtree jumped past it, never fired again.
  Overshoot: unbounded.
- **Entry-level `>=` guard** — better but still wasted a full split-loop
  iteration on every child that entered after budget exhaustion.
- **Post-split `>=` guard + `else` block** — the final form. Checking
  *after* confirming a valid split but *before* allocating a decision
  node avoids wasting a split loop on children that will immediately
  become leaves. The `else` naturally captures left/right placement.

### Alternative considered (Issue #26)

An iterative BFS worklist (priority queue) approach was drafted but
rejected. Depth-first recursion with the `>=` guard bounded overshoot
to 0–1 leaves in practice, while BFS would have required a full
recursion rewrite and introduced a different growth strategy.

### Correctness

- `>=` catches the case where a sibling subtree already exhausted the
  budget (the root cause of the original `==`-only bug)
- The `else` block naturally routes children as left/right based on the
  call site that spawned them — no manual branch routing needed
- `max_depth` uses `==` at entry (stops one level early, per-branch
  resource); `max_leaf_nodes` uses `>=` post-split (shared resource
  needs strict enforcement)
- All other guards (min_samples_split, min_information_gain,
  min_samples_leaf) are local per-node checks unaffected by sibling
  state

# Day summary

## RandomForestClassifier — bootstrap sampling review (Option A)

Audited `bootstrap_features`/`bootstrap_dataset` in
`ensemble/random-forest-classifier/random-forest-classifier.cpp` and the
header `random-forest-classifier.hpp`. Adopted **Option A**: draw a random
position in `[0, pool.size())`, take the value at that position, and append
to the output — no membership/find needed since positions are always valid.
- Sampling is **without replacement** (distinct picks) via
  swap-with-end + shrink boundary (no erase/shift/resize).
- `pool.size()` is the **domain**; `feature_limit` (derived from
  `max_features`) is the **subsample count** — both coexist.
- The obsolete `column_count` parameter is slated for removal.

### Bug-fix status
- **Fixed**: dedup check (C), `X.n_rows` → `X.n_cols` (B, both call sites),
  wrong-variable overwrite (G), `build_bootstrap` → `bootstrap_dataset`
  rename (both files).
- **Unfixed**: index-0 unreachable (A — `uid_feat_generator(1, ...)`
  starts at 1), rejection loop (D), `column_count` removal (E).
- **New regressions**: R1 (loop reads `std::get<int>(max_features)` ignoring
  the `feature_limit` parameter), R2 (throws `bad_variant_access` on the
  default `"sqrt"`), R3 (string branches bypassed).
- **Blockers**: H (dead `holds_alternative<float>` / `get<float>` branch on
  `SIDual` = `variant<string,int>` → compile error), I (uncaught
  `bad_optional_access` when `max_samples` is nullopt, should default to
  `X.n_rows`), and a `max_samples` type mismatch (ctor `optional<float>` vs
  member `optional<variant<int,float>>`).
- **Deferred**: full bootstrapped `bootstrapped_X`/`bootstrapped_Y` logic,
  struct-grouping suggestion, and all crop of TODO functions (`fit`,
  `predict`, etc.).

### Armadillo `submat`/`subvec` finding
The scalar `submat(r1,c1,r2,c2)` and `subvec(first,last)` forms are
**contiguous-only**; `submat(span, span)` allows only fixed-stride spans.
The **index-vector** overloads — `X.rows(uvec)`, `X.cols(uvec)`,
`X.submat(uvec_rows, uvec_cols)` (returns `subview_elem2`) — accept
arbitrary **non-contiguous** (even repeated) indices and are the correct
forms for the future `bootstrapped_X`/`bootstrapped_Y` (with the
`std::vector<int>` → `uvec` conversion noted).

### AGENTS.md structural edits
- Added the `# ML-CUSTOM-AI-CPP` top-level heading.
- Added this `# Day summary` section.

# Agent Instructions: ML/AI R&D Assistant

## Project Philosophy & Scope
- **Primary focus**: "Re-creating from scratch," research & development, and performance optimization.
- **From-scratch recreations**: Stand-alone machine learning / deep learning algorithms or entire frameworks.
- **R&D / performance work**: Start from an existing implementation (either a recreation or a third-party library) and apply critical performance optimizations wherever possible.

## Coding Style & Paradigms
- Be comfortable with both **functional** and **object-oriented** programming.
- **C** projects: prefer a functional style.
- **Python / C++** projects: prefer an OOP style, using classes and modular design.
- **Syntax**: default to modern (C11+/C++17+/Python 3.10+). Use legacy syntax only when the problem, scenario, or dependency strictly requires it.
- **Explicit typing**: always use clear type annotations (Python type hints, C/C++ explicit types).
- **Modularity**: write every component as a self-contained module, with minimal coupling.

## Error Handling & Logging
- **Fatal errors / crashes**: terminate the program immediately, but first log a clear, concise error message.
- **Exceptions**: catch them at the appropriate level; stop execution, log the exception details, and then exit with a non-zero code.
- **Error output**: messages must be clean, human-readable, and contain only the information necessary to understand what went wrong (no stack traces unless explicitly requested).

## Tech Stack & Preferred Libraries

### Languages
- Python
- C
- C++

### Python Libraries
- NumPy, Pandas, Seaborn, Matplotlib
- Scikit-Learn, XGBoost
- PyTorch, TensorFlow

### C++ Libraries
- Armadillo

### C Libraries
- GSL (GNU Scientific Library)

*You may suggest additional well-established libraries if they significantly reduce effort or improve performance, but justify your choice.*

## Domain & Data Rules

### Target Audience
- Fellow ML researchers and engineers (R&D context).

### Data Handling
- **All metrics** (loss, statistics, probability, etc.) must be explicitly labelled and reported as floating-point numbers. If a metric is naturally a percentage (e.g., accuracy), output it both as a float and as a percentage string.
- When logging or printing metrics, always include the metric name, the raw float value, and the percentage equivalent (if applicable).

## Tool Execution Guardrails

### File Modification Boundary
- Respect a `.opencodeignore` file placed in the repository root. Treat it as a strict blacklist: never read, modify, or include in any output the files or directories it lists.

### Loop Breaker
- If any tool or command fails **two consecutive times** for the same step, stop immediately and report the failure to the user with the exact error.

### Pre-Execution Verification
- **Command execution**: You must always pause and confirm with the user any changes or actions or plans before attempting to proceed
- **File/directory existence**: always search (`grep`/`find`) before creating a file or directory to avoid accidental overwrites.
- **Python projects**:
  - Every package directory must contain an `__init__.py`.
  - The project root must contain a `pyproject.toml`. If it does not, determine the project root, scan for packages recursively, and generate a minimal `pyproject.toml` that correctly maps all packages.
- **C/C++ projects**:
  - A `CMakeLists.txt` must exist at the project root and must correctly list all required source files and headers.
  - If it's missing or incomplete, create/update it to reflect the actual source tree.

## Unit Testing & Benchmarking

### Testing Frameworks
- **Python**: `unittest`
- **C++**: Google Test (GTest)
- **C**: Criterion

### Benchmarking Frameworks
- **Python**: `pyperformance` (or `timeit` for micro-benchmarks)
- **C++**: Google Benchmark

### Execution & Output
- **Organisation**: group unit tests and benchmarks in dedicated directories/files, separated from production code.
- **Unit test output**: for every test, print a clear line that includes the test name and a `PASSED` or `FAILED` verdict.
- **Benchmark output**: label each benchmark case explicitly and display the measured time (and any other requested metric) in a human-readable format.

## Definition of Done
- A task is complete when you have successfully fulfilled all the user's request. 
- Leave the execution of any programs to the user.
- If any test fails, the user will report back to you either telling you to run the exact same command or let you run it.
