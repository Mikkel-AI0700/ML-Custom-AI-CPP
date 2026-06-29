# AGENTS.md

## Quick start
- Build: `./run-script.sh -O` (or `cmake -S . -B build && cmake --build build`)
- Generate regression data: `./run-script.sh -G -d regression`
- Full workflow: generate -> build -> run -> evaluate via `run-script.sh`

## Build system
- CMake >=3.31.6, C++23, **requires Armadillo** (`find_package(Armadillo REQUIRED)`)
- Single executable `CPP_SCIKIT_LEARN`; each model cpp has its own `main()` --
  the linker picks the first one per translation unit order
- Header search: `header-source/headers/` and subdirs

## Data workflow (Python -> C++ -> Python)
1. `./run-script.sh -G -d <type>` (regression|classification|clustering) -> sklearn CSVs
2. Run the C++ exe with 8 CLI args: `epochs lr train_x train_y test_x test_y out_filename`
3. `./run-script.sh -C -m <type> -T <truth> -P <predictions> (-S -c <metric> | -A)`

## Python environment
- Venv at `python-utilities/generator-venv/` exists on the `benchmark-tests`
  branch but **not on master** -- create it or switch branches before `-G`
- Dependencies: `scikit-learn pandas numpy`
- `run-script.sh` auto-sets `PYTHONPATH` and activates venv

## C++ quirks
- `TrainTestData` is `std::vector<std::variant<arma::mat, arma::colvec, arma::rowvec>>`
  -- must `std::get<>` on every access, even when the type is already known
- `loader.hpp:8-11` hardcodes absolute `SavePaths` -- update if repo is moved
- All models inherit `BaseEstimator`; params use `HashMapParameters`
  (`std::map<std::string, std::variant<int, float, std::string, bool>>`)

## Conventions
- No test framework (embedded `main()` functions in each model cpp)
- No linter, formatter, or typechecker configured
- CSVs loaded/saved with `arma::csv_ascii`
- Dataset generator params in `python-utilities/json-config-files/<type>/`

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
- A task is complete **only when all unit tests pass** (no errors, no failures).
- Once the tests pass, report to the user: "All tests passed successfully."
- If any test fails, report the failures and do **not** mark the task as done--request further guidance if necessary.
