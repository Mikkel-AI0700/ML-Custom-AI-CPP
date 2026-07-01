cmake_minimum_required(VERSION 3.31.6)

project(CPP_SCIKIT_LEARN VERSION 1.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests (requires Google Test)" OFF)

find_package(Armadillo REQUIRED)

add_library(
    CPP_SCIKIT_LEARN_LIBRARIES
    STATIC
    header-source/header-source-files/base-source-files/base.cpp
    header-source/header-source-files/base-source-files/classifier_mixin.cpp
    header-source/header-source-files/base-source-files/regressor_mixin.cpp
    header-source/header-source-files/base-source-files/transformer_mixin.cpp
    header-source/header-source-files/cpp-utilities-source-files/loader.cpp
    header-source/header-source-files/cpp-utilities-source-files/linalg-operations.cpp
)

add_executable(
    CPP_SCIKIT_LEARN_EXECUTABLES
    tree/decision-tree-classifier/tree.cpp
    linear-regression/source-codes/linear-regression.cpp
    logistic-regression/source-codes/logistic-regression.cpp
)

target_include_directories(
    CPP_SCIKIT_LEARN_LIBRARIES
    PUBLIC
    "header-source/headers"
    "header-source/headers/base-headers"
    "header-source/headers/complex-datatypes"
    "header-source/headers/cpp-utilities"
    "header-source/headers/metrics"
    "header-source/headers/model-headers"
    "header-source/headers/model-headers/decision-tree-header"
    "header-source/headers/model-headers/linear-regression-header"
    "header-source/headers/model-headers/logistic-regression-header"
    ${ARMADILLO_LIBRARIES}
)

target_link_libraries(
    CPP_SCIKIT_LEARN_EXECUTABLES
    CPP_SCIKIT_LEARN_LIBRARIES
    ${ARMADILLO_INCLUDE_DIRS}
)

if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        1.17.0
    )

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest googlebenchmark)

    add_subdirectory(tree/DecisionTreeClassifier_UnitTest)
endif()
