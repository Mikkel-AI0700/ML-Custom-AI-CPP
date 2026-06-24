cmake_minimum_required(VERSION 3.31.6)

project(CPP_SCIKIT_LEARN VERSION 1.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Armadillo REQUIRED)

add_executable(
    CPP_SCIKIT_LEARN
    tree/decision-tree-classifier/tree.cpp
    linear-regression/source-codes/linear-regression.cpp
    logistic-regression/source-codes/logistic-regression.cpp
    header-source/header-source-files/base-source-files/base.cpp
    header-source/header-source-files/base-source-files/classifier_mixin.cpp
    header-source/header-source-files/base-source-files/regressor_mixin.cpp
    header-source/header-source-files/base-source-files/transformer_mixin.cpp
    header-source/header-source-files/cpp-utilities-source-files/loader.cpp
)

target_include_directories(
    CPP_SCIKIT_LEARN
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
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(
    CPP_SCIKIT_LEARN
    ${ARMADILLO_LIBRARIES}
)
