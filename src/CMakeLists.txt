add_library(
    CPP_SCIKIT_LEARN_LIBRARIES
    STATIC
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/base.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/classifier_mixin.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/regressor_mixin.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/transformer_mixin.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/cpp-utilities-source-files/loader.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/cpp-utilities-source-files/linalg-operations.cpp
)

set(
    MODEL_EXECUTABLES
    DecisionTreeClassifier
    LinearRegression
    LogisticRegression
    RandomForestClassifier
)

set(
    MODEL_SOURCES
    ${CMAKE_SOURCE_DIR}/tree/decision-tree-classifier/tree.cpp
    ${CMAKE_SOURCE_DIR}/linear-regression/source-codes/linear-regression.cpp
    ${CMAKE_SOURCE_DIR}/logistic-regression/source-codes/logistic-regression.cpp
    ${CMAKE_SOURCE_DIR}/ensemble/random-forest-classifier/random-forest-classifier.cpp
)

list(LENGTH MODEL_EXECUTABLES NUM_MODELS)
math(EXPR NUM_MODELS "${NUM_MODELS} - 1")
foreach(IDX RANGE ${NUM_MODELS})
    list(GET MODEL_EXECUTABLES ${IDX} EXE_NAME)
    list(GET MODEL_SOURCES ${IDX} EXE_SOURCE)
    add_executable(${EXE_NAME} ${EXE_SOURCE})
    target_link_libraries(${EXE_NAME}
        CPP_SCIKIT_LEARN_LIBRARIES
        ${ARMADILLO_LIBRARIES}
    )
endforeach()

target_include_directories(
    CPP_SCIKIT_LEARN_LIBRARIES
    PUBLIC
    ${CMAKE_SOURCE_DIR}/header-source/headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/base-headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/complex-datatypes
    ${CMAKE_SOURCE_DIR}/header-source/headers/cpp-utilities
    ${CMAKE_SOURCE_DIR}/header-source/headers/metrics
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers/decision-tree-header
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers/linear-regression-header
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers/logistic-regression-header
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers/random-forest-header
    ${ARMADILLO_INCLUDE_DIRS}
)
