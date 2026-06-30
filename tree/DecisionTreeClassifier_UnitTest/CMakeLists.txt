set(SOURCES
    ${CMAKE_SOURCE_DIR}/tree/decision-tree-classifier/tree.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/cpp-utilities-source-files/linalg-operations.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/base.cpp
    ${CMAKE_SOURCE_DIR}/header-source/header-source-files/base-source-files/classifier_mixin.cpp
    src_tests/decision_tree_classifier_test.cpp
)

add_executable(DecisionTreeClassifierTests ${SOURCES})

target_compile_definitions(DecisionTreeClassifierTests PRIVATE SKIP_MAIN)

target_include_directories(DecisionTreeClassifierTests PRIVATE
    ${CMAKE_SOURCE_DIR}/header-source/headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/base-headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/complex-datatypes
    ${CMAKE_SOURCE_DIR}/header-source/headers/cpp-utilities
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers
    ${CMAKE_SOURCE_DIR}/header-source/headers/model-headers/decision-tree-header
    ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(DecisionTreeClassifierTests
    GTest::gtest_main
    ${ARMADILLO_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(DecisionTreeClassifierTests)
