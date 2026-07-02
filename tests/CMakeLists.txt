add_executable(DecisionTreeClassifierTests
    ${CMAKE_SOURCE_DIR}/tree/decision-tree-classifier/tree.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/decision_tree_classifier_test.cpp
)

target_compile_definitions(DecisionTreeClassifierTests PRIVATE SKIP_MAIN)

target_link_libraries(DecisionTreeClassifierTests
    CPP_SCIKIT_LEARN_LIBRARIES
    GTest::gtest_main
    ${ARMADILLO_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(DecisionTreeClassifierTests)
