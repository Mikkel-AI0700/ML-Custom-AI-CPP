add_executable(DecisionTreeClassifierTests
    ${CMAKE_CURRENT_SOURCE_DIR}/decision_tree_classifier_test.cpp
)

target_link_libraries(DecisionTreeClassifierTests
    DECISION_TREE_CLASSIFIER_LIBRARY
    CPP_SCIKIT_LEARN_LIBRARIES
    GTest::gtest_main
    ${ARMADILLO_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(DecisionTreeClassifierTests)
