cmake_minimum_required(VERSION 3.31)
project(DecisionTreeClassifier)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Armadillo REQUIRED)

add_executable(
	main
	"linear-regression/source-codes/linear-regression.cpp"
	"logistic-regression/source-codes/logistic-regression.cpp"
	"tree/decision-tree-classifier/tree.cpp"
	"header-source/header-source-files/base-source-files/base.cpp"
	"header-source/header-source-files/base-source-files/classifier_mixin.cpp"
	"header-source/header-source-files/base-source-files/regressor_mixin.cpp"
	"header-source/header-source-files/base-source-files/transformer_mixin.cpp"
	"header-source/header-source-files/cpp-utilities-source-files/loader.cpp"
	"header-source/header-source-files/cpp-utilities-source-files/linalg-operations.cpp"
)

target_include_directories(
	main
	PRIVATE
	"header-source/headers/"
	"header-source/headers/armadillo"
	"header-source/headers/base-headers"
	"header-source/headers/complex-datatypes"
	"header-source/headers/cpp-utilities"
	"header-source/headers/model-headers"
)

target_link_directories(
	main
	PRIVATE
	${ARMADILLO_LIBRARIES}
)
