cmake_minimum_required(VERSION 3.31.6)

project(CPP_SCIKIT_LEARN VERSION 1.0)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_TESTS "Build unit tests (requires Google Test)" OFF)

find_package(Armadillo REQUIRED)

add_subdirectory(src)

if(BUILD_TESTS)
    enable_testing()
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.0
    )

    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest googlebenchmark)

    add_subdirectory(tests)
endif()
