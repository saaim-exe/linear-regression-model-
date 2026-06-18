# CMakeList.txt : CMake project for linear_regression, include source and define
# project specific logic here.
#

# Add source to this project's executable.


cmake_minimum_required(VERSION 3.15)

project(linear_regression CXX)

find_path(MATPLOTLIBCPP_INCLUDE_DIR matplotlibcpp.h REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


file(GLOB_RECURSE SOURCE_FILES 
    CONFIGURE_DEPENDS 
    "src/*.cpp"
    "src/*.h"
    "data/*.cpp"
    "data/*.h"
    "include/*.cpp"
    "include/*.h"
)

add_executable(linear_regression ${SOURCE_FILES}) 

target_include_directories(linear_regression
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include 
${CMAKE_CURRENT_SOURCE_DIR}/src 
${CMAKE_CURRENT_SOURCE_DIR}/data
${MATPLOTLIBCPP_INCLUDE_DIR}
)

target_link_libraries(linear_regression
PRIVATE
Python3::Python
Eigen3::Eigen
)

target_compile_definitions(linear_regression PRIVATE 
WITHOUT_NUMPY
DATASET_DIR="${CMAKE_SOURCE_DIR}/linear_regression/datasets"
)

target_compile_features(linear_regression PRIVATE cxx_std_20)

