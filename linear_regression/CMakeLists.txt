# CMakeList.txt : CMake project for linear_regression, include source and define
# project specific logic here.
#

# Add source to this project's executable.


cmake_minimum_required(VERSION 3.15)

project(linear_regression CXX)

find_path(MATPLOTLIBCPP_INCLUDE_DIR matplotlibcpp.h REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)


add_executable (linear_regression 
 "main.cpp" 
 "plot.cpp" 
 "plot.h")

 
target_include_directories(linear_regression
PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/include 
${MATPLOTLIBCPP_INCLUDE_DIR}
)

target_link_libraries(linear_regression
PRIVATE
Python3::Python
)

target_compile_definitions(linear_regression PRIVATE WITHOUT_NUMPY)

target_compile_features(linear_regression PRIVATE cxx_std_20)

