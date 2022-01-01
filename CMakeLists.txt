cmake_minimum_required(VERSION 3.5)

project(g2o_practice)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=native")

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(./src)
include_directories(../../libs/g2o)

add_executable(grid_refine ./src/refine.cpp)

target_link_libraries(grid_refine 
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_types_slam3d.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_solver_eigen.so
)
