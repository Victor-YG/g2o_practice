cmake_minimum_required(VERSION 3.5)

project(g2o_practice)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# find gflags
find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

# find g2o
find_package(g2o REQUIRED)
include_directories(../../libs/g2o)

include_directories(./src)

add_executable(grid_refine ./src/refine.cpp)
add_executable(refine_3d ./src/refine_3d.cpp)

target_link_libraries(grid_refine 
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_types_slam3d.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_solver_eigen.so
)

target_link_libraries(refine_3d 
    gflags
    Eigen3::Eigen
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_core.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_stuff.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_types_slam3d.so
    ${PROJECT_SOURCE_DIR}/../../libs/g2o/lib/libg2o_solver_eigen.so
)
