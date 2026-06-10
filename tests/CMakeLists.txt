cmake_minimum_required(VERSION 3.18)
project(dft_tests LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(TEST_SOURCES
    test_plane_wave_basis.cpp
    test_atoms.cpp
    test_hamiltonian.cpp
)

foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SRC})
    target_link_libraries(${TEST_NAME} PRIVATE dft_core Eigen3::Eigen)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
