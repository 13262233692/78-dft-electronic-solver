cmake_minimum_required(VERSION 3.18)
project(dft_electronic_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/O2 /EHsc /bigobj)
else()
    add_compile_options(-O3 -Wall -Wextra -fPIC)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(DFT_CORE_SOURCES
    src/plane_wave_basis.cpp
    src/atoms.cpp
    src/density.cpp
    src/hamiltonian.cpp
)

add_library(dft_core STATIC ${DFT_CORE_SOURCES})
target_link_libraries(dft_core PUBLIC Eigen3::Eigen)

pybind11_add_module(_dft_core python/bindings.cpp)
target_link_libraries(_dft_core PRIVATE dft_core Eigen3::Eigen)
target_compile_definitions(_dft_core PRIVATE VERSION_INFO="0.1.0")

install(TARGETS _dft_core
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/dft_solver
    RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/dft_solver
)

enable_testing()
add_subdirectory(tests EXCLUDE_FROM_ALL)
