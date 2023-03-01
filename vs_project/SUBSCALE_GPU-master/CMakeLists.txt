cmake_minimum_required(VERSION 3.18)

# toolchain file for vcpkg libraries (ATTENTION: install vcpkg and change this path to compile the code)
set(CMAKE_TOOLCHAIN_FILE /home/luser/vcpkg/scripts/buildsystems/vcpkg.cmake)

# additional compiler flags (needed for mlpack)
set(GCC_COVERAGE_COMPILE_FLAGS "-fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")


project("SubscaleGPU")
enable_language(CUDA)
add_subdirectory("SubscaleGPU")