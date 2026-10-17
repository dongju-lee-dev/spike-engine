cmake_minimum_required(VERSION 3.20)
project(study-snn LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "This project is dedicated to the LLVM Clang++ compiler. Currently. now compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(study-snn ${SOURCES})

target_include_directories(study-snn PRIVATE include)

target_compile_options(study-snn PRIVATE
    -stdlib=libc++
    -O3
    -fopenmp-simd
    -fvectorize
    -fslp-vectorize
    -Wall -Wextra -Wpedantic
)

target_link_options(study-snn PRIVATE -stdlib=libc++)