
cmake_minimum_required(VERSION 3.16)
project(pto_test_cpu_sim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})

add_definitions(-D__CPU_SIM)
find_package(GTest REQUIRED)

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -std=c++23
    -Wno-macro-redefined -Wno-ignored-attributes
    -fstack-protector-strong
)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O2)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_options(-O2)
endif()

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
add_link_options(
    -Wl,-z,relro
    -Wl,-z,now
    )


set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

add_subdirectory(testcase)