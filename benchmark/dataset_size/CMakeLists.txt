cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  " -DCLUE_DEBUG -D_GLIBCXX_ASSERTIONS -Wall -Wextra -Wpedantic -Wpedantic -Werror"
)
string(APPEND CMAKE_CXX_FLAGS "-O2")

add_subdirectory(pybind11)

find_package(Boost 1.75.0 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  alpaka
  URL https://github.com/alpaka-group/alpaka/archive/refs/tags/1.2.0.tar.gz)

FetchContent_MakeAvailable(alpaka)

# CPU Serial
add_executable(serial.out main.cpp)
target_include_directories(
  serial.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                     ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
target_link_libraries(serial.out PRIVATE pybind11::embed)
target_compile_definitions(
  serial.out PRIVATE ALPAKA_HOST_ONLY ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)

find_package(TBB)
# CPU TBB
if(TBB_FOUND)
  add_executable(tbb.out main.cpp)
  target_include_directories(
    tbb.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                    ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_link_libraries(tbb.out PRIVATE pybind11::embed TBB::tbb)
  target_compile_definitions(tbb.out PRIVATE ALPAKA_HOST_ONLY
                                             ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
endif()

# check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# GPU CUDA
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(cuda.out main.cpp)
  target_include_directories(
    cuda.out PRIVATE ${CMAKE_SOURCE_DIR}/../../include
                     ${alpaka_SOURCE_DIR}/include ${Boost_INCLUDE_DIR})
  target_compile_definitions(cuda.out PRIVATE ALPAKA_ACC_GPU_CUDA_ENABLED)
  target_compile_options(cuda.out PRIVATE --expt-relaxed-constexpr)
  set_target_properties(cuda.out PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                            CUDA_ARCHITECTURES "50;60;61;62;70")
endif()

add_subdirectory(sycl)
