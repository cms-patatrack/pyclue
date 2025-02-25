
include_directories(..)

# look for sycl compiler and libraries
SET(_sycl_search_dirs ${SYCL_ROOT_DIR} /usr/lib /usr/local/lib /opt/intel/oneapi/compiler/latest/linux)
find_program(SYCL_COMPILER
			NAMES icpx
			HINTS ${_sycl_search_dirs}
			PATH_SUFFIXES bin)
find_path(SYCL_INCLUDE_DIR
		 NAMES sycl/sycl.hpp
		 HINTS ${_sycl_search_dirs}
		 PATH_SUFFIXES include)
find_path(SYCL_LIB_DIR
		 NAMES libsycl.so
		 HINTS ${_sycl_search_dirs}
		 PATH_SUFFIXES lib)
message(STATUS "SYCL compiler: ${SYCL_COMPILER}")
message(STATUS "SYCL include dir: ${SYCL_INCLUDE_DIR}")
message(STATUS "SYCL lib dir: ${SYCL_LIB_DIR}")

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
set(oneDPL_DIR /opt/intel/oneapi/dpl/latest/include)
message(STATUS "oneDPL_DIR: ${oneDPL_DIR}")

pybind11_add_module(CLUE_CPU_SYCL SHARED
                    binding_cpu_sycl.cpp)
target_include_directories(CLUE_CPU_SYCL PRIVATE ${alpaka_SOURCE_DIR}/include
                                                 ${Boost_INCLUDE_DIR})
target_include_directories(CLUE_CPU_SYCL PRIVATE ${oneDPL_DIR})
target_compile_options(CLUE_CPU_SYCL PRIVATE -DALPAKA_ACC_SYCL_ENABLED
                                             -DALPAKA_SYCL_ONEAPI_CPU)
target_compile_options(CLUE_CPU_SYCL PRIVATE -fsycl)
target_link_options(CLUE_CPU_SYCL PRIVATE -fsycl)
set_target_properties(CLUE_CPU_SYCL PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                               ./lib/CLUEstering/lib/)
foreach(CONV_MODULE ${CONV_MODULE})
  execute_process(COMMAND ln -sf ${CONV_MODULE}
                          ./CLUEstering/lib/CLUE_SYCL_CPU.so)
endforeach()

# pybind11_add_module(CLUE_GPU_SYCL SHARED
#                     binding_gpu_sycl.cpp)
# target_include_directories(CLUE_GPU_SYCL PRIVATE ${alpaka_SOURCE_DIR}/include
#                                                  ${Boost_INCLUDE_DIR})
# target_include_directories(CLUE_GPU_SYCL PRIVATE ${oneDPL_DIR})
# target_compile_options(CLUE_GPU_SYCL PRIVATE -DALPAKA_ACC_SYCL_ENABLED
#                                              -DALPAKA_SYCL_ONEAPI_GPU)
# set_target_properties(CLUE_GPU_SYCL PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#                                                ./lib/CLUEstering/lib/)
# target_compile_options(CLUE_GPU_SYCL PRIVATE -fsycl)
# foreach(CONV_MODULE ${CONV_MODULE})
#   execute_process(COMMAND ln -sf ${CONV_MODULE}
#                           ./CLUEstering/lib/CLUE_SYCL_GPU.so)
# endforeach()

