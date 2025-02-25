
include_directories(..)

# CPU Serial compile cpu serial module
pybind11_add_module(CLUE_CPU_Serial SHARED
                    binding_cpu.cpp)
target_include_directories(CLUE_CPU_Serial PRIVATE ${alpaka_SOURCE_DIR}/include
                                                   ${Boost_INCLUDE_DIR})
# alpaka build flags
target_compile_options(
  CLUE_CPU_Serial PRIVATE -DALPAKA_HOST_ONLY
                          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
# set output directory
set_target_properties(CLUE_CPU_Serial PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 ./lib/CLUEstering/lib/)
# create link of shared object to lib folder inside CLUEstering directory
file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_CPU_Serial*.so)
foreach(CONV_MODULE ${CONV_MODULE})
  execute_process(COMMAND ln -sf ${CONV_MODULE}
                          ./CLUEstering/lib/CLUE_CPU_Serial.so)
endforeach()

