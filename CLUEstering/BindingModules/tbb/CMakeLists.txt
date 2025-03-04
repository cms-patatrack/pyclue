
include_directories(..)

# CPU TBB
if(TBB_FOUND)
  # compile cpu tbb module
  pybind11_add_module(CLUE_CPU_TBB SHARED
					  binding_cpu_tbb.cpp)
  target_include_directories(CLUE_CPU_TBB PRIVATE ${alpaka_SOURCE_DIR}/include
												  ${Boost_SOURCE_DIR})
  target_compile_options(CLUE_CPU_TBB
						 PRIVATE -ltbb -DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  target_link_libraries(CLUE_CPU_TBB PRIVATE TBB::tbb)
  set_target_properties(CLUE_CPU_TBB PROPERTIES LIBRARY_OUTPUT_DIRECTORY
												./lib/CLUEstering/lib/)
  # create link of shared object to lib folder inside CLUEstering directory
  file(GLOB CONV_MODULE ./build/lib/CLUEstering/lib/CLUE_CPU_TBB*.so)
  foreach(CONV_MODULE ${CONV_MODULE})
	execute_process(COMMAND ln -sf ${CONV_MODULE}
							./CLUEstering/lib/CLUE_CPU_TBB.so)
  endforeach()
endif()
