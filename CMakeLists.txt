cmake_minimum_required(VERSION 3.0)
project(deepcrunch)
set(CMAKE_CXX_STANDARD 14)
set(DEEPCRUNCH_VERSION 0.1.0)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_PYTHON "Link to Python when building" ON)

if(WITH_CUDA)
  enable_language(CUDA)
  add_definitions(-D__CUDA_NO_HALF_OPERATORS__)
  add_definitions(-DWITH_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (WITH_PYTHON)
  add_definitions(-DWITH_PYTHON)
  find_package(Python3 COMPONENTS Development)
endif()
find_package(Torch REQUIRED)

file(GLOB HEADERS csrc/*.h)
file(GLOB OPERATOR_SOURCES csrc/cpu/*.h csrc/cpu/*.cc csrc/*.cc)
if(WITH_CUDA)
  file(GLOB OPERATOR_SOURCES ${OPERATOR_SOURCES} csrc/cuda/*.cuh csrc/cuda/*.cu)
endif()

add_library(${PROJECT_NAME} SHARED ${OPERATOR_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
if (WITH_PYTHON)
  target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME DeepCrunch)

target_include_directories(${PROJECT_NAME} INTERFACE
  "$<BUILD_INTERFACE:${HEADERS}>"
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(DEEPCRUNCH_CMAKECONFIG_INSTALL_DIR "share/cmake/DeepCrunch" CACHE STRING "install path for DeepCrunchConfig.cmake")

configure_package_config_file(cmake/DeepCrunchConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DeepCrunchConfig.cmake"
  INSTALL_DESTINATION ${DEEPCRUNCH_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/DeepCrunchConfigVersion.cmake
  VERSION ${DEEPCRUNCH_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/DeepCrunchConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/DeepCrunchConfigVersion.cmake
  DESTINATION ${DEEPCRUNCH_CMAKECONFIG_INSTALL_DIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT DeepCrunchTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

install(EXPORT DeepCrunchTargets
  NAMESPACE DeepCrunch::
  DESTINATION ${DEEPCRUNCH_CMAKECONFIG_INSTALL_DIR})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
install(FILES
  csrc/cpu/deepcrunch_cpu.h
  csrc/cpu/segment_coo_cpu.h
  csrc/cpu/segment_csr_cpu.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cpu)
if(WITH_CUDA)
  install(FILES
    csrc/cuda/deepcrunch_cuda.h
    csrc/cuda/segment_coo_cuda.h
    csrc/cuda/segment_csr_cuda.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/cuda)
endif()

if(WITH_CUDA)
  set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
  set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
endif()