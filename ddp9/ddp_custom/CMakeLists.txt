cmake_minimum_required(VERSION 3.20)
project(ddp_backend LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

# ---- NCCL discovery ----
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS /usr/include /usr/local/include /usr/include/nccl ${CUDAToolkit_INCLUDE_DIRS})
find_library(NCCL_LIBRARY NAMES nccl libnccl
  HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/lib64)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Provide -DNCCL_INCLUDE_DIR & -DNCCL_LIBRARY")
endif()

pybind11_add_module(ddp_backend
  src/ddp_backend/bindings.cpp
  src/ddp_backend/process_group_nccl.cpp
  src/ddp_backend/kernels.cu)

target_include_directories(ddp_backend PRIVATE
  ${NCCL_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
  src)

target_link_libraries(ddp_backend PRIVATE
  CUDA::cudart
  ${NCCL_LIBRARY})

set_target_properties(ddp_backend PROPERTIES
  PREFIX ""
  OUTPUT_NAME "ddp_backend")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ddp_backend PRIVATE -fvisibility=hidden -Wno-deprecated-declarations)
endif()

if(UNIX AND NOT APPLE)
  set_target_properties(ddp_backend PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS ddp_backend
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .)
