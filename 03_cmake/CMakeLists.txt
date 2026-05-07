cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(online_softmax_cmake LANGUAGES CXX CUDA)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)

if(NOT DEFINED CMAKE_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

link_directories("${TORCH_INSTALL_PREFIX}/lib")

add_library(online_softmax_cuda SHARED 
    src/binding.cpp 
    src/softmax_kernel.cu
)

target_compile_definitions(online_softmax_cuda PRIVATE 
    TORCH_EXTENSION_NAME=online_softmax_cuda
)

target_include_directories(online_softmax_cuda PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

set_target_properties(online_softmax_cuda PROPERTIES 
    CXX_STANDARD 17
    CUDA_STANDARD 17
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(online_softmax_cuda PRIVATE 
    ${TORCH_LIBRARIES} 
    ${Python3_LIBRARIES}
    torch_python
)

set_target_properties(online_softmax_cuda PROPERTIES
    CXX_VISIBILITY_PRESET default
)