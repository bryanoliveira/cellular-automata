cmake_minimum_required(VERSION 3.8)

# set the project name and version
project(automata LANGUAGES CXX CUDA)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY shared)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_52,code=sm_52 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_86,code=compute_86"
)
# find dependencies
find_package(GLEW REQUIRED)
# include_directories(${GLEW_INCLUDE_DIRS})
# link_libraries(${GLEW_LIBRARIES})

# pkg_check_modules(GLEW libglew2.1)
# find_package(GLEW REQUIRED)
# link_directories(${GLEW_LIBRARY_DIRS})

# pkg_check_modules(GLUT freeglut3-dev)
find_package(Boost REQUIRED COMPONENTS program_options)
# include_directories(${BOOST_INCLUDE_DIRS})

# source & headers
include_directories(include/)

file(GLOB SOURCES src/*.cpp src/*.cu)

# flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -I/usr/local/cuda/include")
# -lGL -lGLU -lglut -lGLEW  -I include")

# make exec
add_executable(automata ${SOURCES})
set_target_properties(automata PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# link libs
target_link_libraries(automata ${Boost_LIBRARIES})
target_link_libraries(automata -lGL -lGLU -lglut -lGLEW)
# target_link_libraries(automata ${CUDA_LIBRARIES})
# if (CUDA_FOUND)
#     target_link_libraries(automata -lcudart)
#     target_link_libraries(automata ${CUDA_LIBRARIES})
# endif ()