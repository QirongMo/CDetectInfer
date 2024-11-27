cmake_minimum_required(VERSION 3.10)  
  
# 设置项目名称和版本  
project(vdetect VERSION 1.0)  
  
# 设置C++标准  
set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED True)  
  
# 查找OpenCV库  
find_package(OpenCV REQUIRED) 

# 添加头文件目录  
include_directories(${OpenCV_INCLUDE_DIRS})  

# yaml-cpp
set(Yaml-cpp_DIR /home/mqr/opt/yaml-cpp/yaml-cpp)
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# onnxruntime
# set(ONNXRUNTIME_ROOT_PATH /home/mqr/opt/onnxruntime/onnxruntime-1.15.1)
# set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
#                              ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
#                              ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
# set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
# include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

set(ONNXRUNTIME_ROOT_PATH /home/mqr/opt/onnxruntime/onnxruntime-linux-x64-gpu-1.12.1)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})

# cuda
set(cuda_home /home/mqr/opt/cuda/cuda-11.8)
include_directories(${cuda_home}/include)
link_directories(${cuda_home}/lib64)

# tensorrt
set(tensorrt_home /home/mqr/opt/tensorrt/TensorRT-8.6.1.6)
include_directories(${tensorrt_home}/include)
link_directories(${tensorrt_home}/lib)

# 如果你的项目还包含其他源文件或库，可以在这里继续添加  
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADS ${PROJECT_SOURCE_DIR}/include/*.h)

# 添加头文件目录  
include_directories(${PROJECT_SOURCE_DIR}/include) 


# 添加可执行文件，并链接所有源文件  
add_executable(main ${SOURCES}) 

# # 添加可执行文件  
# add_executable(vdetect vdetect.cpp) # 假设你的源文件是main.cpp  

# 链接OpenCV库  
target_link_libraries(main ${OpenCV_LIBS})  
# 链接yaml-cpp
# target_link_libraries(main yaml-cpp)
target_link_libraries(main ${YAML_CPP_LIBRARIES})
# 链接onnxruntime
target_link_libraries(main ${ONNXRUNTIME_LIB})
# 链接cuda和tensorrt
target_link_libraries(main nvinfer nvinfer_plugin cudart)

  
# 如果需要额外的编译器或链接器选项，可以在这里添加  
# 例如，对于多线程，你可能需要链接到pthread库（在Unix系统上）  
if(UNIX)  
    target_link_libraries(main pthread)  
endif()  

add_library(detectinfer SHARED ${SOURCES}) 
# 链接OpenCV库  
target_link_libraries(detectinfer ${OpenCV_LIBS})  
# 链接yaml-cpp
# target_link_libraries(main yaml-cpp)
target_link_libraries(detectinfer ${YAML_CPP_LIBRARIES})
# 链接onnxruntime
target_link_libraries(detectinfer ${ONNXRUNTIME_LIB})
# 链接cuda和tensorrt
target_link_libraries(detectinfer nvinfer nvinfer_plugin cudart)