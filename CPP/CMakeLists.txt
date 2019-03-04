cmake_minimum_required (VERSION 2.8.8)
project (tf_example)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")

set(CMAKE_BUILD_TYPE Release)

add_library(tensorflow INTERFACE)
set(Tensorflow_INCLUDE_DIR 
	/usr/local/tensorflow/include/
)
set(Tensorflow_LIBRARIES 
	/usr/local/tensorflow/lib/libtensorflow_cc.so
	#/usr/local/tensorflow/lib/libtensorflow_framework.so
)
target_include_directories(tensorflow INTERFACE ${Tensorflow_INCLUDE_DIR})
target_link_libraries(tensorflow INTERFACE ${Tensorflow_LIBRARIES})
message(STATUS "Tensorflow library status:")
message(STATUS "    Tensorflow_INCLUDE_DIR: ${Tensorflow_INCLUDE_DIR}")
message(STATUS "    Tensorflow_LIBRARIES: ${Tensorflow_LIBRARIES}")


set(CMAKE_PREFIX_PATH "/usr/local/opencv/share/OpenCV")
#find_package(OpenCV REQUIRED)
find_package(OpenCV 3.4.2 REQUIRED)
add_library(opencv INTERFACE)
target_include_directories(opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
list(REMOVE_ITEM OpenCV_LIBS opencv_dnn) # remove caffe libs
list(REMOVE_ITEM OpenCV_LIBS opencv_hdf) # remove hdf5 libs
target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})
message(STATUS "OpenCV library status:")
message(STATUS "    OPENCV_ROOT_DIR: ${OPENCV_ROOT_DIR}")
message(STATUS "    OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "    OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

add_executable(tf_example
    src/detector.cpp
	src/main.cc
)
target_include_directories(tf_example PUBLIC
	include
)
target_link_libraries(tf_example PUBLIC
	tensorflow
    opencv
)

