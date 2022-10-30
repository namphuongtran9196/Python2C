cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(TARGET_NAME yolov5)

project(${TARGET_NAME} C CXX)

# file(GLOB srcs ${CMAKE_SOURCE_DIR}/include/*.c ${CMAKE_SOURCE_DIR}/src/*.cpp ${CMAKE_SOURCE_DIR}/*.c ${CMAKE_SOURCE_DIR}/*.cpp)
# file(GLOB hdrs ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/src/*.hpp ${CMAKE_SOURCE_DIR}/*.h ${CMAKE_SOURCE_DIR}/*.hpp)

# # add library
# add_library(
#         ${TARGET_NAME} SHARED
#         ${hdrs}
#         ${srcs}
#         )

# Set include and source files.
include_directories(include)
file(GLOB SOURCES "src/*.cpp")

# # Build as a shared library.
add_library(${TARGET_NAME} SHARED ${SOURCES})

# Link opencv to library
set(OpenCV_STATIC on)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBS})

# Link tflite to libray
set(TENSORFLOW_SOURCE_DIR "tensorflow_src")
if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../" ABSOLUTE)
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
target_link_libraries(${TARGET_NAME} tensorflow-lite)

add_executable(test test.cpp)
target_link_libraries(test ${TARGET_NAME})