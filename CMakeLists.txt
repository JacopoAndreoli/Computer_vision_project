cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(CV_final_project)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} source/CV_final_project.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

