cmake_minimum_required(VERSION 2.8)
project( DisplayImage_exe )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( test test.cpp )
target_link_libraries( test ${OpenCV_LIBS} )
