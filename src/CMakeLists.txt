cmake_minimum_required(VERSION 2.8)
project( DisplayImage_exe )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( itti itti_model.cpp )
target_link_libraries( itti ${OpenCV_LIBS} )
