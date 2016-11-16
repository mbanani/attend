cmake_minimum_required(VERSION 2.8)
project( attend )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( itti itti_model.cpp  normalize.h normalize.cpp saliency.h saliency.cpp edgeBox_interface.h edgeBox_interface.cpp util.h )
target_link_libraries( itti ${OpenCV_LIBS} )
