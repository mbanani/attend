cmake_minimum_required(VERSION 2.8)
project( attend )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( attend attend.cpp  normalize.h normalize.cpp saliency.h saliency.cpp objectProposal.h objectProposal.cpp util.h )
target_link_libraries( attend ${OpenCV_LIBS} )
