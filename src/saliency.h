#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <cstdio>


void split_input(cv::Mat&, cv::Mat*);
void construct_pyramid(cv::Mat&, cv::Mat*, int);
void across_scale_diff(cv::Mat*, cv::Mat*);
void across_scale_opponency_diff(cv::Mat*, cv::Mat*, cv::Mat*);
void integrate_single_pyramid(cv::Mat*, cv::Mat, int);
void integrate_color_pyamids(cv::Mat*, cv::Mat*, cv::Mat, int);
void integrate_orient_pyamids(cv::Mat*, cv::Mat*, cv::Mat*, cv::Mat*, cv::Mat, int);
