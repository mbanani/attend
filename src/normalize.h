#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <cstdio>


void normalize(cv::Mat);
void normalize_by_maxMeanDiff(cv::Mat);
void normalize_by_maxima_diff(cv::Mat);
void normalize_by_stdev(cv::Mat);
void normalize_pyramid(cv::Mat*, int);
double get_average_local_maxima(cv::Mat, float*, float*);
