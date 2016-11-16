/**
 * Header for interfacing with edgebox (or box proposals in general). Calculates
 * Saliency score for each bounding box.
 *
 * @author Mohamed El Banani
 * @date Nov 15, 2016
 */

 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
 #include <cmath>
 #include <typeinfo>
 #include <iostream>
 #include <cstdio>

double calculateSaliencyScore(cv::Mat&, cv::Rect);
void drawBB(cv::Mat&, cv::Rect, const std::string&, cv::Scalar);
