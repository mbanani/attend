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


/**
 * A structure for object proposals.
 * 	bbox            rectangle object that covers the object proposal.
 * 	confScore       The confidence score output by the proposal Generator
 * 	saliencyScore   The saliency score associated with the object
 * 	label           the object label associated with the proposal
 */
 struct proposal
 {
 	cv::Rect bbox;
 	int confScore;
    int saliencyScore;
    int label;
};


int calculateSaliencyScore(cv::Mat&, proposal);
void drawBB(cv::Mat&, proposal, cv::Scalar);
// void drawBB(cv::Mat&, cv::Rect, const std::string&, cv::Scalar);
proposal* readInProposals(int[][5], int, int);
double  calculateIOU(cv::Rect, cv::Rect);
