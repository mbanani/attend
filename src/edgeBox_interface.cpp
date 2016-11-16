/*
 *	Implementation of methods for integrating bounding boxes with saliency maps
 *
 * @author Mohamed El Banani
 * @date Nov 15, 2016
 */

#include "edgeBox_interface.h"

using namespace std;
using namespace cv;

/**
 * Caclulates the saliency confidence score of every bounding box  and returns
 * a sorted list of bounding boxes in terms of saliency score. Score is calculated
 * via a harris-filter like, then normalized by size
 * 	+1 * pixels inside box - 1 * pixels around box (same area as box)
 *
 * @param  saliencyMap 	a reference to the global saliency map of the scene
 * @param  bboxe      	bounding boxes (top-left x, top-left y, width, height)
 * @param  method      	method of calculating score (m: mean, )
 * @return             	float for saliency confidence score in box
 */
double calculateSaliencyScore(Mat& saliencyMap, Rect bbox) {

	Scalar sumVal = sum(saliencyMap(bbox));

	int x1 = bbox.x - (0.21 * bbox.width);
	int x2 = bbox.x + (1.21 * bbox.width);
	int y1 = bbox.y - (0.21 * bbox.height);
	int y2 = bbox.y + (1.21 * bbox.height);

	x1 = x1 < 0 ? 0: x1;
	x2 = x2 > saliencyMap.cols ? saliencyMap.cols: x2;
	y1 = y1 < 0 ? 0: y1;
	y2 = y2 > saliencyMap.rows ? saliencyMap.rows: y2;


	Scalar surrVal = sum(saliencyMap(Rect(x1, y1, x2-x1, y2-y1))) - sumVal;

	double factor = ((double)((y2-y1) * (x2-x1)) / (double) (bbox.width * bbox.height)) - 1;

	return ((double) (sumVal[0] - (surrVal[0]/factor)))/((double) bbox.width * bbox.height);

}

/**
 * Draws a bounding box with an accompanying description
 * @param  image a reference to the output image
 * @param  bb    a Rect outlining the bounding box
 * @param  text  [description]
 * @param  color [description]
 * @return       [description]
 */
void drawBB(Mat& image, Rect bb, const string& text, Scalar color)
{
	rectangle(image, bb, color, 2);
	putText(image, text, Point(bb.x, bb.y), FONT_HERSHEY_SIMPLEX, 1, color, 2);
}
