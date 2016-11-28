/*
 *	Implementation of methods for integrating bounding boxes with saliency maps
 *
 * @author Mohamed El Banani
 * @date Nov 15, 2016
 */

#include "objectProposal.h"

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
 * @return             	int for saliency confidence score in box x 10000.
 */
int calculateSaliencyScore(Mat& saliencyMap, proposal prop)
{
	Rect bbox = prop.bbox;
	// cout << "Rect x " << bbox.x << " , y " << bbox.y << endl;
	// cout << "Rect w " << bbox.width << " , h " << bbox.height << endl;

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
	int score = prop.confScore * ((double) (sumVal[0] - (surrVal[0]/factor)))/((double) bbox.width * bbox.height);
	return score;

}

/**
 * Draws a bounding box with an accompanying description
 * @param  image a reference to the output image
 * @param  bb    a Rect outlining the bounding box
 * @param  text  Text to be displayed next to bounding box
 * @param  color Color of box
 */
void drawBB(Mat& image, proposal prop, Scalar color)
{
	cout << "drawBB Rect x " << prop.bbox.x << " , y " << prop.bbox.y << endl;
	cout << "drawBB Rect w " << prop.bbox.width << " , h " << prop.bbox.height << endl;

	rectangle(image, prop.bbox, color, 2);
	std::ostringstream strs;
    strs << prop.saliencyScore;
	putText(image, strs.str(), Point(prop.bbox.x, prop.bbox.y), FONT_HERSHEY_SIMPLEX, 1, color, 2);
}


proposal* readInProposals(int propList[][5], int numProposals, int label)
{
	proposal* objProposals = new proposal[numProposals];

	cout << "Goes into Loop" << endl;

	for(int i = 0; i < numProposals; i++)
	{

		objProposals[i].bbox = Rect(propList[i][0], propList[i][1], propList[i][2], propList[i][3]);
		objProposals[i].confScore = propList[i][4];
		objProposals[i].label = label;
		cout << "Ends one loop run" << endl;
	}

	cout << "Returns new address" << endl;
	return objProposals;
}
