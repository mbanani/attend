#include "normalize.h"

using namespace cv;
using namespace std;


/**
* Normalizes image to [0,1] range then multiplies by standard deviation
*
* @param input the input image
*/
void normalize_by_stdev(Mat input)
{
    Scalar mean, stdev;
    normalize(input, input, 0.0, 1.0, NORM_MINMAX, CV_32F);
    meanStdDev(input, mean, stdev);
    multiply(stdev.val[0], input, input);
}

void normalize_by_maxMeanDiff(Mat input)
{
    Scalar mean, stdev;
    double max;

    normalize(input, input, 0.0, 1.0, NORM_MINMAX, CV_32F);
    minMaxLoc(input, NULL, &max, NULL, NULL);
    meanStdDev(input, mean, stdev);
    double nVal = max - mean.val[0];
    multiply(max - mean.val[0], input, input);

}

void normalize_by_maxima_diff(Mat input)
{
    float globalMax, localMaxAvg;
    get_average_local_maxima(input, &globalMax, &localMaxAvg);
    // cout << "New Global Max : " << globalMax << endl;
    // cout << "New Average Max: " << localMaxAvg << endl << endl;
    float multipier = pow(globalMax - localMaxAvg, 2.0);
    multiply(multipier, input, input);

}

/**
* Wrapper class for noramlizer
*
* @param input the input image
*/
void normalize(Mat input)
{
    normalize(input, input, 0.0, 1.0, NORM_MINMAX, CV_32F);
    normalize_by_maxima_diff(input);
}

/**
* Itterates through a pyramid applying the normalize stdev function
* @param pyr       visual pyramid
* @param numLayers number of layers
*/
void normalize_pyramid(Mat* pyramid, int numLayers)
{
    int i;
    for (i = 0; i < numLayers; ++i)
    {
        normalize(pyramid[i]);
    }
}


double get_average_local_maxima(Mat I, float *globalMax, float *localMaxAvg)
{
    float globalM = 0.0, sumLocalM = 0.0;
    int numLocalMax = 0;

    for (int i = 1; i < I.rows-1; i++)
    {
        for(int j = 1; j < I.cols-1; j++)
        {
            if(I.at<float>(i,j) >= I.at<float>(i-1, j-1) && I.at<float>(i,j) >= I.at<float>(i-1, j+1)
                && I.at<float>(i,j) >= I.at<float>(i, j-1) && I.at<float>(i,j) >= I.at<float>(i, j+1)
                && I.at<float>(i,j) >= I.at<float>(i+1, j-1) && I.at<float>(i,j) >= I.at<float>(i+1, j+1)
                && I.at<float>(i,j) >= I.at<float>(i-1, j) && I.at<float>(i,j) >= I.at<float>(i+1, j))
            {
                // cout << sumLocalM << "-" << numLocalMax << " || ";
                numLocalMax++;
                sumLocalM += I.at<float>(i,j);
                if (I.at<float>(i,j) > globalM) {
                    globalM = I.at<float>(i,j);
                }
            }
        }
    }

    // cout << "Number of local maxima is " << numLocalMax << endl;
    if (numLocalMax == 0) {
        *localMaxAvg = 0.0;
        *globalMax = 0.0;
    } else {
        *localMaxAvg = sumLocalM / numLocalMax;
        *globalMax = globalM;
    }
}
