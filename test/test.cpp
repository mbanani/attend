#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;


double get_average_local_maxima(Mat , float* , float*);


int main( int argc, char* argv[])
{
    Mat pic(30,30, CV_32F, Scalar::all(0));
    pic.at<float>(10,10) = 1.0;
    pic.at<float>(10,11) = 13.0;
    pic.at<float>(10,12) = 13.0;
    pic.at<float>(12,1) = 1.0;

    float globalMax, localMax;

    cout << pic << endl << endl;

    get_average_local_maxima(pic, &globalMax, &localMax);

    cout << "local max is  " << localMax << endl;
    cout << "global max is " << globalMax << endl;
}




double get_average_local_maxima(Mat I, float *globalMax, float *localMaxAvg)
{
    float globalM, sumLocalM;
    int numLocalMax;

    for (int i = 1; i < I.rows-1; i++)
    {
        for(int j = 1; j < I.cols-1; j++)
        {
            if(I.at<float>(i,j) >= I.at<float>(i-1, j-1) && I.at<float>(i,j) >= I.at<float>(i-1, j+1)
                && I.at<float>(i,j) >= I.at<float>(i, j-1) && I.at<float>(i,j) >= I.at<float>(i, j+1)
                && I.at<float>(i,j) >= I.at<float>(i+1, j-1) && I.at<float>(i,j) >= I.at<float>(i+1, j+1)
                && I.at<float>(i,j) >= I.at<float>(i-1, j) && I.at<float>(i,j) >= I.at<float>(i+1, j))
            {
                numLocalMax++;
                sumLocalM += I.at<float>(i,j);
                if (I.at<float>(i,j) > globalM) {
                    globalM = I.at<float>(i,j);
                }
            }
        }
    }

    // cout << "Number of local maxima is " << numLocalMax << endl;
    *localMaxAvg = sumLocalM / numLocalMax;
    *globalMax = globalM;
}
