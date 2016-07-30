/**
* Implementation of Itti's model
*
* - Image channels created from input (r_i, g_i, b_i):
* 		1. Intensity = (r_i + g_i + b_i)/3
* 		2. Blue      = b - (r+g)/2,              b = b_i / i , same for r and g
* 		3. Red       = r - (b+g)/2
* 		4. Green     = g - (r+b)/2
* 		5. Yellow    = (r+g)/2 - |r-g|/2 - b
*
* - Image scales needed (i = 2-8, where scale i connotates a ratio of 1:i)
* - Coarser scales are obtained through a low-pass filtering and subsampling
* 		= NTS: use a basic low-pass filter and subsample
*
*
*	** Points to note:
*	- focus now on producing a prototype with very little regard for efficiency
*	- try to modularize as much as possible to allow for simple enhancements
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>

using namespace std;
using namespace cv;

void split_input(Mat&, Mat*);
void split_input_to_visible(Mat&, Mat*);


int main( int argc, char* argv[])
{
    const char* filename = argv[1];

    Mat input = imread(filename, CV_LOAD_IMAGE_COLOR);

    double t = (double)getTickCount();

    Mat channels[5];
    split_input(input, channels);


    t = ((double)getTickCount() - t)/getTickFrequency();

    cout << "Filtering time passed in seconds: " << t << endl;

    namedWindow("R", WINDOW_AUTOSIZE); imshow("R", channels[0]);
    namedWindow("G", WINDOW_AUTOSIZE); imshow("G", channels[1]);
    namedWindow("B", WINDOW_AUTOSIZE); imshow("B", channels[2]);
    namedWindow("Y", WINDOW_AUTOSIZE); imshow("Y", channels[3]);
    namedWindow("I", WINDOW_AUTOSIZE); imshow("I", channels[4]);
    namedWindow("i", WINDOW_AUTOSIZE); imshow("i", input);


    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    waitKey(10000);
    return 0;
}


void split_input_to_visible(Mat& input, Mat* channels)
{
    int nRows = input.rows;
    int nCols = input.cols;

    Mat red         = Mat(nRows, nCols, CV_8U);
    Mat green       = Mat(nRows, nCols, CV_8U);
    Mat blue        = Mat(nRows, nCols, CV_8U);
    Mat yellow      = Mat(nRows, nCols, CV_8U);
    Mat intensity   = Mat(nRows, nCols, CV_8U);

    Vec3b *in_p;

    int x,y;
    int r, g, b, i, R, G, B, Y;
    uchar *r_p, *b_p, *y_p, *g_p, *i_p;

    for( x = 0; x < nRows; ++x)
    {
        in_p  = input.ptr<Vec3b>(x);

        r_p   = red.ptr<uchar>(x);
        g_p   = green.ptr<uchar>(x);
        b_p   = blue.ptr<uchar>(x);
        y_p   = yellow.ptr<uchar>(x);
        i_p   = intensity.ptr<uchar>(x);

        for ( y = 0; y < nCols; ++y)
        {
            i = ((in_p[y])[0] + (in_p[y])[1] + (in_p[y])[2])/(3);

            if (i > 0)
            {

                // For testing reasons. Output images directly
                b = (in_p[y])[0];
                g = (in_p[y])[1];
                r = (in_p[y])[2];

                R = 2*r - (b+g);
                G = 2*g - (b+r);
                B = 2*b - (r+g);
                Y = -B - std::abs(r-g);

                i_p[y] = i;
                b_p[y] = (B>0) ? B : 0;
                r_p[y] = (R>0) ? R : 0;
                g_p[y] = (G>0) ? G : 0;
                y_p[y] = (Y>0) ? Y : 0;
            } else {
                b_p[y] = 0;
                r_p[y] = 0;
                g_p[y] = 0;
                y_p[y] = 0;
            }
        }
    }
    channels[0] = red;
    channels[1] = green;
    channels[2] = blue;
    channels[3] = yellow;
    channels[4] = intensity;


    cout << "reaches output" << endl;

}

void split_input(Mat& input, Mat* channels)
{
    int nRows = input.rows;
    int nCols = input.cols;

    Mat red         = Mat(nRows, nCols, CV_32F);
    Mat green       = Mat(nRows, nCols, CV_32F);
    Mat blue        = Mat(nRows, nCols, CV_32F);
    Mat yellow      = Mat(nRows, nCols, CV_32F);
    Mat intensity   = Mat(nRows, nCols, CV_32F);

    Vec3b *in_p;

    int x,y;
    float r, g, b, i, R, G, B, Y;
    float *r_p, *b_p, *y_p, *g_p, *i_p;

    for( x = 0; x < nRows; ++x)
    {
        in_p  = input.ptr<Vec3b>(x);

        r_p   = red.ptr<float>(x);
        g_p   = green.ptr<float>(x);
        b_p   = blue.ptr<float>(x);
        y_p   = yellow.ptr<float>(x);
        i_p   = intensity.ptr<float>(x);

        for ( y = 0; y < nCols; ++y)
        {
            i = ((in_p[y])[0] + (in_p[y])[1] + (in_p[y])[2])/(3);

            if (i > 0)
            {
                b = (in_p[y])[0]/i;
                g = (in_p[y])[1]/i;
                r = (in_p[y])[2]/i;

                R = 2*r - (b+g);
                G = 2*g - (b+r);
                B = 2*b - (r+g);
                Y = -B - std::abs(r-g);

                b_p[y] = (B>0) ? B : 0;
                r_p[y] = (R>0) ? R : 0;
                g_p[y] = (G>0) ? G : 0;
                y_p[y] = (Y>0) ? Y : 0;
                i_p[y] = (b_p[y]+r_p[y]+g_p[y])/3;
            } else {
                b_p[y] = 0;
                r_p[y] = 0;
                g_p[y] = 0;
                y_p[y] = 0;
            }
        }
    }
    channels[0] = red;
    channels[1] = green;
    channels[2] = blue;
    channels[3] = yellow;
    channels[4] = intensity;
}
