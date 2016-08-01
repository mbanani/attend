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
void construct_pyramid(Mat&, Mat*, int);


int main( int argc, char* argv[])
{
    const char* filename = argv[1];

    Mat input = imread(filename, CV_LOAD_IMAGE_COLOR);

    double t = (double)getTickCount();

    Mat channels[5];
    split_input(input, channels);

    Mat& red       = channels[0];
    Mat& green     = channels[1];
    Mat& blue      = channels[2];
    Mat& yellow    = channels[3];
    Mat& intensity = channels[4];

    t = (double)getTickCount();

    Mat BluePyr[9];
    construct_pyramid(channels[2], BluePyr, 7);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Pyramid takes " << t << " seconds" << endl;
    cout << "print a pyramid" << endl;
    waitKey(100000);
    namedWindow("0", WINDOW_AUTOSIZE); imshow("0", BluePyr[0]);
    cout << "print a pyramid 1" << endl;
    waitKey(100000);
    namedWindow("1", WINDOW_AUTOSIZE); imshow("1", BluePyr[1]);
    cout << "print a pyramid 3" << endl;
    waitKey(100000);
    namedWindow("3", WINDOW_AUTOSIZE); imshow("3", BluePyr[3]);
    cout << "print a pyramid 5" << endl;
    waitKey(100000);
    namedWindow("5", WINDOW_AUTOSIZE); imshow("5", BluePyr[5]);
    cout << "print a pyramid 7" << endl;
    waitKey(100000);
    namedWindow("6", WINDOW_AUTOSIZE); imshow("7", BluePyr[6]);
    // namedWindow("i", WINDOW_AUTOSIZE); imshow("i", input);
    cout << "DONE!" << endl;
    waitKey(100000);
    return 0;
}

/**
 * Splits an input BGR image into 5 channels: red, green, blue, yellow
 * and intenisty. Calculations were done as described in Itti et al (1998)
 *
 * @param input    A BGR image (CV_8U)
 * @param channels An empty array of 5 Mat objects (CV_32F)
 */
void split_input(Mat& input, Mat* channels)
{
    // Initialize 5 Matrix objects as floats
    int nRows = input.rows;
    int nCols = input.cols;

    Mat red         = Mat(nRows, nCols, CV_32F);
    Mat green       = Mat(nRows, nCols, CV_32F);
    Mat blue        = Mat(nRows, nCols, CV_32F);
    Mat yellow      = Mat(nRows, nCols, CV_32F);
    Mat intensity   = Mat(nRows, nCols, CV_32F);

    //define vctors for itterating through the matrices
    Vec3b *in_p;
    int x,y;
    float r, g, b, i, R, G, B, Y;
    float *r_p, *b_p, *y_p, *g_p, *i_p;

    //Itterate through rows and columns of all the matrices to fill out
    //the channels according to the following:
    //  1. Decouple the input colors from their hue through diving by intensity
    //  2. Color values are shown below
    //  3. If the color is less than a threshold value (set to 0), make it 0.
    for( x = 0; x < nRows; ++x)
    {
        // set pointer values to current row
        in_p  = input.ptr<Vec3b>(x);
        r_p   = red.ptr<float>(x);
        g_p   = green.ptr<float>(x);
        b_p   = blue.ptr<float>(x);
        y_p   = yellow.ptr<float>(x);
        i_p   = intensity.ptr<float>(x);

        //iterate through column using previously defined pointers
        for ( y = 0; y < nCols; ++y)
        {
            //calculate initial intensity to make sure it's non-zero
            i = ((in_p[y])[0] + (in_p[y])[1] + (in_p[y])[2])/(3);

            if (i > 0)
            {
                //calculate decoupled color input values from hue
                b = (in_p[y])[0]/i;
                g = (in_p[y])[1]/i;
                r = (in_p[y])[2]/i;

                //seperate colors to 4 colors (RGBY)
                R = r - (b+g)/2;
                G = g - (b+r)/2;
                B = b - (r+g)/2;
                Y = -B - std::abs(r-g)/2;

                //if extracted color value is negative, set to 0
                b_p[y] = (B>0) ? B : 0;
                r_p[y] = (R>0) ? R : 0;
                g_p[y] = (G>0) ? G : 0;
                y_p[y] = (Y>0) ? Y : 0;
                i_p[y] = (b_p[y]+r_p[y]+g_p[y])/3
                ;
            } else {
                b_p[y] = 0;
                r_p[y] = 0;
                g_p[y] = 0;
                y_p[y] = 0;
            }
        }
    }

    // assign the Mat objects to their corresponding channel array index
    channels[0] = red;
    channels[1] = green;
    channels[2] = blue;
    channels[3] = yellow;
    channels[4] = intensity;
}

/**
 * Constructs a gaussing pyramid with numLayers layers from an input image.
 *
 * @param input     An Mat of an image of dimensions larger than 2^numlayers
 * @param pyramid   A Mat array (index corresponds to the reduction factor)
 * @param numLayers Number of layers of the pyramid
 */
void construct_pyramid(Mat& input, Mat* pyramid, int numLayers)
{
    //0th layers in the original image
    pyramid[0] = input.clone();
    int i;

    //Itterate through the pyramid layers with each layer up the pyramid having
    //half the dimensions
    for(i = 1; i < numLayers; ++i)
    {
        pyrDown(pyramid[i-1], pyramid[i], Size(pyramid[i-1].cols/2, pyramid[i-1].rows/2));
    }
}
