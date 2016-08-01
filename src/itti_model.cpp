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
void construct_pyramid(Mat&, Mat*, int);
void across_scale_diff(Mat*, Mat*);
void across_scale_opponency_diff(Mat*, Mat*, Mat*);

int main( int argc, char* argv[])
{
    const char* filename = argv[1];
    Mat input = imread(filename, CV_LOAD_IMAGE_COLOR);

    double t = (double)getTickCount();

    Mat channels[5];

    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow, Intensity
    split_input(input, channels);

    // extract orientations


    //define pyramid variables
    Mat bluePyr[9];
    Mat greenPyr[9];
    Mat redPyr[9];
    Mat yellowPyr[9];
    Mat intensPyr[9];
    Mat or0Pyr[9];
    Mat or45Pyr[9];
    Mat or90Pyr[9];
    Mat or135Pyr[9];

    //Construct pyramids and (maybe release memory for channels)
    construct_pyramid(channels[0], redPyr, 9);
    construct_pyramid(channels[1], greenPyr, 9);
    construct_pyramid(channels[2], bluePyr, 9);
    construct_pyramid(channels[3], yellowPyr, 9);
    construct_pyramid(channels[4], intensPyr, 9);
    //Missing orientation pyramids


    // define feature maps
    Mat oppRG_fm[6];
    Mat oppBY_fm[6];
    Mat or0_fm[6];
    Mat or45_fm[6];
    Mat or90_fm[6];
    Mat or135_fm[6];
    Mat intens_fm[6];

    //calculate feature maps for within pyramid features
    across_scale_diff(intensPyr, intens_fm);
    // across_scale_diff(or0Pyr, or0_fm);
    // across_scale_diff(or45Pyr, or135_fm);
    // across_scale_diff(or90Pyr, or135_fm);
    // across_scale_diff(or135Pyr, or135_fm);
    across_scale_opponency_diff(redPyr, greenPyr, oppRG_fm);
    across_scale_opponency_diff(bluePyr, yellowPyr, oppBY_fm);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Total so far (without read and write) in seconds: " << t << endl;

    namedWindow("in-", WINDOW_AUTOSIZE); imshow("in-", input);
    namedWindow("red-", WINDOW_AUTOSIZE); imshow("red-", redPyr[0]);
    namedWindow("green-", WINDOW_AUTOSIZE); imshow("green-", greenPyr[0]);
    namedWindow("intens0", WINDOW_AUTOSIZE); imshow("intens0", oppRG_fm[0]);
    namedWindow("intens0", WINDOW_AUTOSIZE); imshow("intens1", oppRG_fm[1]);
    namedWindow("intens0", WINDOW_AUTOSIZE); imshow("intens2", oppRG_fm[2]);

    waitKey(100000);
    return 0;
}

/**
* Splits an input BGR image into 5 channels: red, green, blue, yellow
* and intenisty. Calculations were done as described in Itti et al (1998)
*
* @param input    A BGR image (CV_8U)
* @param channels An empty array of 3 Mat objects (CV_32F)
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

/**
* Calculates the across scale difference between multiple layers of a pyramid
* and outputs them into the output pyramid
*
* @param inPyr  Input pyramid
* @param outPyr Output pyramid
*/
void across_scale_diff(Mat* inPyr, Mat* outPyr)
{
    int cL = 2, cU= 4, sL=3, sU=4;
    int c, s, i = 0;
    Mat temp;

    for(c = cL; c <= cU; ++c)
    {
        for(s = sL; s <= sU; ++s)
        {
            temp = inPyr[c+s].clone();

            int j;
            for(j = s-1 ; j >= 0; --j){
                Mat tempX;
                // cout<<"inPyr[c+j]: rows: " <<  inPyr[c+j].rows << ". cols: " << inPyr[c+j].cols << endl;
                // cout<<"temp:       rows: " <<  temp.rows << ". cols: " << temp.cols << endl;
                pyrUp(temp, tempX, Size(inPyr[c+j].cols, inPyr[c+j].rows));
                temp = tempX.clone();
            }
            absdiff(inPyr[c], temp, outPyr[i]);
        }
        ++i;
    }
}

/**
* Calculates the across scale difference between multiple layers of a pyramid
* for an color opponency featyre and outputs them into the output pyramid
*
* @param inPyr1 Input pyramid for first color
* @param inPyr2 Input pyramid for second color
* @param outPyr Output pyramid
*/
void across_scale_opponency_diff(Mat* inPyr1, Mat* inPyr2, Mat* outPyr)
{
    int cL = 2, cU= 4, sL=3, sU=4;
    int c, s, i = 0;
    Mat temp1, temp2, tempC, tempS;

    for(c = cL; c <= cU; ++c)
    {
        for(s = sL; s <= sU; ++s)
        {
            // scale both to information in first pyramid
            // while both pyramids should be identical, this is a way to avoid
            // potential errors and to enforce same dimensions for both
            temp1 = inPyr1[c+s].clone();
            temp2 = inPyr2[c+s].clone();

            int j;
            for(j = s-1 ; j >= 0; --j){
                Mat tempX1, tempX2;
                pyrUp(temp1, tempX1, Size(inPyr1[c+j].cols, inPyr1[c+j].rows));
                pyrUp(temp2, tempX2, Size(inPyr1[c+j].cols, inPyr1[c+j].rows));

                temp1 = tempX1.clone();
                temp2 = tempX2.clone();
            }

            tempC = inPyr1[c] - inPyr2[c];
            tempS = temp2 - temp1;
            absdiff(tempC, tempS, outPyr[i]);
        }
        ++i;
    }
}
