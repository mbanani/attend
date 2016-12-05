#include "saliency.h"
#include "normalize.h"

using namespace std;
using namespace cv;

/**
* Splits an input BGR image into 5 channels: red, green, blue, yellow
* and intenisty. Calculations were done as described in Itti et al (1998)
*
* @param input    A BGR image (CV_8U)
* @param channels An empty array of 3 Mat objects (CV_32F)
*/
void split_rgbyi(Mat& input, Mat* channels)
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
            i = ((in_p[y])[0] + (in_p[y])[1] + (in_p[y])[2]);

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
                // i_p[y] = (b_p[y]+r_p[y]+g_p[y])/3;
                // i_p[y] = (0.7*b_p[y]+ 0.2*r_p[y]+ 0.8*g_p[y])/3;
                // i_p[y] = (b_p[y]+r_p[y]+g_p[y] + y_p[y])/4;
                i_p[y] = i/3;

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
    // as defined in itti's paper
    int cL = 2, cU= 4, sL=3, sU=4;

    //initialize variables
    int c, s, i = 0;
    Mat temp;

    for(c = cL; c <= cU; ++c)
    {
        for(s = sL; s <= sU; ++s)
        {
            temp = inPyr[c+s].clone();

            int j;
            for(j = s-1 ; j >= 0; --j){

                pyrUp(temp, temp, Size(inPyr[c+j].cols, inPyr[c+j].rows));
            }

            if (c == cU - 2)
            {
                Mat tempX1, tempX2;
                absdiff(inPyr[c], temp, tempX1);
                pyrDown(tempX1, tempX2, Size(inPyr[cU-1].cols, inPyr[cU-1].rows));
                pyrDown(tempX2, outPyr[i], Size(inPyr[cU].cols, inPyr[cU].rows));
            } else if (c == cU - 1)
            {
                Mat tempX;
                absdiff(inPyr[c], temp, tempX);
                pyrDown(tempX, outPyr[i], Size(inPyr[cU].cols, inPyr[cU].rows));
            } else {
                absdiff(inPyr[c], temp, outPyr[i]);
            }
            ++i;
        }
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


            if (c == cU - 2)
            {
                Mat tempX1, tempX2;
                absdiff(tempC, tempS, tempX1);
                pyrDown(tempX1, tempX2, Size(inPyr1[cU-1].cols, inPyr1[cU-1].rows));
                pyrDown(tempX2, outPyr[i], Size(inPyr1[cU].cols, inPyr1[cU].rows));
            } else if (c == cU - 1)
            {
                Mat tempX;
                absdiff(tempC, tempS, tempX);
                pyrDown(tempX, outPyr[i], Size(inPyr1[cU].cols, inPyr1[cU].rows));
            } else {
                absdiff(tempC, tempS, outPyr[i]);
            }
            ++i;
        }
    }
}


void integrate_single_pyramid(Mat* pyramid, Mat f_map, int numLayers)
{
    int i;
    add(pyramid[0], pyramid[1], f_map);
    for (i = 2; i < numLayers; ++i)
    {
        f_map = f_map + pyramid[i];
    }
}

void integrate_color_pyamids(Mat* pyramid1, Mat* pyramid2, Mat f_map, int numLayers)
{
    int i;
    add(pyramid1[0], pyramid2[0], f_map);
    for (i = 1; i < numLayers; ++i)
    {
        f_map = f_map + pyramid1[i] + pyramid2[i];
    }

}

void integrate_orient_pyamids(Mat* pyr1, Mat* pyr2, Mat* pyr3, Mat* pyr4, Mat f_map, int numLayers)
{

    Mat fmap1(pyr1[0].rows, pyr1[0].cols, CV_32F, Scalar(0.0));
    Mat fmap2(pyr1[0].rows, pyr1[0].cols, CV_32F, Scalar(0.0));
    Mat fmap3(pyr1[0].rows, pyr1[0].cols, CV_32F, Scalar(0.0));
    Mat fmap4(pyr1[0].rows, pyr1[0].cols, CV_32F, Scalar(0.0));

    integrate_single_pyramid(pyr1, fmap1, 6);
    integrate_single_pyramid(pyr2, fmap2, 6);
    integrate_single_pyramid(pyr3, fmap3, 6);
    integrate_single_pyramid(pyr4, fmap4, 6);

    normalize(fmap1);
    normalize(fmap2);
    normalize(fmap3);
    normalize(fmap4);

    f_map = fmap1 + fmap2;
    f_map = f_map + fmap3;
    f_map = f_map + fmap4;

}
