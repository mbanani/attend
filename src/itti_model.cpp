/**
* Implementation of Itti's model
*
* 	- Image channels created from input (r_i, g_i, b_i):
* 		1. Intensity = (r_i + g_i + b_i)/3
* 		2. Blue      = b - (r+g)/2,              b = b_i / i , same for r and g
* 		3. Red       = r - (b+g)/2
* 		4. Green     = g - (r+b)/2
* 		5. Yellow    = (r+g)/2 - |r-g|/2 - b
*
* 	- Image scales needed (i = 2-8, where scale i connotates a ratio of 1:i)
* 	- Coarser scales are obtained through a low-pass filtering and subsampling
* 		= NTS: use a basic low-pass filter and subsample
*
* 	- Orientation filters are obtained using Gabor filter
* 		= Not sure about gabor filter parameters
*
*	- Normalziation is performed
*		= deviated from itti's model in multiplying by standard deviation
*		  instead of (M - m)^2
*		  	** [where M is global maxima and m is average of all maxima]
*
*
*	** Points to note:
*	- focus now on producing a prototype with very little regard for efficiency
*	- try to modularize as much as possible to allow for simple enhancements
*
* 	To do:
* 		- consider
*
*/

// To do: Implement normalization followed by feature map integration


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

void split_input(Mat&, Mat*);
void construct_pyramid(Mat&, Mat*, int);
void across_scale_diff(Mat*, Mat*);
void across_scale_opponency_diff(Mat*, Mat*, Mat*);
void normalize(Mat);
void normalize_by_maxMeanDiff(Mat);
void normalize_by_stdev(Mat);
void normalize_pyramid(Mat*, int);
void integrate_single_pyramid(Mat*, Mat, int);
void integrate_color_pyamids(Mat*, Mat*, Mat, int);
void integrate_orient_pyamids(Mat*, Mat*, Mat*, Mat*, Mat, int);
void my_imshow(string, Mat, int, int);


int main( int argc, char* argv[])
{

    Mat input = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    bool debug = true;

    if(input.cols > 500 || input.rows > 300)
    {
        resize(input, input, Size(500,300));
    }

    double t = (double)getTickCount();

    Mat channels[5];

    //(output stages)cout << "Spliting input to RGBYI. " << endl;
    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow, Intensity
    split_input(input, channels);

    if (debug)
    {
        my_imshow("input    ", input      , 50  , 50);
        my_imshow("Channel 0", channels[0], 50  , 400);
        my_imshow("Channel 1", channels[1], 600 , 50);
        my_imshow("Channel 2", channels[2], 600 , 400);
        my_imshow("Channel 3", channels[3], 1150, 50);
        my_imshow("Channel 4", channels[4], 1150, 400);
        waitKey(100000);
    }

    //(output stages)cout << "Extracting Orientation. " << endl;
    // extract orientations
    Mat or0, or45, or90, or135;
    Size kerSize = Size(10, 10);
    double sigma = 0.8;
    double lam   = CV_PI;
    double gamma = 1;
    double psi   = CV_PI / 2;


    //some fun i/o and godels method
    if(argc > 2)
    {
        int argumentCode = atoi(argv[2]);

        int index = 3;

        if (argumentCode % 2 == 0)
        {
            sigma = atof(argv[index]);
            cout << "sigma : " << sigma << endl;
            ++index;
        }

        if (argumentCode % 3 == 0)
        {
            lam = atof(argv[index]);
            cout << "lamda : " << lam << endl;
            ++index;
        }

        if (argumentCode % 5 == 0)
        {
            gamma = atof(argv[index]);
            cout << "gamma : " << gamma << endl;
            ++index;
        }

        if (argumentCode % 7 == 0)
        {
            psi = atof(argv[index]);
            cout << "psi : " << psi << endl;
            ++index;
        }

    }

    Mat kern0   = getGaborKernel(kerSize, sigma, 0         , lam, gamma, psi);
    Mat kern45  = getGaborKernel(kerSize, sigma, 0.25*CV_PI, lam, gamma, psi);
    Mat kern90  = getGaborKernel(kerSize, sigma, 0.5*CV_PI , lam, gamma, psi);
    Mat kern135 = getGaborKernel(kerSize, sigma, 0.75*CV_PI, lam, gamma, psi);

    filter2D(channels[4], or0  , CV_32F, kern0);
    filter2D(channels[4], or45 , CV_32F, kern45);
    filter2D(channels[4], or90 , CV_32F, kern90);
    filter2D(channels[4], or135, CV_32F, kern135);

    if (debug)
    {
        my_imshow("input    ", input       , 50  , 50);
        my_imshow("Intensity", channels[4] , 50  , 400);
        my_imshow("Channel 1", or0         , 600 , 50);
        my_imshow("Channel 2", or45        , 600 , 400);
        my_imshow("Channel 3", or90        , 1150, 50);
        my_imshow("Channel 4", or135       , 1150, 400);
        waitKey(100000);
    }

    //(output stages)cout << "Constructing Pyramids. " << endl;
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
    construct_pyramid(or0, or0Pyr, 9);
    construct_pyramid(or45, or45Pyr, 9);
    construct_pyramid(or90, or90Pyr, 9);
    construct_pyramid(or135, or135Pyr, 9);


    //(output stages)cout << "Calculating Across Scale Features. " << endl;
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
    across_scale_diff(or0Pyr, or0_fm);
    across_scale_diff(or45Pyr, or45_fm);
    across_scale_diff(or90Pyr, or90_fm);
    across_scale_diff(or135Pyr, or135_fm);
    across_scale_opponency_diff(redPyr, greenPyr, oppRG_fm);
    across_scale_opponency_diff(bluePyr, yellowPyr, oppBY_fm);

    //(output stages)cout << "Perfoming Normalization on Pyramids. " << endl;
    // Normalize and Integrate
    normalize_pyramid(oppRG_fm, 6);
    normalize_pyramid(oppBY_fm, 6);
    normalize_pyramid(intens_fm, 6);
    normalize_pyramid(or0_fm, 6);
    normalize_pyramid(or45_fm, 6);
    normalize_pyramid(or90_fm, 6);
    normalize_pyramid(or135_fm, 6);

    if (debug)
    {
        Mat* curr = oppRG_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = oppBY_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = intens_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = or0_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = or45_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = or90_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    if (debug)
    {
        Mat* curr = or135_fm;

        my_imshow("level 0", curr[0] , 50  , 50);
        my_imshow("level 1", curr[1] , 50  , 400);
        my_imshow("level 2", curr[2] , 600 , 50);
        my_imshow("level 3", curr[3] , 600 , 400);
        my_imshow("level 4", curr[4] , 1150, 50);
        my_imshow("level 5", curr[5] , 1150, 400);
        waitKey(100000);
    }

    //(output stages)cout << "Integrating Pyramids into single feature maps. " << endl;
    //define overall feature mpas
    Mat intens_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));
    Mat opp_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));
    Mat ori_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));

    //(output stages)cout << "Integrating Features" << endl;
    //integrate feature maps for intenisty and color
    integrate_single_pyramid(intens_fm, intens_FM, 6);
    integrate_color_pyamids(oppBY_fm, oppRG_fm, opp_FM, 6);
    integrate_orient_pyamids(or0_fm, or45_fm,or90_fm, or135_fm, ori_FM, 6);

    //integrate all maps
    Mat global_FM;
    // global_FM = opp_FM + intens_FM + ori_FM;
    max(ori_FM, intens_FM, global_FM);
    max(global_FM, opp_FM, global_FM);
    normalize(global_FM, global_FM, 0.0, 1.0, NORM_MINMAX, CV_32F);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Total so far (without read and write) in seconds: " << t << endl;



    resize(global_FM, global_FM, Size(redPyr[0].cols, redPyr[0].rows));
    resize(ori_FM, ori_FM, Size(redPyr[0].cols, redPyr[0].rows));
    resize(opp_FM, opp_FM, Size(redPyr[0].cols, redPyr[0].rows));
    resize(intens_FM, intens_FM, Size(redPyr[0].cols, redPyr[0].rows));

    double maxOpp, minOpp, maxOri, minOri, maxInt, minInt, maxAll;
    minMaxLoc(opp_FM, &minOpp, &maxOpp, NULL, NULL);
    minMaxLoc(ori_FM, &minOri, &maxOri, NULL, NULL);
    minMaxLoc(intens_FM, &minInt, &maxInt, NULL, NULL);

    maxAll = (maxOpp > maxOri)? maxOpp : maxOri;
    maxAll = (maxAll > maxInt)? maxAll : maxInt;

    normalize(intens_FM, intens_FM      , minInt/maxAll, maxInt/maxAll, NORM_MINMAX, CV_32F);
    normalize(ori_FM, ori_FM            , minOri/maxAll, maxOri/maxAll, NORM_MINMAX, CV_32F);
    normalize(opp_FM, opp_FM            , minOpp/maxAll, maxOpp/maxAll, NORM_MINMAX, CV_32F);
    normalize(intensPyr[0], intensPyr[0], 0.0, 1.0, NORM_MINMAX, CV_32F);

    int x = 50;
    int y = 50;
    int dx = input.cols;
    int dy = input.rows + 100;

    Mat diff = intens_FM - ori_FM;

    my_imshow("input", input, x, y);
    my_imshow("inten raw", intensPyr[0], x, y + dy);
    // my_imshow("diff b/w iten and ori", diff, x, y + dy);
    my_imshow("global", global_FM, x + dx, y);
    my_imshow("opponency", opp_FM, x+dx, y + dy);
    my_imshow("orientation", ori_FM, x + 2 * dx, y);
    my_imshow("intensity", intens_FM, x + 2 * dx,y + dy);

    //(output stages)cout << "Done. " << endl;
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
                Y = -b - std::abs(r-g);

                //if extracted color value is negative, set to 0
                b_p[y] = (B>0) ? B/2 : 0;
                r_p[y] = (R>0) ? R/2 : 0;
                g_p[y] = (G>0) ? G/2 : 0;
                y_p[y] = (Y>0) ? Y/2 : 0;
                // i_p[y] = (b_p[y]+r_p[y]+g_p[y])/3;
                // i_p[y] = (0.7*b_p[y]+ 0.2*r_p[y]+ 0.8*g_p[y])/3;
                // i_p[y] = (b_p[y]+r_p[y]+g_p[y] + y_p[y])/4;
                i_p[y] = i;

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

void normalize_by_max_diff(Mat input)
{

}

/**
* Wrapper class for noramlizer
*
* @param input the input image
*/
void normalize(Mat input)
{
    normalize(input, input, 0.0, 1.0, NORM_MINMAX, CV_32F);
    // normalize_by_stdev(input);
    // normalize_by_maxMeanDiff(input);
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

void my_imshow(string name, Mat matrix, int x, int y)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    moveWindow(name, x, y);
    imshow(name, matrix);

}
