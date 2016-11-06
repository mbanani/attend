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
*		= Normalizes maps with (M - m)^2 [M is global maxima, and m is average of all maxima]
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


#include "saliency.h"
#include "normalize.h"
#include "util.h"

using namespace std;
using namespace cv;


Mat generateSaliency(Mat, float*, bool, bool);
Mat generateRGBYSaliency(Mat, float*, bool);


int main( int argc, char* argv[])
{

    Mat input = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    int width = 1000;
    float ratio = ( (float) input.rows / (float) input.cols);
    cout << "Rows " << input.rows << ". Cols " << input.cols << ". ratio " << ratio << endl;
    int height = ratio * width;

    width = width > input.cols ? width: input.cols;
    height = height > input.rows ? height: input.rows;

    int d_width = 500;
    int d_height = 500 * ratio;

    resize(input, input, Size(width,height));

    float object1[5] = {1.0, 0.0, 0.0, 0.0};
    float object2[5] = {0.0, 1.0, 0.0, 0.0};
    float object3[5] = {0.0, 0.0, 1.0, 0.0};
    float object4[5] = {0.0, 0.0, 0.0, 1.0};
    float object5[5] = {0.9, 0.0, 0.0, 0.1};


    Mat saliency1 = generateRGBYSaliency(input, object1, true);
    Mat saliency2 = generateRGBYSaliency(input, object2, true);
    Mat saliency3 = generateRGBYSaliency(input, object3, true);
    Mat saliency4 = generateRGBYSaliency(input, object4, true);
    Mat saliency5 = generateRGBYSaliency(input, object5, true);

    resize(saliency1, saliency1, Size(d_width,d_height));
    resize(saliency2, saliency2, Size(d_width,d_height));
    resize(saliency3, saliency3, Size(d_width,d_height));
    resize(saliency4, saliency4, Size(d_width,d_height));
    resize(saliency5, saliency5, Size(d_width,d_height));
    resize(input, input, Size(d_width,d_height));


    my_imshow("Red",  saliency1, 50  , 50);
    my_imshow("Green",  saliency2, 50 , 200 + d_height);
    my_imshow("Blue",saliency3, 100 + d_width  ,50);
    my_imshow("Yellow",saliency4, 100 + d_width, 200 + d_height);
    my_imshow("Original",input, 150 + 2*d_width , 50);
    my_imshow("Custom",saliency5, 150 + 2*d_width , 200 + d_height);
    waitKey(100000);

}

/**
 * Initial attempt at outputing normalized saliency maps
 *
 * @param  image          input image
 * @param  objectFeatures an array of floats determining the normalization weights
 * @return                an object-specific saliency map
 */
Mat generateSaliency(Mat input, float* objectFeatures, bool avgGlobal, bool debug)
{

    double t = (double)getTickCount();

    Mat channels[5];

    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow, Intensity
    split_rgbyi(input, channels);

    if (debug)
    {
        my_imshow("input    ",  input      , 50  , 50);
        my_imshow("Red",        channels[0], 50  , 400);
        my_imshow("Green",      channels[1], 600 , 50);
        my_imshow("Blue",       channels[2], 600 , 400);
        my_imshow("Yellow",     channels[3], 1150, 50);
        my_imshow("Intensity",  channels[4], 1150, 400);
        waitKey(100000);
    }

    // extract orientations
    Mat or0, or45, or90, or135;
    Size kerSize = Size(10, 10);
    double sigma = 0.8;
    double lam   = CV_PI;
    double gamma = 1;
    double psi   = CV_PI / 2;


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

    // Normalize and Integrate
    normalize_pyramid(oppRG_fm, 6);
    normalize_pyramid(oppBY_fm, 6);
    normalize_pyramid(intens_fm, 6);
    normalize_pyramid(or0_fm, 6);
    normalize_pyramid(or45_fm, 6);
    normalize_pyramid(or90_fm, 6);
    normalize_pyramid(or135_fm, 6);


    // debug show levels
    if (debug)
    {
        debug_show_imgPyramid(oppRG_fm, "RG Opponency");
        debug_show_imgPyramid(oppBY_fm, "BY Opponency");
        debug_show_imgPyramid(intens_fm, "Intensity");
        debug_show_imgPyramid(or0_fm,   "Orientation 0");
        debug_show_imgPyramid(or45_fm,  "Orientation 45");
        debug_show_imgPyramid(or90_fm,  "Orientation 90");
        debug_show_imgPyramid(or135_fm, "Orientation 135");
    }

    //define overall feature mpas
    Mat intens_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));
    Mat opp_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));
    Mat ori_FM(oppRG_fm[0].rows, oppRG_fm[0].cols, CV_32F, Scalar(0.0));

    //(output stages)cout << "Integrating Features" << endl;
    //integrate feature maps for intenisty and color
    integrate_single_pyramid(intens_fm, intens_FM, 6);
    integrate_color_pyamids(oppBY_fm, oppRG_fm, opp_FM, 6);
    integrate_orient_pyamids(or0_fm, or45_fm,or90_fm, or135_fm, ori_FM, 6);

    normalize(intens_FM);
    normalize(ori_FM);
    normalize(opp_FM);

    normalize(intens_FM,    intens_FM,  0.0, objectFeatures[0], NORM_MINMAX, CV_32F);
    normalize(ori_FM,       ori_FM,     0.0, objectFeatures[1], NORM_MINMAX, CV_32F);
    normalize(opp_FM,       opp_FM,     0.0, objectFeatures[2], NORM_MINMAX, CV_32F);

    //integrate all maps
    Mat global_FM;

    if(avgGlobal){
        global_FM = opp_FM + intens_FM + ori_FM;
    } else {
        max(ori_FM, intens_FM, global_FM);
        max(global_FM, opp_FM, global_FM);
    }

    normalize(global_FM, global_FM, 0.0, 1.0, NORM_MINMAX, CV_32F);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Total so far (without read and write) in seconds: " << t << endl;

    return global_FM;
}

/**
 * Initial attempt at outputing normalized saliency maps
 *
 * @param  input    input image
 * @param  rgby     an array of floats determining the color value
 * @return          a color-specific saliency map
 */
Mat generateRGBYSaliency(Mat input, float* rgbyCoeff, bool avgGlobal)
{

    double t = (double)getTickCount();

    Mat channels[5];

    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow
    split_rgbyi(input, channels);

    Mat newColor = rgbyCoeff[0] * channels[0] + rgbyCoeff[1] * channels[1];
    newColor = newColor + rgbyCoeff[2] * channels[2] +rgbyCoeff[3] * channels[3];

    //define pyramid variables
    Mat colorPyr[9];

    //Construct pyramids and (maybe release memory for channels)
    construct_pyramid(newColor, colorPyr, 9);

    //(output stages)cout << "Calculating Across Scale Features. " << endl;
    // define feature maps
    Mat color_fm[6];

    //calculate feature maps for within pyramid features
    across_scale_diff(colorPyr, color_fm);

    // Normalize and Integrate
    normalize_pyramid(color_fm, 6);

    //define overall feature mpas
    Mat color_FM(color_fm[0].rows, color_fm[0].cols, CV_32F, Scalar(0.0));

    //(output stages)cout << "Integrating Features" << endl;
    //integrate feature maps for intenisty and color
    integrate_single_pyramid(color_fm, color_FM, 6);

    normalize(color_FM);

    normalize(color_FM,  color_FM,   0.0, 1.0, NORM_MINMAX, CV_32F);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Total so far (without read and write) in seconds: " << t << endl;

    return color_FM;
}
