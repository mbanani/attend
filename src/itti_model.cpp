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

using namespace std;
using namespace cv;



void my_imshow(string, Mat, int, int);
void debug_show_imgPyramid(Mat*, string);


int main( int argc, char* argv[])
{

    Mat input = imread(argv[1], CV_LOAD_IMAGE_COLOR);

    bool debug = false;

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
        my_imshow("input    ",  input      , 50  , 50);
        my_imshow("Red",        channels[0], 50  , 400);
        my_imshow("Green",      channels[1], 600 , 50);
        my_imshow("Blue",       channels[2], 600 , 400);
        my_imshow("Yellow",     channels[3], 1150, 50);
        my_imshow("Intensity",  channels[4], 1150, 400);
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



    normalize(intens_FM);
    normalize(ori_FM);
    normalize(opp_FM);

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
    // normalize(intensPyr[0], intensPyr[0], 0.0, 1.0, NORM_MINMAX, CV_32F);
    //


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




void my_imshow(string name, Mat matrix, int x, int y)
{
    namedWindow(name, WINDOW_AUTOSIZE);
    moveWindow(name, x, y);

    Mat newMatrix;
    normalize(matrix, newMatrix, 0.0, 1.0, NORM_MINMAX, CV_32F);
    imshow(name, newMatrix);

}

void debug_show_imgPyramid(Mat* imgPyramid, string pyramidInfo)
{

    for (int i = 0; i < 6; i++) {
        resize(imgPyramid[i], imgPyramid[i], Size(500,300));
    }

    my_imshow(pyramidInfo, imgPyramid[0] , 50  , 50);
    my_imshow("level 1", imgPyramid[1] , 50  , 400);
    my_imshow("level 2", imgPyramid[2] , 600 , 50);
    my_imshow("level 3", imgPyramid[3] , 600 , 400);
    my_imshow("level 4", imgPyramid[4] , 1150, 50);
    my_imshow("level 5", imgPyramid[5] , 1150, 400);
    waitKey(100000);
    destroyAllWindows();
}
