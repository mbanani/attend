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
*		= Normalizes maps with (M - m)^2
*		     * [M is global maxima, and m is mean of all maxima]
*		= After normalization, conspicuity maps are multiplied by the learned
*		  weights for the object being searched for.
*
*	** Points to note:
*	- focus now on producing a prototype with very little regard for efficiency
*	- try to modularize as much as possible to allow for simple enhancements
*	- All the normalization makes it hard to compare across images; an object
*	  with low saliency in a bland image with have the same saliency as a highly
*	  salient object in a very exciting image.
*
* 	To do:
* 		- Consider memory allocation issues if this is to be implemented
*/

#include "saliency.h"
#include "normalize.h"
#include "util.h"
#include "objectProposal.h"
#include <dirent.h>

using namespace std;
using namespace cv;


Mat generateSaliency(Mat, float*, bool, bool);
Mat generateSaliencyProto(Mat, float*, bool, bool);
proposal topPropoal(Mat&, proposal*, int, float*, int);
float* learnFeature(Mat&, proposal);
float* learnFeatureProto(Mat&, proposal);
float* learnFeaturefromDataset(const char *, int);
float* calculateSaliencyFeaturesProto(Mat& );
void printFeatureValues(float* );


int main( int argc, char* argv[])
{

    double t = (double)getTickCount();

    // Path parameters;
    string object = argv[1];
    string picture = argv[2];
    string folderPath = "/home/mohamed/attend/img/4Progress_dataset/" + object;
    string CSVpath = folderPath + "/bboxes/" + picture + ".csv";
    string IMGpath = folderPath + "/image/" + picture + ".jpg";
    string trainPath = folderPath + "/image/positive";

    // Learn features from training set;
    float* features = learnFeaturefromDataset(trainPath.c_str() , 11);

    t = ((double)getTickCount() - t);
    cout << "Time to learn features in seconds: " << t/getTickFrequency() << endl;

    int propList[NUM_PROPOSALS][5] = {0};

    csvToProposalList(CSVpath.c_str(), propList);

    Mat input = imread(IMGpath.c_str(), CV_LOAD_IMAGE_COLOR);

    proposal* objProps = arrayToProposals(propList, NUM_PROPOSALS, 1);

    t = ((double)getTickCount() - t);
    cout << "Time to parse propoals in seconds: " << t/getTickFrequency() << endl;

    proposal topProp = topPropoal(input, objProps, NUM_PROPOSALS, features, 10000);

    t = ((double)getTickCount() - t);
    cout << "Time to calculate top proposal in seconds: " << t/getTickFrequency() << endl;

    Mat output;

    resize(input, output, input.size());
    drawBB(output, topProp, Scalar(0,0,255));
    drawBB(output, Rect(470,90,240,340), Scalar(255,0,0));
    my_imshow("output",  output, 50  , 50);
    cout << "Top Proposal: " << topProp.bbox.x << ", " << topProp.bbox.y <<", " << topProp.bbox.width <<", " << topProp.bbox.height << endl;


    for(int i = 0; i < 100; i++)
    {
        if (calculateIOU(objProps[i].bbox, Rect(470,90,240, 340)) > 0.8)
        {
            drawBB(output, objProps[i], Scalar(255,0,0 ));

        } else {
            drawBB(output, objProps[i], Scalar(0,255,0));

        }
    }
    my_imshow("output with all",  output, 550  , 50);

    cout << "Saliency calculations in seconds: " << t/getTickFrequency() << endl;

    waitKey(100000);
}

/**
 * Initial attempt at outputing normalized saliency maps
 *
 * @param  image          input image
 * @param  objectFeatures float-array determining the feature weights
 * @param  avgGlobal      true if maps are average, false for winner-take-all
 * @param  debug          if set to true, show images produced at each stage
 * @return                an object-specific saliency map
 */
Mat generateSaliency(Mat input, float* objectFeatures, bool avgGlobal, bool debug)
{
    // get time (to be used for calculating time for saliency generation)
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

    // Gabor Filter Parameters
    Mat or0, or45, or90, or135;
    Size kerSize = Size(10, 10);
    double sigma = 0.8;
    double lam   = CV_PI;
    double gamma = 1;
    double psi   = CV_PI / 2;

    // Generate Gabor kernels for the different orientations
    Mat kern0   = getGaborKernel(kerSize, sigma, 0         , lam, gamma, psi);
    Mat kern45  = getGaborKernel(kerSize, sigma, 0.25*CV_PI, lam, gamma, psi);
    Mat kern90  = getGaborKernel(kerSize, sigma, 0.5*CV_PI , lam, gamma, psi);
    Mat kern135 = getGaborKernel(kerSize, sigma, 0.75*CV_PI, lam, gamma, psi);

    // Calculate orientation feature maps
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

    // Define Pyramid variables
    Mat bluePyr[9];
    Mat greenPyr[9];
    Mat redPyr[9];
    Mat yellowPyr[9];
    Mat intensPyr[9];
    Mat or0Pyr[9];
    Mat or45Pyr[9];
    Mat or90Pyr[9];
    Mat or135Pyr[9];

    //Construct pyramids
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


    // define conspicuity map pyramids
    Mat oppRG_cm[6];
    Mat oppBY_cm[6];
    Mat or0_cm[6];
    Mat or45_cm[6];
    Mat or90_cm[6];
    Mat or135_cm[6];
    Mat intens_cm[6];

    //calculate conspituity map pyramids
    across_scale_diff(intensPyr, intens_cm);
    across_scale_diff(or0Pyr, or0_cm);
    across_scale_diff(or45Pyr, or45_cm);
    across_scale_diff(or90Pyr, or90_cm);
    across_scale_diff(or135Pyr, or135_cm);
    across_scale_opponency_diff(redPyr, greenPyr, oppRG_cm);
    across_scale_opponency_diff(bluePyr, yellowPyr, oppBY_cm);

    // Normalize
    normalize_pyramid(oppRG_cm, 6);
    normalize_pyramid(oppBY_cm, 6);
    normalize_pyramid(intens_cm, 6);
    normalize_pyramid(or0_cm, 6);
    normalize_pyramid(or45_cm, 6);
    normalize_pyramid(or90_cm, 6);
    normalize_pyramid(or135_cm, 6);


    // debug show levels
    if (debug)
    {
        debug_show_imgPyramid(oppRG_cm, "RG Opponency");
        debug_show_imgPyramid(oppBY_cm, "BY Opponency");
        debug_show_imgPyramid(intens_cm, "Intensity");
        debug_show_imgPyramid(or0_cm,   "Orientation 0");
        debug_show_imgPyramid(or45_cm,  "Orientation 45");
        debug_show_imgPyramid(or90_cm,  "Orientation 90");
        debug_show_imgPyramid(or135_cm, "Orientation 135");
    }

    //define overall conspicuity maps (initialized size is the same for all)
    Mat intens_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat opp_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat ori_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));

    //integrate conspicuity maps
    integrate_single_pyramid(intens_cm, intens_CM, 6);
    integrate_color_pyamids(oppBY_cm, oppRG_cm, opp_CM, 6);
    integrate_orient_pyamids(or0_cm, or45_cm, or90_cm, or135_cm, ori_CM, 6);

    // normalize again ?!
    normalize(intens_CM);
    normalize(ori_CM);
    normalize(opp_CM);

    // Multiply by feature weights
    intens_CM = intens_CM * objectFeatures[0];
    ori_CM = ori_CM * objectFeatures[1];
    opp_CM = opp_CM * objectFeatures[2];

    //integrate all maps
    Mat global_CM;

    if(avgGlobal){
        global_CM = opp_CM + intens_CM + ori_CM;
    } else {
        max(ori_CM, intens_CM, global_CM);
        max(global_CM, opp_CM, global_CM);
    }

    // Normalize final output ?
    normalize(global_CM, global_CM, 0.0, 1.0, NORM_MINMAX, CV_32F);

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Total so far (without read and write) in seconds: " << t << endl;

    return global_CM;
}

Mat generateSaliencyProto(Mat input, float* objectFeatures, bool avgGlobal, bool debug)
{
    // // get time (to be used for calculating time for saliency generation)
    // double t = (double)getTickCount();

    Mat channels[5];

    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow, Intensity
    split_rgbyi(input, channels);

    if (debug)
    {
        cout << "Debug generateSaliencyProto 1: show raw channels" << endl;
        my_imshow("input    ",  input      , 50  , 50);
        my_imshow("Red",        channels[0], 50  , 400);
        my_imshow("Green",      channels[1], 600 , 50);
        my_imshow("Blue",       channels[2], 600 , 400);
        my_imshow("Yellow",     channels[3], 1150, 50);
        my_imshow("Intensity",  channels[4], 1150, 400);
        waitKey(100000);
    }

    // Gabor Filter Parameters
    Mat or0, or45, or90, or135;
    Size kerSize = Size(10, 10);
    double sigma = 0.8;
    double lam   = CV_PI;
    double gamma = 1;
    double psi   = CV_PI / 2;

    // Generate Gabor kernels for the different orientations
    Mat kern0   = getGaborKernel(kerSize, sigma, 0         , lam, gamma, psi);
    Mat kern45  = getGaborKernel(kerSize, sigma, 0.25*CV_PI, lam, gamma, psi);
    Mat kern90  = getGaborKernel(kerSize, sigma, 0.5*CV_PI , lam, gamma, psi);
    Mat kern135 = getGaborKernel(kerSize, sigma, 0.75*CV_PI, lam, gamma, psi);

    // Calculate orientation feature maps
    filter2D(channels[4], or0  , CV_32F, kern0);
    filter2D(channels[4], or45 , CV_32F, kern45);
    filter2D(channels[4], or90 , CV_32F, kern90);
    filter2D(channels[4], or135, CV_32F, kern135);

    if (debug)
    {
        cout << "Debug generateSaliencyProto 1: show orientation channels" << endl;
        my_imshow("input    ", input       , 50  , 50);
        my_imshow("Intensity", channels[4] , 50  , 400);
        my_imshow("Channel 1", or0         , 600 , 50);
        my_imshow("Channel 2", or45        , 600 , 400);
        my_imshow("Channel 3", or90        , 1150, 50);
        my_imshow("Channel 4", or135       , 1150, 400);
        waitKey(100000);
    }

    // Define Pyramid variables
    Mat bluePyr[9];
    Mat greenPyr[9];
    Mat redPyr[9];
    Mat yellowPyr[9];
    Mat intensPyr[9];
    Mat or0Pyr[9];
    Mat or45Pyr[9];
    Mat or90Pyr[9];
    Mat or135Pyr[9];

    //Construct pyramids
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

    // define conspicuity map pyramids
    Mat oppRG_cm[6];
    Mat oppBY_cm[6];
    Mat or0_cm[6];
    Mat or45_cm[6];
    Mat or90_cm[6];
    Mat or135_cm[6];
    Mat intens_cm[6];

    //calculate conspituity map pyramids
    across_scale_diff(intensPyr, intens_cm);
    across_scale_diff(or0Pyr, or0_cm);
    across_scale_diff(or45Pyr, or45_cm);
    across_scale_diff(or90Pyr, or90_cm);
    across_scale_diff(or135Pyr, or135_cm);
    across_scale_opponency_diff(redPyr, greenPyr, oppRG_cm);
    across_scale_opponency_diff(bluePyr, yellowPyr, oppBY_cm);

    // Normalize
    normalize_pyramid(oppRG_cm, 6);
    normalize_pyramid(oppBY_cm, 6);
    normalize_pyramid(intens_cm, 6);
    normalize_pyramid(or0_cm, 6);
    normalize_pyramid(or45_cm, 6);
    normalize_pyramid(or90_cm, 6);
    normalize_pyramid(or135_cm, 6);


    // debug show levels
    if (debug)
    {
        cout << "Debug generateSaliencyProto 1: show conspicuity channels" << endl;
        debug_show_imgPyramid(oppRG_cm, "RG Opponency");
        debug_show_imgPyramid(oppBY_cm, "BY Opponency");
        debug_show_imgPyramid(intens_cm, "Intensity");
        debug_show_imgPyramid(or0_cm,   "Orientation 0");
        debug_show_imgPyramid(or45_cm,  "Orientation 45");
        debug_show_imgPyramid(or90_cm,  "Orientation 90");
        debug_show_imgPyramid(or135_cm, "Orientation 135");
    }

    //define overall conspicuity maps (initialized size is the same for all)
    Mat intens_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat opp_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat ori_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));

    //integrate conspicuity maps
    integrate_single_pyramid(intens_cm, intens_CM, 6);
    integrate_color_pyamids(oppBY_cm, oppRG_cm, opp_CM, 6);
    integrate_orient_pyamids(or0_cm, or45_cm, or90_cm, or135_cm, ori_CM, 6);

    // normalize again ?!
    normalize(intens_CM);
    normalize(ori_CM);
    normalize(opp_CM);

    //normalize all feature maps .. hues are already normalized, normalize orientation
    normalize(or0);
    normalize(or45);
    normalize(or90);
    normalize(or135);

    //resize all maps
    resize(intens_CM, intens_CM, input.size());
    resize(ori_CM, ori_CM, input.size());
    resize(opp_CM, opp_CM, input.size());

    printFeatureValues(objectFeatures);

    // Multiply by feature weights
    intens_CM = intens_CM * objectFeatures[0];
    ori_CM = ori_CM * objectFeatures[1];
    opp_CM = opp_CM * objectFeatures[2];
    channels[0] = channels[0] * objectFeatures[3];

    channels[1] = channels[1] * objectFeatures[4];
    channels[2] = channels[2] * objectFeatures[5];
    channels[3] = channels[3] * objectFeatures[6];
    or0 = or0 * objectFeatures[7];
    or45 = or45 * objectFeatures[8];
    or90 = or90 * objectFeatures[9];
    or135 = or135 * objectFeatures[10];


    //integrate all maps
    Mat global_CM;

    if(avgGlobal){
        global_CM = opp_CM + intens_CM + ori_CM + channels[0] + channels[1] + channels[2] + channels[3] + or0 + or45 + or90 + or135;
    } else {
        max(ori_CM, intens_CM, global_CM);
        max(global_CM, opp_CM, global_CM);
    }

    // Normalize final output ?
    normalize(global_CM, global_CM, 0.0, 1.0, NORM_MINMAX, CV_32F);

    // t = ((double)getTickCount() - t)/getTickFrequency();
    // cout << "Total so far (without read and write) in seconds: " << t << endl;

    return global_CM;
}


/**
 * Outputs the best proposal (or the first to exceed a specific threshold)
 * @param  image    [description]
 * @param  objProps [description]
 * @param  thresh   a percentage confidence multipled by 10,000
 * @return          The best proposal (or first to exceed threshold)
 */
proposal topPropoal(Mat& image, proposal* objProps, int numProposals, float* features, int thresh)
{
	proposal topProp;

    Mat saliencyMap = generateSaliencyProto(image, features, true, false);
    resize(saliencyMap, saliencyMap, image.size());

    // Display the saliency map
    my_imshow("output",  saliencyMap, 50  , 50);
    waitKey(100000);


    objProps[0].saliencyScore = calculateSaliencyScore(saliencyMap, objProps[0]);
    topProp = objProps[0];

    cout << "calculated score" << endl;
	for(int i = 1; i < numProposals; i++)
	{
        objProps[i].saliencyScore = calculateSaliencyScore(saliencyMap, objProps[i]);

        if(objProps[i].saliencyScore > thresh){
            return objProps[i];
        } else if(objProps[i].saliencyScore > topProp.saliencyScore)
        {
            topProp = objProps[i];
        }
	}
    return topProp;
}

float* learnFeature(Mat& image, proposal prop)
{
    float* score = new float[3];

    // initialize feature vectors for each feature
    float features0[3] = {1.0, 0.0, 0.0};
    float features1[3] = {0.0, 1.0, 0.0};
    float features2[3] = {0.0, 0.0, 1.0};

    //generate saliency map for each object
    Mat saliency0 = generateSaliency(image, features0, false, false);
    Mat saliency1 = generateSaliency(image, features1, false, false);
    Mat saliency2 = generateSaliency(image, features2, false, false);

    // resize image to fit it
    // (consider resizing box instead, as loss in accuracy should be negligable)
    resize(saliency0, saliency0, image.size());
    resize(saliency1, saliency1, image.size());
    resize(saliency2, saliency2, image.size());

    // Score calculated from the same Saliency evaluation function
    score[0] = (float) calculateSaliencyScore(saliency0, prop);
    score[1] = (float) calculateSaliencyScore(saliency1, prop);
    score[2] = (float) calculateSaliencyScore(saliency2, prop);

    // Normalize score vector
    float sum = score[0] + score[1] + score[2];
    score[0] = score[0] / sum;
    score[1] = score[1] / sum;
    score[2] = score[2] / sum;

    // cout << "score for 0: " << calculateSaliencyScore(saliency0, prop) << endl;
    // cout << "score for 1: " << calculateSaliencyScore(saliency1, prop) << endl;
    // cout << "score for 2: " << calculateSaliencyScore(saliency2, prop) << endl;
    //
    // cout << "sum: " << sum << endl;
    return score;
}

float* learnFeaturefromDataset(const char *databasePath, int numFeatures)
{
    float* featureSums = new float[numFeatures];
    int numExamples = 0;
    string imgName, imgPath;
    int maxExamples = 1000;

    for(int i = 0; i < numFeatures; i++)
    {
        featureSums[i] = 0;
    }

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (databasePath)) != NULL) {
      /* print all the files and directories within directory */
      while ((ent = readdir (dir)) != NULL && numExamples < maxExamples) {
        imgName = ent -> d_name;
        if((imgName).find("jpg") != std::string::npos)
        {

            imgPath = (string) databasePath + "/" + imgName;
            // cout << imgPath << endl;
            Mat input = imread(imgPath , CV_LOAD_IMAGE_COLOR);
            // cout << imgPath << endl;
            float* instanceFeatures = calculateSaliencyFeaturesProto(input);

            numExamples = numExamples + 1;
            for(int i = 0; i < numFeatures; i++)
            {
                featureSums[i] = featureSums[i] + instanceFeatures[i];
            }

        }
      }
      closedir (dir);

      if (numExamples > 0)
      {
          for(int i = 0; i < numFeatures; i++)
          {
              featureSums[i] = featureSums[i] / numExamples;
              cout << i << ":\t" << featureSums[i] << endl;
          }
      } else
      {
          for(int i = 0; i < numFeatures; i++)
          {
              featureSums[i] = 0.0;
          }
      }

    } else {
      /* could not open directory */
      perror ("");
    }

    return featureSums;
}


float* calculateSaliencyFeaturesProto(Mat& input)
{
    Mat channels[5];

    //extract colors and intensity.
    //Channels in order: Red, Green, Blue, Yellow, Intensity
    split_rgbyi(input, channels);


    // Gabor Filter Parameters
    Mat or0, or45, or90, or135;
    Size kerSize = Size(10, 10);
    double sigma = 0.8;
    double lam   = CV_PI;
    double gamma = 1;
    double psi   = CV_PI / 2;

    // Generate Gabor kernels for the different orientations
    Mat kern0   = getGaborKernel(kerSize, sigma, 0         , lam, gamma, psi);
    Mat kern45  = getGaborKernel(kerSize, sigma, 0.25*CV_PI, lam, gamma, psi);
    Mat kern90  = getGaborKernel(kerSize, sigma, 0.5*CV_PI , lam, gamma, psi);
    Mat kern135 = getGaborKernel(kerSize, sigma, 0.75*CV_PI, lam, gamma, psi);

    // Calculate orientation feature maps
    filter2D(channels[4], or0  , CV_32F, kern0);
    filter2D(channels[4], or45 , CV_32F, kern45);
    filter2D(channels[4], or90 , CV_32F, kern90);
    filter2D(channels[4], or135, CV_32F, kern135);


    // Define Pyramid variables
    Mat bluePyr[9];
    Mat greenPyr[9];
    Mat redPyr[9];
    Mat yellowPyr[9];
    Mat intensPyr[9];
    Mat or0Pyr[9];
    Mat or45Pyr[9];
    Mat or90Pyr[9];
    Mat or135Pyr[9];

    //Construct pyramids
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


    // define conspicuity map pyramids
    Mat oppRG_cm[6];
    Mat oppBY_cm[6];
    Mat or0_cm[6];
    Mat or45_cm[6];
    Mat or90_cm[6];
    Mat or135_cm[6];
    Mat intens_cm[6];

    //calculate conspituity map pyramids
    across_scale_diff(intensPyr, intens_cm);
    across_scale_diff(or0Pyr, or0_cm);
    across_scale_diff(or45Pyr, or45_cm);
    across_scale_diff(or90Pyr, or90_cm);
    across_scale_diff(or135Pyr, or135_cm);
    across_scale_opponency_diff(redPyr, greenPyr, oppRG_cm);
    across_scale_opponency_diff(bluePyr, yellowPyr, oppBY_cm);

    // Normalize
    normalize_pyramid(oppRG_cm, 6);
    normalize_pyramid(oppBY_cm, 6);
    normalize_pyramid(intens_cm, 6);
    normalize_pyramid(or0_cm, 6);
    normalize_pyramid(or45_cm, 6);
    normalize_pyramid(or90_cm, 6);
    normalize_pyramid(or135_cm, 6);



    //define overall conspicuity maps (initialized size is the same for all)
    Mat intens_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat opp_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));
    Mat ori_CM(oppRG_cm[0].rows, oppRG_cm[0].cols, CV_32F, Scalar(0.0));

    //integrate conspicuity maps
    integrate_single_pyramid(intens_cm, intens_CM, 6);
    integrate_color_pyamids(oppBY_cm, oppRG_cm, opp_CM, 6);
    integrate_orient_pyamids(or0_cm, or45_cm, or90_cm, or135_cm, ori_CM, 6);

    // normalize again ?!
    normalize(intens_CM);
    normalize(ori_CM);
    normalize(opp_CM);

    //resize all maps
    resize(intens_CM, intens_CM, input.size());
    resize(ori_CM, ori_CM, input.size());
    resize(opp_CM, opp_CM, input.size());

    // Multiply by feature weights
    Scalar features[11];
    float* featureVec = new float[11];
    //
    // my_imshow("intenseCM", intens_CM, 50, 50);
    //
    // cout << "sum vale "<< sum(intens_CM) << endl;
    // waitKey(10000);

    features[0] = mean(intens_CM);
    features[1] = mean(ori_CM);
    features[2] = mean(opp_CM);
    features[3] = mean(channels[0]);
    features[4] = mean(channels[1]);
    features[5] = mean(channels[2]);
    features[6] = mean(channels[3]);
    features[7] = mean(or0);
    features[8] = mean(or45);
    features[9] = mean(or90);
    features[10] = mean(or135);

    for(int i = 0; i < 11; i++)
    {

        featureVec[i] = (float) (features[i])[0];
        // cout << "feature " << i << ": "<< featureVec[i] << endl;
        // cout << "features       "<< features[i] << endl;
        // cout << "features       "<< features[i] << endl;
    }
    // cout << "--------" << endl;
    // waitKey(100000);

    float sumFeat1,sumFeat2,sumFeat3,featParity1;
    sumFeat1 = abs(featureVec[0]) + abs(featureVec[1]) + abs(featureVec[2]);
    sumFeat2 = abs(featureVec[3]) + abs(featureVec[4]) + abs(featureVec[5]) + abs(featureVec[6]);
    sumFeat3 = abs(featureVec[7]) + abs(featureVec[8]) + abs(featureVec[9]) + abs(featureVec[10]);

    if(sumFeat1 != 0){
        featureVec[0] = featureVec[0]/sumFeat1;
        featureVec[1] = featureVec[1]/sumFeat1;
        featureVec[2] = featureVec[2]/sumFeat1;
    }

    if(sumFeat2 != 0){
        featureVec[3] = featureVec[3]/sumFeat2;
        featureVec[4] = featureVec[4]/sumFeat2;
        featureVec[5] = featureVec[5]/sumFeat2;
        featureVec[6] = featureVec[6]/sumFeat2;

    }

    if(sumFeat3 != 0){

        featureVec[7] = featureVec[7]/sumFeat3;
        featureVec[8] = featureVec[8]/sumFeat3;
        featureVec[9] = featureVec[9]/sumFeat3;
        featureVec[10] = featureVec[10]/sumFeat3;
    }



    return featureVec;

}

void printFeatureValues(float* features)
{
    cout << endl << "Saliency Feature Values: " << endl;
    cout << "Intensity conspicuity :    \t"    <<  features[0] << endl;
    cout << "Orientation conspicuity :  \t"    <<  features[1] << endl;
    cout << "Opponency conspicuity :    \t"    <<  features[2] << endl;
    cout << "Red Filter :               \t"    <<  features[3] << endl;
    cout << "Green Filter :             \t"    <<  features[4] << endl;
    cout << "Blue Filter :              \t"    <<  features[5] << endl;
    cout << "Yellow Filter :            \t"    <<  features[6] << endl;
    cout << "0-degree orientation :     \t"    <<  features[7] << endl;
    cout << "45-degree orientation :    \t"    <<  features[8] << endl;
    cout << "90-degree orientation :    \t"    <<  features[9] << endl;
    cout << "135-degree orientation :   \t"    <<  features[10] << endl;
}
