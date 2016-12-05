
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

    //
    // float* learnFeatureProto(Mat& image)
    // {
    //
    //     int numFeatures = 11;
    //
    //     float* score = new float[numFeatures];
    //     float sum = 0;
    //
    //     for(int i = 0; i < numFeatures; i++)
    //     {
    //         // initialize feature vectore for each feature
    //         float features[11] = {0};
    //         features[i] = 1.0;
    //
    //         //generate saliency map for each feature
    //         Mat saliency = generateSaliency(image, features, true, false);
    //
    //         // resize image to fit it
    //         // (consider resizing box instead, as loss in accuracy should be negligable)
    //         resize(saliency, saliency, image.size());
    //
    //         score[i] = (float) calculateSaliencyScoreProto(saliency);
    //         sum = sum + score[i];
    //     }
    //
    //     for(int i = 0; i < numFeatures; i++)
    //     {
    //         score[i] = score[i] / sum;
    //     }
    //
    //     // cout << "score for 0: " << calculateSaliencyScore(saliency0, prop) << endl;
    //     // cout << "score for 1: " << calculateSaliencyScore(saliency1, prop) << endl;
    //     // cout << "score for 2: " << calculateSaliencyScore(saliency2, prop) << endl;
    //     //
    //     // cout << "sum: " << sum << endl;
    //     return score;
    // }
