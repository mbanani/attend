#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

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
