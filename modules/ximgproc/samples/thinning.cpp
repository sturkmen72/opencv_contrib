
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <map>

using namespace std;
using namespace cv;

// look up table - there is one entry for each of the 2^8=256 possible
// combinations of 8 binary neighbors.
static uint8_t lut_zhang_iter0[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1};

static uint8_t lut_zhang_iter1[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    0, 1, 1, 1};

static uint8_t lut_guo_iter0[] = {
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1};

static uint8_t lut_guo_iter1[] = {
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1};

// Applies a thinning iteration to a binary image
static void thinningIteration(Mat &img, Mat &marker, const uint8_t* const lut) {
    int rows = img.rows;
    int cols = img.cols;
    marker.col(0).setTo(1);
    marker.col(cols - 1).setTo(1);
    marker.row(0).setTo(1);
    marker.row(rows - 1).setTo(1);

    marker.forEach<uchar>([=](uchar& value, const int position[]) {
        int i = position[0];
        int j = position[1];
        if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1) { return; }

        auto ptr = img.ptr(i, j); // p1
        if (ptr[0]) {
            uchar p2 = ptr[-cols] != 0;
            uchar p3 = ptr[-cols + 1] != 0;
            uchar p4 = ptr[1] != 0;
            uchar p5 = ptr[cols + 1] != 0;
            uchar p6 = ptr[cols] != 0;
            uchar p7 = ptr[cols - 1] != 0;
            uchar p8 = ptr[-1] != 0;
            uchar p9 = ptr[-cols - 1] != 0;

            int neighbors = p9 | (p2 << 1) | (p3 << 2) | (p4 << 3) | (p5 << 4) | (p6 << 5) | (p7 << 6) | (p8 << 7);
            value = 1;
        }
    });

    img &= marker;
    marker.setTo(0);
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType){
    Mat processed = input.getMat().clone();
    CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;
    Mat prev = processed.clone();
    Mat marker = Mat::zeros(processed.size(), CV_8UC1);
    const auto lutIter0 = (thinningType == 1) ? lut_guo_iter0 : lut_zhang_iter0;
    const auto lutIter1 = (thinningType == 1) ? lut_guo_iter1 : lut_zhang_iter1;
    do {
        thinningIteration(processed, marker, lutIter0);
        thinningIteration(processed, marker, lutIter1);
        const auto res = cv::norm(processed, prev, cv::NORM_L1);
        if (res <= 0) { break; }
        processed.copyTo(prev);
    } while (true);

    processed *= 255;
    output.assign(processed);
}

int main(int argc, const char** argv)
{
   // for (int i = 0; i < 256; i++)
  //  if (lut_zhang_iter1[i] == 0)
    //    printf("%d\n", i);
    Mat src = imread("08.png", IMREAD_GRAYSCALE);
    Mat dst,check_img;
    TickMeter tm;
    tm.start();
    thinning(~src, dst, 0);
    tm.stop();
    cout << tm << endl;

    tm.reset();
    tm.start();
    thinning(src, dst, 0);
    tm.stop();
    cout << tm << endl;

    tm.reset();
    tm.start();
    thinning(~src, dst, 0);
    tm.stop();
    cout << tm << endl;

    tm.reset();
    tm.start();
    thinning(src, dst, 0);
    tm.stop();
    cout << tm << endl;

    check_img = imread("Thinning_ZHANGSUEN.png", IMREAD_GRAYSCALE);
    cout << cv::norm(check_img, dst, cv::NORM_L1) << endl;
    imshow("dst", dst);
    waitKey();
    thinning(~src, dst, 0);

    check_img = imread("Thinning_inv_ZHANGSUEN.png", IMREAD_GRAYSCALE);
    cout << cv::norm(check_img, dst, cv::NORM_L1) << endl;
    imshow("dst", dst);
    waitKey(100);
        thinning(src, dst, 1);

    check_img = imread("Thinning_GUOHALL.png", IMREAD_GRAYSCALE);
    cout << cv::norm(check_img, dst, cv::NORM_L1) << endl;
        imshow("dst", dst);
    waitKey(100);
    thinning(~src, dst, 1);

    check_img = imread("Thinning_inv_GUOHALL.png", IMREAD_GRAYSCALE);
    cout << cv::norm(check_img, dst, cv::NORM_L1) << endl;

    imshow("dst", dst);
    waitKey(100);
        exit(0);
    vector<Mat> imgs;
    Mat img(100, 100, CV_8UC1);
    imgs.push_back(img);
    std::vector<uchar> buf;
    bool ret_image = imencode(".tiff", imgs, buf);
    bool ret_multipage = imencode(".tiff", imgs, buf);
    printf("%d\n", ret_image);
    printf("%d\n", ret_multipage);
    imgs.push_back(img.clone());
    ret_multipage = imencode(".tiff", imgs, buf);
    printf("%d\n", ret_multipage);
    return 0;
}
