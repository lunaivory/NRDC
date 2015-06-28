#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

const int segN = 6;

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<Point2d> &regions);
void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, double a[][4]);
void _ApplyGlobalColor(Mat &src, Mat &ret, double a[][4]);
int _GetRange(uchar val);

Mat GlobalColorTransformation(Mat src, Mat ref, vector<Point2d> regions) {
  double par[segN][4];

  vector<Mat> rgbSrc, rgbRef, rgbRet;
  rgbSrc.resize(3), rgbRef.resize(3), rgbRet.resize(3);
  vector<Point2d> pt;
  split(src, rgbSrc); //BGR
  split(ref, rgbRef);

  _GetPointsInside(rgbSrc[0], rgbSrc[0], pt, regions);
  _GetParameters(rgbSrc[0], rgbRef[0], pt, par);
  _ApplyGlobalColor(rgbSrc[0], rgbRet[0], par);

  _GetPointsInside(rgbSrc[1], rgbSrc[1], pt, regions);
  _GetParameters(rgbSrc[1], rgbRef[1], pt, par);
  _ApplyGlobalColor(rgbSrc[1], rgbRet[1], par);

  _GetPointsInside(rgbSrc[2], rgbSrc[2], pt, regions);
  _GetParameters(rgbSrc[2], rgbRef[2], pt, par);
  _ApplyGlobalColor(rgbSrc[2], rgbRet[2], par);

  Mat ret;
  merge(rgbRet, ret);
  return ret;
}

bool _pointSort(Point2d a, Point2d b) {
  return a.x == b.x ? a.y < b.y : a.x < b.x;
}

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<Point2d> &regions) {
  for (int i = 0; i < regions.size(); i++) {
    int x = regions[i].x, y = regions[i].y;
    //TODO yy xx cause src and ref have different coordinate!
    Point2d val((int)src.at<uchar>(y, x), (int)ref.at<uchar>(y, x));
    pt.push_back(val);
  }

  sort(pt.begin(), pt.end(), _pointSort);

  return;
}

void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, double a[][4]) {
  
  double size = 255.0 / segN;

  int itr = 0;
  double prev = 0.0;
  for (int i = 0; i < segN; i++) {
    double range[2] = {prev, (i == segN - 1) ? 255 : (prev + size)};
    
    Mat A, x = Mat::ones(4, 1, CV_64F), b;
    printf("Range %f %f\n", range[0], range[1]);
    for (; itr < pt.size() && pt[itr].x < range[1]; itr++) {
      double x = pt[itr].x, y = pt[itr].y;
      double aa[4] = {pow(x, 3), pow(x, 2), x, 1}, bb[1] = {y};
      A.push_back(Mat(1, 4, CV_64F, aa));
      b.push_back(Mat(1, 1, CV_64F, bb));
    }
   printf("JO A %d %d B %d %d x %d %d\n", A.cols, A.rows, b.cols, b.rows, x.cols, x.rows); 
    try {
      solve(A, b, x, DECOMP_SVD);
    } catch(Exception &e) {
      x.data[0] = x.data[1] = 0;
      x.data[2] = 1;
      x.data[3] = 0;
      const char *err_msg = e.what();
      printf("[OPEN_CV] %s\n", err_msg);
    }
  printf("done\n");
    for (int ii = 0; ii < 4; ii++)
      a[i][ii] = x.at<double>(0,ii);
    prev = range[1];
  }
}

void _ApplyGlobalColor(Mat &src, Mat &ret, double a[][4]) {
  ret = Mat::ones(src.rows, src.cols, CV_8U);

  for (int i = 0; i < src.cols; i++) 
    for (int j = 0; j < src.rows; j++) {
      int k = _GetRange(src.at<uchar>(j, i));
      double x = (double)src.at<uchar>(j, i);
      ret.at<uchar>(j, i) = (uchar)(a[k][0] + a[k][1] * x + a[k][2] * pow(x, 2) + a[k][3] * pow(x, 3));
    }
  return;
}

int _GetRange(uchar val) {
  double size = 255.0 / segN;
  for (int i = 0; i < segN; i++) {
    if ((double) val <= size * ((double)(i + 1)) + 1e-9)  return i;
  }
  return segN - 1;
}

//testing
int main() {
  Mat src = imread("./image/src.png", CV_LOAD_IMAGE_COLOR);
  Mat ref = imread("./image/ref.png", CV_LOAD_IMAGE_COLOR);
  vector<Point2d> pt;
  for (int i = 0 ; i < src.cols; i++)
    for (int j = 0; j < src.rows; j++) {
      pt.push_back(Point2d(i, j));
    }
  Mat ret = GlobalColorTransformation(src, ref, pt);

  namedWindow("final", 1);
  imshow("final", ret);
  waitKey(0);
}
