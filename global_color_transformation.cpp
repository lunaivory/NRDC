//#define GLOBAL_COLOR_TEST

#include <cstring>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

const int segN = 6;

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<Point2d> &regions);
void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, vector<vector<double> > &a);
void _ApplyGlobalColor(Mat &src, Mat &ret, vector<vector<double> > &a);
int _GetRange(uchar val);

Mat GlobalColorTransformation(Mat src, Mat ref, vector<Point2d> regions) {

  vector<Mat> rgbSrc, rgbRef, rgbRet;
  rgbSrc.resize(3), rgbRef.resize(3), rgbRet.resize(3);
  vector<Point2d> pt;
  vector<vector<double> > par(segN, vector<double>(4, 0));
  split(src, rgbSrc); //BGR
  split(ref, rgbRef);

  _GetPointsInside(rgbSrc[0], rgbRef[0], pt, regions);
  _GetParameters(rgbSrc[0], rgbRef[0], pt, par);
  _ApplyGlobalColor(rgbSrc[0], rgbRet[0], par);

  _GetPointsInside(rgbSrc[1], rgbRef[1], pt, regions);
  _GetParameters(rgbSrc[1], rgbRef[1], pt, par);
  _ApplyGlobalColor(rgbSrc[1], rgbRet[1], par);
  
  _GetPointsInside(rgbSrc[2], rgbRef[2], pt, regions);
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

void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, vector<vector<double> > &a) {
  
  double size = 255.0 / segN;

  int itr = 0;
  double prev = 0.0;
  for (int i = 0; i < segN; i++) {
    double range[2] = {prev, (i == segN - 1) ? 255 : (prev + size)};
    
    Mat A, x, b;
    printf("[COLOR] Range %f %f\n", range[0], range[1]);
    int cnt = 0;
    for (; itr < pt.size() && pt[itr].x < range[1]; itr++) {
      //if (cnt > 500 || itr % segN != i) continue;
      //else  cnt++;
      double x = (double)pt[itr].x, y = (double)pt[itr].y;
      //double aa[4] = {pow(x, 3), pow(x, 2), x, 1}, bb[1] = {y};
      //A.push_back(Mat(1, 4, CV_64F, aa));
      double aa[2] = {1, x}, bb[1] = {y};
      A.push_back(Mat(1, 2, CV_64F, aa));

      b.push_back(Mat(1, 1, CV_64F, bb));
    }
    try {
      //x = Mat::ones(4, 1, CV_64F);
      x = Mat::ones(2, 1, CV_64F);
      //x = A.inv() * b;
      solve(A, b, x, DECOMP_SVD);
    } catch(Exception &e) {
      x.at<double>(0, 0) = x.at<double>(0, 1) = 0;
      //x.data[2] = 1;
     // x.data[3] = 0;
      const char *err_msg = e.what();
      printf("[OPEN_CV] %s\n", err_msg);
    }
    for (int ii = 0; ii < 2; ii++) {
      double val = x.at<double>(0, ii);
      if (ii == 0) a[i][ii] = val;
      else         a[i][ii] = val; //a[i][ii] = val < 0? 0: val;
    }
    a[i][2] = a[i][3] = 0;
    prev = range[1];
  }
#ifdef GLOBAL_COLOR_TEST
  for(int i = 0; i < segN; i++)
    printf("%d = %f %f\n", i, a[i][1], a[i][0]);
#endif //GLOBAL_COLOR_TEST
}

void _ApplyGlobalColor(Mat &src, Mat &ret, vector<vector<double> > &a) {

  ret = Mat::ones(src.rows, src.cols, CV_8U);
  for (int i = 0; i < src.cols; i++) 
    for (int j = 0; j < src.rows; j++) {
      int k = _GetRange(src.at<uchar>(j, i));
      double x = (double)src.at<uchar>(j, i);
      //if (i > 10 && j > 10) {
        //printf("pixel (%d) %f * %f + %f = %f => %hhu\n", k, a[k][1], x, a[k][0], a[k][1] * x + a[k][0], (uchar)(a[k][1] * x + a[k][0]));
      //}
      double v = (a[k][0] + a[k][1] * x + a[k][2] * pow(x, 2) + a[k][3] * pow(x, 3));
      ret.at<uchar>(j, i) = (uchar) (v < 0.9? 0 : (v > 255? 255 : v));
    } return;
}

int _GetRange(uchar val) {
  double size = 255.0 / segN;
  for (int i = 0; i < segN; i++) {
    if ((double) val <= size * ((double)(i + 1)) + 1e-9)  return i;
  } 
  return segN - 1;
}

#ifdef GLOBAL_COLOR_TEST
//testing
int main() {
  Mat src = imread("./image/src.png", CV_LOAD_IMAGE_COLOR);
  Mat ref = imread("./image/ref.png", CV_LOAD_IMAGE_COLOR);
  resize(src, src, Size(src.cols / 2, src.rows / 2));
  resize(ref, ref, Size(ref.cols / 2, ref.rows / 2));

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
#endif //GLOBAL_COLOR_TEST
