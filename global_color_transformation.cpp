#include <cstdio>
#include <opencv2/opencv>
#include <vector>

using namespace std;
using namespace cv;

const int segN = 6;

Mat GlobalColorTransformation(Mat src, Mat ref, vector<Point2d> regions) {
  double par[segN][4];

  Mat rgbSrc[3], rgbRef[3], rgbRet[3];
  vector<Point2d> pt[3];
  split(src, rgbSrc); //BGR
  split(ref, rgbRef);

  _GetPointsInside(rgbSrc[0], rgbSrc[0], pt, regions);
  _GetParameters(rgbSrc[0], rgbRef[0], pt, par[][4]);
  _ApplyGlobalColor(rgbSrc[0], rgbRet[0], par[][4]);

  _GetPointsInside(rgbSrc[1], rgbSrc[1], pt, regions);
  _GetParameters(rgbSrc[1], rgbRef[1], pt, par[][4]);
  _ApplyGlobalColor(rgbSrc[1], rgbRet[1], par[][4]);

  _GetPointsInside(rgbSrc[2], rgbSrc[2], pt, regions);
  _GetParameters(rgbSrc[2], rgbRef[2], pt, par[][4]);
  _ApplyGlobalColor(rgbSrc[2], rgbRet[2], par[][4]);

  Mat ret;
  merge(rgbRet, ret);
  return ret;
}

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<Point2d> &regions) {
  for (int i = 0; i < regions.size(); i++) {
    int x = regions[i].x, y = regions[i].y;
    //TODO yy xx cause src and ref have different coordinate!
    Point2d val((int)src.at<uchar>(y, x), (int)ref.at<uchar>(y, x));
    pt.push_back(val);
  }

  sort(pt.begin(), pt.end());

  return;
}

void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, double *a[][4]) {
  
  double size = 255.0 / segN;

  int itr = 0;
  double prev = 0.0;
  for (int i = 0; i < segN; i++) {
    double range[2] = {prev, (i == segN - 1) ? 255 : (prev + size)};
    
    Mat A, x = ones(1, 3, CV_64F), b;

    for (; itr < pt.size() && pt[itr].x < range[1]; itr++) {
      double x = pt[itr].x, y = pt[itr].y;
      A.push_back(Mat(1, 4, CV_64F, {pow(x, 3), pow(x, 2), x, 1}));
      b.push_back(Mat(1, 1, CV_64F, {y}));
    }
    
    solve(A, b, x);

    for (int ii = 0; ii < 4; ii++)
      a[i][ii] = x.at(ii);
  }

}

void _ApplyGlobalColor(Mat &src, Mat &ret, double *a[][4]) {
  ret = ones(src.height, src.width, CV_8U);

  for (int i = 0; i < src.width; i++) 
    for (int j = 0; j < src.height; j++) {
      int k = _GetRange(src.at(j, i));
      double x = (double)src.at(j, i);
      ret.at(j, i) = (uchar)(a[k][0] + a[k][1] * x + a[k][2] * pow(x, 2) + a[k][3] * pow(x, 3));
    }
  return;
}

int _GetRange(uchar val) {
  double size = 255.0 / segN;
  for (int i = 0; i < segN; i++) {
    if ((double) val <= size * ((double)(i + 1)) + 1e-9)  return i;
  }
}

