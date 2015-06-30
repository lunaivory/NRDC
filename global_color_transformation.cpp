
 //#define GLOBAL_COLOR_TEST

#include <cstring>
#include <cstdio>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <algorithm>
#include <utility>

using namespace std;
using namespace cv;

const int segN = 6;

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<pair<Point2d, Point2d> > &regions);
void _GetParameters(Mat &src, Mat &ref, vector<Point2d> &pt, vector<vector<double> > &a);
void _ApplyGlobalColor(Mat &src, Mat &ret, vector<vector<double> > &a);

void _GetSaturationPoints(Mat src, Mat ref, vector<pair<Point2d, Point2d> > &regions, Vec3f gray, vector<Point2f> &pt);
void _GetSaturationScale(vector<Point2f> &pt, double &s, double &ss, double &v);
void _ApplySaturationColor(Mat ret, double s, double ss, Vec3f gray);
int _GetRange(uchar val);

/**
 * pair<Point2d src, Point2d ref> >
 */
Mat GlobalColorTransformation(Mat src, Mat ref, vector<pair<Point2d, Point2d> > regions) {

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

  #ifdef GLOBAL_COLOR_TEST
  printf("RGB done\n");
  #endif

  double sUni, sYuv, vUni, vYuv, ssUni, ssYuv;
  Vec3f UNI(1.0/3, 1.0/3, 1.0/3), YUV(0.2989, 0.587, 0.144);
  vector<Point2f> ptUni, ptYuv;
  
  _GetSaturationPoints(ret, ref, regions, UNI, ptUni);
  _GetSaturationScale(ptUni, sUni, ssUni, vUni);
  
  _GetSaturationPoints(ret, ref, regions, YUV, ptYuv);
  _GetSaturationScale(ptYuv, sYuv, ssYuv, vYuv);
 
  #ifdef GLOBAL_COLOR_TEST
  printf("UNI Scale(%f +%f) = %f, YUV Scale(%f +%f) = %f\n", sUni, ssUni, vUni, sYuv, ssYuv, vYuv);
  #endif

  if (vUni < vYuv) _ApplySaturationColor(ret, sUni, ssUni, UNI);
  else             _ApplySaturationColor(ret, sYuv, ssYuv, YUV);
  
  return ret;
}

bool _pointSort(Point2d a, Point2d b) {
  return a.x == b.x ? a.y < b.y : a.x < b.x;
}

bool _point2fSort(Point2f a, Point2f b) {
  return a.x == b.x ? a.y < b.y : a.x < b.x;
}

void _GetPointsInside(Mat &src, Mat &ref, vector<Point2d> &pt, vector<pair<Point2d, Point2d> > &regions) {
  for (int i = 0; i < regions.size(); i++) {
    int x = regions[i].first.x, y = regions[i].first.y;
    int xp = regions[i].second.x, yp = regions[i].second.y;
    Point2d val((int)src.at<uchar>(y, x), (int)ref.at<uchar>(yp, xp));
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
    // int cnt = 0;
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

void _GetSaturationPoints(Mat src, Mat ref, vector<pair<Point2d, Point2d> > &regions, Vec3f gray, vector<Point2f> &pt) {
  for (int i = 0; i < regions.size(); i++) {
    int x = regions[i].first.x, y = regions[i].first.y;
    double v = gray.dot(src.at<Vec3b>(y, x));

    int xp = regions[i].second.x, yp = regions[i].second.y;
    double vp = gray.dot(ref.at<Vec3b>(yp, xp));
    
    pt.push_back(Point2f(v, vp));
  }
  sort(pt.begin(), pt.end(), _point2fSort);
}

void _GetSaturationScale(vector<Point2f> &pt, double &s, double &ss, double &v) {
  // can use trinary search
  Mat A, b, x = Mat::ones(2, 1, CV_64F);
  for (int i = 0; i < pt.size(); i++) {
    double aa[2] = {1, pt[i].x}, bb[1] = {pt[i].y};
    A.push_back(Mat(1, 2, CV_64F, aa));
    b.push_back(Mat(1, 1, CV_64F, bb));
  }
  try {
    solve(A, b, x, DECOMP_SVD);
  } catch (Exception &e) {
    const char *err_msg= e.what();
    printf("[OPEN_CV2] %s\n", err_msg);
  }
  
  cout << x << endl;
  ss = x.at<double>(0, 0);
  s = x.at<double>(1, 0);
  v = 0;
  for (int i = 0; i < pt.size(); i++) {
    v += pow((s * pt[i].x + ss - pt[i].y), 2);
  }
}

void _ApplySaturationColor(Mat ret, double s, double ss, Vec3f gray) {
//  double data[3][3] = {{s - gray[0], gray[0], gray[0]},
//                       {gray[1], s - gray[1], gray[1]},
//                       {gray[2], gray[2], s - gray[2]} };

  for (int i = 0; i < ret.cols; i++)
    for (int j = 0; j < ret.rows; j++) {
      Vec3b orig = ret.at<Vec3b>(j, i);
      for (int c = 0; c < 3; c++) {
        double val = s * orig[c] + ss;
        ret.at<Vec3b>(j, i)[c] = (uchar) (val < 0.9 ? 0 : (val > 254? 255 : val));
      }
    }
      
}



#ifdef GLOBAL_COLOR_TEST
//testing
int main() {
  Mat src = imread("./image/src.png", CV_LOAD_IMAGE_COLOR);
  Mat ref = imread("./image/ref.png", CV_LOAD_IMAGE_COLOR);
  //resize(src, src, Size(src.cols / 2, src.rows / 2));
  //resize(ref, ref, Size(ref.cols / 2, ref.rows / 2));

  vector<pair<Point2d, Point2d> > pt;
  for (int i = 0 ; i < src.cols; i++)
    for (int j = 0; j < src.rows; j++) {
      pt.push_back(make_pair(Point2d(i, j), Point2d(i, j)));
    }
  Mat ret = GlobalColorTransformation(src, ref, pt);

  namedWindow("final", 1);
  imshow("final", ret);
  waitKey(0);
}
#endif //GLOBAL_COLOR_TEST
