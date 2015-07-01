#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;
using namespace cv;

const int ITER_NUM = 1;
const double INF = 1e20;
int MAX_RADIUS;
double ALPHA = 0.5;
int P = 4; //patchsize = P * 2 + 1;


void _Init(Mat &src, Mat &ref, Mat &match);
void Propogation(int x, int y, int dx, int dy, Mat &src, Mat &ref, Mat &match, vector<vector<double> > &mins);
void RandomSearch(int x, int y, int dx, int dy, Mat &src, Mat &ref, Mat &match, vector<vector<double> > &mins);
bool GetDistance(Mat &src, Mat &ref, int x, int y, int vx, int vy, double &val);
void ShowImage(Mat &ref, Mat &match);
void MakeTransform(Mat &src, Mat &match, vector<vector<Mat> > &T);

void NNS(Mat src, Mat ref, Mat &match, vector<vector<Mat> > &T) {

  _Init(src, ref, match);
  
  vector<vector<double> > mins(src.rows, vector<double>(src.cols, INF));

  for (int itr = 0; itr < ITER_NUM; itr++) {
    printf("INTERATION %d\n", itr);

    for (int i = P + 1; i < src.cols - P; i++) {
      for (int j = P + 1; j < src.rows - P; j++) {
        double val = mins[j][i];
        int posX, posY;
        int d = 1;
        
        Propogation(i, j, d, 0, src, ref, match, mins);
        
        Propogation(i, j, d, d, src, ref, match, mins);

        Propogation(i, j, 0, d, src, ref, match, mins);
        RandomSearch(i, j, 0, 0, src, ref, match, mins);

      }
    }
    
    for (int i = src.cols - P - 1; i >= P + 1; i--) {
      for (int j = src.rows - P - 1; j >= P + 1; j--) {
        double val = mins[j][i];
        int posX, posY;
        int d = -1;
        
        Propogation(i, j, d, 0, src, ref, match, mins);
        
        Propogation(i, j, d, d, src, ref, match, mins);

        Propogation(i, j, 0, d, src, ref, match, mins);
        RandomSearch(i, j, 0, 0, src, ref, match, mins);

      }
    }
    ShowImage(ref, match);
  }
  MakeTransform(src, match, T);

  cvtColor(src, src, CV_Lab2BGR);
  cvtColor(ref, ref, CV_Lab2BGR);
}

void MakeTransform(Mat &src, Mat &match, vector<vector<Mat> > &T) {
  for (int i = 0; i < src.cols; i++) {
    for (int j = 0; j < src.rows; j++) {
      T[j][i] = Mat::eye(3, 3, CV_64F);
      T[j][i].at<double>(0, 2) = match.at<Vec2i>(j, i)[0] - i;
      T[j][i].at<double>(1, 2) = match.at<Vec2i>(j, i)[1] - j;
    }
  }
}

void _Init(Mat &src, Mat &ref, Mat &match) {
  MAX_RADIUS = max(src.rows, src.cols);
  cvtColor(src, src, CV_BGR2Lab);
  cvtColor(ref, ref, CV_BGR2Lab);
  match = Mat::zeros(src.rows, src.cols, CV_32SC2);
  
  int all = src.cols * src.rows;

  for (int i = 0; i < src.cols; i++) {
    for (int j = 0; j < src.rows; j++) {
      int pos = rand() % all;
      match.at<Vec2i>(j, i)[0] = pos % src.cols;
      match.at<Vec2i>(j, i)[1] = pos / src.cols;
    }
  }
}

void Propogation(int x, int y, int dx, int dy, Mat &src, Mat &ref, Mat &match, vector<vector<double> > &mins) {
  int vx = match.at<Vec2i>(y + dy, x + dx)[0] - (x + dx);
  int vy = match.at<Vec2i>(y + dy, x + dx)[1] - (y + dy);
  double val;
  if (!GetDistance(src, ref, x, y, vx, vy, val)) return;
  if (val < mins[y][x]) {
    mins[y][x] = val;
    match.at<Vec2i>(y, x)[0] = x + vx;
    match.at<Vec2i>(y, x)[1] = y + vy;
  }
}


void RandomSearch(int x, int y, int dx, int dy, Mat &src, Mat &ref, Mat &match, vector<vector<double> > &mins) {
  double ratio = MAX_RADIUS;

  while (ratio > 1.5) {
    int vx = (int) ( ((double) rand() / (double)RAND_MAX - 0.5 ) * ratio );
    int vy = (int) ( ((double) rand() / (double)RAND_MAX - 0.5 ) * ratio );
    double val;
    ratio *= ALPHA;

    if (!GetDistance(src, ref, x, y, vx, vy, val)) continue;
    if (val < mins[y][x]) {
      mins[y][x] = val;
      match.at<Vec2i>(y, x)[0] = x + vx;
      match.at<Vec2i>(y, x)[1] = y + vy;
    }
  }
}

bool GetDistance(Mat &src, Mat &ref, int x, int y, int vx, int vy, double &val) {
  double tmp = 0;
  for (int i = -P; i < P; i++)
    for (int j = -P; j < P; j++) {
      int rx = x + vx + i, ry = y + vy + j;
      int sx = x + i, sy = y + j;
      if (rx >= src.cols || ry >= src.rows || rx < 0 || ry < 0) return false;
      Vec3b s = src.at<Vec3b>(sy, sx);
      Vec3b r = ref.at<Vec3b>(ry, rx);
      tmp += pow((double)s[0] - (double)r[0], 2) + pow((double)s[1] - (double)r[1], 2) + pow((double)s[2] - (double)r[2], 2);
    }
  val = tmp;
  return true;
}

void ShowImage(Mat &ref, Mat &match) {
  Mat img = Mat::zeros(ref.rows, ref.cols, CV_8UC3);
  
  for (int i = 0; i < img.cols; i++) 
    for (int j = 0; j < img.rows; j++) {
      int x = match.at<Vec2i>(j, i)[0];
      int y = match.at<Vec2i>(j, i)[1];
      img.at<Vec3b>(j, i) = ref.at<Vec3b>(y, x);
    }
  
  cvtColor(img, img, CV_Lab2BGR);

  namedWindow("result");
  imshow("result", img);
  //waitKey(0);
}

#ifdef TEST
int main() {
  Mat src = imread("./src.png", CV_LOAD_IMAGE_COLOR);
  Mat ref = imread("./ref.png", CV_LOAD_IMAGE_COLOR);

  Size sz(src.cols / 2, src.rows / 2);
  resize(src, src, sz);
  resize(ref, ref, sz);

  Mat _match;
  vector<vector<Mat> > _T;
  
  NNS(src, ref, _match, _T);
}
#endif
