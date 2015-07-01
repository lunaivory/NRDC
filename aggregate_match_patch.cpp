//#define _AGGREGATE_MATCH_PATCH_TEST
#define _SHOW_AMP_IMG

#include <cstdio>
#include <opencv2/opencv.hpp>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>

using namespace std;
using namespace cv;

double parLocal = 3;
double parGlobal = 0.8;
double parRatio = 0.5;
double parSize = 50;
double parSmall2 = 2 * 2;
double parLarge2 = 16 * 16;

void _CreatePatchSet(Size sz, Mat match, vector<vector<Mat> > T, vector<pair<Point2d, Point2d> > &region);
bool _HaveMatch(Vec2i pt, Size sz);
bool _Consistent(Mat u, Mat v, Mat Tu, Mat Tv, bool local);
bool _Compare(Point2d a, Point2d b);

void AggregateMatchPatch(Size sz, Mat match, vector<vector <Mat> > T, vector<pair<Point2d, Point2d> > &region) {
  
  printf("AGR T: %lu %lu, Mat: %d %d\n", T.size(), T[0].size(), match.rows, match.cols);

  region.clear();
  
  #ifdef _SHOW_AMP_IMG


  Mat dSrc= Mat::zeros(sz, CV_8U), dRef = Mat::zeros(sz, CV_8U);
  Mat empty = Mat::zeros(sz, CV_8U), imgShow;
  vector<Mat> show;
  
  for (int i = 0; i < sz.width; i++) {
    for (int j = 0; j < sz.height; j++) {
      Vec2i p = match.at<Vec2i>(j, i);
      if (!_HaveMatch(p, sz)) continue;
      dSrc.at<uchar>(j, i) = (uchar)253;
      dRef.at<uchar>(p[1], p[0]) = (uchar)253;
    }
  }
 
  imshow("src", dSrc), imshow("ref", dRef);
  
  waitKey(0);
  #endif //_SHOW_AMP_IMG

  _CreatePatchSet(sz, match, T, region); 

  #ifdef _SHOW_AMP_IMG 
  dSrc = Mat::zeros(sz, CV_8U), dRef = Mat::zeros(sz, CV_8U); 
  printf("Found %lu pairs\n", region.size());
  for (int i = 0; i < region.size(); i++) { 
    Point2d ps = region[i].first, pr = region[i].second; 
    dSrc.at<uchar>(ps.y, ps.x) = 254;
    dRef.at<uchar>(pr.y, pr.x) = 254;
  }

  imshow("src", dSrc), imshow("ref", dRef);

  waitKey(0);
  #endif //_SHOW_AMP_IMG
}

//other functions

void _CreatePatchSet(Size sz, Mat match, vector<vector<Mat> > T, vector<pair<Point2d, Point2d> > &region) {

  const int px = sz.width * sz.height;
  vector<vector<int> > edge(px, vector<int>());

  int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  for (int i = 1; i < sz.width - 1; i++)
    for (int j = 1; j < sz.height - 1; j++) {
      if (!_HaveMatch(match.at<Vec2i>(j, i), sz)) continue;

      double uData[3] = {(double)i, (double)j, 1};
      Mat u(3, 1, CV_64F, uData);

      for (int k = 0; k < 8; k++) {

        int ii = i + dx[k], jj = j + dy[k];

        if (ii < 0 || jj< 0 || ii>= sz.width || jj >= sz.height)  continue;
        if (!_HaveMatch(match.at<Vec2i>(jj, ii), sz))  continue;

        double vData[3] = {(double)ii, (double)jj, 1};
        Mat v(3, 1, CV_64F, vData);
        if (_Consistent(u, v, T[j][i], T[jj][ii], true))
          edge[i + j * sz.width].push_back(ii + jj * sz.width);
      }
    }

  //join sets and analyze
  vector< vector<bool> > added(sz.height, vector<bool>(sz.width, false));
  for (int i = 0; i < sz.width; i++)
    for (int j = 0; j < sz.height; j++) {
      if (added[j][i]) continue; // will be true if in a set already

      int itr = i + j * sz.width;

      if (edge[itr].size() == 0)  continue;

      //find out the connected region using BFS
      queue<int> q;
      vector<Point2d> set;

      q.push(itr); added[j][i] = true;

      while(!q.empty()) {
        int now = q.front();  q.pop();
        set.push_back(Point2d(now % sz.width, now / sz.width));
        for(int k = 0; k < edge[now].size(); k++) {
          int ii = edge[now][k] % sz.width;
          int jj = edge[now][k] / sz.width;
          if (added[jj][ii]) continue;
          q.push(edge[now][k]); added[jj][ii] = true;
        }
      }

      //eliminate small regions
//      printf("size connected %lu\n", set.size());
      if (set.size() < parSize) continue;

      //get sample of size sqrt(N) from region Z
      sort(set.begin(), set.end(), _Compare);
      vector<int> in;
      int mid = set.size() / 2;
      for (int num = 0; num < set.size(); num++) {
        double dist2 = (set[mid] - set[num]).dot(set[mid] - set[num]);
        if (dist2 >= parSmall2 && dist2 <= parLarge2) in.push_back(num);
      }

      #ifdef _SHOW_AMP_IMG
      printf("Sample Z size from %lu to %lu\n", set.size(), in.size());
      #endif

      //get global consistency of region Z
      int cnt = 0;
      for (int ii = 0; ii < in.size(); ii++)
        for (int jj = ii + 1; jj < in.size(); jj++) {
          int ux = set[ii].x, uy = set[ii].y;
          int vx = set[jj].x, vy = set[jj].y;
          double uData[3] = {(double)ux, (double)uy, 1.0};
          double vData[3] = {(double)vx, (double)vy, 1.0};
          Mat u(3, 1, CV_64F, uData);
          Mat v(3, 1, CV_64F, vData);
          if (!_Consistent(u, v, T[uy][ux], T[vy][vx], false))
            cnt++;
        }

      //add set if
      printf("ratio %f\n", (double)cnt / in.size() / in.size());
      if ((double)cnt / in.size() / in.size() < parRatio) {
        for (int last = 0; last < set.size(); last++) {
          int x = set[last].x, y = set[last].y;
          Point2d ss(x, y), rr(match.at<Vec2i>(y, x)[0], match.at<Vec2i>(y, x)[1]);
          region.push_back(make_pair(ss, rr));
        }
      }
    }
  return;
}

bool _HaveMatch(Vec2i pt, Size sz) {
  return (pt[0] > 0 && pt[1] > 0 && pt[0] < sz.width && pt[1] < sz.height);
}

bool _Consistent(Mat u, Mat v, Mat Tu, Mat Tv, bool local) {

  Mat vv = Tv * v - Tu * v;
  Mat uv = Tu * u - Tu * v;
  return (vv.dot(vv) / uv.dot(uv)) < (local ? parLocal : parGlobal);
}

bool _Compare(Point2d a, Point2d b) {
  return a.x == b.x? (a.y < b.y) : (a.x < b.x);
}

#ifdef _AGGREGATE_MATCH_PATCH_TEST
Size _sz(300, 200);
Mat _match = Mat::zeros(_sz.height, _sz.width, CV_32SC2);
vector<vector<Mat> > _T(_sz.height, vector<Mat>(_sz.width));
vector<pair<Point2d, Point2d> > _region;

void _addPoint(int x0, int x1, int y0, int y1, int dx, int dy);

int main() {
  _addPoint(10, 100, 20, 100, 1, 1);
  _addPoint(150, 180, 20, 50, 10, -10);
  _addPoint(240, 260, 180, 190, -10, -29);
  _addPoint(240, 260, 170, 179, -200, 0);
  printf("Aggregation Starts\n");
  AggregateMatchPatch(_sz, _match, _T, _region);
}

void _addPoint(int x0, int x1, int y0, int y1, int dx, int dy) {
  for (int i = x0; i < x1; i++)
    for (int j = y0; j < y1; j++) {
//      printf("%d %d\n", j, i);
      _match.at<Vec2i>(j, i)[0] = dx + i;
      _match.at<Vec2i>(j, i)[1] = dy + j;

      _T[j][i] = Mat::eye(3, 3, CV_64F);
      _T[j][i].at<double>(0, 2) = dx + rand() % 4;
      _T[j][i].at<double>(1, 2) = dy + rand() % 4;
    }
}
#endif
