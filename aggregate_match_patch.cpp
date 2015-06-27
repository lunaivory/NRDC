#include <cstdio>
#include <opencv2/opencv>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;

double parLocal = 3;
double parGlobal = 0.8;
double parRatio = 0.5;
double parSize = 500;
double parSmall2 = 8 * 8;
double parLarge2 = 64 * 64;


void AggregateMatchPatch(Size sz, Mat T, vector<Point2d> &region) {
  
  patches.clear();
  
  _CreatePatchSet(sz, T, region);
}

//other functions

void _CreatePatchSet(Size sz, vector<Mat> T, vector<Point2d> &region) {
  
  vector<int> edge[sz.width * sz.height];

  int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  for (int i = 1; i < sz.width - 1; i++)
    for (int j = 1; j < sz.height - 1; j++) {
      if (i % 2 == 0 || j % 2 == 0) continue;
      
      Point3f u = Point3f(i, j, 1);
      
      for (int k = 0; k < 8; k++) {
      
        int ii = i + dx[k], jj = j + dy[k];
        
        if (ii <= 0 || jj<= 0 || ii>= sz.width || jj >= sz.height)  continue;
        
        if (_Consistent(u, Point3f(ii, jj, 1), Tu, Tv, true))
          edge[i + j * sz.width].push_back(ii + jj * sz.width);
      }
    }

  //join sets and analyze
  bool added[sz.height][sz.width] = {false};
  vector<set<Point2d> > region;
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
        set.insert(Point2d(now % sz.width, now / sz.width));
        for(int k = 0; k < edge[now].size(); k++) {
          int ii = edge[now][k] % sz.width;
          int jj = edge[now][k] / sz.width;
          q.push(edge[now][k]); added[jj][ii] = true;
        }
      }

      //eliminate small regions
      if (set.size() < parSize) continue;

      //get sample of size sqrt(N) from region Z
      vector<int> in;
      int mid = set.size() / 2;
      for (int num = 0; num < set.size(); num++) {
        double dist2 = (set[mid] - set[num]).dot(set[mid] - set[num]);
        if (dist2 >= parSmall2 && dist2 <= parLarge2) in.push_back(num);
      }

      //get global consistency of region Z
      int cnt = 0;
      for (int ii = 0; ii < in.size(); ii++)
        for (int jj = ii + 1; jj < in.size(); jj++) {
          Point3f u((double)set[ii].x, (double)set[ii].y, 1);
          Point3f v((double)set[jj].x, (double)set[jj].y, 1);
          if (!_consistent(u, v, Tu, Tv, false))  cnt++;
        }
      
      //add set if 
      if ((double)cnt / in.size() < parRatio) {
        for (int last = 0; last < set.size(); last++)
          region.push_back(set[last]);
      }
    }
  return;
}

bool _Consistent(Point3f u, Point3f v, Mat Tu, Mat Tv, bool local) {
  Point3f vv = Tv * v - Tu * v;
  Point3f uv = Tu * u - Tu * v;
  return (vv.dot(vv) / uv.dot(uv)) < (local ? parLocal : parGlobal);
}

