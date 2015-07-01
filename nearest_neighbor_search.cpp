#include "nearest_neighbor_search.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <vector>
#include <random>
#include <iostream>

#ifndef MAX
#define MAX(x, y) ((x)>(y)?(x):(y))
#define MIN(x, y) ((x)<(y)?(x):(y))
#endif

int patch_w = 8;
int iterations = 5;
int rs_max = INT_MAX;
int center_x = 0;
int center_y = 0;

void _rotate(cv::Mat & src, cv::Mat & dst, double angle){

    cv::Mat r = cv::getRotationMatrix2D(cv::Point(center_x, center_y), angle, 1.0);

    cv::warpAffine(src, dst, r, src.size());
}

int _dist2(cv::Mat * a, cv::Mat * b, int ax, int ay, int bx, int by, int threshold=INT_MAX){
  int distance = 0;

  for (int dy = 0; dy < patch_w; ++dy){
    cv::Vec3b * a_row = a->ptr<cv::Vec3b>(ay+dy);
    cv::Vec3b * b_row = b->ptr<cv::Vec3b>(by+dy);
    for (int dx = 0; dx < patch_w; ++dx){
      int d_R = (a_row[ax + dx][0]) - (b_row[bx + dx][0]);
      int d_G = (a_row[ax + dx][1]) - (b_row[bx + dx][1]);
      int d_B = (a_row[ax + dx][2]) - (b_row[bx + dx][2]);
      distance += d_R*d_R + d_G*d_G + d_B*d_B;
    }
    if(distance >= threshold) return threshold;
  }
  return distance;
}

int _dist(cv::Mat * a, cv::Mat * b, int ax, int ay, int bx, int by, double & r_best, int threshold=INT_MAX){
  int best_dist = INT_MAX;
  cv::Mat b_rotated;
  double limit = 180.;
  double step = 30.;
  for (double theta = -limit; theta <= limit; theta += step){
    int distance = 0;

    _rotate((*b), b_rotated, theta);

    for (int dy = 0; dy < patch_w; ++dy){
      cv::Vec3b * a_row = a->ptr<cv::Vec3b>(ay+dy);
      cv::Vec3b * b_row = b_rotated.ptr<cv::Vec3b>(by+dy);
      for (int dx = 0; dx < patch_w; ++dx){
        if(a_row[ax + dx][0] == 0 && a_row[ax + dx][1] == 0 && a_row[ax + dx][2] == 0) continue;
        int d_R = (a_row[ax + dx][0]) - (b_row[bx + dx][0]);
        int d_G = (a_row[ax + dx][1]) - (b_row[bx + dx][1]);
        int d_B = (a_row[ax + dx][2]) - (b_row[bx + dx][2]);
        distance += d_R*d_R + d_G*d_G + d_B*d_B;
      }
      if(distance >= threshold) break;
    }

    if(distance < best_dist){
      best_dist = distance;
      r_best = theta;
    }

  }
  return best_dist;
}

void _improve_guess(cv::Mat * a, cv::Mat * b, int ax, int ay, int & x_best, int & y_best, int & d_best, int bx, int by, double & r_best) {
  // int d = _dist(a, b, ax, ay, bx, by, r_best, d_best);
  int d = _dist2(a, b, ax, ay, bx, by, d_best);
  if (d < d_best) {
    d_best = d;
    x_best = bx;
    y_best = by;
  }
}

cv::Mat calculate_transformation_matrix(double dx, double dy, double ang) {
  ang = ang / 180 * M_PI;
  cv::Mat ret = cv::Mat::eye(3, 3, CV_64F);
  double s = sin(ang), c = cos(ang);
  ret.at<double>(0, 0) = s;
  ret.at<double>(0, 1) = -c;
  ret.at<double>(0, 2) = dx;
  ret.at<double>(1, 0) = c;
  ret.at<double>(1, 1) = s;
  ret.at<double>(1, 2) = dy;

  return ret;
}

void nearest_neighbor_search(cv::Mat * a, cv::Mat * b, cv::Mat * &a_nn, cv::Mat * &a_nnd, std::vector<std::vector<cv::Mat>> &T){
  center_x = a->cols / 2.;
  center_y = a->rows / 2.;
  a_nn = new cv::Mat(a->rows, a->cols, CV_32SC2);

  a_nnd = new cv::Mat(a->rows, a->cols, CV_32SC1);
  cv::Mat * a_nnr = new cv::Mat(a->rows, a->cols, CV_64FC1);

  int aew = a->cols - patch_w+1, aeh = a->rows - patch_w+1;
  int bew = b->cols - patch_w+1, beh = b->rows - patch_w+1;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> dist1(0, bew), dist2(0, beh);
  generator.seed(1337);
  auto rand1 = std::bind(dist1, generator), rand2 = std::bind(dist2, generator);

  cv::namedWindow( "w1", cv::WINDOW_AUTOSIZE );
  cv::Mat rand_img(a->clone());
  // cv::Mat x_img(a->rows, a->cols, CV_8UC1);
  // cv::Mat y_img(a->rows, a->cols, CV_8UC1);

  std::cout << "Initializing values" << std::endl;
  for (int ay = 0; ay < aeh; ++ay){
    cv::Vec2i * a_nn_ptr = a_nn->ptr<cv::Vec2i>(ay);
    int * a_nnd_ptr = a_nnd->ptr<int>(ay);
    // uchar * x_ptr = x_img.ptr<uchar>(ay);
    // uchar * y_ptr = y_img.ptr<uchar>(ay);
    cv::Vec3b * rand_ptr = rand_img.ptr<cv::Vec3b>(ay);
    for (int ax = 0; ax < aew; ++ax){
      // randomize initial values
      int bx = rand1();
      int by = rand2();
      a_nn_ptr[ax][0] = bx;
      a_nn_ptr[ax][1] = by;
      a_nnd_ptr[ax] = _dist2(a, b, ax, ay, bx, by);
      // x_ptr[ax] = bx % 255;
      // y_ptr[ax] = by % 255;
      cv::Vec3b * b_ptr = b->ptr<cv::Vec3b>(by);
      rand_ptr[ax] = b_ptr[bx];
    }
  }

  cv::imshow("w1", rand_img);
  cv::waitKey(0);

  std::cout << "starting search" << std::endl;
  for (int iter = 0; iter < iterations; ++iter){
    std::cout << "Iteration: " << iter << std::endl;
    int y_start = 0, y_end = aeh, y_change = 1;
    int x_start = 0, x_end = aew, x_change = 1;
    if(iter % 2 == 0){
      y_start = y_end-1; y_end = -1; y_change = -1;
      x_start = x_end-1; x_end = -1; x_change = -1;
    }

    for (int ay = y_start; ay != y_end; ay += y_change){
      cv::Vec3b * rand_ptr = rand_img.ptr<cv::Vec3b>(ay);
      cv::Vec2i * v = a_nn->ptr<cv::Vec2i>(ay);
      int * d = a_nnd->ptr<int>(ay);
      double * r = a_nnr->ptr<double>(ay);

      for (int ax = x_start; ax != x_end; ax += x_change){
        // best guess so far
        int x_best = v[ax][0];
        int y_best = v[ax][1];
        int d_best = d[ax];
        double r_best = r[ax];

        // propagation: improve the current best guess by trying correspondences from left and above (right and down on even iterations)
        if((ax - x_change) >= 0 && (ax - x_change) < aew){
          cv::Vec2i v_prop = v[ax][ax-x_change];
          int x_prop = v_prop[0] + x_change;
          int y_prop = v_prop[1];
          if(x_prop >= 0 && x_prop < bew){
            _improve_guess(a,b,ax,ay,x_best,y_best,d_best,x_prop,y_prop, r_best);
          }
        }

        if((ay - y_change) >= 0 && (ay - y_change) < aeh){
          cv::Vec2i * v_prop = a_nn->ptr<cv::Vec2i>(ay-y_change);
          int x_prop = v_prop[ax][0];
          int y_prop = v_prop[ax][1] + y_change;
          if(y_prop >= 0 && y_prop < beh){
            _improve_guess(a,b,ax,ay,x_best,y_best,d_best,x_prop,y_prop, r_best);
          }
        }

        // Random Search
        int rs_start = rs_max;
        if(rs_start > MAX(b->cols, b->rows)) rs_start = MAX(b->cols, b->rows);
        for (int mag = rs_start; mag >= 1; mag /= 2){
          // sample window
          int x_min = MAX(x_best-mag, 0), x_max = MIN(x_best+mag+1, bew);
          int y_min = MAX(y_best-mag, 0), y_max = MIN(y_best+mag+1, beh);

          int x_p = x_min + (rand() % (x_max - x_min));
          int y_p = y_min + (rand() % (y_max - y_min));
          _improve_guess(a,b,ax,ay,x_best,y_best,d_best,x_p,y_p, r_best);
        }

        v[ax][0] = x_best;
        v[ax][1] = y_best;
        d[ax] = d_best;
        r[ax] = r_best;
        cv::Vec3b * b_ptr = b->ptr<cv::Vec3b>(y_best);
        rand_ptr[ax] = b_ptr[x_best];
      }
    }

    cv::imshow("w1", rand_img);
    cv::waitKey(0);
  }

  printf("a_nn %d %d, a %d %d\n", a_nn->rows, a_nn->cols, a->rows, a->cols);
  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      cv::Vec2i v = a_nn->at<cv::Vec2i>(i, j);
      double dx = (double)(j - v[0]);
      double dy = (double)(j - v[1]);
      double ang = 0;
      T[i][j] = calculate_transformation_matrix(dx, dy, ang);
    }
  }
}
