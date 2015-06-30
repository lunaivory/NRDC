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
  int d = _dist(a, b, ax, ay, bx, by, r_best, d_best);
  // int d = _dist2(a, b, ax, ay, bx, by, d_best);
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

  for (int i = 0; i < a->rows; i++){
    cv::Vec2i v = a_nn->ptr<cv::Vec2i>(i);
    for (int j = 0; j < a->cols; j++) {
      double dx = (j - v[j][0]);
      double dy = (i - v[j][1]);
      double ang = 0;
      T[j][i] = calculate_transformation_matrix(dx, dy, ang);
    }
  }
}

/*
// overlapping patches ox size 8x8 pixels
cv::Size patch_size(dimension, dimension);
std::vector<std::vector<Patch>> src_patches;
std::vector<std::vector<Patch>> ref_patches;
std::vector<std::vector<FeatureVector>> src_features;
std::vector<std::vector<FeatureVector>> ref_features;

src_patches.reserve(src.cols-8);

int scale = 1;
int delta = 0;
int ddepth = CV_16S;

// Image gradients for magnitude of luminance gradient
cv::Mat src_gray, src_lab, src_grad_x, src_grad_y, src_abs_grad_x, src_abs_grad_y, src_grad;
cv::Mat ref_gray, ref_lab, ref_grad_x, ref_grad_y, ref_abs_grad_x, ref_abs_grad_y, ref_grad;

// src
cvtColor( src, src_gray, CV_RGB2GRAY );
cvtColor( src, src_lab, CV_RGB2Lab );
// Gradient X & Y
cv::Scharr( src_gray, src_grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
cv::Scharr( src_gray, src_grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
// approximate for gradient
cv::convertScaleAbs( src_grad_x, src_abs_grad_x );
cv::convertScaleAbs( src_grad_y, src_abs_grad_y );
addWeighted( src_abs_grad_x, 0.5, src_abs_grad_y, 0.5, 0, src_grad );

// ref
cvtColor( ref, ref_gray, CV_RGB2GRAY );
cvtColor( ref, ref_lab, CV_RGB2Lab );
// Gradient X & Y
cv::Scharr( ref_gray, ref_grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
cv::Scharr( ref_gray, ref_grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
// approximate for gradient
cv::convertScaleAbs( ref_grad_x, ref_abs_grad_x );
cv::convertScaleAbs( ref_grad_y, ref_abs_grad_y );
addWeighted( ref_abs_grad_x, 0.5, ref_abs_grad_y, 0.5, 0, ref_grad );

std::cerr << "Starting for-loop" << std::endl;
for (int i = 0; i < src.rows; ++i){
  double * src_grad_ptr = src_grad.ptr<double>(i);
  double * ref_grad_ptr = ref_grad.ptr<double>(i);

  cv::Vec3b * src_lab1 = src_lab.ptr<cv::Vec3b>(i);
  cv::Vec3f * src_lab2 = src_lab.ptr<cv::Vec3f>(i);

  cv::Vec3b * ref_lab1 = ref_lab.ptr<cv::Vec3b>(i);
  cv::Vec3f * ref_lab2 = ref_lab.ptr<cv::Vec3f>(i);

  std::vector<FeatureVector> src_row_features;
  std::vector<FeatureVector> ref_row_features;
  std::vector<Patch> src_row_patches;
  std::vector<Patch> ref_row_patches;

  if(i < src.rows-dimension){
    src_row_patches.reserve(src.rows-dimension);
    ref_row_patches.reserve(ref.rows-dimension);
  }
  for (int j = 0; j < src.cols; ++j){
    if(i < src.rows-dimension && j < src.cols-dimension){
      // 8x8 patch for every pixel
      cv::Rect rect = cv::Rect(j,i, patch_size.width, patch_size.height);
      struct Patch src_patch = {cv::Point(j+4,i+4), cv::Mat(src, rect).clone()};
      struct Patch ref_patch = {cv::Point(j+4,i+4), cv::Mat(ref, rect).clone()};
      src_row_patches.push_back(src_patch);
      ref_row_patches.push_back(ref_patch);
    }

    FeatureVector src_feature = {src_lab1[j][0], src_lab2[j][1], src_lab2[j][2], (*src_grad_ptr)};
    FeatureVector ref_feature = {ref_lab1[j][0], ref_lab2[j][1], ref_lab2[j][2], (*ref_grad_ptr)};
    src_row_features.push_back(src_feature);
    ref_row_features.push_back(ref_feature);
  }
  if(i < src.rows-dimension){
    src_patches.push_back(src_row_patches);
    ref_patches.push_back(ref_row_patches);
  }
  src_features.push_back(src_row_features);
  ref_features.push_back(ref_row_features);
}

std::cout << "done" << std::endl;

// TODO
template <size_t rows, size_t cols>
void calc_bias_gain(Patch & src_patch, Patch & ref_patch, FeatureVector (&features)[rows][cols]){
  CvScalar src_mean, src_var, ref_mean, ref_var;
  cvAvgSdv(&src_patch.patch_img, &src_mean, &src_var);
  cvAvgSdv(&ref_patch.patch_img, &ref_mean, &ref_var);
  CvScalar src_std_dev(sqrt(src_var.val[0]), sqrt(src_var.val[1]), sqrt(src_var.val[2]));
  CvScalar ref_std_dev(sqrt(ref_var.val[0]), sqrt(ref_var.val[1]), sqrt(ref_var.val[2]));

  // gain
  // src_patch.L_gain = gain(src_std_dev.val[0], ref_std_dev.val[0]);
  // src_patch.L_gain = (src_patch.L_gain < gainMin[0] ? gainMin[0] : (src_patch.L_gain > gainMax[0] ? gainMax[0] : src_patch.L_gain);
  // src_patch.a_gain = 1.0;
  // src_patch.b_gain = 1.0;
  // // double mag_gain = gain();

  // // bias
  // src_patch.L_bias = bias(src_mean.val[0], src_patch.L_gain, ref_mean.val[0]);
  // src_patch.a_bias = 1.0;
  // src_patch.b_bias = 1.0;
  // src_patch.mag_bias = 0.0;

}

// double gain(double src_dev, double ref_dev){
//   return (double)(src_dev / ref_dev);
// }
// double bias(double src_mean, double src_gain, double ref_mean){
//   return src_mean - (src_gain * ref_mean);
// }

typedef unsigned int uint;
// values from NRDC source code
// static std::vector<float> gainMin = {0.2f, 1.0f, 1.0f, 0.5f};  // L, a, b, ||Dx+Dy||
// static std::vector<float> gainMax = {1.0f, 1.0f, 1.0f, 2.0f};     // L, a, b, ||Dx+Dy||
// static std::vector<int> biasMin = {-30, -20, -20, -0};    // L, a, b, ||Dx+Dy||
// static std::vector<int> biasMax = {20, 20, 20, 0};        // L, a, b, ||Dx+Dy||
// static std::vector<double> logScaleRange = {log(0.33f), log(3.0f)};
// static std::vector<double> rotationRange = {(-190)*0.01745329252, 190 * 0.01745329252}; // Rotation range in radians

*/
