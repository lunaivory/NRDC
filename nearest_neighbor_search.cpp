#include "nearest_neighbor_search.hpp"
#include <vector>
#include <cmath>
#include <iostream>

// values from NRDC source code
static std::vector<float> gainMin = {0.2f, 1.0f, 1.0f, 0.5f};  // L, a, b, ||Dx+Dy||
static std::vector<float> gainMax = {1.0f, 1.0f, 1.0f, 2.0f};     // L, a, b, ||Dx+Dy||
static std::vector<int> biasMin = {-30, -20, -20, -0};    // L, a, b, ||Dx+Dy||
static std::vector<int> biasMax = {20, 20, 20, 0};        // L, a, b, ||Dx+Dy||
static std::vector<double> logScaleRange = {log(0.33f), log(3.0f)};
static std::vector<double> rotationRange = {(-190)*0.01745329252, 190 * 0.01745329252}; // Rotation range in radians

void nearest_neighbor_search(cv::Mat & src, cv::Mat & ref){
  Patch patches[src.rows-8][src.cols-8];
  FeatureVector features[src.rows][src.cols];

  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  // Image gradients for magnitude of luminance gradient
  cv::Mat gray, grad_x, grad_y, abs_grad_x, abs_grad_y, lab_img;
  cvtColor( src, gray, CV_RGB2GRAY );
  cvtColor( src, lab_img, CV_RGB2Lab );
  /// Gradient X
  cv::Scharr( gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT );
  // cv::convertScaleAbs( grad_x, abs_grad_x );
  /// Gradient Y
  cv::Scharr( gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT );
  // cv::convertScaleAbs( grad_y, abs_grad_y );

  for (int i = 0; i < gray.rows; ++i){
    uchar * x_pixel = abs_grad_x.ptr<uchar>(i);
    uchar * y_pixel = abs_grad_y.ptr<uchar>(i);
    cv::Vec3b * lab = lab_img.ptr<Vec3b>(i);
    for (int j = 0; j < gray.cols; ++j){
      FeatureVector f = {lab[j][0], lab[j][1], lab[j][2], sqrt((*x_pixel)*(*x_pixel) + (*y_pixel)*(*y_pixel))};
      features[i][j] = f;
    }
  }
}