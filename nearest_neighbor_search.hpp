#ifndef NNS_H_
#define NNS_H_

#include <opencv2/imgproc/imgproc.hpp>

struct Patch{
  // coordinates
  // cv::Point2i point;
  // cv::Mat patch_img;
  // patch scale and rotation
  double scale, rotation;


};

struct FeatureVector{
  // Lab-color space values
  int L;
  double a;
  double b;
  // luminance gradiant magnitude
  double lg_magnitude;

  // TODO: bias and gain
  // double L_bias, L_gain,a_bias, a_gain, b_bias, b_gain, mag_bias, mag_gain;
};

void nearest_neighbor_search(cv::Mat * a, cv::Mat * b, cv::Mat * &a_nn, cv::Mat * &a_nnd);
int dist(cv::Mat * a, cv::Mat * b, int ax, int ay, int bx, int by, int threshold);

// double bias(double src_mean, double gain, double ref_mean);
// double gain(double src_dev, double ref_dev);

// void calc_bias_gain(cv::Mat & patch);

#endif
