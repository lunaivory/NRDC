#ifndef NNS_H_
#define NNS_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct Patch{
  // coordinates
  cv::Point p;
  // color bias and gain (?)
  float bias, gain;
  // patch scale and rotation
  double scale, rotation;
};

struct FeatureVector{
  // Lab-color space values
  int L;
  double a, b;
  // luminance gradiant magnitude
  double lg_magnitude;
};

void nearest_neighbor_search(cv::Mat & src, cv::Mat & ref);

#endif
