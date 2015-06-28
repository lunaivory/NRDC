#ifndef NNS_H_
#define NNS_H_

struct Patch{
  int x, y;
  float bias, gain;
  double scale, rotation;
  // Lab-color space values
  int L;
  double a, b;
  // luminance gradiant magnitude
  double lg_magnitude;
};

void nearest_neighbor_search(cv::Mat src, cv::Mat ref);

#endif
