#include "nearest_neighbor_search.hpp"
#include <vector>
#include <cmath>

// values from NRDC source code
static std::vector<float> gainMin = {0.2f, 1.0f, 1.0f, 0.5f};  // L, a, b, ||Dx+Dy||
static std::vector<float> gainMax = {1.0f, 1.0f, 1.0f, 2.0f};     // L, a, b, ||Dx+Dy||
static std::vector<int> biasMin = {-30, -20, -20, -0};    // L, a, b, ||Dx+Dy||
static std::vector<int> biasMax = {20, 20, 20, 0};        // L, a, b, ||Dx+Dy||
static std::vector<float> logScaleRange = {log(0.33f), log(3.0f)};
static std::vector<float> rotationRange = {(-190) * 0.01745329252d, 190 * 0.01745329252d}; // Rotation range in radians


void nearest_neighbor_search(cv::Mat src, cv::Mat ref){

}