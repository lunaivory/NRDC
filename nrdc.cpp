#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include "nearest_neighbor_search.hpp"
#include "aggregate_match_patch.hpp"
#include "global_color_transformation.hpp"
// #include "narrow_seaerch_range.hpp"

using namespace cv;

const static std::string source_path = "src.png";
const static std::string reference_path = "ref.png";

void load_images(Mat & source, Mat & reference){
  bool error = false;
  std::cerr << "Loading images..." << std::endl;

  source = imread(source_path, CV_LOAD_IMAGE_COLOR);
  reference = imread(reference_path, CV_LOAD_IMAGE_COLOR);
  if(! (source.data && reference.data) ){ // Check for invalid input
    std::cout <<  "Could not open or find the image" << std::endl;
    error = true;
  }
  if(!error) std::cerr << "Images loaded" << std::endl;
  else std::cerr << "An error occurred";
}

Mat nrdc(Mat src, Mat ref) {
  Mat srcS, refS;

  for (int i = (1<<6); i > 0; i >>= 1) {
    // Size srcSz = src.size();
    // Size refSz = ref.size();

    // srcSz.height /= i, srcSz.width /= i;
    // refSz.height /= i, srcSz.width /= i;

    // resize(src, srcS, srcSz);
    // resize(ref, refS, refSz);

    // nearest_neighbor_search(srcS, refS);

    // AggregateMatchPatch();

    // GlobalColorTransformation();

    // NarrowSearchRange();
  }
  return refS;
}

int main(int argc, char const *argv[])
{
  Mat a, b, a_nn, a_nnd;

  load_images(a, b);

  Mat * a_nn_ptr = &a_nn;
  Mat * a_nnd_ptr = &a_nnd;

  nearest_neighbor_search(&a, &b, a_nn_ptr, a_nnd_ptr);
  std::cerr << "nn-search done" << std::endl;
  int x = 300;
  int y = 400;
  cv::Vec2b v = a_nn_ptr->at<cv::Vec2b>(x, y);
  cv::Rect src_rect = cv::Rect(x, y, 20, 20);
  cv::Rect ref_rect = cv::Rect(v[0], v[1], 20, 20);
  cv::Mat src = cv::Mat(a, src_rect).clone();
  cv::Mat ref = cv::Mat(b, ref_rect).clone();

  cv::namedWindow( "w1", WINDOW_AUTOSIZE );// Create a window for display.
  cv::namedWindow( "w2", WINDOW_AUTOSIZE );// Create a window for display.

  cv::imshow( "w1", src );

  cv::imshow( "w2", ref );

  waitKey(0);
  return 0;
}