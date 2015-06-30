#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
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
  Mat a, b;

  load_images(a, b);

  Mat * a_nn_ptr = NULL;
  Mat * a_nnd_ptr = NULL;

  nearest_neighbor_search(&a, &b, a_nn_ptr, a_nnd_ptr);
  std::cerr << "nn-search done" << std::endl;

  // cv::namedWindow( "w1", WINDOW_AUTOSIZE );// Create a window for display.
  // Size sz1 = a.size();
  // Size sz2 = b.size();
  // Mat combo(sz1.height, sz1.width+sz2.width, CV_8UC3);
  // Mat left(combo, Rect(0, 0, sz1.width, sz1.height));
  // a.copyTo(left);
  // Mat right(combo, Rect(sz1.width, 0, sz2.width, sz2.height));
  // b.copyTo(right);

  // typedef std::chrono::high_resolution_clock myclock;
  // myclock::time_point beginning = myclock::now();
  // std::default_random_engine gen1, gen2;
  // std::uniform_int_distribution<int> dist1(0, sz1.width), dist2(0, sz1.height);
  // gen1.seed((myclock::now() - beginning).count()); gen2.seed((myclock::now() - beginning).count());
  // auto rand1 = std::bind(dist1, gen1), rand2 = std::bind(dist2, gen2);
  // for (int i = 0; i < 100; ++i){
  //   // int x1 = rand() % sz1.width, y1 = rand() % sz1.height;
  //   int x1 = rand1(), y1 = rand2();
  //   cv::Vec2b v = a_nn_ptr->at<cv::Vec2b>(x1, y1);
  //   int x2 = v[0], y2 = v[1];
  //   std::cerr << "Plotting (" << x1 << "," << y1 << ")<->(" << x2+sz1.width << "," << y2 << ")" << std::endl;
  //   cv::line(combo, cv::Point2d(x1, y1), cv::Point2d(x2+sz1.width, y2), Scalar( 0, 0, 0 ), 1, 8);
  // }

  // cv::imshow( "w1",  combo);
  // waitKey(0);
  return 0;
}