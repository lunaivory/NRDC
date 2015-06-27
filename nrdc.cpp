#include <opencv2/opencv.hpp>
#include "nearest_neighbor_search.hpp"
// #include "aggregate_match_patch.hpp"
// #include "global_color_transformation.hpp"
// #include "narrow_seaerch_range.hpp"

using namespace cv;

const static std::string source_path = "source.jpg";
const static std::string reference_path = "reference.jpg";

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

  for (var i = (1<<6); i > 0; i >>= 1) {
    Size srcSz = src.size();
    Size refSz = ref.size();

    srcSz.height /= i, srcSz.width /= i;
    refSz.height /= i, srcSz.width /= i;

    resize(src, srcS, srcSz);
    resize(ref, refS, refSz);

    NearestNeighborSearch();

    AggregateMatchPatch();

    GlobalColorTransformation();

    NarrowSearchRange();
  }
}

int main(int argc, char const *argv[])
{
  Mat src, ref;

  load_images(src, ref);

  return 0;
}