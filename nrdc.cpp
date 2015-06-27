#include <opencv2/opencv.hpp>
#include "nearest_neighbor_search.hpp"
#include "aggregate_match_patch.hpp"
#include "global_color_transformation.hpp"
#include "narrow_seaerch_range.hpp"

using namespace cv;

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
