#ifndef GLOBAL_COLOR_TRANSFORMATION_HPP
#define GLOBAL_COLOR_TRANSFORMATION_HPP

using namespace std;
using namespace cv;

/**
 * pair<Point2d src, Point2d ref>>
 */
Mat GlobalColorTransformation(Mat src, Mat ref, vector<pair<Point2d, Point2d> > regions);

#endif //GLOBAL_COLOR_TRANSFORMATION_HPP
