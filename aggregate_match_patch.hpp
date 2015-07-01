#ifndef AGGREGATE_MATCH_PATCH_HPP
#define AGGREGATE_MATCH_PATCH_HPP

using namespace std;
using namespace cv;

void AggregateMatchPatch(Size sz, Mat match, vector<vector <Mat> > T, vector<pair<Point2d, Point2d> > &region);

#endif //AGGREGATE_MATCH_PATCH_HPP
