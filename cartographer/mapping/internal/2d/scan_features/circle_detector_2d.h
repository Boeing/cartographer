#ifndef CARTOGRAPHER_MAPPING_INTERNAL_SCAN_FEATURES_CIRCLE_DETECTOR_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_SCAN_FEATURES_CIRCLE_DETECTOR_2D_H_

#include <vector>

#include "cartographer/sensor/point_cloud.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/transform/transform.h"

#include "cartographer/mapping/2d/probability_grid.h"

namespace cartographer {
namespace mapping {

template <typename T>
struct Circle {
  float mse;
  int count;

  float radius;
  Eigen::Matrix<T, 2, 1> position;

  std::vector<Eigen::Matrix<T, 2, 1>> points;
};

std::vector<Circle<float>> DetectReflectivePoles(
    const sensor::PointCloud& point_cloud, const float radius);

Circle<float> FitCircle(const Circle<float>& circle);

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_SCAN_FEATURES_CIRCLE_DETECTOR_2D_H_
