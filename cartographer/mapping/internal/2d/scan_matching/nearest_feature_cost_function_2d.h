#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NEAREST_FEATURE_COST_FUNCTION_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NEAREST_FEATURE_COST_FUNCTION_2D_H_

#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "ceres/ceres.h"

#include "cartographer/common/nanoflann.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

struct CircleFeatureSet {
  std::vector<CircleFeature> data;

  inline size_t kdtree_get_point_count() const { return data.size(); }

  inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return data[idx].keypoint.position.x();
    else if (dim == 1)
      return data[idx].keypoint.position.y();
    else
      return data[idx].fdescriptor.radius;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};

struct CircleFeatureDistanceAdapter {
  typedef float ElementType;
  typedef float DistanceType;
  const CircleFeatureSet& data_source;

  CircleFeatureDistanceAdapter(const CircleFeatureSet& _data_source)
      : data_source(_data_source) {}

  inline DistanceType evalMetric(const float* a, const size_t b_idx,
                                 size_t size) const {
    DistanceType result = DistanceType();
    const DistanceType diff_x = a[0] - data_source.kdtree_get_pt(b_idx, 0);
    const DistanceType diff_y = a[1] - data_source.kdtree_get_pt(b_idx, 1);
    const DistanceType diff_r = a[2] - data_source.kdtree_get_pt(b_idx, 2);
    // distance of radius is weighted higher to make dissimilar radii further
    // apart
    result = diff_x * diff_x + diff_y * diff_y + 4.0f * (diff_r * diff_r);
    return result;
  }

  template <typename U, typename V>
  inline DistanceType accum_dist(const U a, const V b, const size_t) const {
    return (a - b) * (a - b);
  }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<CircleFeatureDistanceAdapter,
                                            CircleFeatureSet, 3>
    CircleFeatureKDTree;

struct CircleFeatureIndex {
  CircleFeatureSet feature_set;
  std::unique_ptr<CircleFeatureKDTree> kdtree;
};

class NearestFeatureCostFunction2D {
 public:
  NearestFeatureCostFunction2D(const double scaling_factor,
                               const CircleFeature& src,
                               const CircleFeatureKDTree& index)
      : scaling_factor_(scaling_factor), src_(src), index_(index) {}

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const Eigen::Matrix<T, 3, 1> point((T(src_.keypoint.position[0])),
                                       (T(src_.keypoint.position[1])), T(1.));
    const Eigen::Matrix<T, 3, 1> world = transform * point;

    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(num_results);

    result_set.init(&ret_index, &out_dist_sqr);

    query_pt[0] = world[0];
    query_pt[1] = world[1];
    query_pt[2] = src_.fdescriptor.radius;

    index_.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParams(10));

    residual[0] =
        T(scaling_factor_) *
        (world[0] - T(index_.dataset.data[ret_index].keypoint.position[0]));
    residual[1] =
        T(scaling_factor_) *
        (world[1] - T(index_.dataset.data[ret_index].keypoint.position[1]));

    return true;
  }

 private:
  NearestFeatureCostFunction2D(const NearestFeatureCostFunction2D&) = delete;
  NearestFeatureCostFunction2D& operator=(const NearestFeatureCostFunction2D&) =
      delete;

  const double scaling_factor_;
  const CircleFeature src_;
  const CircleFeatureKDTree& index_;
};

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif
