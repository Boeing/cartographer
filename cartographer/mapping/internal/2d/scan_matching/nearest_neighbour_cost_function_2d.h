#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NN_COST_FUNCTION_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NN_COST_FUNCTION_2D_H_

#include "cartographer/common/nanoflann.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "ceres/ceres.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

template <typename T>
struct DataSet {
  struct Cell {
    T x;
    T y;
  };

  std::vector<Cell> cells;

  inline size_t kdtree_get_point_count() const { return cells.size(); }

  inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return cells[idx].x;
    else
      return cells[idx].y;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, DataSet<double>, double>,
    DataSet<double>, 2>
    RealKDTree;

struct RealIndex {
  DataSet<double> cells;
  std::unique_ptr<RealKDTree> kdtree;
};

class SingleNearestNeighbourCostFunction2D {
 public:
  SingleNearestNeighbourCostFunction2D(const double scaling_factor,
                                       const Eigen::Vector2d& src,
                                       const RealKDTree& kdtree)
      : scaling_factor_(scaling_factor), src_(src), kdtree_(kdtree) {}

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const Eigen::Matrix<T, 3, 1> point((T(src_[0])), (T(src_[1])), T(1.));
    const Eigen::Matrix<T, 3, 1> world = transform * point;

    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);

    result_set.init(&ret_index, &out_dist_sqr);

    query_pt[0] = world[0];
    query_pt[1] = world[1];

    kdtree_.findNeighbors(result_set, &query_pt[0],
                          nanoflann::SearchParams(10));

    residual[0] =
        T(scaling_factor_) * (world[0] - T(kdtree_.dataset.cells[ret_index].x));
    residual[1] =
        T(scaling_factor_) * (world[1] - T(kdtree_.dataset.cells[ret_index].y));

    return true;
  }

 private:
  SingleNearestNeighbourCostFunction2D(
      const SingleNearestNeighbourCostFunction2D&) = delete;
  SingleNearestNeighbourCostFunction2D& operator=(
      const SingleNearestNeighbourCostFunction2D&) = delete;

  const double scaling_factor_;
  const Eigen::Vector2d src_;
  const RealKDTree& kdtree_;
};

class PointPairCostFunction2D {
 public:
  PointPairCostFunction2D(const double scaling_factor,
                          const Eigen::Vector2d& src,
                          const Eigen::Vector2d& dst)
      : scaling_factor_(scaling_factor), src_(src), dst_(dst) {}

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const Eigen::Matrix<T, 3, 1> point((T(src_[0])), (T(src_[1])), T(1.));
    const Eigen::Matrix<T, 3, 1> world = transform * point;

    residual[0] = T(scaling_factor_) * (world[0] - T(dst_[0]));
    residual[1] = T(scaling_factor_) * (world[1] - T(dst_[1]));

    return true;
  }

 private:
  PointPairCostFunction2D(const PointPairCostFunction2D&) = delete;
  PointPairCostFunction2D& operator=(const PointPairCostFunction2D&) = delete;

  const double scaling_factor_;
  const Eigen::Vector2d src_;
  const Eigen::Vector2d dst_;
};

RealIndex CreateRealIndexForGrid(const Grid2D& grid);

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif
