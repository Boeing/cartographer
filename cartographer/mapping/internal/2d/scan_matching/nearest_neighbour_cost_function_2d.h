#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NN_COST_FUNCTION_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_NN_COST_FUNCTION_2D_H_

#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "ceres/ceres.h"

#include "nanoflann.h"

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
    nanoflann::L2_Simple_Adaptor<int, DataSet<int>, double>, DataSet<int>, 2>
    CellKDTree;
typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, DataSet<double>, double>,
    DataSet<double>, 2>
    RealKDTree;

struct CellIndex {
  DataSet<int> cells;
  std::unique_ptr<CellKDTree> kdtree;
};

struct RealIndex {
  DataSet<double> cells;
  std::unique_ptr<RealKDTree> kdtree;
};

class NearestNeighbourCostFunction2D {
 public:
  NearestNeighbourCostFunction2D(const double scaling_factor,
                                 const sensor::PointCloud& point_cloud,
                                 const MapLimits& limits,
                                 const RealKDTree& kdtree)
      : scaling_factor_(scaling_factor),
        point_cloud_(point_cloud),
        limits_(limits),
        kdtree_(kdtree) {}

  template <typename T>
  bool operator()(const T* const pose, T* residual) const {
    Eigen::Matrix<T, 2, 1> translation(pose[0], pose[1]);
    Eigen::Rotation2D<T> rotation(pose[2]);
    Eigen::Matrix<T, 2, 2> rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix<T, 3, 3> transform;
    transform << rotation_matrix, translation, T(0.), T(0.), T(1.);

    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);

    for (size_t i = 0; i < point_cloud_.size(); ++i) {
      const Eigen::Matrix<T, 3, 1> point((T(point_cloud_[i].position.x())),
                                         (T(point_cloud_[i].position.y())),
                                         T(1.));
      const Eigen::Matrix<T, 3, 1> world = transform * point;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = world[0];
      query_pt[1] = world[1];

      kdtree_.findNeighbors(result_set, &query_pt[0],
                            nanoflann::SearchParams(10));

      residual[i] = std::sqrt(out_dist_sqr);  // scaling_factor_ *
    }
    return true;
  }

 private:
  NearestNeighbourCostFunction2D(const NearestNeighbourCostFunction2D&) =
      delete;
  NearestNeighbourCostFunction2D& operator=(
      const NearestNeighbourCostFunction2D&) = delete;

  const double scaling_factor_;
  const sensor::PointCloud& point_cloud_;
  const MapLimits limits_;
  const RealKDTree& kdtree_;
};

class SingleNearestNeighbourCostFunction2D {
 public:
  SingleNearestNeighbourCostFunction2D(const Eigen::Vector2d& src,
                                       const RealKDTree& kdtree)
      : src_(src), kdtree_(kdtree) {}

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

    residual[0] = world[0] - T(kdtree_.dataset.cells[ret_index].x);
    residual[1] = world[1] - T(kdtree_.dataset.cells[ret_index].y);

    return true;
  }

 private:
  SingleNearestNeighbourCostFunction2D(
      const SingleNearestNeighbourCostFunction2D&) = delete;
  SingleNearestNeighbourCostFunction2D& operator=(
      const SingleNearestNeighbourCostFunction2D&) = delete;

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

    residual[0] = world[0] - T(dst_[0]);  // scaling_factor_ *
    residual[1] = world[1] - T(dst_[1]);  // scaling_factor_ *

    return true;
  }

 private:
  PointPairCostFunction2D(const NearestNeighbourCostFunction2D&) = delete;
  PointPairCostFunction2D& operator=(const NearestNeighbourCostFunction2D&) =
      delete;

  const double scaling_factor_;
  const Eigen::Vector2d src_;
  const Eigen::Vector2d dst_;
};

CellIndex CreateCellIndexForGrid(const Grid2D& grid);

RealIndex CreateRealIndexForGrid(const Grid2D& grid);

// Creates a cost function for matching the 'point_cloud' to the 'grid' with
// a 'pose'. The cost increases with poorer correspondence of the grid and the
// point observation (e.g. points falling into less occupied space).
ceres::CostFunction* CreateNearestNeighbourCostFunction2D(
    const double scaling_factor, const sensor::PointCloud& point_cloud,
    const MapLimits& limits, const RealKDTree& kdtree);

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif
