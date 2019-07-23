#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_GLOBAL_ICP_SCAN_MATCHER_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_GLOBAL_ICP_SCAN_MATCHER_2D_H_

#include <memory>
#include <random>
#include <vector>

#include "Eigen/Core"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/icp_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/nearest_neighbour_cost_function_2d.h"
#include "cartographer/mapping/proto/scan_matching/global_icp_scan_matcher_options_2d.pb.h"
#include "cartographer/sensor/point_cloud.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::GlobalICPScanMatcherOptions2D CreateGlobalICPScanMatcherOptions2D(
    common::LuaParameterDictionary* parameter_dictionary);

std::vector<Eigen::Array2i> FreeCells(const Grid2D& grid);

class EmptySpaceSampler {
 public:
  explicit EmptySpaceSampler(const Grid2D& grid)
      : gen_(42),
        free_cells_(FreeCells(grid)),
        dist_(0, static_cast<int>(free_cells_.size() - 1)) {}
  ~EmptySpaceSampler() = default;

  Eigen::Array2i sample() {
    return free_cells_[static_cast<std::size_t>(dist_(gen_))];
  }

 private:
  std::mt19937 gen_;
  std::vector<Eigen::Array2i> free_cells_;
  std::uniform_int_distribution<int> dist_;
};

class GlobalICPScanMatcher2D {
 public:
  explicit GlobalICPScanMatcher2D(
      const Submap2D& submap,
      const proto::GlobalICPScanMatcherOptions2D& options);
  virtual ~GlobalICPScanMatcher2D();

  GlobalICPScanMatcher2D(const GlobalICPScanMatcher2D&) = delete;
  GlobalICPScanMatcher2D& operator=(const GlobalICPScanMatcher2D&) = delete;

  struct RotatedScan {
    double rotation;
    sensor::PointCloud scan_data;
  };

  struct SamplePose {
    double score;
    double x;
    double y;
    double rotation;
  };

  struct PoseCluster {
    double x;
    double y;
    double rotation;

    std::vector<SamplePose> poses;
  };

  struct ClusterData {
    std::vector<SamplePose> data;

    inline size_t kdtree_get_point_count() const { return data.size(); }

    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
      if (dim == 0) return data[idx].x;
      if (dim == 1)
        return data[idx].y;
      else
        return data[idx].rotation;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
      return false;
    }
  };

  typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, ClusterData, double>, ClusterData, 3>
      ClusterTree;

  struct Result {
    std::vector<SamplePose> poses;
  };

  Result Match(const transform::Rigid2d pose_estimate,
               const sensor::PointCloud& point_cloud);

  Result Match(const sensor::PointCloud& point_cloud);

  std::vector<PoseCluster> DBScanCluster(const std::vector<SamplePose>& poses);

  const ICPScanMatcher2D& IcpSolver() const { return icp_solver_; }

 private:
  const proto::GlobalICPScanMatcherOptions2D options_;
  const MapLimits limits_;
  EmptySpaceSampler sampler_;
  const ICPScanMatcher2D icp_solver_;
};

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_GLOBAL_ICP_SCAN_MATCHER_2D_H_
