#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_ICP_SCAN_MATCHER_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_ICP_SCAN_MATCHER_2D_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/feature.h"
#include "cartographer/mapping/internal/2d/scan_matching/nearest_feature_cost_function_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/nearest_neighbour_cost_function_2d.h"
#include "cartographer/mapping/proto/scan_matching/icp_scan_matcher_options_2d.pb.h"
#include "cartographer/sensor/point_cloud.h"
#include "ceres/ceres.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::ICPScanMatcherOptions2D CreateICPScanMatcherOptions2D(
    common::LuaParameterDictionary* parameter_dictionary);

class ICPScanMatcher2D {
 public:
  explicit ICPScanMatcher2D(const Submap2D& submap,
                            const proto::ICPScanMatcherOptions2D& options);
  virtual ~ICPScanMatcher2D();

  ICPScanMatcher2D(const ICPScanMatcher2D&) = delete;
  ICPScanMatcher2D& operator=(const ICPScanMatcher2D&) = delete;

  struct Result {
    transform::Rigid2d pose_estimate;
    ceres::Solver::Summary summary;

    size_t num_inlier_points;

    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> pairs;
  };

  Result Match(const transform::Rigid2d& initial_pose_estimate,
               const sensor::PointCloud& point_cloud,
               const std::vector<CircleFeature>& features = {}) const;

  Result MatchPointPair(const transform::Rigid2d& initial_pose_estimate,
                        const sensor::PointCloud& point_cloud,
                        const std::vector<CircleFeature>& features = {}) const;

  const RealIndex& kdtree() const { return kdtree_; }
  const CircleFeatureIndex& circle_feature_index() const {
    return circle_feature_index_;
  }

 private:
  const proto::ICPScanMatcherOptions2D options_;
  const ceres::Solver::Options ceres_solver_options_;
  const MapLimits limits_;

  const RealIndex kdtree_;
  const CircleFeatureIndex circle_feature_index_;
};

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_ICP_SCAN_MATCHER_2D_H_
