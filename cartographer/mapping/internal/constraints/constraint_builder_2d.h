#ifndef CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_2D_H_

#include <array>
#include <deque>
#include <functional>
#include <limits>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/math.h"
#include "cartographer/common/task.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/global_icp_scan_matcher_2d.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/sensor/point_cloud.h"

namespace cartographer {
namespace mapping {
namespace constraints {

class ConstraintBuilder2D {
 public:
  using Constraint = PoseGraphInterface::Constraint;

  ConstraintBuilder2D(const proto::ConstraintBuilderOptions& options);
  ~ConstraintBuilder2D();

  ConstraintBuilder2D(const ConstraintBuilder2D&) = delete;
  ConstraintBuilder2D& operator=(const ConstraintBuilder2D&) = delete;

  absl::optional<Constraint> LocalSearchForConstraint(
      const NodeId node_id, const SubmapId submap_id,
      const transform::Rigid2d& initial_relative_pose, const Submap2D& submap,
      const TrajectoryNode::Data& constant_data);

  absl::optional<Constraint> GlobalSearchForConstraint(
      const NodeId node_id, const SubmapId submap_id, const Submap2D& submap,
      const TrajectoryNode::Data& constant_data);

  absl::optional<Constraint> GlobalSearchForConstraint(
      const NodeId node_id, const MapById<SubmapId, const Submap2D*>& submaps,
      const TrajectoryNode::Data& constant_data);

  void DeleteScanMatcher(const SubmapId& submap_id);

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);

 private:
  struct SubmapScanMatcher {
    const SubmapId submap_id;
    const Submap2D& submap;
    std::unique_ptr<scan_matching::GlobalICPScanMatcher2D>
        global_icp_scan_matcher;
  };

  struct FoundConstraint {
    double score;
    Constraint constraint;
  };

  const SubmapScanMatcher* GetScanMatcher(const SubmapId& submap_id,
                                          const Submap2D& submap)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::optional<FoundConstraint> ComputeConstraint(
      const NodeId node_id, const SubmapId submap_id,
      const transform::Rigid2d initial_relative_pose, const Submap2D& submap,
      const TrajectoryNode::Data& constant_data, const bool match_full_submap,
      const SubmapScanMatcher& submap_scan_matcher) LOCKS_EXCLUDED(mutex_);

  const constraints::proto::ConstraintBuilderOptions options_;
  absl::Mutex mutex_;

  std::map<SubmapId, SubmapScanMatcher> submap_scan_matchers_
      GUARDED_BY(mutex_);
  scan_matching::CeresScanMatcher2D ceres_scan_matcher_;
};

}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_CONSTRAINTS_CONSTRAINT_BUILDER_2D_H_
