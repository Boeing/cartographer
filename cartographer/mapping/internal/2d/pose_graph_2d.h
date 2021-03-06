#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_POSE_GRAPH_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_POSE_GRAPH_2D_H_

#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "cartographer/common/fixed_ratio_sampler.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/internal/constraints/constraint_builder_2d.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_2d.h"
#include "cartographer/mapping/internal/trajectory_connectivity_state.h"
#include "cartographer/mapping/internal/work_queue.h"
#include "cartographer/mapping/pose_graph.h"
#include "cartographer/mapping/pose_graph_data.h"
#include "cartographer/mapping/pose_graph_trimmer.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/fixed_frame_pose_data.h"
#include "cartographer/sensor/landmark_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"

namespace cartographer {
namespace mapping {

// Implements the loop closure method called Sparse Pose Adjustment (SPA) from
// Konolige, Kurt, et al. "Efficient sparse pose adjustment for 2d mapping."
// Intelligent Robots and Systems (IROS), 2010 IEEE/RSJ International Conference
// on (pp. 22--29). IEEE, 2010.
//
// It is extended for submapping:
// Each node has been matched against one or more submaps (adding a constraint
// for each match), both poses of nodes and of submaps are to be optimized.
// All constraints are between a submap i and a node j.
class PoseGraph2D : public PoseGraph {
 public:
  PoseGraph2D(const proto::PoseGraphOptions& options);
  ~PoseGraph2D() override;

  PoseGraph2D(const PoseGraph2D&) = delete;
  PoseGraph2D& operator=(const PoseGraph2D&) = delete;

  void StartBackgroundThread();
  void StopBackgroundThread();

  NodeId AddNode(
      std::shared_ptr<const TrajectoryNode::Data> constant_data,
      int trajectory_id,
      const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps)
      LOCKS_EXCLUDED(mutex_);

  void AddFixedFramePoseData(
      int trajectory_id,
      const sensor::FixedFramePoseData& fixed_frame_pose_data) override
      LOCKS_EXCLUDED(mutex_);

  void AddLandmarkData(int trajectory_id,
                       const sensor::LandmarkData& landmark_data) override
      LOCKS_EXCLUDED(mutex_);

  void DeleteTrajectory(int trajectory_id) override;

  void FinishTrajectory(int trajectory_id) override;

  bool IsTrajectoryFinished(int trajectory_id) const override
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void FreezeTrajectory(int trajectory_id) override;

  bool IsTrajectoryFrozen(int trajectory_id) const override
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void AddSubmapFromProto(const transform::Rigid3d& global_submap_pose,
                          const proto::Submap& submap) override;

  void AddNodeFromProto(const transform::Rigid3d& global_pose,
                        const proto::Node& node) override;

  void SetTrajectoryDataFromProto(const proto::TrajectoryData& data) override;

  void AddNodeToSubmap(const NodeId& node_id,
                       const SubmapId& submap_id) override;

  void AddSerializedConstraints(
      const std::vector<Constraint>& constraints) override;

  void AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) override;

  void RunFinalOptimization() override;

  std::vector<std::vector<int>> GetConnectedTrajectories() const override
      LOCKS_EXCLUDED(mutex_);

  PoseGraphInterface::SubmapData GetSubmapData(const SubmapId& submap_id) const
      LOCKS_EXCLUDED(mutex_) override;

  MapById<SubmapId, PoseGraphInterface::SubmapData> GetAllSubmapData() const
      LOCKS_EXCLUDED(mutex_) override;

  MapById<SubmapId, SubmapPose> GetAllSubmapPoses() const
      LOCKS_EXCLUDED(mutex_) override;

  transform::Rigid3d GetLocalToGlobalTransform(int trajectory_id) const
      LOCKS_EXCLUDED(mutex_) override;

  MapById<NodeId, TrajectoryNode> GetTrajectoryNodes() const override
      LOCKS_EXCLUDED(mutex_);

  MapById<NodeId, TrajectoryNodePose> GetTrajectoryNodePoses() const override
      LOCKS_EXCLUDED(mutex_);

  std::map<int, TrajectoryState> GetTrajectoryStates() const override
      LOCKS_EXCLUDED(mutex_);

  std::map<std::string, transform::Rigid3d> GetLandmarkPoses() const override
      LOCKS_EXCLUDED(mutex_);

  void SetLandmarkPose(const std::string& landmark_id,
                       const transform::Rigid3d& global_pose,
                       const bool frozen = false) override
      LOCKS_EXCLUDED(mutex_);

  sensor::MapByTime<sensor::FixedFramePoseData> GetFixedFramePoseData()
      const override LOCKS_EXCLUDED(mutex_);

  std::map<std::string /* landmark ID */, PoseGraph::LandmarkNode>
  GetLandmarkNodes() const override LOCKS_EXCLUDED(mutex_);

  std::map<int, TrajectoryData> GetTrajectoryData() const override
      LOCKS_EXCLUDED(mutex_);

  std::vector<Constraint> constraints() const override LOCKS_EXCLUDED(mutex_);

  void SetInitialTrajectoryPose(int from_trajectory_id, int to_trajectory_id,
                                const transform::Rigid3d& pose,
                                const common::Time time) override
      LOCKS_EXCLUDED(mutex_);

  void SetGlobalSlamOptimizationCallback(
      PoseGraphInterface::GlobalSlamOptimizationCallback callback) override;

  transform::Rigid3d GetInterpolatedGlobalTrajectoryPose(
      int trajectory_id, const common::Time time) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static void RegisterMetrics(metrics::FamilyFactory* family_factory);

 private:
  struct Statistics {
    // map of nodes to constraints
    MapById<NodeId, std::vector<Constraint>> node_inter_constraints;
    // number of local constraints for traj
    std::unordered_map<int, size_t> local_traj_constraint_count;
    // number of global constraints for traj
    std::unordered_map<int, size_t> global_traj_constraint_count;
    // number of local constraints for submap
    MapById<SubmapId, size_t> local_inter_constraint_count;
    // number of global constraints for submap
    MapById<SubmapId, size_t> global_inter_constraint_count;
  };

  Statistics CalculateStatistics() const;

  std::vector<PoseGraphInterface::Constraint> FindNewConstraints();

  std::atomic_bool background_thread_running_;
  std::thread background_thread_;
  void BackgroundThread();

  void RunOptimization(const int32 num_of_iterations = -1);

  MapById<SubmapId, PoseGraphInterface::SubmapData> GetSubmapDataUnderLock()
      const EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Handles a new work item.
  void AddWorkItem(const std::function<WorkItem::Result()>& work_item)
      LOCKS_EXCLUDED(mutex_) LOCKS_EXCLUDED(work_queue_mutex_);

  // Adds connectivity and sampler for a trajectory if it does not exist.
  void AddTrajectoryIfNeeded(int trajectory_id)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Deletes trajectories waiting for deletion. Must not be called during
  // constraint search.
  void DeleteTrajectoriesIfNeeded() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  bool CanAddWorkItemModifying(int trajectory_id)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Computes the local to global map frame transform based on the given
  // 'global_submap_poses'.
  transform::Rigid3d ComputeLocalToGlobalTransform(
      const MapById<SubmapId, optimization::SubmapSpec2D>& global_submap_poses,
      int trajectory_id) const EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  SubmapData GetSubmapDataUnderLock(const SubmapId& submap_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  common::Time GetLatestNodeTime(const NodeId& node_id,
                                 const SubmapId& submap_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Updates the trajectory connectivity structure with a new constraint.
  void UpdateTrajectoryConnectivity(const Constraint& constraint)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  const proto::PoseGraphOptions options_;
  GlobalSlamOptimizationCallback global_slam_optimization_callback_;

  mutable absl::Mutex mutex_;

  constraints::ConstraintBuilder2D constraint_builder_;

  // List of all trimmers to consult when optimizations finish.
  std::vector<std::unique_ptr<PoseGraphTrimmer>> trimmers_ GUARDED_BY(mutex_);

  PoseGraphData data_ GUARDED_BY(mutex_);

  ValueConversionTables conversion_tables_;

  // Allows querying and manipulating the pose graph by the 'trimmers_'. The
  // 'mutex_' of the pose graph is held while this class is used.
  class TrimmingHandle : public Trimmable {
   public:
    TrimmingHandle(PoseGraph2D* parent);
    ~TrimmingHandle() override {}

    int num_submaps(int trajectory_id) const override;
    std::vector<SubmapId> GetSubmapIds(int trajectory_id) const override;
    MapById<SubmapId, SubmapData> GetOptimizedSubmapData() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);
    const MapById<NodeId, TrajectoryNode>& GetTrajectoryNodes() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);
    const std::vector<Constraint>& GetConstraints() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);
    const MapById<SubmapId, std::set<NodeId>> GetSubmapNodes() const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);
    void TrimSubmap(const SubmapId& submap_id)
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_) override;
    bool IsFinished(int trajectory_id) const override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);
    void SetTrajectoryState(int trajectory_id, TrajectoryState state) override
        EXCLUSIVE_LOCKS_REQUIRED(parent_->mutex_);

   private:
    PoseGraph2D* const parent_;
  };
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_2D_POSE_GRAPH_2D_H_
