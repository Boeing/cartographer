#ifndef CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_2D_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_2D_H_

#include <array>
#include <deque>
#include <map>
#include <set>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "cartographer/common/port.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/id.h"
#include "cartographer/mapping/internal/optimization/optimization_problem_interface.h"
#include "cartographer/mapping/pose_graph_interface.h"
#include "cartographer/mapping/proto/pose_graph/optimization_problem_options.pb.h"
#include "cartographer/sensor/map_by_time.h"
#include "cartographer/transform/timestamped_transform.h"

namespace cartographer {
namespace mapping {
namespace optimization {

struct NodeSpec2D {
  common::Time time;
  transform::Rigid2d local_pose_2d;
  transform::Rigid2d global_pose_2d;
  Eigen::Quaterniond gravity_alignment;
};

struct SubmapSpec2D {
  transform::Rigid2d global_pose;
};

class OptimizationProblem2D {
 public:
  using Constraint = PoseGraphInterface::Constraint;
  using LandmarkNode = PoseGraphInterface::LandmarkNode;

  explicit OptimizationProblem2D(
      const optimization::proto::OptimizationProblemOptions& options);
  ~OptimizationProblem2D();

  OptimizationProblem2D(const OptimizationProblem2D&) = delete;
  OptimizationProblem2D& operator=(const OptimizationProblem2D&) = delete;

  void SetMaxNumIterations(int32 max_num_iterations);

  struct Result {
    bool success;
    MapById<NodeId, transform::Rigid2d> node_poses;
    MapById<SubmapId, transform::Rigid2d> submap_poses;
    std::map<std::string, transform::Rigid3d> landmark_poses;
  };

  Result Solve(const MapById<NodeId, NodeSpec2D>& node_data,
               const MapById<SubmapId, SubmapSpec2D>& submap_data,
               const std::vector<Constraint>& constraints,
               const std::map<int, PoseGraphInterface::TrajectoryState>&
                   trajectories_state,
               const std::map<std::string, LandmarkNode>& landmark_nodes);

 private:
  optimization::proto::OptimizationProblemOptions options_;
};

}  // namespace optimization
}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_INTERNAL_OPTIMIZATION_OPTIMIZATION_PROBLEM_2D_H_
