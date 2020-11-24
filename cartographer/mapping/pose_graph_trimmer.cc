#include "cartographer/mapping/pose_graph_trimmer.h"

#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PureLocalizationTrimmer::PureLocalizationTrimmer(const int trajectory_id,
                                                 const int num_submaps_to_keep)
    : trajectory_id_(trajectory_id), num_submaps_to_keep_(num_submaps_to_keep) {
  CHECK_GE(num_submaps_to_keep, 2) << "Cannot trim with less than 2 submaps";
}

void PureLocalizationTrimmer::Trim(Trimmable* const pose_graph) {
  if (pose_graph->IsFinished(trajectory_id_)) {
    num_submaps_to_keep_ = 0;
  }

  std::map<NodeId, std::vector<PoseGraphInterface::Constraint>>
      node_global_constraints;
  const auto constraints = pose_graph->GetConstraints();
  for (size_t i = 0; i < constraints.size(); ++i) {
    const auto constraint = constraints[i];
    if (constraint.node_id.trajectory_id == trajectory_id_ &&
        constraint.tag == PoseGraphInterface::Constraint::INTER_SUBMAP &&
        constraint.submap_id.trajectory_id != trajectory_id_) {
      if (node_global_constraints.count(constraint.node_id))
        node_global_constraints.at(constraint.node_id).push_back(constraint);
      else
        node_global_constraints.insert(
            {constraint.node_id,
             std::vector<PoseGraphInterface::Constraint>{constraint}});
    }
  }

  const auto all_submap_nodes = pose_graph->GetSubmapNodes();

  auto submap_ids = pose_graph->GetSubmapIds(trajectory_id_);
  std::sort(submap_ids.begin(), submap_ids.end());

  for (std::size_t i = 0; i + num_submaps_to_keep_ < submap_ids.size(); ++i) {
    //
    // If there are global constraints
    // Only trim if there are still global constraints after removal
    //
    if (!node_global_constraints.empty()) {
      const auto submap_nodes = all_submap_nodes.at(submap_ids.at(i));
      const NodeId last_global_constraint_node =
          std::prev(node_global_constraints.end())->first;
      if (submap_nodes.count(last_global_constraint_node)) {
        LOG(WARNING) << "Not trimming submap: " << submap_ids.at(i)
                     << " as the last global constraint belongs to node on it: "
                     << last_global_constraint_node;
        break;
      }
    }

    pose_graph->TrimSubmap(submap_ids.at(i));
  }

  if (num_submaps_to_keep_ == 0) {
    finished_ = true;
    pose_graph->SetTrajectoryState(
        trajectory_id_, PoseGraphInterface::TrajectoryState::DELETED);
  }
}

bool PureLocalizationTrimmer::IsFinished() { return finished_; }

}  // namespace mapping
}  // namespace cartographer
