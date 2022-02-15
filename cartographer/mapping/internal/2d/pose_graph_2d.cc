#include "cartographer/mapping/internal/2d/pose_graph_2d.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/mapping/internal/2d/overlapping_submaps_trimmer_2d.h"
#include "cartographer/mapping/proto/pose_graph/constraint_builder_options.pb.h"
#include "cartographer/sensor/compressed_point_cloud.h"
#include "cartographer/sensor/internal/voxel_filter.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PoseGraph2D::PoseGraph2D(const proto::PoseGraphOptions& options)
    : background_thread_running_(false),
      options_(options),
      constraint_builder_(options_.constraint_builder_options()) {
  if (options.has_overlapping_submaps_trimmer_2d()) {
    const auto& trimmer_options = options.overlapping_submaps_trimmer_2d();
    AddTrimmer(absl::make_unique<OverlappingSubmapsTrimmer2D>(
        trimmer_options.fresh_submaps_count(),
        trimmer_options.min_covered_area(),
        trimmer_options.min_added_submaps_count()));
  }
  StartBackgroundThread();
}

PoseGraph2D::~PoseGraph2D() {
  if (background_thread_running_) {
    StopBackgroundThread();
  }
}

void PoseGraph2D::StartBackgroundThread() {
  CHECK(!background_thread_running_);
  background_thread_running_ = true;
  background_thread_ = std::thread(&PoseGraph2D::BackgroundThread, this);
}

void PoseGraph2D::StopBackgroundThread() {
  CHECK(background_thread_running_);
  {
    // We take the mutex here despite the atomic bool because
    // the thread will wait on a condition that will wake on Unlock()
    absl::MutexLock locker(&mutex_);
    background_thread_running_ = false;
  }
  background_thread_.join();
}

NodeId PoseGraph2D::AddNode(
    std::shared_ptr<const TrajectoryNode::Data> constant_data,
    const int trajectory_id,
    const std::vector<std::shared_ptr<const Submap2D>>& insertion_submaps) {
  absl::MutexLock locker(&mutex_);

  AddTrajectoryIfNeeded(trajectory_id);

  CHECK(CanAddWorkItemModifying(trajectory_id));

  //
  // insertion_submaps are 1 or a 2 submaps
  // front() is the old submap = used for matching as it has more data
  // back() is a new submap = will transition to front() eventually
  //

  //
  // Determine the correct submap IDs
  // Add a new submap if necessary
  //
  std::vector<SubmapId> submap_ids;
  if (data_.submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
    const InternalSubmapData nd{insertion_submaps.back(),
                                SubmapState::kNoConstraintSearch,
                                std::set<NodeId>{}};
    const SubmapId submap_id = data_.submap_data.Append(trajectory_id, nd);
    const transform::Rigid2d global_submap_pose =
        transform::Project2D(ComputeLocalToGlobalTransform(
                                 data_.global_submap_poses_2d, trajectory_id) *
                             insertion_submaps.back()->local_pose());
    data_.global_submap_poses_2d.Insert(
        submap_id, optimization::SubmapSpec2D{global_submap_pose});
    submap_ids.push_back(submap_id);
    CHECK(submap_id.submap_index == 0);
  } else if (insertion_submaps.size() == 1) {
    CHECK_EQ(1, data_.submap_data.SizeOfTrajectoryOrZero(trajectory_id));
    const SubmapId submap_id{trajectory_id, 0};
    CHECK(data_.submap_data.at(submap_id).submap == insertion_submaps.front());
    submap_ids.push_back(submap_id);

  } else if (std::prev(data_.submap_data.EndOfTrajectory(trajectory_id))
                 ->data.submap != insertion_submaps.back()) {
    const SubmapId submap_id = data_.submap_data.Append(
        trajectory_id,
        InternalSubmapData{
            insertion_submaps.back(), SubmapState::kNoConstraintSearch, {}});

    auto end_it = data_.submap_data.EndOfTrajectory(trajectory_id);
    const SubmapId back_submap = std::prev(end_it, 1)->id;
    const SubmapId front_submap = std::prev(end_it, 2)->id;

    CHECK_EQ(back_submap, submap_id);

    const transform::Rigid2d front_pose = transform::Project2D(
        data_.submap_data.at(front_submap).submap->local_pose());
    const transform::Rigid2d back_pose = transform::Project2D(
        data_.submap_data.at(back_submap).submap->local_pose());

    const transform::Rigid2d last_global_pose =
        data_.global_submap_poses_2d.at(front_submap).global_pose;
    const transform::Rigid2d global_submap_pose =
        last_global_pose * front_pose.inverse() * back_pose;

    data_.global_submap_poses_2d.Insert(
        submap_id, optimization::SubmapSpec2D{global_submap_pose});
    submap_ids.push_back(front_submap);
    submap_ids.push_back(back_submap);
    CHECK(front_submap.submap_index == back_submap.submap_index - 1);
  } else {
    auto end_it = data_.submap_data.EndOfTrajectory(trajectory_id);
    const SubmapId back_submap = std::prev(end_it, 1)->id;
    const SubmapId front_submap = std::prev(end_it, 2)->id;
    submap_ids.push_back(front_submap);
    submap_ids.push_back(back_submap);
    CHECK(front_submap.submap_index == back_submap.submap_index - 1);
  }

  CHECK_EQ(submap_ids.size(), insertion_submaps.size());

  //
  // Create a new TrajectoryNode
  //
  const transform::Rigid3d global_pose(
      ComputeLocalToGlobalTransform(data_.global_submap_poses_2d,
                                    trajectory_id) *
      constant_data->local_pose);
  const NodeId node_id = data_.trajectory_nodes.Append(
      trajectory_id, TrajectoryNode{constant_data, global_pose});

  //
  // Add a constraint and a Node to the current and previous submaps
  //
  const transform::Rigid2d local_pose_2d =
      transform::Project2D(constant_data->local_pose);
  for (size_t i = 0; i < insertion_submaps.size(); ++i) {
    const SubmapId submap_id = submap_ids[i];
    // Even if this was the last node added to 'submap_id', the submap will
    // only be marked as finished in 'data_.submap_data' further below.
    CHECK(data_.submap_data.at(submap_id).state ==
          SubmapState::kNoConstraintSearch);
    data_.submap_data.at(submap_id).node_ids.emplace(node_id);
    const transform::Rigid2d constraint_transform =
        transform::Project2D(insertion_submaps[i]->local_pose()).inverse() *
        local_pose_2d;
    data_.constraints.push_back(
        Constraint{submap_id,
                   node_id,
                   {transform::Embed3D(constraint_transform),
                    options_.matcher_translation_weight(),
                    options_.matcher_rotation_weight()},
                   Constraint::INTRA_SUBMAP});
  }

  if (insertion_submaps.front()->insertion_finished()) {
    const SubmapId newly_finished_submap_id = submap_ids.front();
    InternalSubmapData& finished_submap_data =
        data_.submap_data.at(newly_finished_submap_id);
    CHECK(finished_submap_data.state == SubmapState::kNoConstraintSearch);
    finished_submap_data.state = SubmapState::kFinished;
  }

  return node_id;
}

PoseGraph2D::Statistics PoseGraph2D::CalculateStatistics() const {
  Statistics stats;
  //
  // Count the number of INTER constraints
  // Map the constraints by node
  //
  for (const int trajectory_id : data_.trajectory_nodes.trajectory_ids()) {
    stats.global_traj_constraint_count.insert({trajectory_id, 0});
    stats.local_traj_constraint_count.insert({trajectory_id, 0});
  }

  for (size_t i = 0; i < data_.constraints.size(); ++i) {
    const auto constraint = data_.constraints[i];
    if (constraint.tag == Constraint::INTER_SUBMAP) {
      if (stats.node_inter_constraints.Contains(constraint.node_id))
        stats.node_inter_constraints.at(constraint.node_id)
            .push_back(constraint);
      else
        stats.node_inter_constraints.Insert(
            constraint.node_id, std::vector<Constraint>{constraint});

      std::unordered_map<int, size_t>* traj_count = nullptr;
      MapById<SubmapId, size_t>* count = nullptr;
      if (constraint.node_id.trajectory_id ==
          constraint.submap_id.trajectory_id) {
        traj_count = &stats.local_traj_constraint_count;
        count = &stats.local_inter_constraint_count;
      } else {
        traj_count = &stats.global_traj_constraint_count;
        count = &stats.global_inter_constraint_count;
      }

      if (count->Contains(constraint.submap_id))
        count->at(constraint.submap_id)++;
      else
        count->Insert(constraint.submap_id, 1);

      auto it = traj_count->find(constraint.node_id.trajectory_id);
      if (it != traj_count->end())
        it->second++;
      else
        traj_count->insert({constraint.node_id.trajectory_id, 1});
    }
  }
  return stats;
}

std::vector<PoseGraphInterface::Constraint> PoseGraph2D::FindNewConstraints() {
  mutex_.Lock();

  std::vector<PoseGraphInterface::Constraint> new_constraints;

  const Statistics stats = CalculateStatistics();

  for (const int trajectory_id : data_.trajectory_nodes.trajectory_ids()) {
    if (data_.trajectories_state.at(trajectory_id).state ==
        PoseGraphInterface::TrajectoryState::ACTIVE) {
      if (data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id) == 0)
        continue;

      //
      // Search for constraints on the last node
      //
      auto node_it =
          std::prev(data_.trajectory_nodes.EndOfTrajectory(trajectory_id));
      const NodeId node_id = node_it->id;

      const size_t traj_global_constraint_count =
          stats.global_traj_constraint_count.at(node_id.trajectory_id);

      const bool globally_localized =
          trajectory_id == 0 ||
          static_cast<int>(traj_global_constraint_count) >=
              options_.min_globally_searched_constraints_for_trajectory();

      if (!globally_localized) {
        MapById<SubmapId, const Submap2D*> global_submaps;
        for (const auto submap_id_data : data_.submap_data) {
          if (submap_id_data.id.trajectory_id != node_id.trajectory_id &&
              submap_id_data.data.state == SubmapState::kFinished) {
            auto ptr = dynamic_cast<const Submap2D*>(
                data_.submap_data.at(submap_id_data.id).submap.get());
            global_submaps.Insert(submap_id_data.id, ptr);
          }
        }

        LOG(INFO) << "Search (globally) for constraints for Node: " << node_id;

        mutex_.Unlock();
        const TrajectoryNode::Data& constant_data =
            *data_.trajectory_nodes.at(node_id).constant_data.get();
        const auto result = constraint_builder_.GlobalSearchForConstraint(
            node_id, global_submaps, constant_data);
        mutex_.Lock();

        if (result) new_constraints.push_back(result.value());
      } else {
        // globally localised
        // decide whether to look for a new constraint
        const int num_of_nodes_with_constraints =
            stats.node_inter_constraints.SizeOfTrajectoryOrZero(trajectory_id);

        int nodes_since_last_local_constraint = std::numeric_limits<int>::max();
        int nodes_since_last_global_constraint =
            std::numeric_limits<int>::max();
        if (num_of_nodes_with_constraints > 0) {
          auto it = stats.node_inter_constraints.EndOfTrajectory(trajectory_id);
          while (it != stats.node_inter_constraints.BeginOfTrajectory(
                           trajectory_id)) {
            it = std::prev(it);
            const bool has_local_constraint =
                std::any_of(it->data.begin(), it->data.end(),
                            [trajectory_id](const Constraint& c) {
                              return c.submap_id.trajectory_id == trajectory_id;
                            });
            int distance = node_id.node_index - it->id.node_index;
            if (has_local_constraint &&
                distance < nodes_since_last_local_constraint) {
              nodes_since_last_local_constraint = distance;
              break;
            }
          }

          it = stats.node_inter_constraints.EndOfTrajectory(trajectory_id);
          while (it != stats.node_inter_constraints.BeginOfTrajectory(
                           trajectory_id)) {
            it = std::prev(it);
            const bool has_global_constraint =
                std::any_of(it->data.begin(), it->data.end(),
                            [trajectory_id](const Constraint& c) {
                              return c.submap_id.trajectory_id != trajectory_id;
                            });
            int distance = node_id.node_index - it->id.node_index;
            if (has_global_constraint &&
                distance < nodes_since_last_global_constraint) {
              nodes_since_last_global_constraint = distance;
              break;
            }
          }
        }

        // iterate through own submaps
        // look for ones which are geometrically close to this node
        // if few constraints then look for one
        if (nodes_since_last_local_constraint >
            options_.local_constraint_every_n_nodes()) {
          for (auto submap_itr =
                   data_.submap_data.BeginOfTrajectory(trajectory_id);
               submap_itr != data_.submap_data.EndOfTrajectory(trajectory_id);
               ++submap_itr) {
            if (submap_itr->data.state == SubmapState::kFinished) {
              const transform::Rigid2d initial_relative_pose =
                  data_.global_submap_poses_2d.at(submap_itr->id)
                      .global_pose.inverse() *
                  transform::Project2D(node_it->data.global_pose);
              
              const double resolution = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                   .submap.get())->grid()->limits().resolution();
              
              // The grid is row-major and poorly named. num_x_cells actually refers to the number of rows (y)
                // and num_y_cells is the number of columns (x)
              const double submap_size_x = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                   .submap.get())->grid()->limits().cell_limits().num_y_cells * resolution;
              const double submap_size_y = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                  .submap.get())->grid()->limits().cell_limits().num_x_cells * resolution;

              const bool in_submap = std::abs(initial_relative_pose.translation().x()) <
                                     submap_size_x / 2.0 &&
                                     std::abs(initial_relative_pose.translation().y()) <
                                     submap_size_y / 2.0;

              if (in_submap) {
                LOG(INFO)
                    << "Search (locally) for local constraint between Node: "
                    << node_id << " and Submap: " << submap_itr->id
                    << " relative_pose: " << initial_relative_pose;

                mutex_.Unlock();
                const auto result =
                    constraint_builder_.LocalSearchForConstraint(
                        node_id, submap_itr->id, initial_relative_pose,
                        static_cast<const Submap2D&>(
                            *data_.submap_data.at(submap_itr->id).submap.get()),
                        *node_it->data.constant_data.get());
                mutex_.Lock();

                if (result) new_constraints.push_back(result.value());
              }
            }
          }
        }

        // iterate through submaps of other trajectories
        // look for ones which are geometrically close to this node
        // if few constraints then look for one
        if (nodes_since_last_global_constraint >
            options_.global_constraint_every_n_nodes()) {
          for (const int other_trajectory_id :
               data_.trajectory_nodes.trajectory_ids()) {
            if (other_trajectory_id == trajectory_id) continue;
            for (auto submap_itr =
                     data_.submap_data.BeginOfTrajectory(other_trajectory_id);
                 submap_itr !=
                 data_.submap_data.EndOfTrajectory(other_trajectory_id);
                 ++submap_itr) {
              if (submap_itr->data.state == SubmapState::kFinished) {
                const transform::Rigid2d initial_relative_pose =
                    data_.global_submap_poses_2d.at(submap_itr->id)
                        .global_pose.inverse() *
                    transform::Project2D(node_it->data.global_pose);

                const double resolution = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                   .submap.get())->grid()->limits().resolution();
                
                // The grid is row-major and poorly named. num_x_cells actually refers to the number of rows (y)
                // and num_y_cells is the number of columns (x)
                const double submap_size_x = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                    .submap.get())->grid()->limits().cell_limits().num_y_cells * resolution;
                const double submap_size_y = static_cast<const Submap2D*>(data_.submap_data.at(submap_itr->id)
                                    .submap.get())->grid()->limits().cell_limits().num_x_cells * resolution;

                const bool in_submap = std::abs(initial_relative_pose.translation().x()) <
                                       submap_size_x / 2.0 &&
                                       std::abs(initial_relative_pose.translation().y()) <
                                       submap_size_y / 2.0;
                
                LOG(INFO)
                      << "Distance to submap x: " << initial_relative_pose.translation().x() 
                      << " Distance to submap y: " << initial_relative_pose.translation().y()
                      << " Submap size x: " << submap_size_x
                      << " Submap size y: " << submap_size_y;

                if (in_submap) {
                  LOG(INFO)
                      << "Search (locally) for global constraint between Node: "
                      << node_id << " and Submap: " << submap_itr->id
                      << " relative_pose: " << initial_relative_pose;

                  mutex_.Unlock();
                  const auto result =
                      constraint_builder_.LocalSearchForConstraint(
                          node_id, submap_itr->id, initial_relative_pose,
                          static_cast<const Submap2D&>(
                              *data_.submap_data.at(submap_itr->id)
                                   .submap.get()),
                          *node_it->data.constant_data.get());
                  mutex_.Lock();

                  if (result) new_constraints.push_back(result.value());
                }
              }
            }
          }
        }
      }
    }
  }
  mutex_.Unlock();

  return new_constraints;
}

void PoseGraph2D::BackgroundThread() {
  // The job of this thread is to periodically search for global constraints and
  // run the optimisation

  while (background_thread_running_) {
    //
    // Search for new constraints
    //
    std::vector<Constraint> new_constraints = FindNewConstraints();

    //
    // Add the new constraints to internal data
    //
    {
      absl::MutexLock locker(&mutex_);

      // add the new constraints
      data_.constraints.insert(data_.constraints.end(), new_constraints.begin(),
                               new_constraints.end());

      for (const Constraint& constraint : new_constraints) {
        UpdateTrajectoryConnectivity(constraint);
      }
    }

    //
    // Run Optimization
    //
    if (!new_constraints.empty()) {
      LOG(INFO) << "Running optimisation with new constraints: "
                << new_constraints.size();
      RunOptimization();

      //
      // Execute trajectory trimmers
      //
      {
        absl::MutexLock locker(&mutex_);

        LOG(INFO) << "Execute trajectory trimmers";
        {
          DeleteTrajectoriesIfNeeded();

          TrimmingHandle trimming_handle(this);
          for (auto& trimmer : trimmers_) {
            trimmer->Trim(&trimming_handle);
          }

          // Remove any finished trimmers
          trimmers_.erase(
              std::remove_if(trimmers_.begin(), trimmers_.end(),
                             [](std::unique_ptr<PoseGraphTrimmer>& trimmer) {
                               return trimmer->IsFinished();
                             }),
              trimmers_.end());
        }
      }

    } else {
      std::size_t num_of_nodes = 0;
      {
        absl::MutexLock locker(&mutex_);
        num_of_nodes = data_.trajectory_nodes.size();
      }

      auto predicate = [this, num_of_nodes]() -> bool {
        return !background_thread_running_ ||
               data_.trajectory_nodes.size() != num_of_nodes;
      };

      if (mutex_.LockWhenWithTimeout(absl::Condition(&predicate),
                                     absl::Milliseconds(10000))) {
        //        LOG(INFO) << "Number of nodes changed!";
      } else {
        //        LOG(INFO) << "Timeout expired!";
      }
      mutex_.Unlock();
    }
  }
}

void PoseGraph2D::RunOptimization(const int32 num_of_iterations) {
  auto options = options_.optimization_problem_options();

  if (num_of_iterations > 0)
    options.mutable_ceres_solver_options()->set_max_num_iterations(
        num_of_iterations);

  optimization::OptimizationProblem2D optimization_problem(options);

  //
  // Copy required data for solve under mutex lock
  //
  MapById<NodeId, optimization::NodeSpec2D> node_data;
  MapById<SubmapId, optimization::SubmapSpec2D> submap_data;
  std::vector<Constraint> constraints;
  std::map<int, PoseGraphInterface::TrajectoryState> trajectories_states;
  std::map<std::string, LandmarkNode> landmark_nodes;

  // before optimization
  std::map<int, transform::Rigid3d> trajectory_local_to_global;
  {
    absl::MutexLock locker(&mutex_);

    for (const auto& item : data_.trajectory_nodes) {
      const transform::Rigid2d local_pose_2d =
          transform::Project2D(item.data.constant_data->local_pose);
      const transform::Rigid2d global_pose_2d =
          transform::Project2D(item.data.global_pose);
      node_data.Insert(
          item.id, optimization::NodeSpec2D{item.data.constant_data->time,
                                            local_pose_2d, global_pose_2d,
                                            Eigen::Quaterniond::Identity()});
    }

    submap_data = data_.global_submap_poses_2d;
    landmark_nodes = data_.landmark_nodes;
    constraints = data_.constraints;

    for (const auto& it : data_.trajectories_state) {
      trajectories_states[it.first] = it.second.state;
    }

    for (const int trajectory_id : data_.trajectory_nodes.trajectory_ids()) {
      trajectory_local_to_global[trajectory_id] = ComputeLocalToGlobalTransform(
          data_.global_submap_poses_2d, trajectory_id);
    }
  }

  //
  // Run Solver
  //
  LOG(INFO) << "Optimization: nodes: " << node_data.size()
            << " submaps: " << submap_data.size()
            << " constraints: " << constraints.size()
            << " landmarks: " << landmark_nodes.size();
  const auto result = optimization_problem.Solve(
      node_data, submap_data, constraints, trajectories_states, landmark_nodes);

  //
  // Post Optimization data management
  //
  if (result.success) {
    {
      absl::MutexLock locker(&mutex_);

      for (const auto item : result.node_poses) {
        data_.trajectory_nodes.at(item.id).global_pose =
            transform::Embed3D(item.data);
      }

      for (const auto item : result.submap_poses) {
        data_.global_submap_poses_2d.at(item.id).global_pose = item.data;
      }

      for (const auto item : result.landmark_poses) {
        data_.landmark_nodes.at(item.first).global_landmark_pose = item.second;
      }

      // Extrapolate all point cloud poses that were not included in the
      // 'optimization_problem_' yet.
      for (const int trajectory_id : data_.trajectory_nodes.trajectory_ids()) {
        const auto local_to_new_global = ComputeLocalToGlobalTransform(
            data_.global_submap_poses_2d, trajectory_id);
        const auto local_to_old_global =
            trajectory_local_to_global.at(trajectory_id);
        const transform::Rigid3d old_global_to_new_global =
            local_to_new_global * local_to_old_global.inverse();

        const NodeId last_optimized_node_id =
            std::prev(node_data.EndOfTrajectory(trajectory_id))->id;
        auto node_it =
            std::next(data_.trajectory_nodes.find(last_optimized_node_id));
        for (; node_it != data_.trajectory_nodes.EndOfTrajectory(trajectory_id);
             ++node_it) {
          auto& mutable_trajectory_node =
              data_.trajectory_nodes.at(node_it->id);
          mutable_trajectory_node.global_pose =
              old_global_to_new_global * mutable_trajectory_node.global_pose;
        }
      }
    }

    if (global_slam_optimization_callback_) {
      std::map<int, NodeId> trajectory_id_to_last_optimized_node_id;
      std::map<int, SubmapId> trajectory_id_to_last_optimized_submap_id;
      {
        for (const int trajectory_id : node_data.trajectory_ids()) {
          if (node_data.SizeOfTrajectoryOrZero(trajectory_id) == 0 ||
              submap_data.SizeOfTrajectoryOrZero(trajectory_id) == 0) {
            continue;
          }
          trajectory_id_to_last_optimized_node_id.emplace(
              trajectory_id,
              std::prev(node_data.EndOfTrajectory(trajectory_id))->id);
          trajectory_id_to_last_optimized_submap_id.emplace(
              trajectory_id,
              std::prev(submap_data.EndOfTrajectory(trajectory_id))->id);
        }
      }
      global_slam_optimization_callback_(
          trajectory_id_to_last_optimized_submap_id,
          trajectory_id_to_last_optimized_node_id);
    }
  }
}

void PoseGraph2D::AddTrajectoryIfNeeded(const int trajectory_id) {
  data_.trajectories_state[trajectory_id];
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::FINISHED);
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::DELETED);
  CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
        InternalTrajectoryState::DeletionState::NORMAL);
  data_.trajectory_connectivity_state.Add(trajectory_id);
}

void PoseGraph2D::AddFixedFramePoseData(
    const int trajectory_id,
    const sensor::FixedFramePoseData& fixed_frame_pose_data) {
  LOG(FATAL) << "Not yet implemented for 2D.";
}

void PoseGraph2D::AddLandmarkData(int trajectory_id,
                                  const sensor::LandmarkData& landmark_data) {
  absl::MutexLock locker(&mutex_);
  CHECK(CanAddWorkItemModifying(trajectory_id));
  for (const auto& observation : landmark_data.landmark_observations) {
    data_.landmark_nodes[observation.id].landmark_observations.emplace_back(
        PoseGraphInterface::LandmarkNode::LandmarkObservation{
            trajectory_id, landmark_data.time,
            observation.landmark_to_tracking_transform,
            observation.translation_weight, observation.rotation_weight});
  }
}

common::Time PoseGraph2D::GetLatestNodeTime(const NodeId& node_id,
                                            const SubmapId& submap_id) const {
  common::Time time = data_.trajectory_nodes.at(node_id).constant_data->time;
  const InternalSubmapData& submap_data = data_.submap_data.at(submap_id);
  if (!submap_data.node_ids.empty()) {
    const NodeId last_submap_node_id =
        *data_.submap_data.at(submap_id).node_ids.rbegin();
    time = std::max(
        time,
        data_.trajectory_nodes.at(last_submap_node_id).constant_data->time);
  }
  return time;
}

void PoseGraph2D::UpdateTrajectoryConnectivity(const Constraint& constraint) {
  CHECK_EQ(constraint.tag, Constraint::INTER_SUBMAP);
  const common::Time time =
      GetLatestNodeTime(constraint.node_id, constraint.submap_id);
  data_.trajectory_connectivity_state.Connect(
      constraint.node_id.trajectory_id, constraint.submap_id.trajectory_id,
      time);
}

void PoseGraph2D::DeleteTrajectoriesIfNeeded() {
  TrimmingHandle trimming_handle(this);
  for (auto& it : data_.trajectories_state) {
    if (it.second.deletion_state ==
        InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION) {
      LOG(INFO) << "trajectory: " << it.first
                << " deletion_state: WAIT_FOR_DELETION";

      auto submap_ids = trimming_handle.GetSubmapIds(it.first);

      for (auto& submap_id : submap_ids) {
        LOG(INFO) << "Trimming submap: " << submap_id;

        trimming_handle.TrimSubmap(submap_id);
      }

      it.second.state = TrajectoryState::DELETED;
      it.second.deletion_state = InternalTrajectoryState::DeletionState::NORMAL;
    }
  }
}

void PoseGraph2D::DeleteTrajectory(const int trajectory_id) {
  // TODO deletion threading needs work

  LOG(INFO) << "DeleteTrajectory: " << trajectory_id;

  absl::MutexLock locker(&mutex_);
  auto it = data_.trajectories_state.find(trajectory_id);
  if (it == data_.trajectories_state.end()) {
    LOG(WARNING) << "Skipping request to delete non-existing trajectory_id: "
                 << trajectory_id;
    return;
  }
  it->second.deletion_state =
      InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION;

  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::ACTIVE);
  CHECK(data_.trajectories_state.at(trajectory_id).state !=
        TrajectoryState::DELETED);
  CHECK(data_.trajectories_state.at(trajectory_id).deletion_state ==
        InternalTrajectoryState::DeletionState::SCHEDULED_FOR_DELETION);
  data_.trajectories_state.at(trajectory_id).deletion_state =
      InternalTrajectoryState::DeletionState::WAIT_FOR_DELETION;
}

void PoseGraph2D::FinishTrajectory(const int trajectory_id) {
  absl::MutexLock locker(&mutex_);
  CHECK(!IsTrajectoryFinished(trajectory_id));
  data_.trajectories_state[trajectory_id].state = TrajectoryState::FINISHED;
  for (const auto& submap : data_.submap_data.trajectory(trajectory_id)) {
    data_.submap_data.at(submap.id).state = SubmapState::kFinished;
  }

  // TODO maybe trigger an optimisation?
}

bool PoseGraph2D::IsTrajectoryFinished(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FINISHED;
}

void PoseGraph2D::FreezeTrajectory(const int trajectory_id) {
  absl::MutexLock locker(&mutex_);
  AddTrajectoryIfNeeded(trajectory_id);
  data_.trajectory_connectivity_state.Add(trajectory_id);
  CHECK(!IsTrajectoryFrozen(trajectory_id));
  data_.trajectories_state[trajectory_id].state = TrajectoryState::FROZEN;
}

bool PoseGraph2D::IsTrajectoryFrozen(const int trajectory_id) const {
  return data_.trajectories_state.count(trajectory_id) != 0 &&
         data_.trajectories_state.at(trajectory_id).state ==
             TrajectoryState::FROZEN;
}

void PoseGraph2D::AddSubmapFromProto(
    const transform::Rigid3d& global_submap_pose, const proto::Submap& submap) {
  CHECK(submap.has_submap_2d());

  const SubmapId submap_id = {submap.submap_id().trajectory_id(),
                              submap.submap_id().submap_index()};

  const transform::Rigid2d global_submap_pose_2d =
      transform::Project2D(global_submap_pose);
  {
    absl::MutexLock locker(&mutex_);
    const std::shared_ptr<const Submap2D> submap_ptr =
        std::make_shared<const Submap2D>(submap.submap_2d(),
                                         &conversion_tables_);

    AddTrajectoryIfNeeded(submap_id.trajectory_id);

    CHECK(IsTrajectoryFrozen(submap_id.trajectory_id));
    CHECK(CanAddWorkItemModifying(submap_id.trajectory_id));

    data_.submap_data.Insert(submap_id, InternalSubmapData());
    data_.submap_data.at(submap_id).submap = submap_ptr;

    // Immediately show the submap at the 'global_submap_pose'.
    data_.global_submap_poses_2d.Insert(
        submap_id, optimization::SubmapSpec2D{global_submap_pose_2d});

    data_.submap_data.at(submap_id).state = SubmapState::kFinished;
    //    optimization_problem_->InsertSubmap(submap_id, global_submap_pose_2d);
  }
}

void PoseGraph2D::AddNodeFromProto(const transform::Rigid3d& global_pose,
                                   const proto::Node& node) {
  const NodeId node_id = {node.node_id().trajectory_id(),
                          node.node_id().node_index()};
  std::shared_ptr<const TrajectoryNode::Data> constant_data =
      std::make_shared<const TrajectoryNode::Data>(FromProto(node.node_data()));

  {
    absl::MutexLock locker(&mutex_);
    AddTrajectoryIfNeeded(node_id.trajectory_id);
    CHECK(CanAddWorkItemModifying(node_id.trajectory_id));
    data_.trajectory_nodes.Insert(node_id,
                                  TrajectoryNode{constant_data, global_pose});
  }
}

void PoseGraph2D::SetTrajectoryDataFromProto(
    const proto::TrajectoryData& data) {
  LOG(ERROR) << "not implemented";
}

void PoseGraph2D::AddNodeToSubmap(const NodeId& node_id,
                                  const SubmapId& submap_id) {
  absl::MutexLock locker(&mutex_);
  CHECK(CanAddWorkItemModifying(submap_id.trajectory_id));
  data_.submap_data.at(submap_id).node_ids.insert(node_id);
}

void PoseGraph2D::AddSerializedConstraints(
    const std::vector<Constraint>& constraints) {
  absl::MutexLock locker(&mutex_);
  for (const auto& constraint : constraints) {
    CHECK(data_.trajectory_nodes.Contains(constraint.node_id));
    CHECK(data_.submap_data.Contains(constraint.submap_id));
    CHECK(data_.trajectory_nodes.at(constraint.node_id).constant_data !=
          nullptr);
    CHECK(data_.submap_data.at(constraint.submap_id).submap != nullptr);
    switch (constraint.tag) {
      case Constraint::Tag::INTRA_SUBMAP:
        CHECK(data_.submap_data.at(constraint.submap_id)
                  .node_ids.emplace(constraint.node_id)
                  .second);
        break;
      case Constraint::Tag::INTER_SUBMAP:
        UpdateTrajectoryConnectivity(constraint);
        break;
    }
    const Constraint::Pose pose = {constraint.pose.zbar_ij,
                                   constraint.pose.translation_weight,
                                   constraint.pose.rotation_weight};
    data_.constraints.push_back(Constraint{
        constraint.submap_id, constraint.node_id, pose, constraint.tag});
  }
  LOG(INFO) << "Loaded " << constraints.size() << " constraints.";
}

void PoseGraph2D::AddTrimmer(std::unique_ptr<PoseGraphTrimmer> trimmer) {
  PoseGraphTrimmer* const trimmer_ptr = trimmer.release();
  absl::MutexLock locker(&mutex_);
  trimmers_.emplace_back(trimmer_ptr);
}

void PoseGraph2D::RunFinalOptimization() {
  StopBackgroundThread();
  RunOptimization(options_.max_num_final_iterations());
  StartBackgroundThread();
}

bool PoseGraph2D::CanAddWorkItemModifying(int trajectory_id) {
  auto it = data_.trajectories_state.find(trajectory_id);
  if (it == data_.trajectories_state.end()) {
    return true;
  }
  if (it->second.state == TrajectoryState::FINISHED) {
    // TODO(gaschler): Replace all FATAL to WARNING after some testing.
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has finished "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.deletion_state !=
      InternalTrajectoryState::DeletionState::NORMAL) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been scheduled for deletion "
                  "but modification is requested, skipping.";
    return false;
  }
  if (it->second.state == TrajectoryState::DELETED) {
    LOG(FATAL) << "trajectory_id " << trajectory_id
               << " has been deleted "
                  "but modification is requested, skipping.";
    return false;
  }
  return true;
}

MapById<NodeId, TrajectoryNode> PoseGraph2D::GetTrajectoryNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_nodes;
}

MapById<NodeId, TrajectoryNodePose> PoseGraph2D::GetTrajectoryNodePoses()
    const {
  MapById<NodeId, TrajectoryNodePose> node_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& node_id_data : data_.trajectory_nodes) {
    absl::optional<TrajectoryNodePose::ConstantPoseData> constant_pose_data;
    if (node_id_data.data.constant_data != nullptr) {
      constant_pose_data = TrajectoryNodePose::ConstantPoseData{
          node_id_data.data.constant_data->time,
          node_id_data.data.constant_data->local_pose};
    }
    node_poses.Insert(
        node_id_data.id,
        TrajectoryNodePose{node_id_data.data.global_pose, constant_pose_data});
  }
  return node_poses;
}

std::map<int, PoseGraphInterface::TrajectoryState>
PoseGraph2D::GetTrajectoryStates() const {
  std::map<int, PoseGraphInterface::TrajectoryState> trajectories_state;
  absl::MutexLock locker(&mutex_);
  for (const auto& it : data_.trajectories_state) {
    trajectories_state[it.first] = it.second.state;
  }
  return trajectories_state;
}

std::map<std::string, transform::Rigid3d> PoseGraph2D::GetLandmarkPoses()
    const {
  std::map<std::string, transform::Rigid3d> landmark_poses;
  absl::MutexLock locker(&mutex_);
  for (const auto& landmark : data_.landmark_nodes) {
    // Landmark without value has not been optimized yet.
    if (!landmark.second.global_landmark_pose.has_value()) continue;
    landmark_poses[landmark.first] =
        landmark.second.global_landmark_pose.value();
  }
  return landmark_poses;
}

void PoseGraph2D::SetLandmarkPose(const std::string& landmark_id,
                                  const transform::Rigid3d& global_pose,
                                  const bool frozen) {
  absl::MutexLock locker(&mutex_);
  data_.landmark_nodes[landmark_id].global_landmark_pose = global_pose;
  data_.landmark_nodes[landmark_id].frozen = frozen;
}

std::map<std::string /* landmark ID */, PoseGraphInterface::LandmarkNode>
PoseGraph2D::GetLandmarkNodes() const {
  absl::MutexLock locker(&mutex_);
  return data_.landmark_nodes;
}

std::map<int, PoseGraphInterface::TrajectoryData>
PoseGraph2D::GetTrajectoryData() const {
  // The 2D optimization problem does not have any 'TrajectoryData'.
  return {};
}

sensor::MapByTime<sensor::FixedFramePoseData>
PoseGraph2D::GetFixedFramePoseData() const {
  // FixedFramePoseData is not yet implemented for 2D. We need to return empty
  // so serialization works.
  return {};
}

std::vector<PoseGraphInterface::Constraint> PoseGraph2D::constraints() const {
  std::vector<PoseGraphInterface::Constraint> result;
  absl::MutexLock locker(&mutex_);
  for (const Constraint& constraint : data_.constraints) {
    result.push_back(
        Constraint{constraint.submap_id, constraint.node_id,
                   Constraint::Pose{constraint.pose.zbar_ij,
                                    constraint.pose.translation_weight,
                                    constraint.pose.rotation_weight},
                   constraint.tag});
  }
  return result;
}

void PoseGraph2D::SetInitialTrajectoryPose(const int from_trajectory_id,
                                           const int to_trajectory_id,
                                           const transform::Rigid3d& pose,
                                           const common::Time time) {
  absl::MutexLock locker(&mutex_);
  data_.initial_trajectory_poses[from_trajectory_id] =
      InitialTrajectoryPose{to_trajectory_id, pose, time};
}

transform::Rigid3d PoseGraph2D::GetInterpolatedGlobalTrajectoryPose(
    const int trajectory_id, const common::Time time) const {
  CHECK_GT(data_.trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 0);
  const auto it = data_.trajectory_nodes.lower_bound(trajectory_id, time);
  if (it == data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)) {
    return data_.trajectory_nodes.BeginOfTrajectory(trajectory_id)
        ->data.global_pose;
  }
  if (it == data_.trajectory_nodes.EndOfTrajectory(trajectory_id)) {
    return std::prev(data_.trajectory_nodes.EndOfTrajectory(trajectory_id))
        ->data.global_pose;
  }
  return transform::Interpolate(
             transform::TimestampedTransform{std::prev(it)->data.time(),
                                             std::prev(it)->data.global_pose},
             transform::TimestampedTransform{it->data.time(),
                                             it->data.global_pose},
             time)
      .transform;
}

transform::Rigid3d PoseGraph2D::GetLocalToGlobalTransform(
    const int trajectory_id) const {
  absl::MutexLock locker(&mutex_);
  return ComputeLocalToGlobalTransform(data_.global_submap_poses_2d,
                                       trajectory_id);
}

std::vector<std::vector<int>> PoseGraph2D::GetConnectedTrajectories() const {
  absl::MutexLock locker(&mutex_);
  return data_.trajectory_connectivity_state.Components();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapData(
    const SubmapId& submap_id) const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock(submap_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetAllSubmapData() const {
  absl::MutexLock locker(&mutex_);
  return GetSubmapDataUnderLock();
}

MapById<SubmapId, PoseGraphInterface::SubmapPose>
PoseGraph2D::GetAllSubmapPoses() const {
  absl::MutexLock locker(&mutex_);
  MapById<SubmapId, SubmapPose> submap_poses;
  for (const auto& submap_id_data : data_.submap_data) {
    auto submap_data = GetSubmapDataUnderLock(submap_id_data.id);
    submap_poses.Insert(
        submap_id_data.id,
        PoseGraph::SubmapPose{submap_data.submap->num_range_data(),
                              submap_data.pose});
  }
  return submap_poses;
}

transform::Rigid3d PoseGraph2D::ComputeLocalToGlobalTransform(
    const MapById<SubmapId, optimization::SubmapSpec2D>& global_submap_poses,
    const int trajectory_id) const {
  auto begin_it = global_submap_poses.BeginOfTrajectory(trajectory_id);
  auto end_it = global_submap_poses.EndOfTrajectory(trajectory_id);
  if (begin_it == end_it) {
    const auto it = data_.initial_trajectory_poses.find(trajectory_id);
    if (it != data_.initial_trajectory_poses.end()) {
      return GetInterpolatedGlobalTrajectoryPose(it->second.to_trajectory_id,
                                                 it->second.time) *
             it->second.relative_pose;
    } else {
      return transform::Rigid3d::Identity();
    }
  }
  const SubmapId last_optimized_submap_id = std::prev(end_it)->id;
  // Accessing 'local_pose' in Submap is okay, since the member is const.
  return transform::Embed3D(
             global_submap_poses.at(last_optimized_submap_id).global_pose) *
         data_.submap_data.at(last_optimized_submap_id)
             .submap->local_pose()
             .inverse();
}

PoseGraphInterface::SubmapData PoseGraph2D::GetSubmapDataUnderLock(
    const SubmapId& submap_id) const {
  const auto it = data_.submap_data.find(submap_id);
  if (it == data_.submap_data.end()) {
    return {};
  }
  auto submap = it->data.submap;
  if (data_.global_submap_poses_2d.Contains(submap_id)) {
    // We already have an optimized pose.
    return {submap,
            transform::Embed3D(
                data_.global_submap_poses_2d.at(submap_id).global_pose)};
  }
  // We have to extrapolate.
  return {submap, ComputeLocalToGlobalTransform(data_.global_submap_poses_2d,
                                                submap_id.trajectory_id) *
                      submap->local_pose()};
}

PoseGraph2D::TrimmingHandle::TrimmingHandle(PoseGraph2D* const parent)
    : parent_(parent) {}

int PoseGraph2D::TrimmingHandle::num_submaps(const int trajectory_id) const {
  const auto& submap_data = parent_->data_.global_submap_poses_2d;
  return submap_data.SizeOfTrajectoryOrZero(trajectory_id);
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::TrimmingHandle::GetOptimizedSubmapData() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : parent_->data_.submap_data) {
    if (submap_id_data.data.state != SubmapState::kFinished ||
        !parent_->data_.global_submap_poses_2d.Contains(submap_id_data.id)) {
      continue;
    }
    submaps.Insert(
        submap_id_data.id,
        SubmapData{submap_id_data.data.submap,
                   transform::Embed3D(parent_->data_.global_submap_poses_2d
                                          .at(submap_id_data.id)
                                          .global_pose)});
  }
  return submaps;
}

std::vector<SubmapId> PoseGraph2D::TrimmingHandle::GetSubmapIds(
    int trajectory_id) const {
  std::vector<SubmapId> submap_ids;
  const auto& submap_data = parent_->data_.global_submap_poses_2d;
  for (const auto& it : submap_data.trajectory(trajectory_id)) {
    submap_ids.push_back(it.id);
  }
  return submap_ids;
}

const MapById<NodeId, TrajectoryNode>&
PoseGraph2D::TrimmingHandle::GetTrajectoryNodes() const {
  return parent_->data_.trajectory_nodes;
}

const std::vector<PoseGraphInterface::Constraint>&
PoseGraph2D::TrimmingHandle::GetConstraints() const {
  return parent_->data_.constraints;
}

const MapById<SubmapId, std::set<NodeId>>
PoseGraph2D::TrimmingHandle::GetSubmapNodes() const {
  MapById<SubmapId, std::set<NodeId>> submap_nodes;
  for (const auto item : parent_->data_.submap_data)
    submap_nodes.Insert(item.id, item.data.node_ids);
  return submap_nodes;
}

bool PoseGraph2D::TrimmingHandle::IsFinished(const int trajectory_id) const {
  return parent_->IsTrajectoryFinished(trajectory_id);
}

void PoseGraph2D::TrimmingHandle::SetTrajectoryState(int trajectory_id,
                                                     TrajectoryState state) {
  parent_->data_.trajectories_state[trajectory_id].state = state;
}

void PoseGraph2D::TrimmingHandle::TrimSubmap(const SubmapId& submap_id) {
  LOG(INFO) << "TrimSubmap: " << submap_id;

  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);

  // Compile all nodes that are still INTRA_SUBMAP constrained once the submap
  // with 'submap_id' is gone.
  std::set<NodeId> nodes_to_retain;
  for (const Constraint& constraint : parent_->data_.constraints) {
    if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
        constraint.submap_id != submap_id) {
      nodes_to_retain.insert(constraint.node_id);
    }
  }

  // Remove all 'data_.constraints' related to 'submap_id'.
  std::set<NodeId> nodes_to_remove;
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (constraint.submap_id == submap_id) {
        if (constraint.tag == Constraint::Tag::INTRA_SUBMAP &&
            nodes_to_retain.count(constraint.node_id) == 0) {
          // This node will no longer be INTRA_SUBMAP constrained and has to be
          // removed.
          nodes_to_remove.insert(constraint.node_id);
        }
      } else {
        constraints.push_back(constraint);
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }
  // Remove all 'data_.constraints' related to 'nodes_to_remove'.
  {
    std::vector<Constraint> constraints;
    for (const Constraint& constraint : parent_->data_.constraints) {
      if (nodes_to_remove.count(constraint.node_id) == 0) {
        constraints.push_back(constraint);
      }
    }
    parent_->data_.constraints = std::move(constraints);
  }

  // Mark the submap with 'submap_id' as trimmed and remove its data.
  CHECK(parent_->data_.submap_data.at(submap_id).state ==
        SubmapState::kFinished);
  parent_->data_.submap_data.Trim(submap_id);
  parent_->data_.global_submap_poses_2d.Trim(submap_id);
  parent_->constraint_builder_.DeleteScanMatcher(submap_id);

  // Remove the 'nodes_to_remove' from the pose graph and the optimization
  // problem.
  for (const NodeId& node_id : nodes_to_remove) {
    parent_->data_.trajectory_nodes.Trim(node_id);
  }
}

MapById<SubmapId, PoseGraphInterface::SubmapData>
PoseGraph2D::GetSubmapDataUnderLock() const {
  MapById<SubmapId, PoseGraphInterface::SubmapData> submaps;
  for (const auto& submap_id_data : data_.submap_data) {
    submaps.Insert(submap_id_data.id,
                   GetSubmapDataUnderLock(submap_id_data.id));
  }
  return submaps;
}

void PoseGraph2D::SetGlobalSlamOptimizationCallback(
    PoseGraphInterface::GlobalSlamOptimizationCallback callback) {
  global_slam_optimization_callback_ = callback;
}

void PoseGraph2D::RegisterMetrics(metrics::FamilyFactory* family_factory) {}

}  // namespace mapping
}  // namespace cartographer
