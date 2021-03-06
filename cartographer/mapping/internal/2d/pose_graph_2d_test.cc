/*
 * Copyright 2016 The Cartographer Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cartographer/mapping/internal/2d/pose_graph_2d.h"

#include <cmath>
#include <memory>
#include <random>

#include "absl/memory/memory.h"
#include "cartographer/common/lua_parameter_dictionary_test_helpers.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/2d/active_submaps_2d.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/rigid_transform_test_helpers.h"
#include "cartographer/transform/transform.h"
#include "gmock/gmock.h"

namespace cartographer {
namespace mapping {
namespace {

class PoseGraph2DTest : public ::testing::Test {
 protected:
  PoseGraph2DTest() : thread_pool_(1) {
    // Builds a wavy, irregularly circular point cloud that is unique
    // rotationally. This gives us good rotational texture and avoids any
    // possibility of the CeresScanMatcher2D preferring free space (>
    // kMinProbability) to unknown space (== kMinProbability).
    for (float t = 0.f; t < 2.f * M_PI; t += 0.005f) {
      const float r = (std::sin(20.f * t) + 2.f) * std::sin(t + 2.f);
      point_cloud_.push_back(
          {Eigen::Vector3f{r * std::sin(t), r * std::cos(t), 0.f}});
    }

    {
      auto parameter_dictionary = common::MakeDictionary(R"text(
          return {
            num_range_data = 100,
            grid_options_2d = {
              grid_type = "PROBABILITY_GRID",
              resolution = 0.05,
            },
            range_data_inserter = {
              range_data_inserter_type = "PROBABILITY_GRID_INSERTER_2D",
              probability_grid_range_data_inserter = {
                insert_free_space = true,
                hit_probability = 0.53,
                miss_probability = 0.495,
              },
            tsdf_range_data_inserter = {
              truncation_distance = 0.3,
              maximum_weight = 10.,
              update_free_space = false,
              normal_estimation_options = {
                num_normal_samples = 4,
                sample_radius = 0.5,
              },
              project_sdf_distance_to_scan_normal = false,
              update_weight_range_exponent = 0,
              update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0,
              update_weight_distance_cell_to_hit_kernel_bandwidth = 0,
            },
          },
          min_feature_observations = 15,
          max_feature_score = 0.5,
        })text");
      active_submaps_ = absl::make_unique<ActiveSubmaps2D>(
          mapping::CreateSubmapsOptions2D(parameter_dictionary.get()));
    }

    {
      auto parameter_dictionary = common::MakeDictionary(R"text(
          return {
             constraint_builder = {
                min_local_search_score = 0.40,
                min_global_search_score = 0.30,

                -- used when adding INTER submap constraints
                constraint_translation_weight = 2,
                constraint_rotation_weight = 2,
                ceres_scan_matcher = {
                    occupied_space_weight = 1,
                    translation_weight = 1,
                    rotation_weight = 1,
                    ceres_solver_options = {
                        use_nonmonotonic_steps = true,
                        max_num_iterations = 100,
                        num_threads = 1,
                    },
                },
                global_icp_scan_matcher_options_2d = {
                    num_global_samples_per_sq_m = 3,
                    num_global_rotations = 128,

                    proposal_point_inlier_threshold = 1.0,
                    proposal_feature_inlier_threshold = 1.0,

                    proposal_min_points_inlier_fraction = 0.4,
                    proposal_min_features_inlier_fraction = 0.5,

                    proposal_features_weight = 2.0,
                    proposal_points_weight = 1.0,

                    proposal_raytracing_max_error = 0.4,

                    proposal_max_points_error = 0.5,
                    proposal_max_features_error = 1.2,
                    proposal_max_error = 0.5,

                    min_cluster_size = 1,
                    max_cluster_size = 4,
                    min_cluster_distance = 0.4,

                    num_local_samples = 8,

                    local_sample_linear_distance = 0.2,
                    local_sample_angular_distance = 0.1,

                    icp_options = {
                        nearest_neighbour_point_huber_loss = 0.01,
                        nearest_neighbour_feature_huber_loss = 0.01,

                        point_pair_point_huber_loss = 0.01,
                        point_pair_feature_huber_loss = 0.01,

                        point_weight = 1.0,
                        feature_weight = 0.5,

                        point_inlier_threshold = 1.0,
                        feature_inlier_threshold = 1.0,

                        -- Used for evaluating match
                        raytrace_threshold = 0.3;
                        hit_threshold = 0.3;
                        feature_match_threshold = 0.2,
                    }
                },
                min_icp_score = 0.97,
                min_icp_points_inlier_fraction = 0.6,
                min_icp_features_inlier_fraction = 0.5,
                min_hit_fraction = 0.65,
                min_ray_trace_fraction = 0.85,
                min_icp_features_match_fraction = 0.6,
             },

            -- used when adding INTRA submap constraints
            -- these are from node to current submap and previous submap
            matcher_translation_weight = 1,
            matcher_rotation_weight = 1,
            optimization_problem = {
                huber_scale = 1e1, -- only for inter-submap constraints

                -- these are between nodes based on front-end mapping
                local_slam_pose_translation_weight = 0,
                local_slam_pose_rotation_weight = 0,

                fixed_frame_pose_translation_weight = 1e1, -- only in 3d
                fixed_frame_pose_rotation_weight = 1e2, -- only in 3d

                log_solver_summary = false,
                ceres_solver_options = {
                    use_nonmonotonic_steps = false,
                    max_num_iterations = 50,
                    num_threads = 7,
                },
            },
            max_num_final_iterations = 200,

            --  overlapping_submaps_trimmer_2d = {
            --    fresh_submaps_count = 1,
            --    min_covered_area = 2,
            --    min_added_submaps_count = 5,
            --  },

            -- keep searching globally until this many found in total
            min_globally_searched_constraints_for_trajectory = 1,

            -- keep searching locally until this many inside submap
            local_constraint_every_n_nodes = 8,

            -- keep searching globally until this many inside submap
            global_constraint_every_n_nodes = 8,

            max_constraint_match_distance = 9.0,
          })text");
      auto options = CreatePoseGraphOptions(parameter_dictionary.get());
      pose_graph_ = absl::make_unique<PoseGraph2D>(options);
    }

    current_pose_ = transform::Rigid2d::Identity();
  }

  void MoveRelativeWithNoise(const transform::Rigid2d& movement,
                             const transform::Rigid2d& noise) {
    current_pose_ = current_pose_ * movement;
    const sensor::PointCloud new_point_cloud = sensor::TransformPointCloud(
        point_cloud_,
        transform::Embed3D(current_pose_.inverse().cast<float>()));
    const sensor::RangeData range_data{
        Eigen::Vector3f::Zero(), new_point_cloud, {}};
    const transform::Rigid2d pose_estimate = noise * current_pose_;
    constexpr int kTrajectoryId = 0;
    active_submaps_->InsertRangeData(TransformRangeData(
        range_data, transform::Embed3D(pose_estimate.cast<float>())));
    std::vector<std::shared_ptr<const Submap2D>> insertion_submaps;
    for (const auto& submap : active_submaps_->submaps()) {
      insertion_submaps.push_back(submap);
    }
    pose_graph_->AddNode(
        std::make_shared<const TrajectoryNode::Data>(
            TrajectoryNode::Data{common::FromUniversal(0),
                                 {},
                                 range_data.returns,
                                 {},
                                 transform::Embed3D(pose_estimate)}),
        kTrajectoryId, insertion_submaps);
  }

  void MoveRelative(const transform::Rigid2d& movement) {
    MoveRelativeWithNoise(movement, transform::Rigid2d::Identity());
  }

  template <typename Range>
  std::vector<int> ToVectorInt(const Range& range) {
    return std::vector<int>(range.begin(), range.end());
  }

  sensor::PointCloud point_cloud_;
  std::unique_ptr<ActiveSubmaps2D> active_submaps_;
  common::ThreadPool thread_pool_;
  std::unique_ptr<PoseGraph2D> pose_graph_;
  transform::Rigid2d current_pose_;
};

TEST_F(PoseGraph2DTest, EmptyMap) {
  pose_graph_->RunFinalOptimization();
  const auto nodes = pose_graph_->GetTrajectoryNodes();
  EXPECT_TRUE(nodes.empty());
}

TEST_F(PoseGraph2DTest, NoMovement) {
  MoveRelative(transform::Rigid2d::Identity());
  MoveRelative(transform::Rigid2d::Identity());
  MoveRelative(transform::Rigid2d::Identity());
  pose_graph_->RunFinalOptimization();
  const auto nodes = pose_graph_->GetTrajectoryNodes();
  ASSERT_THAT(ToVectorInt(nodes.trajectory_ids()),
              ::testing::ContainerEq(std::vector<int>{0}));
  EXPECT_THAT(nodes.SizeOfTrajectoryOrZero(0), ::testing::Eq(3u));
  EXPECT_THAT(nodes.at(NodeId{0, 0}).global_pose,
              transform::IsNearly(transform::Rigid3d::Identity(), 1e-2));
  EXPECT_THAT(nodes.at(NodeId{0, 1}).global_pose,
              transform::IsNearly(transform::Rigid3d::Identity(), 1e-2));
  EXPECT_THAT(nodes.at(NodeId{0, 2}).global_pose,
              transform::IsNearly(transform::Rigid3d::Identity(), 1e-2));
}

TEST_F(PoseGraph2DTest, NoOverlappingNodes) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  std::vector<transform::Rigid2d> poses;
  for (int i = 0; i != 4; ++i) {
    MoveRelative(transform::Rigid2d({0.25 * distribution(rng), 5.}, 0.));
    poses.emplace_back(current_pose_);
  }
  pose_graph_->RunFinalOptimization();
  const auto nodes = pose_graph_->GetTrajectoryNodes();
  ASSERT_THAT(ToVectorInt(nodes.trajectory_ids()),
              ::testing::ContainerEq(std::vector<int>{0}));
  for (int i = 0; i != 4; ++i) {
    EXPECT_THAT(
        poses[i],
        IsNearly(transform::Project2D(nodes.at(NodeId{0, i}).global_pose),
                 1e-2))
        << i;
  }
}

TEST_F(PoseGraph2DTest, ConsecutivelyOverlappingNodes) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  std::vector<transform::Rigid2d> poses;
  for (int i = 0; i != 5; ++i) {
    MoveRelative(transform::Rigid2d({0.25 * distribution(rng), 2.}, 0.));
    poses.emplace_back(current_pose_);
  }
  pose_graph_->RunFinalOptimization();
  const auto nodes = pose_graph_->GetTrajectoryNodes();
  ASSERT_THAT(ToVectorInt(nodes.trajectory_ids()),
              ::testing::ContainerEq(std::vector<int>{0}));
  for (int i = 0; i != 5; ++i) {
    EXPECT_THAT(
        poses[i],
        IsNearly(transform::Project2D(nodes.at(NodeId{0, i}).global_pose),
                 1e-2))
        << i;
  }
}

TEST_F(PoseGraph2DTest, OverlappingNodes) {
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  std::vector<transform::Rigid2d> ground_truth;
  std::vector<transform::Rigid2d> poses;

  const int number_of_nodes = 5;

  for (int i = 0; i < number_of_nodes; ++i) {
    const double noise_x = 0.0001 * distribution(rng);
    const double noise_y = 0.0001 * distribution(rng);
    const double noise_orientation = 0.0001 * distribution(rng);
    transform::Rigid2d noise({noise_x, noise_y}, noise_orientation);
    MoveRelativeWithNoise(
        transform::Rigid2d({0.15 * distribution(rng), 0.0}, 0.), noise);
    ground_truth.emplace_back(current_pose_);
    poses.emplace_back(noise * current_pose_);
  }
  pose_graph_->RunFinalOptimization();

  LOG(INFO) << "Number of submaps: " << active_submaps_->submaps().size();

  auto submap = active_submaps_->submaps().front();
  //  const ProbabilityGrid* pg =
  //      dynamic_cast<const ProbabilityGrid*>(submap->grid());

  //  auto surface = pg->DrawSurface();
  //  cairo_surface_write_to_png(surface.get(), "test.png");
  const auto nodes = pose_graph_->GetTrajectoryNodes();

  // check how many nodes exist
  ASSERT_THAT(nodes.size(), number_of_nodes);
  ASSERT_THAT(ToVectorInt(nodes.trajectory_ids()),
              ::testing::ContainerEq(std::vector<int>{0}));

  transform::Rigid2d true_movement =
      ground_truth.front().inverse() * ground_truth.back();
  transform::Rigid2d movement_before = poses.front().inverse() * poses.back();
  transform::Rigid2d error_before = movement_before.inverse() * true_movement;

  transform::Rigid3d optimized_movement =
      nodes.BeginOfTrajectory(0)->data.global_pose.inverse() *
      std::prev(nodes.EndOfTrajectory(0))->data.global_pose;
  transform::Rigid2d optimized_error =
      transform::Project2D(optimized_movement).inverse() * true_movement;

  LOG(INFO) << "true_movement: " << true_movement;
  LOG(INFO) << "movement_before: " << movement_before;
  LOG(INFO) << "error_before: " << error_before;

  LOG(INFO) << "optimized_movement: " << optimized_movement;
  LOG(INFO) << "optimized_error: " << optimized_error;

  //  EXPECT_THAT(std::abs(optimized_error.normalized_angle()),
  //  ::testing::Lt(std::abs(error_before.normalized_angle())));
  //  EXPECT_THAT(optimized_error.translation().norm(),
  //  ::testing::Lt(error_before.translation().norm()));
}

}  // namespace
}  // namespace mapping
}  // namespace cartographer
