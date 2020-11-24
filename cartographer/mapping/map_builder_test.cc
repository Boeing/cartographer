/*
 * Copyright 2017 The Cartographer Authors
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

#include "cartographer/mapping/map_builder.h"

#include "cartographer/common/config.h"
#include "cartographer/io/proto_stream.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/internal/testing/test_helpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cartographer {
namespace mapping {
namespace {

using SensorId = cartographer::mapping::TrajectoryBuilderInterface::SensorId;

const SensorId kRangeSensorId{SensorId::SensorType::RANGE, "range"};
const SensorId kOdomSensorId{SensorId::SensorType::ODOMETRY, "odom"};

constexpr double kDuration = 4.;         // Seconds.
constexpr double kTimeStep = 0.1;        // Seconds.
constexpr double kTravelDistance = 2.0;  // Meters.

template <class T>
class MapBuilderTestBase : public T {
 protected:
  void SetUp() override {
    // Global SLAM optimization is not executed.
    const std::string kMapBuilderLua = R"text(
      return {
        pose_graph = {
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
            },
            collate_by_trajectory = false,
       })text";
    auto map_builder_parameters = testing::ResolveLuaParameters(kMapBuilderLua);
    map_builder_options_ =
        CreateMapBuilderOptions(map_builder_parameters.get());
    // Multiple submaps are created because of a small 'num_range_data'.
    const std::string kTrajectoryBuilderLua = R"text(
          return {
            trajectory_builder_2d = {
                min_range = 0.,
                max_range = 28.,
                min_z = -0.8,
                max_z = 2.,
                missing_data_ray_length = 12.,

                circle_feature_options = {
                    detect_radii = {0.06}
                },

                -- used before scan matching
                voxel_filter_size = 0.01,

                -- used before scan matching
                adaptive_voxel_filter = {
                    max_length = 0.5,
                    min_num_points = 200,
                    max_range = 50.,
                },
                ceres_scan_matcher = {
                    occupied_space_weight = 10.,
                    translation_weight = 1.,
                    rotation_weight = 1.,
                    ceres_solver_options = {
                        use_nonmonotonic_steps = false,
                        max_num_iterations = 20,
                        num_threads = 1,
                    },
                },
                motion_filter = {
                    max_time_seconds = 5.,
                    max_distance_meters = 0.1,
                    max_angle_radians = math.rad(10.0),
                },
                submaps = {
                    num_range_data = 30,
                    grid_options_2d = {
                        grid_type = "PROBABILITY_GRID",
                        resolution = 0.02,
                    },
                    range_data_inserter = {
                        range_data_inserter_type = "PROBABILITY_GRID_INSERTER_2D",
                        probability_grid_range_data_inserter = {
                            insert_free_space = true,
                            hit_probability = 0.55,
                            miss_probability = 0.49,
                        },
                        tsdf_range_data_inserter = {
                            truncation_distance = 0.3,
                            maximum_weight = 10.,
                            update_free_space = false,
                            normal_estimation_options = {
                                num_normal_samples = 4,
                                sample_radius = 0.5,
                            },
                            project_sdf_distance_to_scan_normal = true,
                            update_weight_range_exponent = 0,
                            update_weight_angle_scan_normal_to_ray_kernel_bandwidth = 0.5,
                            update_weight_distance_cell_to_hit_kernel_bandwidth = 0.5,
                        },
                    },
                    min_feature_observations = 15,
                    max_feature_score = 0.5,
                },
            },
            --  pure_localization_trimmer = {
            --    max_submaps_to_keep = 3,
            --  },
            collate_fixed_frame = true,
            collate_landmarks = false,
          })text";
    auto trajectory_builder_parameters =
        testing::ResolveLuaParameters(kTrajectoryBuilderLua);
    trajectory_builder_options_ =
        CreateTrajectoryBuilderOptions(trajectory_builder_parameters.get());
  }

  void BuildMapBuilder() {
    map_builder_ = absl::make_unique<MapBuilder>(map_builder_options_);
  }

  void SetOptionsToTSDF2D() {
    trajectory_builder_options_.mutable_trajectory_builder_2d_options()
        ->mutable_submaps_options()
        ->mutable_range_data_inserter_options()
        ->set_range_data_inserter_type(
            proto::RangeDataInserterOptions::TSDF_INSERTER_2D);
    trajectory_builder_options_.mutable_trajectory_builder_2d_options()
        ->mutable_submaps_options()
        ->mutable_grid_options_2d()
        ->set_grid_type(proto::GridOptions2D::TSDF);
    trajectory_builder_options_.mutable_trajectory_builder_2d_options()
        ->mutable_ceres_scan_matcher_options()
        ->set_occupied_space_weight(10.0);
    map_builder_options_.mutable_pose_graph_options()
        ->mutable_constraint_builder_options()
        ->mutable_ceres_scan_matcher_options()
        ->set_occupied_space_weight(50.0);
  }

  void SetOptionsEnableGlobalOptimization() {
    trajectory_builder_options_.mutable_trajectory_builder_2d_options()
        ->mutable_motion_filter_options()
        ->set_max_distance_meters(0);
  }

  MapBuilderInterface::LocalSlamResultCallback GetLocalSlamResultCallback() {
    return [=](const int trajectory_id, const ::cartographer::common::Time time,
               const ::cartographer::transform::Rigid3d local_pose,
               const ::cartographer::transform::Rigid3d odom,
               const std::unique_ptr<
                   const cartographer::mapping::TrajectoryBuilderInterface::
                       InsertionResult>) {
      local_slam_result_poses_.push_back(local_pose);
    };
  }

  int CreateTrajectoryWithFakeData(double timestamps_add_duration = 0.) {
    int trajectory_id = map_builder_->AddTrajectoryBuilder(
        {kRangeSensorId}, trajectory_builder_options_,
        GetLocalSlamResultCallback());
    TrajectoryBuilderInterface* trajectory_builder =
        map_builder_->GetTrajectoryBuilder(trajectory_id);
    auto measurements = testing::GenerateFakeRangeMeasurements(
        kTravelDistance, kDuration, kTimeStep);
    for (auto& measurement : measurements) {
      measurement.time += common::FromSeconds(timestamps_add_duration);
      trajectory_builder->AddSensorData(kRangeSensorId.id, measurement);
    }
    map_builder_->FinishTrajectory(trajectory_id);
    return trajectory_id;
  }

  std::unique_ptr<MapBuilderInterface> map_builder_;
  proto::MapBuilderOptions map_builder_options_;
  proto::TrajectoryBuilderOptions trajectory_builder_options_;
  std::vector<::cartographer::transform::Rigid3d> local_slam_result_poses_;
};

class MapBuilderTest : public MapBuilderTestBase<::testing::Test> {};
class MapBuilderTestByGridType
    : public MapBuilderTestBase<::testing::TestWithParam<GridType>> {};
class MapBuilderTestByGridTypeAndDimensions
    : public MapBuilderTestBase<
          ::testing::TestWithParam<std::pair<GridType, int /* dimensions */>>> {
};
INSTANTIATE_TEST_CASE_P(MapBuilderTestByGridType, MapBuilderTestByGridType,
                        ::testing::Values(GridType::PROBABILITY_GRID,
                                          GridType::TSDF));
INSTANTIATE_TEST_CASE_P(
    MapBuilderTestByGridTypeAndDimensions,
    MapBuilderTestByGridTypeAndDimensions,
    ::testing::Values(std::make_pair(GridType::PROBABILITY_GRID, 2),
                      std::make_pair(GridType::PROBABILITY_GRID, 3),
                      std::make_pair(GridType::TSDF, 2)));

TEST_P(MapBuilderTestByGridTypeAndDimensions, TrajectoryAddFinish) {
  if (GetParam().first == GridType::TSDF) SetOptionsToTSDF2D();
  BuildMapBuilder();
  int trajectory_id = map_builder_->AddTrajectoryBuilder(
      {kRangeSensorId}, trajectory_builder_options_,
      nullptr /* GetLocalSlamResultCallbackForSubscriptions */);
  EXPECT_EQ(1, map_builder_->num_trajectory_builders());
  EXPECT_TRUE(map_builder_->GetTrajectoryBuilder(trajectory_id) != nullptr);
  EXPECT_TRUE(map_builder_->pose_graph() != nullptr);
  map_builder_->FinishTrajectory(trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_TRUE(map_builder_->pose_graph()->IsTrajectoryFinished(trajectory_id));
}

TEST_P(MapBuilderTestByGridType, LocalSlam2D) {
  if (GetParam() == GridType::TSDF) SetOptionsToTSDF2D();
  BuildMapBuilder();

  int trajectory_id = map_builder_->AddTrajectoryBuilder(
      {kRangeSensorId, kOdomSensorId}, trajectory_builder_options_,
      GetLocalSlamResultCallback());

  TrajectoryBuilderInterface* trajectory_builder =
      map_builder_->GetTrajectoryBuilder(trajectory_id);

  //  const auto measurements =
  //  testing::GenerateFakeRangeMeasurements(kTravelDistance, kDuration,
  //  kTimeStep);
  const auto measurements = testing::GenerateSimulationData();

  for (const auto& measurement : measurements.data) {
    trajectory_builder->AddSensorData(kRangeSensorId.id,
                                      measurement.laser_scan);
    trajectory_builder->AddSensorData(kOdomSensorId.id, measurement.odom);
  }
  map_builder_->FinishTrajectory(trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();

  const auto travelled = (measurements.data.back().odom.pose.inverse() *
                          measurements.data.front().odom.pose);
  const auto estimated = (local_slam_result_poses_.back().inverse() *
                          local_slam_result_poses_.front());

  LOG(INFO) << "travelled: " << travelled.DebugString();
  LOG(INFO) << "estimated: " << estimated.DebugString();

  // print the submap
  const auto submap =
      map_builder_->pose_graph()->GetAllSubmapData().at(SubmapId(0, 0));
  auto surface = submap.submap->DrawSurface();
  cairo_surface_write_to_png(surface.get(), "test.png");

  EXPECT_NEAR(travelled.translation().norm(), estimated.translation().norm(),
              0.06);
}

TEST_P(MapBuilderTestByGridType, GlobalSlam2D) {
  if (GetParam() == GridType::TSDF) SetOptionsToTSDF2D();
  SetOptionsEnableGlobalOptimization();
  BuildMapBuilder();
  int trajectory_id = map_builder_->AddTrajectoryBuilder(
      {kRangeSensorId}, trajectory_builder_options_,
      GetLocalSlamResultCallback());
  TrajectoryBuilderInterface* trajectory_builder =
      map_builder_->GetTrajectoryBuilder(trajectory_id);
  const auto measurements = testing::GenerateFakeRangeMeasurements(
      kTravelDistance, kDuration, kTimeStep);
  for (const auto& measurement : measurements) {
    trajectory_builder->AddSensorData(kRangeSensorId.id, measurement);
  }
  map_builder_->FinishTrajectory(trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_EQ(local_slam_result_poses_.size(), measurements.size());
  EXPECT_NEAR(kTravelDistance,
              (local_slam_result_poses_.back().translation() -
               local_slam_result_poses_.front().translation())
                  .norm(),
              0.1 * kTravelDistance);
  EXPECT_GE(map_builder_->pose_graph()->constraints().size(), 50);
  EXPECT_THAT(map_builder_->pose_graph()->constraints(),
              ::testing::Contains(::testing::Field(
                  &PoseGraphInterface::Constraint::tag,
                  PoseGraphInterface::Constraint::INTER_SUBMAP)));
  const auto trajectory_nodes =
      map_builder_->pose_graph()->GetTrajectoryNodes();
  EXPECT_GE(trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 20);
  const auto submap_data = map_builder_->pose_graph()->GetAllSubmapData();
  EXPECT_GE(submap_data.SizeOfTrajectoryOrZero(trajectory_id), 5);
  const transform::Rigid3d final_pose =
      map_builder_->pose_graph()->GetLocalToGlobalTransform(trajectory_id) *
      local_slam_result_poses_.back();
  EXPECT_NEAR(kTravelDistance, final_pose.translation().norm(),
              0.1 * kTravelDistance);
}

TEST_P(MapBuilderTestByGridType, DeleteFinishedTrajectory2D) {
  if (GetParam() == GridType::TSDF) SetOptionsToTSDF2D();
  SetOptionsEnableGlobalOptimization();
  BuildMapBuilder();
  int trajectory_id = CreateTrajectoryWithFakeData();
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_TRUE(map_builder_->pose_graph()->IsTrajectoryFinished(trajectory_id));
  EXPECT_GE(map_builder_->pose_graph()->constraints().size(), 50);
  EXPECT_GE(
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          trajectory_id),
      20);
  EXPECT_GE(
      map_builder_->pose_graph()->GetAllSubmapData().SizeOfTrajectoryOrZero(
          trajectory_id),
      5);
  map_builder_->pose_graph()->DeleteTrajectory(trajectory_id);
  int another_trajectory_id = CreateTrajectoryWithFakeData(100.);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_TRUE(
      map_builder_->pose_graph()->IsTrajectoryFinished(another_trajectory_id));
  EXPECT_EQ(
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          trajectory_id),
      0);
  EXPECT_EQ(
      map_builder_->pose_graph()->GetAllSubmapData().SizeOfTrajectoryOrZero(
          trajectory_id),
      0);
  map_builder_->pose_graph()->DeleteTrajectory(another_trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_EQ(map_builder_->pose_graph()->constraints().size(), 0);
  EXPECT_EQ(
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          another_trajectory_id),
      0);
  EXPECT_EQ(
      map_builder_->pose_graph()->GetAllSubmapData().SizeOfTrajectoryOrZero(
          another_trajectory_id),
      0);
}

TEST_P(MapBuilderTestByGridTypeAndDimensions, SaveLoadState) {
  if (GetParam().first == GridType::TSDF) SetOptionsToTSDF2D();
  BuildMapBuilder();
  int trajectory_id = map_builder_->AddTrajectoryBuilder(
      {kRangeSensorId}, trajectory_builder_options_,
      GetLocalSlamResultCallback());
  TrajectoryBuilderInterface* trajectory_builder =
      map_builder_->GetTrajectoryBuilder(trajectory_id);
  const auto measurements = testing::GenerateFakeRangeMeasurements(
      kTravelDistance, kDuration, kTimeStep);
  for (const auto& measurement : measurements) {
    trajectory_builder->AddSensorData(kRangeSensorId.id, measurement);
  }
  map_builder_->FinishTrajectory(trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();
  int num_constraints = map_builder_->pose_graph()->constraints().size();
  int num_nodes =
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          trajectory_id);
  EXPECT_GT(num_constraints, 0);
  EXPECT_GT(num_nodes, 0);
  // TODO(gaschler): Consider using in-memory to avoid side effects.
  const std::string filename = "temp-SaveLoadState.pbstream";
  io::ProtoStreamWriter writer(filename);
  map_builder_->SerializeState(/*include_unfinished_submaps=*/true, &writer);
  writer.Close();

  // Reset 'map_builder_'.
  BuildMapBuilder();
  io::ProtoStreamReader reader(filename);
  auto trajectory_remapping =
      map_builder_->LoadState(&reader, false /* load_frozen_state */);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_EQ(num_constraints, map_builder_->pose_graph()->constraints().size());
  ASSERT_EQ(trajectory_remapping.size(), 1);
  int new_trajectory_id = trajectory_remapping.begin()->second;
  EXPECT_EQ(
      num_nodes,
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          new_trajectory_id));
}

TEST_P(MapBuilderTestByGridType, LocalizationOnFrozenTrajectory2D) {
  if (GetParam() == GridType::TSDF) SetOptionsToTSDF2D();
  BuildMapBuilder();
  int temp_trajectory_id = CreateTrajectoryWithFakeData();
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_GT(map_builder_->pose_graph()->constraints().size(), 0);
  EXPECT_GT(
      map_builder_->pose_graph()->GetTrajectoryNodes().SizeOfTrajectoryOrZero(
          temp_trajectory_id),
      0);
  const std::string filename = "temp-LocalizationOnFrozenTrajectory2D.pbstream";
  io::ProtoStreamWriter writer(filename);
  map_builder_->SerializeState(/*include_unfinished_submaps=*/true, &writer);
  writer.Close();

  // Reset 'map_builder_'.
  local_slam_result_poses_.clear();
  SetOptionsEnableGlobalOptimization();
  BuildMapBuilder();
  io::ProtoStreamReader reader(filename);
  map_builder_->LoadState(&reader, true /* load_frozen_state */);
  map_builder_->pose_graph()->RunFinalOptimization();
  int trajectory_id = map_builder_->AddTrajectoryBuilder(
      {kRangeSensorId}, trajectory_builder_options_,
      GetLocalSlamResultCallback());
  TrajectoryBuilderInterface* trajectory_builder =
      map_builder_->GetTrajectoryBuilder(trajectory_id);
  transform::Rigid3d frozen_trajectory_to_global(
      Eigen::Vector3d(0.5, 0.4, 0),
      Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Eigen::Vector3d::UnitZ())));
  Eigen::Vector3d travel_translation =
      Eigen::Vector3d(2., 1., 0.).normalized() * kTravelDistance;
  auto measurements = testing::GenerateFakeRangeMeasurements(
      travel_translation.cast<float>(), kDuration, kTimeStep,
      frozen_trajectory_to_global.cast<float>());
  for (auto& measurement : measurements) {
    measurement.time += common::FromSeconds(100.);
    trajectory_builder->AddSensorData(kRangeSensorId.id, measurement);
  }
  map_builder_->FinishTrajectory(trajectory_id);
  map_builder_->pose_graph()->RunFinalOptimization();
  EXPECT_EQ(local_slam_result_poses_.size(), measurements.size());
  EXPECT_NEAR(kTravelDistance,
              (local_slam_result_poses_.back().translation() -
               local_slam_result_poses_.front().translation())
                  .norm(),
              0.15 * kTravelDistance);
  EXPECT_GE(map_builder_->pose_graph()->constraints().size(), 50);
  auto constraints = map_builder_->pose_graph()->constraints();
  int num_cross_trajectory_constraints = 0;
  for (const auto& constraint : constraints) {
    if (constraint.node_id.trajectory_id !=
        constraint.submap_id.trajectory_id) {
      ++num_cross_trajectory_constraints;
    }
  }
  EXPECT_GE(num_cross_trajectory_constraints, 3);
  // TODO(gaschler): Subscribe global slam callback, verify that all nodes are
  // optimized.
  EXPECT_THAT(constraints, ::testing::Contains(::testing::Field(
                               &PoseGraphInterface::Constraint::tag,
                               PoseGraphInterface::Constraint::INTER_SUBMAP)));
  const auto trajectory_nodes =
      map_builder_->pose_graph()->GetTrajectoryNodes();
  EXPECT_GE(trajectory_nodes.SizeOfTrajectoryOrZero(trajectory_id), 20);
  const auto submap_data = map_builder_->pose_graph()->GetAllSubmapData();
  EXPECT_GE(submap_data.SizeOfTrajectoryOrZero(trajectory_id), 5);
  const transform::Rigid3d global_pose =
      map_builder_->pose_graph()->GetLocalToGlobalTransform(trajectory_id) *
      local_slam_result_poses_.back();
  EXPECT_NEAR(frozen_trajectory_to_global.translation().norm(),
              map_builder_->pose_graph()
                  ->GetLocalToGlobalTransform(trajectory_id)
                  .translation()
                  .norm(),
              0.1);
  const transform::Rigid3d expected_global_pose =
      frozen_trajectory_to_global *
      transform::Rigid3d::Translation(travel_translation);
  EXPECT_NEAR(
      0.,
      (global_pose.translation() - expected_global_pose.translation()).norm(),
      0.3)
      << "global_pose: " << global_pose
      << "expected_global_pose: " << expected_global_pose;
}

}  // namespace
}  // namespace mapping
}  // namespace cartographer
