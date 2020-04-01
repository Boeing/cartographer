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

#include <random>

#include "cartographer/mapping/internal/testing/test_helpers.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"
#include "absl/memory/memory.h"
#include "cartographer/common/config.h"
#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/lua_parameter_dictionary_test_helpers.h"
#include "cartographer/sensor/timed_point_cloud_data.h"
#include "cartographer/transform/transform.h"


namespace cartographer {
namespace mapping {
namespace testing {

std::unique_ptr<::cartographer::common::LuaParameterDictionary>
ResolveLuaParameters(const std::string& lua_code) {
  auto file_resolver =
      absl::make_unique<::cartographer::common::ConfigurationFileResolver>(
          std::vector<std::string>{
              std::string(::cartographer::common::kSourceDirectory) +
              "/configuration_files"});
  return absl::make_unique<::cartographer::common::LuaParameterDictionary>(
      lua_code, std::move(file_resolver));
}

std::vector<cartographer::sensor::TimedPointCloudData>
GenerateFakeRangeMeasurements(double travel_distance, double duration,
                              double time_step) {
  const Eigen::Vector3f kDirection = Eigen::Vector3f(2., 1., 0.).normalized();
  return GenerateFakeRangeMeasurements(kDirection * travel_distance, duration,
                                       time_step,
                                       transform::Rigid3f::Identity());
}

std::vector<cartographer::sensor::TimedPointCloudData>
GenerateFakeRangeMeasurements(const Eigen::Vector3f& translation,
                              double duration, double time_step,
                              const transform::Rigid3f& local_to_global) {
  std::vector<cartographer::sensor::TimedPointCloudData> measurements;
  cartographer::sensor::TimedPointCloud point_cloud;
  for (double angle = 0.; angle < M_PI; angle += 0.01) {
    for (double height : {-0.4, -0.2, 0.0, 0.2, 0.4}) {
      constexpr double kRadius = 5;
      point_cloud.push_back({Eigen::Vector3d{kRadius * std::cos(angle),
                                             kRadius * std::sin(angle), height}
                                 .cast<float>(),
                             0., 0.});
    }
  }
  const Eigen::Vector3f kVelocity = translation / duration;
  for (double elapsed_time = 0.; elapsed_time < duration;
       elapsed_time += time_step) {
    cartographer::common::Time time =
        cartographer::common::FromUniversal(123) +
        cartographer::common::FromSeconds(elapsed_time);
    cartographer::transform::Rigid3f global_pose =
        local_to_global *
        cartographer::transform::Rigid3f::Translation(elapsed_time * kVelocity);
    cartographer::sensor::TimedPointCloud ranges =
        cartographer::sensor::TransformTimedPointCloud(point_cloud,
                                                       global_pose.inverse());
    measurements.emplace_back(cartographer::sensor::TimedPointCloudData{
        time, Eigen::Vector3f::Zero(), ranges});
  }
  return measurements;
}

proto::Node CreateFakeNode(int trajectory_id, int node_index) {
  proto::Node proto;
  proto.mutable_node_id()->set_trajectory_id(trajectory_id);
  proto.mutable_node_id()->set_node_index(node_index);
  proto.mutable_node_data()->set_timestamp(42);
  *proto.mutable_node_data()->mutable_local_pose() =
      transform::ToProto(transform::Rigid3d::Identity());
  return proto;
}

proto::PoseGraph::Constraint CreateFakeConstraint(const proto::Node& node,
                                                  const proto::Submap& submap) {
  proto::PoseGraph::Constraint proto;
  proto.mutable_submap_id()->set_submap_index(
      submap.submap_id().submap_index());
  proto.mutable_submap_id()->set_trajectory_id(
      submap.submap_id().trajectory_id());
  proto.mutable_node_id()->set_node_index(node.node_id().node_index());
  proto.mutable_node_id()->set_trajectory_id(node.node_id().trajectory_id());
  transform::Rigid3d pose(
      Eigen::Vector3d(2., 3., 4.),
      Eigen::AngleAxisd(M_PI / 8., Eigen::Vector3d::UnitX()));
  *proto.mutable_relative_pose() = transform::ToProto(pose);
  proto.set_translation_weight(0.2);
  proto.set_rotation_weight(0.1);
  proto.set_tag(proto::PoseGraph::Constraint::INTER_SUBMAP);
  return proto;
}

proto::Trajectory* CreateTrajectoryIfNeeded(int trajectory_id,
                                            proto::PoseGraph* pose_graph) {
  for (int i = 0; i < pose_graph->trajectory_size(); ++i) {
    proto::Trajectory* trajectory = pose_graph->mutable_trajectory(i);
    if (trajectory->trajectory_id() == trajectory_id) {
      return trajectory;
    }
  }
  proto::Trajectory* trajectory = pose_graph->add_trajectory();
  trajectory->set_trajectory_id(trajectory_id);
  return trajectory;
}

proto::PoseGraph::LandmarkPose CreateFakeLandmark(
    const std::string& landmark_id, const transform::Rigid3d& global_pose) {
  proto::PoseGraph::LandmarkPose landmark;
  landmark.set_landmark_id(landmark_id);
  *landmark.mutable_global_pose() = transform::ToProto(global_pose);
  return landmark;
}

void AddToProtoGraph(const proto::Node& node_data,
                     proto::PoseGraph* pose_graph) {
  auto* trajectory =
      CreateTrajectoryIfNeeded(node_data.node_id().trajectory_id(), pose_graph);
  auto* node = trajectory->add_node();
  node->set_timestamp(node_data.node_data().timestamp());
  node->set_node_index(node_data.node_id().node_index());
  *node->mutable_pose() = node_data.node_data().local_pose();
}

void AddToProtoGraph(const proto::Submap& submap_data,
                     proto::PoseGraph* pose_graph) {
  auto* trajectory = CreateTrajectoryIfNeeded(
      submap_data.submap_id().trajectory_id(), pose_graph);
  auto* submap = trajectory->add_submap();
  submap->set_submap_index(submap_data.submap_id().submap_index());
  *submap->mutable_pose() = submap_data.submap_2d().local_pose();
}

void AddToProtoGraph(const proto::PoseGraph::Constraint& constraint,
                     proto::PoseGraph* pose_graph) {
  *pose_graph->add_constraint() = constraint;
}

void AddToProtoGraph(const proto::PoseGraph::LandmarkPose& landmark,
                     proto::PoseGraph* pose_graph) {
  *pose_graph->add_landmark_poses() = landmark;
}

SimulationData GenerateSimulationData()
{
    std::mt19937 prng(42);
    std::uniform_real_distribution<double> distribution(-1., 1.);

    const double resolution = 0.02;
    const double side_length = 40.0;
    const int cells = static_cast<int>(side_length / resolution);

    SimulationData sim_data;

    ValueConversionTables conversion_tables;
    sim_data.ground_truth = absl::make_unique<ProbabilityGrid>(
        MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                  CellLimits(cells, cells)),
        &conversion_tables);
    ProbabilityGrid& probability_grid = *sim_data.ground_truth;

    for (int ii = 0; ii < cells; ++ii)
      for (int jj = 0; jj < cells; ++jj)
        probability_grid.SetProbability({ii, jj}, 0.5);

    // insert some random box shapes
    std::uniform_int_distribution<int> box_dist;
    using param_t = std::uniform_int_distribution<>::param_type;
    for (int i = 0; i < 40; ++i) {
      const int box_size = box_dist(prng, param_t(50, 200));
      const int box_x = box_dist(prng, param_t(0, cells));
      const int box_y = box_dist(prng, param_t(0, cells));

      const float prob = 1.0;

      for (int jj = box_y; jj < box_y + box_size; ++jj) {
        if (jj >= cells) break;

        for (int ii = box_x; ii < box_x + box_size; ++ii) {
          if (ii >= cells) break;

          if (jj > box_y && jj < box_y + box_size - 1) {
            if (ii > box_x && ii < box_x + box_size - 1) {
              continue;
            }
          }

          probability_grid.SetProbability({ii, jj}, prob);
        }
      }
      probability_grid.FinishUpdate();
    }

    // insert some random retro reflective poles
    const double pole_radius = 0.06;
    std::vector<std::pair<int, int>> pole_centers;
    std::set<std::pair<int, int>> reflective_cells;
    std::uniform_int_distribution<int> pole_dist;
    using param_t = std::uniform_int_distribution<>::param_type;
    for (int i = 0; i < 30; ++i) {
      const int pole_size = static_cast<int>(pole_radius / resolution);
      const int pole_x = pole_dist(prng, param_t(0, cells));
      const int pole_y = pole_dist(prng, param_t(0, cells));

      const float prob = 1.0;
      const double r2 =
          static_cast<double>(pole_size) * static_cast<double>(pole_size);

      pole_centers.push_back({pole_x, pole_y});

      for (int jj = pole_y - pole_size; jj <= pole_y + pole_size; ++jj) {
        if (jj >= cells || jj < 0) break;

        for (int ii = pole_x - pole_size; ii <= pole_x + pole_size; ++ii) {
          if (ii >= cells || ii < 0) break;

          const double w = static_cast<double>(ii - pole_x);
          const double h = static_cast<double>(jj - pole_y);
          const double _r2 = w * w + h * h;
          if (_r2 <= r2) {
            probability_grid.SetProbability({ii, jj}, prob);
            reflective_cells.insert({ii, jj});
          }
        }
      }
      probability_grid.FinishUpdate();
    }

    transform::Rigid2f start_pose({side_length / 2.0, side_length / 2.0}, 0.);

    const double sensor_hz = 10;
    const double duration = 40;
    const int num_of_data = sensor_hz * duration;
    const int scan_points = 800;

    const double velocity_x = 0.01;
    const double velocity_w = 0.2;

    for (int i=0; i<num_of_data; ++i)
    {
        const double seconds = (1.0 / sensor_hz) * i;
        const common::Time time(common::FromSeconds(seconds));

        const transform::Rigid2f pose(start_pose.translation() + Eigen::Vector2f(seconds*velocity_x, 0.), velocity_w*seconds);

        const cartographer::sensor::OdometryData odom{
          time,
          transform::Embed3D(pose.cast<double>()),
          Eigen::Vector3d{0, 0, 0},
          Eigen::Vector3d{0, 0, 0}
        };

        sensor::TimedPointCloud laser_scan;
        for (int i = 0; i < scan_points; ++i) {
          const float angle = static_cast<float>(i * 2. * M_PI / scan_points);
          const float x = std::cos(angle);
          const float y = std::sin(angle);
          const Eigen::Vector2f dir(x, y);
          const Eigen::Vector2f end = pose.translation() + dir * 30.;

          const auto map_start =
              probability_grid.limits().GetCellIndex(pose.translation());
          const auto map_end = probability_grid.limits().GetCellIndex(end);

          auto p = cartographer::mapping::scan_matching::raytraceLine(probability_grid, map_start.x(), map_start.y(), map_end.x(), map_end.y(), cells, 30.f / resolution);
          if (p.x > 0 && p.y > 0) {
            const auto real_p = probability_grid.limits().GetCellCenter({p.x, p.y});
            const auto diff = real_p - pose.translation();
            auto r = Eigen::AngleAxisf(-pose.rotation().angle(),
                                       Eigen::Vector3f::UnitZ()) *
                     Eigen::Vector3f{diff.x(), diff.y(), 0.f};
            float intensity = 0;
            if (reflective_cells.find({p.x, p.y}) != reflective_cells.end())
              intensity = 9000;
            laser_scan.push_back({r, 0.f, intensity});
          }
        }

        LOG(INFO) << "time: " << time << " " << seconds << " odom: " << odom.pose.DebugString();

        sim_data.data.push_back({cartographer::sensor::TimedPointCloudData{time, Eigen::Vector3f{0, 0, 0}, laser_scan}, odom});
    }

    return sim_data;
}

}  // namespace testing
}  // namespace mapping
}  // namespace cartographer
