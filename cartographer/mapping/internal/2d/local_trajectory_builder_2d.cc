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

#include "cartographer/mapping/internal/2d/local_trajectory_builder_2d.h"
#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"

#include <limits>
#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/metrics/family_factory.h"
#include "cartographer/sensor/range_data.h"

namespace cartographer {
namespace mapping {

static auto* kLocalSlamLatencyMetric = metrics::Gauge::Null();
static auto* kLocalSlamRealTimeRatio = metrics::Gauge::Null();
static auto* kLocalSlamCpuRealTimeRatio = metrics::Gauge::Null();
static auto* kRealTimeCorrelativeScanMatcherScoreMetric =
    metrics::Histogram::Null();
static auto* kCeresScanMatcherCostMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualDistanceMetric = metrics::Histogram::Null();
static auto* kScanMatcherResidualAngleMetric = metrics::Histogram::Null();

LocalTrajectoryBuilder2D::LocalTrajectoryBuilder2D(
    const proto::LocalTrajectoryBuilderOptions2D& options,
    const std::vector<std::string>& expected_range_sensor_ids)
    : options_(options),
      active_submaps_(options.submaps_options()),
      motion_filter_(options_.motion_filter_options()),
      ceres_scan_matcher_(options_.ceres_scan_matcher_options()),
      range_data_collator_(expected_range_sensor_ids) {
  extrapolator_ = absl::make_unique<PoseExtrapolator>();
  extrapolator_->AddPose(common::Time::min(), transform::Rigid3d::Identity());
}

LocalTrajectoryBuilder2D::~LocalTrajectoryBuilder2D() {}

std::unique_ptr<LocalTrajectoryBuilder2D::MatchingResult>
LocalTrajectoryBuilder2D::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& unsynchronized_data) {
  const auto synchronized_data =
      range_data_collator_.AddRangeData(sensor_id, unsynchronized_data);

  if (synchronized_data.ranges.empty()) return nullptr;

  // time should refer to the last point
  const common::Time& time = synchronized_data.time;

  const common::Time& start_time =
      time +
      common::FromSeconds(synchronized_data.ranges.front().point_time.time);

  CHECK(start_time <= time);

  const auto pose_prediction = extrapolator_->ExtrapolatePose(time);

  std::vector<transform::Rigid3f> range_data_poses;
  range_data_poses.reserve(synchronized_data.ranges.size());
  for (const auto& range : synchronized_data.ranges) {
    common::Time time_point = time + common::FromSeconds(range.point_time.time);
    range_data_poses.push_back(
        extrapolator_->ExtrapolatePose(time_point).pose.cast<float>());
  }

  sensor::RangeData accumulated_range_data = sensor::RangeData{{}, {}, {}};

  // Drop any returns below the minimum range and convert returns beyond the
  // maximum range into misses
  for (size_t i = 0; i < synchronized_data.ranges.size(); ++i) {
    const sensor::TimedRangefinderPoint& hit =
        synchronized_data.ranges[i].point_time;
    const Eigen::Vector3f origin_in_local =
        range_data_poses[i] *
        synchronized_data.origins.at(synchronized_data.ranges[i].origin_index);
    sensor::RangefinderPoint hit_in_local =
        range_data_poses[i] * sensor::ToRangefinderPoint(hit);
    const Eigen::Vector3f delta = hit_in_local.position - origin_in_local;
    const float range = delta.norm();
    if (range >= options_.min_range()) {
      if (range <= options_.max_range()) {
        accumulated_range_data.returns.push_back(hit_in_local);
      } else {
        hit_in_local.position =
            origin_in_local +
            options_.missing_data_ray_length() / range * delta;
        accumulated_range_data.misses.push_back(hit_in_local);
      }
    }
  }

  const common::Time current_sensor_time = synchronized_data.time;
  absl::optional<common::Duration> sensor_duration;
  if (last_sensor_time_.has_value()) {
    sensor_duration = current_sensor_time - last_sensor_time_.value();
  }
  last_sensor_time_ = current_sensor_time;

  accumulated_range_data.origin =
      pose_prediction.pose.translation().cast<float>();

  const auto accumulated_range_data_wrt_tracking = sensor::TransformRangeData(
      accumulated_range_data, pose_prediction.pose.inverse().cast<float>());

  const auto cropped = sensor::CropRangeData(
      accumulated_range_data_wrt_tracking, options_.min_z(), options_.max_z());

  const auto voxel_filtered = sensor::RangeData{
      cropped.origin,
      sensor::VoxelFilter(options_.voxel_filter_size()).Filter(cropped.returns),
      sensor::VoxelFilter(options_.voxel_filter_size()).Filter(cropped.misses)};

  const sensor::RangeData& range_data_wrt_tracking = voxel_filtered;

  CHECK(!range_data_wrt_tracking.returns.empty());

  const transform::Rigid2d pose_prediction_2d =
      transform::Project2D(pose_prediction.pose);

  const sensor::PointCloud& adaptive_filtered =
      sensor::AdaptiveVoxelFilter(options_.adaptive_voxel_filter_options())
          .Filter(range_data_wrt_tracking.returns);

  if (adaptive_filtered.empty()) return nullptr;

  transform::Rigid2d pose_estimate_2d = pose_prediction_2d;

  // The first submap needs a minimum amount of range data before a scan match
  // will be accurate
  const int min_num_range_data = 20;
  const bool submap_init =
      !active_submaps_.submaps().empty() &&
      active_submaps_.submaps().front()->num_range_data() > min_num_range_data;

  // TODO scale scan match weight based on trust with odom
  // TODO put everything into the first submap until sensor is stabilized

  // scan match
  if (submap_init) {
    std::shared_ptr<const Submap2D> matching_submap =
        active_submaps_.submaps().front();

    const transform::Rigid2d initial_ceres_pose = pose_prediction_2d;
    transform::Rigid2d pose_observation;
    ceres::Solver::Summary summary;
    ceres_scan_matcher_.Match(
        pose_prediction_2d.translation(), initial_ceres_pose, adaptive_filtered,
        *matching_submap->grid(), &pose_observation, &summary);
    {
      kCeresScanMatcherCostMetric->Observe(summary.final_cost);
      const double residual_distance =
          (pose_observation.translation() - pose_prediction_2d.translation())
              .norm();
      kScanMatcherResidualDistanceMetric->Observe(residual_distance);
      const double residual_angle =
          std::abs(pose_observation.rotation().angle() -
                   pose_prediction_2d.rotation().angle());
      kScanMatcherResidualAngleMetric->Observe(residual_angle);
    }
    pose_estimate_2d = pose_observation;
  }

  const auto scan_match_shift = pose_prediction_2d.inverse() * pose_estimate_2d;

  const transform::Rigid3d pose_estimate = transform::Embed3D(pose_estimate_2d);

  if (scan_match_shift.translation().norm() > 0.08) {
    LOG(WARNING) << "Excessive scan match shift: "
                 << scan_match_shift.translation().transpose();
  }

  const bool is_similar = motion_filter_.IsSimilar(time, pose_estimate);
  if (submap_init && is_similar) {
    return nullptr;
  }

  extrapolator_->AddPose(time, pose_estimate);

  const auto range_data_in_local =
      TransformRangeData(range_data_wrt_tracking,
                         transform::Embed3D(pose_estimate_2d.cast<float>()));

  // Detect features
  std::vector<CircleFeature> circle_features;
  for (const auto radius : options_.circle_feature_options().detect_radii()) {
    // LOG(INFO) << "Searching for circles of radius: " << radius;
    const auto pole_features =
        DetectReflectivePoles(range_data_wrt_tracking.returns, radius);
    for (const auto& f : pole_features) {
      const auto p = f;
      const float xy_covariance = p.mse * p.position.norm();
      // LOG(INFO) << "Found circle: " << p.position.transpose() << " mse: " <<
      // p.mse << " xy_cov: " << xy_covariance;
      circle_features.push_back(
          CircleFeature{Keypoint{{p.position.x(), p.position.y(), 0.f},
                                 {xy_covariance, xy_covariance, 0.f}},
                        CircleDescriptor{p.mse, p.radius}});
    }
  }

  // Transform features to local trajectory frame
  std::vector<CircleFeature> circle_features_in_local;
  std::transform(circle_features.begin(), circle_features.end(),
                 std::back_inserter(circle_features_in_local),
                 [&pose_estimate](const CircleFeature& cf) {
                   return CircleFeature{Keypoint{pose_estimate.cast<float>() *
                                                     cf.keypoint.position,
                                                 cf.keypoint.covariance},
                                        cf.fdescriptor};
                 });

  std::vector<std::shared_ptr<const Submap2D>> insertion_submaps =
      active_submaps_.InsertRangeData(range_data_in_local,
                                      circle_features_in_local);

  if (is_similar) {
    return nullptr;
  }

  auto insertion_result = absl::make_unique<InsertionResult>(InsertionResult{
      std::make_shared<const TrajectoryNode::Data>(
          TrajectoryNode::Data{time, range_data_wrt_tracking, adaptive_filtered,
                               circle_features, pose_estimate}),
      std::move(insertion_submaps)});

  const auto wall_time = std::chrono::steady_clock::now();
  if (last_wall_time_.has_value()) {
    const auto wall_time_duration = wall_time - last_wall_time_.value();
    kLocalSlamLatencyMetric->Set(common::ToSeconds(wall_time_duration));
    if (sensor_duration.has_value()) {
      kLocalSlamRealTimeRatio->Set(common::ToSeconds(sensor_duration.value()) /
                                   common::ToSeconds(wall_time_duration));
    }
  }
  const double thread_cpu_time_seconds = common::GetThreadCpuTimeSeconds();
  if (last_thread_cpu_time_seconds_.has_value()) {
    const double thread_cpu_duration_seconds =
        thread_cpu_time_seconds - last_thread_cpu_time_seconds_.value();
    if (sensor_duration.has_value()) {
      kLocalSlamCpuRealTimeRatio->Set(
          common::ToSeconds(sensor_duration.value()) /
          thread_cpu_duration_seconds);
    }
  }
  last_wall_time_ = wall_time;
  last_thread_cpu_time_seconds_ = thread_cpu_time_seconds;

  const auto odom = extrapolator_->odom(time);

  return absl::make_unique<MatchingResult>(
      MatchingResult{time, pose_estimate, odom, std::move(insertion_result)});
}

void LocalTrajectoryBuilder2D::AddImuData(const sensor::ImuData&) {
  LOG(FATAL) << "Imu Data is not supported";
}

void LocalTrajectoryBuilder2D::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  extrapolator_->AddOdometryData(odometry_data);
}

void LocalTrajectoryBuilder2D::RegisterMetrics(
    metrics::FamilyFactory* family_factory) {
  auto* latency = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_latency",
      "Duration from first incoming point cloud in accumulation to local slam "
      "result");
  kLocalSlamLatencyMetric = latency->Add({});
  auto* real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_real_time_ratio",
      "sensor duration / wall clock duration.");
  kLocalSlamRealTimeRatio = real_time_ratio->Add({});

  auto* cpu_real_time_ratio = family_factory->NewGaugeFamily(
      "mapping_2d_local_trajectory_builder_cpu_real_time_ratio",
      "sensor duration / cpu duration.");
  kLocalSlamCpuRealTimeRatio = cpu_real_time_ratio->Add({});
  auto score_boundaries = metrics::Histogram::FixedWidth(0.05, 20);
  auto* scores = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_scores", "Local scan matcher scores",
      score_boundaries);
  kRealTimeCorrelativeScanMatcherScoreMetric =
      scores->Add({{"scan_matcher", "real_time_correlative"}});
  auto cost_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 100);
  auto* costs = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_costs", "Local scan matcher costs",
      cost_boundaries);
  kCeresScanMatcherCostMetric = costs->Add({{"scan_matcher", "ceres"}});
  auto distance_boundaries = metrics::Histogram::ScaledPowersOf(2, 0.01, 10);
  auto* residuals = family_factory->NewHistogramFamily(
      "mapping_2d_local_trajectory_builder_residuals",
      "Local scan matcher residuals", distance_boundaries);
  kScanMatcherResidualDistanceMetric =
      residuals->Add({{"component", "distance"}});
  kScanMatcherResidualAngleMetric = residuals->Add({{"component", "angle"}});
}

}  // namespace mapping
}  // namespace cartographer
