#include "cartographer/mapping/pose_extrapolator.h"

#include <algorithm>

#include "absl/memory/memory.h"
#include "cartographer/transform/timestamped_transform.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

PoseExtrapolator::PoseExtrapolator()
    : cached_extrapolated_pose_{common::Time::min(),
                                transform::Rigid3d::Identity(),
                                transform::Rigid3d::Identity()},
      odometry_data_(2000) {}

void PoseExtrapolator::AddPose(const common::Time time,
                               const transform::Rigid3d& pose) {
  reference_pose_.reset(new TimedPose{time, {pose}});
}

void PoseExtrapolator::AddOdometryData(
    const sensor::OdometryData& odometry_data) {
  odometry_data_.push_back(odometry_data);
}

PoseExtrapolator::Extrapolation PoseExtrapolator::ExtrapolatePose(
    const common::Time time) {
  CHECK(reference_pose_);

  if (odometry_data_.empty()) {
    cached_extrapolated_pose_.time = time;
    cached_extrapolated_pose_.pose = reference_pose_->pose;
    cached_extrapolated_pose_.motion = transform::Rigid3d::Identity();
    return cached_extrapolated_pose_;
  }

  if (cached_extrapolated_pose_.time != time) {
    const TimedPose& newest_timed_pose = *reference_pose_;
    CHECK_GE(time, newest_timed_pose.time);

    CHECK(!odometry_data_.empty());

    auto& odometry_data = odometry_data_;
    auto interpolate = [&odometry_data](const common::Time time) {
      transform::Rigid3d odom;
      auto it = odometry_data.begin();
      while (it != odometry_data.end() && it->time < time) {
        it++;
      }

      if (it == odometry_data.begin()) {
        LOG(WARNING) << "No odometry data for time: " << time
                     << " (earliest: " << odometry_data.front().time << ")";
        odom = it->pose;
      } else if (it == odometry_data.end()) {
        auto prev_it = it - 1;
        const double t_diff = common::ToSeconds(time - prev_it->time);
        const Eigen::Quaterniond rot =
            Eigen::AngleAxisd(t_diff * prev_it->angular_velocity.x(),
                              Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(t_diff * prev_it->angular_velocity.y(),
                              Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(t_diff * prev_it->angular_velocity.z(),
                              Eigen::Vector3d::UnitZ());
        const Eigen::Vector3d current_t =
            prev_it->pose.translation() +
            rot * (prev_it->linear_velocity * t_diff);
        const Eigen::Quaterniond current_r = rot * prev_it->pose.rotation();
        odom = transform::Rigid3d(current_t, current_r);
      } else {
        auto prev_it = it - 1;
        odom =
            Interpolate(
                transform::TimestampedTransform{prev_it->time, prev_it->pose},
                transform::TimestampedTransform{it->time, it->pose}, time)
                .transform;
      }
      return odom;
    };

    // estimate odometry at the last known pose
    transform::Rigid3d reference_odom = interpolate(newest_timed_pose.time);

    // estimate odometry at the queried time
    transform::Rigid3d current_odom = interpolate(time);

    // calculate the odom diff
    transform::Rigid3d odom_diff = reference_odom.inverse() * current_odom;
    transform::Rigid3d extrapolated = newest_timed_pose.pose * odom_diff;

    cached_extrapolated_pose_ = Extrapolation{time, extrapolated, odom_diff};
  }
  return cached_extrapolated_pose_;
}

}  // namespace mapping
}  // namespace cartographer
