#ifndef CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
#define CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_

#include <boost/circular_buffer.hpp>
#include <deque>
#include <memory>

#include "cartographer/common/time.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

class PoseExtrapolator {
 public:
  explicit PoseExtrapolator();

  PoseExtrapolator(const PoseExtrapolator&) = delete;
  PoseExtrapolator& operator=(const PoseExtrapolator&) = delete;

  void AddPose(common::Time time, const transform::Rigid3d& pose);
  void AddOdometryData(const sensor::OdometryData& odometry_data);

  struct Extrapolation {
    common::Time time;
    transform::Rigid3d pose;
    transform::Rigid3d motion;
  };

  Extrapolation ExtrapolatePose(common::Time time);

 private:
  struct TimedPose {
    common::Time time;
    transform::Rigid3d pose;
  };

  Extrapolation cached_extrapolated_pose_;

  std::unique_ptr<TimedPose> reference_pose_;
  boost::circular_buffer<sensor::OdometryData> odometry_data_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
