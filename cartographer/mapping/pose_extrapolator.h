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

#ifndef CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
#define CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_

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

  std::deque<TimedPose> timed_pose_queue_;
  std::deque<sensor::OdometryData> odometry_data_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_POSE_EXTRAPOLATOR_H_
