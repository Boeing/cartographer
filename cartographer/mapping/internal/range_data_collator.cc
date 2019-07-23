/*
 * Copyright 2018 The Cartographer Authors
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

#include "cartographer/mapping/internal/range_data_collator.h"

#include <memory>

#include "absl/memory/memory.h"
#include "cartographer/mapping/local_slam_result_data.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

sensor::TimedPointCloudOriginData RangeDataCollator::AddRangeData(
    const std::string& sensor_id,
    const sensor::TimedPointCloudData& timed_point_cloud_data) {
  CHECK_NE(expected_sensor_ids_.count(sensor_id), 0);

  auto it = id_to_pending_data_.find(sensor_id);
  if (it == id_to_pending_data_.end())
      id_to_pending_data_.emplace(sensor_id, timed_point_cloud_data);
  else
      it->second = timed_point_cloud_data;

  if (id_to_pending_data_.size() != expected_sensor_ids_.size())
      return {};

  common::Time oldest_timestamp = common::Time::max();
  for (const auto& pair : id_to_pending_data_) {
    oldest_timestamp = std::min(oldest_timestamp, pair.second.time);
  }

  sensor::TimedPointCloudOriginData result{oldest_timestamp, {}, {}};

  size_t idx = 0;
  for (const auto& pair : id_to_pending_data_) {
    result.origins.push_back(pair.second.origin);
    const float time_correction = static_cast<float>(common::ToSeconds(pair.second.time - oldest_timestamp));
    for (const auto& p : pair.second.ranges)
    {
        result.ranges.push_back(sensor::TimedPointCloudOriginData::RangeMeasurement{p, idx});
        result.ranges.back().point_time.time += time_correction;
    }
    ++idx;
  }

  std::sort(result.ranges.begin(), result.ranges.end(),
            [](const sensor::TimedPointCloudOriginData::RangeMeasurement& a,
               const sensor::TimedPointCloudOriginData::RangeMeasurement& b) {
              return a.point_time.time < b.point_time.time;
            });

  id_to_pending_data_.clear();

  return result;
}

}  // namespace mapping
}  // namespace cartographer
