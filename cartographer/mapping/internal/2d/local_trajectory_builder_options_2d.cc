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

#include "cartographer/mapping/internal/2d/local_trajectory_builder_options_2d.h"

#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"
#include "cartographer/mapping/internal/motion_filter.h"
#include "cartographer/mapping/internal/scan_matching/real_time_correlative_scan_matcher.h"
#include "cartographer/sensor/internal/voxel_filter.h"

namespace cartographer {
namespace mapping {

proto::CircleFeatureOptions CreateCircleFeatureOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::CircleFeatureOptions options;
  const auto& ref = parameter_dictionary->GetDictionary("detect_radii")
                        ->GetArrayValuesAsDoubles();
  options.mutable_detect_radii()->Reserve(ref.size());
  for (const auto& d : ref) options.add_detect_radii(d);

  options.set_min_reflective_points_far(parameter_dictionary->GetDouble("min_reflective_points_far"));
  options.set_min_reflective_points_near(parameter_dictionary->GetDouble("min_reflective_points_near"));
  options.set_max_detection_distance(parameter_dictionary->GetDouble("max_detection_distance"));
  return options;
}

proto::LocalTrajectoryBuilderOptions2D CreateLocalTrajectoryBuilderOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::LocalTrajectoryBuilderOptions2D options;
  options.set_min_range(parameter_dictionary->GetDouble("min_range"));
  options.set_max_range(parameter_dictionary->GetDouble("max_range"));
  options.set_min_z(parameter_dictionary->GetDouble("min_z"));
  options.set_max_z(parameter_dictionary->GetDouble("max_z"));
  options.set_missing_data_ray_length(
      parameter_dictionary->GetDouble("missing_data_ray_length"));

  *options.mutable_circle_feature_options() = CreateCircleFeatureOptions(
      parameter_dictionary->GetDictionary("circle_feature_options").get());

  options.set_voxel_filter_size(
      parameter_dictionary->GetDouble("voxel_filter_size"));
  *options.mutable_adaptive_voxel_filter_options() =
      sensor::CreateAdaptiveVoxelFilterOptions(
          parameter_dictionary->GetDictionary("adaptive_voxel_filter").get());
  *options.mutable_ceres_scan_matcher_options() =
      mapping::scan_matching::CreateCeresScanMatcherOptions2D(
          parameter_dictionary->GetDictionary("ceres_scan_matcher").get());
  *options.mutable_motion_filter_options() = mapping::CreateMotionFilterOptions(
      parameter_dictionary->GetDictionary("motion_filter").get());
  *options.mutable_submaps_options() = CreateSubmapsOptions2D(
      parameter_dictionary->GetDictionary("submaps").get());
  return options;
}

}  // namespace mapping
}  // namespace cartographer
