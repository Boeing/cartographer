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

#include "cartographer/mapping/internal/constraints/constraint_builder.h"

#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/fast_correlative_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/global_icp_scan_matcher_2d.h"
#include "cartographer/sensor/internal/voxel_filter.h"

namespace cartographer {
namespace mapping {
namespace constraints {

proto::ConstraintBuilderOptions CreateConstraintBuilderOptions(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::ConstraintBuilderOptions options;

  options.set_min_local_search_score(
      parameter_dictionary->GetDouble("min_local_search_score"));
  options.set_min_global_search_score(
      parameter_dictionary->GetDouble("min_global_search_score"));

  options.set_constraint_translation_weight(
      parameter_dictionary->GetDouble("constraint_translation_weight"));
  options.set_constraint_rotation_weight(
      parameter_dictionary->GetDouble("constraint_rotation_weight"));

  options.set_log_matches(parameter_dictionary->GetBool("log_matches"));

  *options.mutable_ceres_scan_matcher_options() =
      scan_matching::CreateCeresScanMatcherOptions2D(
          parameter_dictionary->GetDictionary("ceres_scan_matcher").get());

  *options.mutable_global_icp_scan_matcher_options_2d() =
      scan_matching::CreateGlobalICPScanMatcherOptions2D(
          parameter_dictionary
              ->GetDictionary("global_icp_scan_matcher_options_2d")
              .get());

  options.set_min_icp_score(parameter_dictionary->GetDouble("min_icp_score"));
  options.set_min_icp_points_inlier_fraction(
      parameter_dictionary->GetDouble("min_icp_points_inlier_fraction"));
  options.set_min_icp_features_inlier_fraction(
      parameter_dictionary->GetDouble("min_icp_features_inlier_fraction"));

  options.set_min_hit_fraction(
      parameter_dictionary->GetDouble("min_hit_fraction"));

  return options;
}

}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer
