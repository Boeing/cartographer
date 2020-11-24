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

#include "cartographer/mapping/2d/submap_2d.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>

#include "Eigen/Geometry"
#include "absl/memory/memory.h"
#include "cartographer/common/port.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/2d/tsdf_range_data_inserter_2d.h"
#include "cartographer/mapping/feature.h"
#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {

proto::SubmapsOptions2D CreateSubmapsOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::SubmapsOptions2D options;
  options.set_num_range_data(
      parameter_dictionary->GetNonNegativeInt("num_range_data"));

  options.set_min_feature_observations(
      parameter_dictionary->GetNonNegativeInt("min_feature_observations"));

  options.set_max_feature_score(
      parameter_dictionary->GetDouble("max_feature_score"));

  *options.mutable_grid_options_2d() = CreateGridOptions2D(
      parameter_dictionary->GetDictionary("grid_options_2d").get());
  *options.mutable_range_data_inserter_options() =
      CreateRangeDataInserterOptions(
          parameter_dictionary->GetDictionary("range_data_inserter").get());

  bool valid_range_data_inserter_grid_combination = false;
  const proto::GridOptions2D_GridType& grid_type =
      options.grid_options_2d().grid_type();
  const proto::RangeDataInserterOptions_RangeDataInserterType&
      range_data_inserter_type =
          options.range_data_inserter_options().range_data_inserter_type();
  if (grid_type == proto::GridOptions2D::PROBABILITY_GRID &&
      range_data_inserter_type ==
          proto::RangeDataInserterOptions::PROBABILITY_GRID_INSERTER_2D) {
    valid_range_data_inserter_grid_combination = true;
  }
  if (grid_type == proto::GridOptions2D::TSDF &&
      range_data_inserter_type ==
          proto::RangeDataInserterOptions::TSDF_INSERTER_2D) {
    valid_range_data_inserter_grid_combination = true;
  }
  CHECK(valid_range_data_inserter_grid_combination)
      << "Invalid combination grid_type " << grid_type
      << " with range_data_inserter_type " << range_data_inserter_type;
  CHECK_GT(options.num_range_data(), 0);
  return options;
}

Submap2D::Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
                   ValueConversionTables* conversion_tables,
                   const proto::SubmapsOptions2D& options)
    : Submap(transform::Rigid3d::Translation(
          Eigen::Vector3d(origin.x(), origin.y(), 0.))),
      conversion_tables_(conversion_tables),
      options_(options) {
  grid_ = std::move(grid);
}

Submap2D::Submap2D(const proto::Submap2D& proto,
                   ValueConversionTables* conversion_tables)
    : Submap(transform::ToRigid3(proto.local_pose())),
      conversion_tables_(conversion_tables) {
  if (proto.has_grid()) {
    if (proto.grid().has_probability_grid_2d()) {
      grid_ =
          absl::make_unique<ProbabilityGrid>(proto.grid(), conversion_tables_);
    } else if (proto.grid().has_tsdf_2d()) {
      grid_ = absl::make_unique<TSDF2D>(proto.grid(), conversion_tables_);
    } else {
      LOG(FATAL) << "proto::Submap2D has grid with unknown type.";
    }
  }
  set_num_range_data(proto.num_range_data());
  set_insertion_finished(proto.finished());
  circle_features_.reserve(proto.circle_features_size());
  for (const auto& cf : proto.circle_features())
    circle_features_.push_back(FromProto(cf));
}

proto::Submap Submap2D::ToProto(const bool include_grid_data) const {
  proto::Submap proto;
  auto* const submap_2d = proto.mutable_submap_2d();
  *submap_2d->mutable_local_pose() = transform::ToProto(local_pose());
  submap_2d->set_num_range_data(num_range_data());
  submap_2d->set_finished(insertion_finished());
  if (include_grid_data) {
    CHECK(grid_);
    *submap_2d->mutable_grid() = grid_->ToProto();
  }
  submap_2d->mutable_circle_features()->Reserve(circle_features_.size());
  for (const CircleFeature& cf : circle_features_)
    *submap_2d->add_circle_features() = ::cartographer::mapping::ToProto(cf);
  return proto;
}

void Submap2D::UpdateFromProto(const proto::Submap& proto) {
  CHECK(proto.has_submap_2d());
  const auto& submap_2d = proto.submap_2d();
  set_num_range_data(submap_2d.num_range_data());
  set_insertion_finished(submap_2d.finished());
  if (proto.submap_2d().has_grid()) {
    if (proto.submap_2d().grid().has_probability_grid_2d()) {
      grid_ = absl::make_unique<ProbabilityGrid>(proto.submap_2d().grid(),
                                                 conversion_tables_);
    } else if (proto.submap_2d().grid().has_tsdf_2d()) {
      grid_ = absl::make_unique<TSDF2D>(proto.submap_2d().grid(),
                                        conversion_tables_);
    } else {
      LOG(FATAL) << "proto::Submap2D has grid with unknown type.";
    }
  }
  if (submap_2d.circle_features_size() > 0) {
    circle_features_.clear();
    for (const auto& cf : submap_2d.circle_features())
      circle_features_.push_back(FromProto(cf));
  }
}

void Submap2D::ToResponseProto(
    const transform::Rigid3d&,
    proto::SubmapQuery::Response* const response) const {
  if (!grid_) return;
  response->set_submap_version(num_range_data());
  proto::SubmapQuery::Response::SubmapTexture* const texture =
      response->add_textures();
  grid()->DrawToSubmapTexture(texture, local_pose());
}

void Submap2D::InsertRangeData(
    const sensor::RangeData& range_data,
    const RangeDataInserterInterface* range_data_inserter) {
  CHECK(grid_);
  CHECK(!insertion_finished());
  range_data_inserter->Insert(range_data, grid_.get());
  set_num_range_data(num_range_data() + 1);
}

void Submap2D::InsertCircleFeatures(
    const std::vector<CircleFeature>& circle_features) {
  CHECK(!insertion_finished());
  for (const CircleFeature& feature : circle_features) {
    // TODO this could be made more preformat with a spatial index if necessary
    // data association
    bool found = false;
    for (CircleFeatureSmoother& s : circle_feature_smoothers_) {
      const float dist =
          (feature.keypoint.position - s.feature().keypoint.position).norm();
      if (dist < 3.f * feature.fdescriptor.radius) {
        s.AddObservation(feature);
        found = true;
        break;
      }
    }
    if (!found) {
      circle_feature_smoothers_.emplace_back(feature);
    }
  }
}

void Submap2D::Finish() {
  CHECK(grid_);
  CHECK(!insertion_finished());
  grid_ = grid_->ComputeCroppedGrid();

  std::vector<CircleFeatureSmoother> filtered_features;
  std::copy_if(
      circle_feature_smoothers_.begin(), circle_feature_smoothers_.end(),
      std::back_inserter(filtered_features),
      [this](const CircleFeatureSmoother& s) {
        return s.observations_count > options_.min_feature_observations() &&
               s.feature().fdescriptor.score < options_.max_feature_score();
      });

  std::transform(filtered_features.begin(), filtered_features.end(),
                 std::back_inserter(circle_features_),
                 [](const CircleFeatureSmoother& s) { return s.feature(); });
  circle_feature_smoothers_.clear();

  set_insertion_finished(true);
}

}  // namespace mapping
}  // namespace cartographer
