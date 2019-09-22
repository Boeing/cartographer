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

#ifndef CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
#define CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/map_limits.h"
#include "cartographer/mapping/proto/2d/submaps_options_2d.pb.h"
#include "cartographer/mapping/proto/serialization.pb.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "cartographer/mapping/submaps.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

proto::SubmapsOptions2D CreateSubmapsOptions2D(
    common::LuaParameterDictionary* parameter_dictionary);

class CircleFeatureSmoother {
 public:
  CircleFeatureSmoother(const CircleFeature& feature)
      : initial_feature_(feature) {
    x = feature.keypoint.position.head<2>();
    P.diagonal() << feature.keypoint.covariance.head<2>();
  }

  CircleFeature feature() const {
    CircleFeature f = initial_feature_;
    f.keypoint.position.x() = x.x();
    f.keypoint.position.y() = x.y();
    f.keypoint.position.z() = 0;
    return f;
  }
  void AddObservation(const CircleFeature& feature) {
    const Eigen::Vector2f x_1 = A * x;

    // state prediction covariance
    P = A * P * A.transpose() + Q;

    // measurement noise
    Eigen::Matrix<float, 2, 2> R = Eigen::Matrix<float, 2, 2>::Zero();
    R.diagonal() << feature.keypoint.covariance.head<2>();

    // measurement prediction covariance
    const Eigen::Matrix<float, 2, 2> S = H * P * H.transpose() + R;

    // filter gain
    const Eigen::Matrix<float, 2, 2> K = P * H.transpose() * S.inverse();

    // measurement
    const Eigen::Vector2f z = feature.keypoint.position.head<2>();

    //    LOG(INFO) << "AddObservation: z: " << z.transpose();
    //    LOG(INFO) << " x: " << x.transpose() << " P: " <<
    //    P.diagonal().transpose();

    // measurement residual
    const Eigen::Vector2f V = z - H * x_1;

    // update state
    x += K * V;

    // update state covariance
    P = (Eigen::Matrix<float, 2, 2>::Identity() - K * H) * P;

    //    LOG(INFO) << " x: " << x.transpose() << " P: " <<
    //    P.diagonal().transpose();
  }

 private:
  Eigen::Vector2f x;
  Eigen::Matrix<float, 2, 2> P =
      Eigen::Matrix<float, 2, 2>::Zero();  // estimate error covariance

  const Eigen::Matrix<float, 2, 2> A =
      Eigen::Matrix<float, 2, 2>::Identity();  // dynamics model
  const Eigen::Matrix<float, 2, 2> Q =
      Eigen::Matrix<float, 2, 2>::Zero();  // process noise
  const Eigen::Matrix<float, 2, 2> H =
      Eigen::Matrix<float, 2, 2>::Identity();  // observation model

  CircleFeature initial_feature_;
};

class Submap2D : public Submap {
 public:
  Submap2D(const Eigen::Vector2f& origin, std::unique_ptr<Grid2D> grid,
           ValueConversionTables* conversion_tables);
  explicit Submap2D(const proto::Submap2D& proto,
                    ValueConversionTables* conversion_tables);

  proto::Submap ToProto(bool include_grid_data) const override;
  void UpdateFromProto(const proto::Submap& proto) override;

  void ToResponseProto(const transform::Rigid3d& global_submap_pose,
                       proto::SubmapQuery::Response* response) const override;

  const Grid2D* grid() const { return grid_.get(); }

  // Insert 'range_data' into this submap using 'range_data_inserter'. The
  // submap must not be finished yet.
  void InsertRangeData(const sensor::RangeData& range_data,
                       const RangeDataInserterInterface* range_data_inserter);

  void InsertCircleFeatures(const std::vector<CircleFeature>& circle_features);

  void Finish();

  const std::vector<CircleFeature> SetCircleFeatures(
      const std::vector<CircleFeature>& circle_features) {
    return circle_features_ = circle_features;
  }

  const std::vector<CircleFeature>& CircleFeatures() const {
    return circle_features_;
  }

 private:
  std::vector<CircleFeature> circle_features_;
  std::vector<CircleFeatureSmoother> circle_feature_smoothers_;

  std::unique_ptr<Grid2D> grid_;
  ValueConversionTables* conversion_tables_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
