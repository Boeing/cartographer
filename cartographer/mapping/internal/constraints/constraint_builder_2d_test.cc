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

#include "cartographer/mapping/internal/constraints/constraint_builder_2d.h"

#include <functional>

#include "cartographer/common/internal/testing/thread_pool_for_testing.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/internal/constraints/constraint_builder.h"
#include "cartographer/mapping/internal/testing/test_helpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cartographer {
namespace mapping {
namespace constraints {
namespace {

class MockCallback {
 public:
  MOCK_METHOD1(Run, void(const ConstraintBuilder2D::Result&));
};

class ConstraintBuilder2DTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto constraint_builder_parameters = testing::ResolveLuaParameters(R"text(
           return {
           min_local_search_score = 0.40,
           min_global_search_score = 0.45,

           -- used when adding INTER submap constraints
           constraint_translation_weight = 2,
           constraint_rotation_weight = 2,
           log_matches = true,
           ceres_scan_matcher = {
               occupied_space_weight = 1,
               translation_weight = 1,
               rotation_weight = 1,
               ceres_solver_options = {
                   use_nonmonotonic_steps = true,
                   max_num_iterations = 100,
                   num_threads = 1,
               },
           },
           global_icp_scan_matcher_options_2d = {
               num_global_samples = 400,
               num_global_rotations = 32,

               proposal_point_inlier_threshold = 0.8,
               proposal_feature_inlier_threshold = 0.8,

               proposal_min_points_inlier_fraction = 0.2,
               proposal_min_features_inlier_fraction = 0.5,

               proposal_features_weight = 1.0,
               proposal_points_weight = 1.0,

               proposal_raytracing_max_error = 1.0,

               proposal_max_points_error = 0.8,
               proposal_max_features_error = 0.8,
               proposal_max_error = 0.8,

               min_cluster_size = 1,
               min_cluster_distance = 1.0,

               num_local_samples = 40,

               local_sample_linear_distance = 0.2,
               local_sample_angular_distance = 0.2,

               icp_options = {
                   nearest_neighbour_point_huber_loss = 0.01,
                   nearest_neighbour_feature_huber_loss = 0.01,

                   point_pair_point_huber_loss = 0.01,
                   point_pair_feature_huber_loss = 0.01,

                   point_weight = 1.0,
                   feature_weight = 10.0,

                   point_inlier_threshold = 0.4,
                   feature_inlier_threshold = 0.4,
               }
           },
           min_icp_score = 0.98,
           min_icp_points_inlier_fraction = 0.3,
           min_icp_features_inlier_fraction = 0.5,
           min_hit_fraction = 0.50})text");
    constraint_builder_ = absl::make_unique<ConstraintBuilder2D>(
        CreateConstraintBuilderOptions(constraint_builder_parameters.get()),
        &thread_pool_);
  }

  std::unique_ptr<ConstraintBuilder2D> constraint_builder_;
  MockCallback mock_;
  common::testing::ThreadPoolForTesting thread_pool_;
};

TEST_F(ConstraintBuilder2DTest, CallsBack) {
  EXPECT_EQ(constraint_builder_->GetNumFinishedNodes(), 0);
  EXPECT_CALL(mock_, Run(::testing::IsEmpty()));
  constraint_builder_->NotifyEndOfNode();
  constraint_builder_->WhenDone(
      [this](const constraints::ConstraintBuilder2D::Result& result) {
        mock_.Run(result);
      });
  thread_pool_.WaitUntilIdle();
  EXPECT_EQ(constraint_builder_->GetNumFinishedNodes(), 1);
}

/*
TEST_F(ConstraintBuilder2DTest, FindsConstraints) {
  TrajectoryNode::Data node_data;
  node_data.filtered_gravity_aligned_point_cloud.push_back(
      {Eigen::Vector3f(0.1, 0.2, 0.3)});
  node_data.gravity_alignment = Eigen::Quaterniond::Identity();
  node_data.local_pose = transform::Rigid3d::Identity();
  SubmapId submap_id{0, 1};
  MapLimits map_limits(1., Eigen::Vector2d(2., 3.), CellLimits(100, 110));
  ValueConversionTables conversion_tables;
  Submap2D submap(
      Eigen::Vector2f(4.f, 5.f),
      absl::make_unique<ProbabilityGrid>(map_limits, &conversion_tables),
      &conversion_tables);
  int expected_nodes = 0;
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(constraint_builder_->GetNumFinishedNodes(), expected_nodes);
    for (int j = 0; j < 2; ++j) {
      constraint_builder_->MaybeAddConstraint(submap_id, &submap, NodeId{0, 0},
                                              &node_data,
                                              transform::Rigid2d::Identity());
    }
    constraint_builder_->MaybeAddGlobalConstraint(submap_id, &submap,
                                                  NodeId{0, 0}, &node_data);
    constraint_builder_->NotifyEndOfNode();
    thread_pool_.WaitUntilIdle();
    EXPECT_EQ(constraint_builder_->GetNumFinishedNodes(), ++expected_nodes);
    constraint_builder_->NotifyEndOfNode();
    thread_pool_.WaitUntilIdle();
    EXPECT_EQ(constraint_builder_->GetNumFinishedNodes(), ++expected_nodes);
    EXPECT_CALL(mock_,
                Run(::testing::AllOf(
                    ::testing::SizeIs(3),
                    ::testing::Each(::testing::Field(
                        &PoseGraphInterface::Constraint::tag,
                        PoseGraphInterface::Constraint::INTER_SUBMAP)))));
    constraint_builder_->WhenDone(
        [this](const constraints::ConstraintBuilder2D::Result& result) {
          mock_.Run(result);
        });
    thread_pool_.WaitUntilIdle();
    constraint_builder_->DeleteScanMatcher(submap_id);
  }
}
*/

}  // namespace
}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer
