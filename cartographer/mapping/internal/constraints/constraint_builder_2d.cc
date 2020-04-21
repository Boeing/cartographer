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

#include "cartographer/mapping/internal/constraints/constraint_builder_2d.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "Eigen/Eigenvalues"
#include "absl/memory/memory.h"
#include "cartographer/common/math.h"
#include "cartographer/common/thread_pool.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"
#include "cartographer/mapping/proto/scan_matching/ceres_scan_matcher_options_2d.pb.h"
#include "cartographer/mapping/proto/scan_matching/fast_correlative_scan_matcher_options_2d.pb.h"
#include "cartographer/metrics/counter.h"
#include "cartographer/metrics/gauge.h"
#include "cartographer/metrics/histogram.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace constraints {

namespace {

double calculateICPscore(const double cost) {
  return std::max(0.01, std::min(1., 1. - cost));
}

}  // namespace

static auto* kConstraintsSearchedMetric = metrics::Counter::Null();
static auto* kConstraintsFoundMetric = metrics::Counter::Null();
static auto* kGlobalConstraintsSearchedMetric = metrics::Counter::Null();
static auto* kGlobalConstraintsFoundMetric = metrics::Counter::Null();
static auto* kQueueLengthMetric = metrics::Gauge::Null();
static auto* kConstraintScoresMetric = metrics::Histogram::Null();
static auto* kGlobalConstraintScoresMetric = metrics::Histogram::Null();

transform::Rigid2d ComputeSubmapPose(const Submap2D& submap) {
  return transform::Project2D(submap.local_pose());
}

ConstraintBuilder2D::ConstraintBuilder2D(
    const constraints::proto::ConstraintBuilderOptions& options,
    common::ThreadPoolInterface* const thread_pool)
    : options_(options),
      thread_pool_(thread_pool),
      finish_node_task_(absl::make_unique<common::Task>()),
      when_done_task_(absl::make_unique<common::Task>()),
      ceres_scan_matcher_(options.ceres_scan_matcher_options()) {}

ConstraintBuilder2D::~ConstraintBuilder2D() {
  absl::MutexLock locker(&mutex_);
  CHECK_EQ(finish_node_task_->GetState(), common::Task::NEW);
  CHECK_EQ(when_done_task_->GetState(), common::Task::NEW);
  CHECK_EQ(constraints_.size(), 0) << "WhenDone() was not called";
  CHECK_EQ(num_started_nodes_, num_finished_nodes_);
  CHECK(when_done_ == nullptr);
}

void ConstraintBuilder2D::LocalSearchForConstraint(
    const NodeId node_id, const SubmapId submap_id,
    const transform::Rigid2d& initial_relative_pose, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data) {
  absl::MutexLock locker(&mutex_);
  CHECK(!when_done_);

  constraints_.emplace_back();
  kQueueLengthMetric->Set(constraints_.size());
  auto* const constraint = &constraints_.back();
  const auto* scan_matcher = DispatchScanMatcherConstruction(submap_id, submap);
  auto constraint_task = absl::make_unique<common::Task>();
  constraint_task->SetWorkItem([node_id, submap_id, initial_relative_pose,
                                &submap, &constant_data, constraint,
                                scan_matcher, this]() LOCKS_EXCLUDED(mutex_) {
    this->ComputeConstraint(node_id, submap_id, initial_relative_pose, submap,
                            constant_data, false, scan_matcher, constraint);
  });
  constraint_task->AddDependency(scan_matcher->creation_task_handle);
  auto constraint_task_handle =
      thread_pool_->Schedule(std::move(constraint_task));
  finish_node_task_->AddDependency(constraint_task_handle);
}

void ConstraintBuilder2D::GlobalSearchForConstraint(
    const NodeId node_id, const SubmapId submap_id, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data) {
  absl::MutexLock locker(&mutex_);
  CHECK(!when_done_);

  constraints_.emplace_back();
  kQueueLengthMetric->Set(constraints_.size());
  auto* const constraint = &constraints_.back();
  const auto* scan_matcher = DispatchScanMatcherConstruction(submap_id, submap);
  auto constraint_task = absl::make_unique<common::Task>();
  constraint_task->SetWorkItem([node_id, submap_id, &submap, &constant_data,
                                constraint, scan_matcher,
                                this]() LOCKS_EXCLUDED(mutex_) {
    this->ComputeConstraint(node_id, submap_id, transform::Rigid2d::Identity(),
                            submap, constant_data, true, scan_matcher,
                            constraint);
  });
  constraint_task->AddDependency(scan_matcher->creation_task_handle);
  auto constraint_task_handle =
      thread_pool_->Schedule(std::move(constraint_task));
  finish_node_task_->AddDependency(constraint_task_handle);
}

void ConstraintBuilder2D::NotifyEndOfNode() {
  absl::MutexLock locker(&mutex_);
  CHECK(finish_node_task_ != nullptr);
  finish_node_task_->SetWorkItem([this] {
    absl::MutexLock locker(&mutex_);
    ++num_finished_nodes_;
  });
  auto finish_node_task_handle =
      thread_pool_->Schedule(std::move(finish_node_task_));
  finish_node_task_ = absl::make_unique<common::Task>();
  when_done_task_->AddDependency(finish_node_task_handle);
  ++num_started_nodes_;
}

void ConstraintBuilder2D::WhenDone(
    const std::function<void(const ConstraintBuilder2D::Result&)>& callback) {
  absl::MutexLock locker(&mutex_);
  CHECK(when_done_ == nullptr);
  // TODO(gaschler): Consider using just std::function, it can also be empty.
  when_done_ = absl::make_unique<std::function<void(const Result&)>>(callback);
  CHECK(when_done_task_ != nullptr);
  when_done_task_->SetWorkItem([this] { RunWhenDoneCallback(); });
  thread_pool_->Schedule(std::move(when_done_task_));
  when_done_task_ = absl::make_unique<common::Task>();
}

const ConstraintBuilder2D::SubmapScanMatcher*
ConstraintBuilder2D::DispatchScanMatcherConstruction(const SubmapId& submap_id,
                                                     const Submap2D& submap) {
  if (submap_scan_matchers_.count(submap_id) != 0) {
    return &submap_scan_matchers_.at(submap_id);
  }
  auto ret = submap_scan_matchers_.emplace(
      std::make_pair(submap_id, SubmapScanMatcher{submap, {}, {}}));
  SubmapScanMatcher& submap_scan_matcher = ret.first->second;

  std::unique_ptr<common::Task> scan_matcher_task;

  auto& scan_matcher_options = options_.global_icp_scan_matcher_options_2d();
  scan_matcher_task = absl::make_unique<common::Task>();
  scan_matcher_task->SetWorkItem(
      [&submap_scan_matcher, &scan_matcher_options]() {
        submap_scan_matcher.global_icp_scan_matcher =
            absl::make_unique<scan_matching::GlobalICPScanMatcher2D>(
                submap_scan_matcher.submap, scan_matcher_options);
      });

  submap_scan_matcher.creation_task_handle =
      thread_pool_->Schedule(std::move(scan_matcher_task));
  return &submap_scan_matchers_.at(submap_id);
}

void ConstraintBuilder2D::ComputeConstraint(
    const NodeId node_id, const SubmapId submap_id,
    const transform::Rigid2d initial_relative_pose, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data, const bool match_full_submap,
    const SubmapScanMatcher* submap_scan_matcher,
    std::unique_ptr<Constraint>* constraint) {
  CHECK(submap_scan_matcher);
  CHECK(constraint);

  const transform::Rigid2d initial_pose =
      ComputeSubmapPose(submap) * initial_relative_pose;

  // The 'constraint_transform' (submap i <- node j) is computed from:
  // - a 'filtered_gravity_aligned_point_cloud' in node j,
  // - the initial guess 'initial_pose' for (map <- node j),
  // - the result 'pose_estimate' of Match() (map <- node j).
  // - the ComputeSubmapPose() (map <- submap i)

  CHECK(submap_scan_matcher->global_icp_scan_matcher);
  const auto t0 = std::chrono::steady_clock::now();
  auto _t0 = std::chrono::steady_clock::now();

  cartographer::mapping::scan_matching::GlobalICPScanMatcher2D::Result
      match_result;

  if (match_full_submap) {
    kGlobalConstraintsSearchedMetric->Increment();
    match_result = submap_scan_matcher->global_icp_scan_matcher->Match(
        constant_data.filtered_point_cloud, constant_data.circle_features);
  } else {
    kConstraintsSearchedMetric->Increment();
    match_result = submap_scan_matcher->global_icp_scan_matcher->Match(
        initial_pose, constant_data.filtered_point_cloud,
        constant_data.circle_features);
  }

  const auto clusters =
      submap_scan_matcher->global_icp_scan_matcher->DBScanCluster(
          match_result.poses, constant_data.filtered_point_cloud,
          constant_data.circle_features);

  LOG(INFO) << "Found " << match_result.poses.size() << " good proposals"
            << " (took: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::steady_clock::now() - _t0)
                   .count()
            << ") with " << clusters.size() << " clusters";

  struct MatchResult {
    float score;
    transform::Rigid2d pose_estimate;
    transform::Rigid2d icp_pose_proposal;
    cartographer::mapping::scan_matching::ICPScanMatcher2D::Result icp_result;
    bool success;
  };

  MatchResult* best_match = nullptr;
  std::vector<MatchResult> cluster_results(clusters.size());
  for (size_t i = 0; i < clusters.size(); ++i) {
    const transform::Rigid2d cluster_estimate({clusters[i].x, clusters[i].y},
                                              clusters[i].rotation);

    // ICP the proposal
    // Run ICP at a few different initial rotations about the proposal
    // ICP is very sensitive to the initial conditions
    // Sometimes the best convergence is a nearby rotation
    // I'm not convinced this is necessary yet - more data required to confirm
    cartographer::mapping::scan_matching::ICPScanMatcher2D::Result icp_match;
    icp_match.summary.final_cost = std::numeric_limits<double>::max();
    double best_score = 0;
    for (int i = -2; i <= 2; ++i) {
      const double angle_diff = 0.2 * i;
      const transform::Rigid2d estimate(
          cluster_estimate.translation(),
          cluster_estimate.rotation().smallestAngle() + angle_diff);

      auto this_icp_match =
          submap_scan_matcher->global_icp_scan_matcher->IcpSolver().Match(
              estimate, constant_data.filtered_point_cloud,
              constant_data.circle_features);
      const double score =
          calculateICPscore(this_icp_match.summary.final_cost) *
          this_icp_match.points_inlier_fraction;
      //        LOG(INFO) << "i: " << i << " icp: " <<
      //        calculateICPscore(this_icp_match.summary.final_cost) << "
      //        p_inlier: " << this_icp_match.points_inlier_fraction << " score:
      //        " << score;
      if (score > best_score) {
        best_score = score;
        icp_match = this_icp_match;
      }
    }

    const double icp_score = calculateICPscore(icp_match.summary.final_cost);

    const auto statistics =
        submap_scan_matcher->global_icp_scan_matcher->IcpSolver()
            .EvalutateMatch(icp_match, constant_data.range_data);

    const bool icp_good = icp_score >= options_.min_icp_score();
    const bool icp_points_inlier_good =
        icp_match.points_inlier_fraction >=
        options_.min_icp_points_inlier_fraction();
    const bool icp_features_inlier_good =
        icp_match.features_inlier_fraction >=
        options_.min_icp_features_inlier_fraction();

    // More features mean we need less actual points to match
    const double require_hit = std::max(
        0.15, options_.min_hit_fraction() - 0.01 * icp_match.features_count);

    const bool hit_good = statistics.hit_fraction >= require_hit;

    const bool rt_good =
        statistics.ray_trace_fraction >= options_.min_ray_trace_fraction();

    const bool match_successful = icp_good && icp_points_inlier_good &&
                                  icp_features_inlier_good && hit_good &&
                                  rt_good;

    const float overall_score =
        static_cast<float>(icp_score * statistics.hit_fraction);

    LOG(INFO) << match_successful << " (" << clusters[i].origin.error << " -> "
              << overall_score << ")"
              << " icp: " << icp_score << "(" << icp_good << ")"
              << " p: (" << icp_match.points_count << "/"
              << constant_data.filtered_point_cloud.size()
              << ")=" << icp_match.points_inlier_fraction << "("
              << icp_points_inlier_good << ")"
              << " f: (" << icp_match.features_count << "/"
              << constant_data.circle_features.size()
              << ")=" << icp_match.features_inlier_fraction << "("
              << icp_features_inlier_good << ")"
              << " rt: " << statistics.ray_trace_fraction << "(" << rt_good
              << ")"
              << " hit: " << statistics.hit_fraction << "(" << hit_good << ")";

    cluster_results[i] =
        MatchResult{overall_score, icp_match.pose_estimate, cluster_estimate,
                    icp_match, match_successful};

    if (match_successful &&
        (!best_match || overall_score > best_match->score)) {
      best_match = &cluster_results[i];
    }
  }

  if (!best_match) {
    for (auto& cr : cluster_results)
      if (!best_match || cr.score > best_match->score) best_match = &cr;
  }

  if (best_match) {
    LOG(INFO) << "best_match:  score: " << best_match->score
              << " success: " << best_match->success;
  }

  //
  // Debug visualisation
  //
  {
    const auto grid = dynamic_cast<const mapping::ProbabilityGrid*>(
        submap_scan_matcher->submap.grid());

    auto surface = grid->DrawSurface();

    cairo_t* cr = cairo_create(surface.get());

    for (const auto& f : submap.CircleFeatures()) {
      cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.5);
      cairo_set_line_width(cr, 1.0);
      const auto mp = grid->limits().GetCellIndex(
          {f.keypoint.position.x(), f.keypoint.position.y()});
      cairo_arc(cr, mp.x(), mp.y(), 20.0, 0, 2 * M_PI);
      cairo_stroke(cr);
      cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      cairo_fill(cr);
    }

    for (const auto cell :
         submap_scan_matcher->global_icp_scan_matcher->IcpSolver()
             .kdtree()
             .cells.cells) {
      cairo_set_source_rgba(cr, 1.0, 1.0, 0.3, 1);
      cairo_set_line_width(cr, 1.0);
      const auto mp = grid->limits().GetCellIndex({cell.x, cell.y});
      cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      cairo_fill(cr);
    }

    for (const auto& cluster : clusters) {
      cairo_new_path(cr);
      cairo_set_source_rgba(cr, 1, 0, 0, 1);
      cairo_set_line_width(cr, 1.0);
      const auto mp = grid->limits().GetCellIndex({cluster.x, cluster.y});
      cairo_arc(cr, mp.x(), mp.y(), 8.0, 0, 2 * M_PI);
      auto dir = Eigen::Rotation2Df(cluster.poses.front().rotation) *
                 Eigen::Vector2f(10.0, 0);
      cairo_stroke(cr);
      cairo_new_path(cr);
      cairo_move_to(cr, mp.x(), mp.y());
      cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
      cairo_stroke(cr);

      std::uniform_real_distribution<double> dist(0, 1);
      std::random_device rd;
      std::mt19937 gen(rd());

      const double c_b = dist(gen);

      for (const auto& pose : cluster.poses) {
        const auto _mp = grid->limits().GetCellIndex({pose.x, pose.y});
        cairo_set_source_rgba(cr, 0.2, 0.2, c_b, 1.0);
        cairo_new_path(cr);
        cairo_arc(cr, _mp.x(), _mp.y(), 4.0, 0, 2 * M_PI);
        cairo_stroke(cr);

        auto dir = Eigen::Rotation2Df(pose.rotation) * Eigen::Vector2f(5.0, 0);

        cairo_new_path(cr);
        cairo_move_to(cr, _mp.x(), _mp.y());
        cairo_line_to(cr, _mp.x() - dir.y(), _mp.y() - dir.x());
        cairo_stroke(cr);
      }
    }

    for (const auto& match : cluster_results) {
      if (match.success)
        cairo_set_source_rgba(cr, 0.5, 1.0, 0, match.score);
      else
        cairo_set_source_rgba(cr, 1.0, 0.0, 0, match.score);
      cairo_set_line_width(cr, 2.0);
      cairo_new_path(cr);
      const auto mp =
          grid->limits().GetCellIndex({match.pose_estimate.translation().x(),
                                       match.pose_estimate.translation().y()});
      cairo_arc(cr, mp.x(), mp.y(), 14.0, 0, 2 * M_PI);
      const auto dir = match.pose_estimate.rotation().cast<float>() *
                       Eigen::Vector2f(30.0, 0);
      cairo_stroke(cr);
      cairo_new_path(cr);
      cairo_move_to(cr, mp.x(), mp.y());
      cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
      cairo_stroke(cr);
    }

    // Visualise the best match
    if (best_match) {
      if (best_match->success)
        cairo_set_source_rgba(cr, 0.5, 1.0, 0, 1);
      else
        cairo_set_source_rgba(cr, 1.0, 0.0, 0, 1);
      cairo_set_line_width(cr, 2.0);
      cairo_new_path(cr);
      const auto mp = grid->limits().GetCellIndex(
          {best_match->pose_estimate.translation().x(),
           best_match->pose_estimate.translation().y()});
      cairo_arc(cr, mp.x(), mp.y(), 30.0, 0, 2 * M_PI);
      const auto dir = best_match->pose_estimate.rotation().cast<float>() *
                       Eigen::Vector2f(30.0, 0);
      cairo_stroke(cr);
      cairo_new_path(cr);
      cairo_move_to(cr, mp.x(), mp.y());
      cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
      cairo_stroke(cr);

      cairo_set_line_width(cr, 1.0);

      const sensor::PointCloud actual_tpc = sensor::TransformPointCloud(
          constant_data.range_data.returns,
          transform::Embed3D(best_match->pose_estimate.cast<float>()));
      for (const auto& point : actual_tpc) {
        const auto mp = grid->limits().GetCellIndex(
            {point.position.x(), point.position.y()});
        if (point.intensity > 0)
          cairo_set_source_rgba(cr, 1, 1, 1, 1);
        else
          cairo_set_source_rgba(cr, 0, 0, 1, 1);
        cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
        cairo_fill(cr);
      }

      const sensor::PointCloud actual_icp_tpc = sensor::TransformPointCloud(
          constant_data.filtered_point_cloud,
          transform::Embed3D(best_match->pose_estimate.cast<float>()));
      for (const auto& point : actual_icp_tpc) {
        const auto mp = grid->limits().GetCellIndex(
            {point.position.x(), point.position.y()});
        cairo_set_source_rgba(cr, 1, 0.75, 0.79, 1);
        cairo_arc(cr, mp.x(), mp.y(), 4, 0, 2 * M_PI);
        cairo_stroke(cr);
      }

      for (const auto& pair : best_match->icp_result.pairs) {
        cairo_set_source_rgba(cr, 0, 1, 0, 1.0);
        cairo_set_line_width(cr, 1.0);
        const auto src = grid->limits().GetCellIndex(pair.first.cast<float>());
        const auto dst = grid->limits().GetCellIndex(pair.second.cast<float>());
        cairo_move_to(cr, src.x(), src.y());
        cairo_line_to(cr, dst.x(), dst.y());
        cairo_stroke(cr);
      }

      const auto proposal_tpc = sensor::TransformPointCloud(
          constant_data.range_data.returns,
          transform::Embed3D(best_match->icp_pose_proposal.cast<float>()));
      for (const auto& point : proposal_tpc) {
        cairo_set_source_rgba(cr, 1, 0, 0, 1);
        const auto mp = grid->limits().GetCellIndex(
            {point.position.x(), point.position.y()});
        cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
        cairo_fill(cr);
      }

      for (const auto& f : constant_data.circle_features) {
        cairo_set_source_rgba(cr, 1, 0.8, 0.2, 1.0);
        cairo_set_line_width(cr, 2.0);
        const auto tr =
            best_match->pose_estimate.cast<float>() *
            Eigen::Vector2f(f.keypoint.position.x(), f.keypoint.position.y());
        const auto mp = grid->limits().GetCellIndex(tr);
        cairo_arc(cr, mp.x(), mp.y(), 10.0, 0, 2 * M_PI);
        cairo_stroke(cr);
      }
    }

    cairo_destroy(cr);

    const std::string fname = "cartographer_debug/match_t" +
                              std::to_string(submap_id.trajectory_id) + "_n" +
                              std::to_string(node_id.node_index) + "_s" +
                              std::to_string(submap_id.submap_index) + ".png";
    cairo_surface_write_to_png(surface.get(), fname.c_str());
  }

  if (best_match && best_match->success &&
      ((match_full_submap &&
        best_match->score > options_.min_global_search_score()) ||
       (!match_full_submap &&
        best_match->score > options_.min_local_search_score()))) {
    CHECK_GE(node_id.trajectory_id, 0);
    CHECK_GE(submap_id.trajectory_id, 0);

    if (match_full_submap) {
      kGlobalConstraintsFoundMetric->Increment();
      kGlobalConstraintScoresMetric->Observe(best_match->score);
    } else {
      kConstraintsFoundMetric->Increment();
      kConstraintScoresMetric->Observe(best_match->score);
    }

    LOG(INFO) << "SUCCESS matching node_id: " << node_id
              << " submap_id: " << submap_id << " score: " << best_match->score
              << " took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
  } else {
    LOG(INFO) << "FAILED matching node_id: " << node_id
              << " submap_id: " << submap_id << " took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return;
  }

  {
    absl::MutexLock locker(&mutex_);
    score_histogram_.Add(best_match->score);
  }

  // Use the CSM estimate as both the initial and previous pose. This has the
  // effect that, in the absence of better information, we prefer the original
  // CSM estimate.
  ceres::Solver::Summary unused_summary;
  ceres_scan_matcher_.Match(
      best_match->pose_estimate.translation(), best_match->pose_estimate,
      constant_data.filtered_point_cloud, *submap_scan_matcher->submap.grid(),
      &best_match->pose_estimate, &unused_summary);

  const transform::Rigid2d constraint_transform =
      ComputeSubmapPose(submap).inverse() * best_match->pose_estimate;
  constraint->reset(new Constraint{
      submap_id,
      node_id,
      {transform::Embed3D(constraint_transform),
       best_match->score * options_.constraint_translation_weight(),
       best_match->score * options_.constraint_rotation_weight()},
      Constraint::INTER_SUBMAP});

  if (options_.log_matches()) {
    std::ostringstream info;
    info << "Node " << node_id << " with "
         << constant_data.filtered_point_cloud.size() << " points on submap "
         << submap_id << std::fixed;
    if (match_full_submap) {
      info << " matches";
    } else {
      const transform::Rigid2d difference =
          initial_pose.inverse() * best_match->pose_estimate;
      info << " differs by translation " << std::setprecision(2)
           << difference.translation().norm() << " rotation "
           << std::setprecision(3) << std::abs(difference.normalized_angle());
    }
    info << " with score " << std::setprecision(1) << 100. * best_match->score
         << "%.";
    LOG(INFO) << info.str();
  }
}

void ConstraintBuilder2D::RunWhenDoneCallback() {
  Result result;
  std::unique_ptr<std::function<void(const Result&)>> callback;
  {
    absl::MutexLock locker(&mutex_);
    CHECK(when_done_ != nullptr);
    for (const std::unique_ptr<Constraint>& constraint : constraints_) {
      if (constraint == nullptr) continue;
      result.push_back(*constraint);
    }
    if (options_.log_matches()) {
      LOG(INFO) << constraints_.size() << " computations resulted in "
                << result.size() << " additional constraints.";
      LOG(INFO) << "Score histogram:\n" << score_histogram_.ToString(10);
    }
    constraints_.clear();
    callback = std::move(when_done_);
    when_done_.reset();
    kQueueLengthMetric->Set(constraints_.size());
  }
  (*callback)(result);
}

int ConstraintBuilder2D::GetNumFinishedNodes() {
  absl::MutexLock locker(&mutex_);
  return num_finished_nodes_;
}

void ConstraintBuilder2D::DeleteScanMatcher(const SubmapId& submap_id) {
  absl::MutexLock locker(&mutex_);
  if (when_done_) {
    LOG(WARNING)
        << "DeleteScanMatcher was called while WhenDone was scheduled.";
  }
  submap_scan_matchers_.erase(submap_id);
}

void ConstraintBuilder2D::RegisterMetrics(metrics::FamilyFactory* factory) {
  auto* counts = factory->NewCounterFamily(
      "mapping_constraints_constraint_builder_2d_constraints",
      "Constraints computed");
  kConstraintsSearchedMetric =
      counts->Add({{"search_region", "local"}, {"matcher", "searched"}});
  kConstraintsFoundMetric =
      counts->Add({{"search_region", "local"}, {"matcher", "found"}});
  kGlobalConstraintsSearchedMetric =
      counts->Add({{"search_region", "global"}, {"matcher", "searched"}});
  kGlobalConstraintsFoundMetric =
      counts->Add({{"search_region", "global"}, {"matcher", "found"}});
  auto* queue_length = factory->NewGaugeFamily(
      "mapping_constraints_constraint_builder_2d_queue_length", "Queue length");
  kQueueLengthMetric = queue_length->Add({});
  auto boundaries = metrics::Histogram::FixedWidth(0.05, 20);
  auto* scores = factory->NewHistogramFamily(
      "mapping_constraints_constraint_builder_2d_scores",
      "Constraint scores built", boundaries);
  kConstraintScoresMetric = scores->Add({{"search_region", "local"}});
  kGlobalConstraintScoresMetric = scores->Add({{"search_region", "global"}});
}

}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer
