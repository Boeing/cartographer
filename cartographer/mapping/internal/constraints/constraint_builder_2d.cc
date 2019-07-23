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
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"
#include "cartographer/metrics/counter.h"
#include "cartographer/metrics/gauge.h"
#include "cartographer/metrics/histogram.h"
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace constraints {

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
    const NodeId node_id,
    const SubmapId submap_id,
    const transform::Rigid2d& initial_relative_pose,
    const Submap2D& submap,
    const TrajectoryNode::Data& constant_data)
{
    absl::MutexLock locker(&mutex_);
    CHECK(!when_done_);

    constraints_.emplace_back();
    kQueueLengthMetric->Set(constraints_.size());
    auto* const constraint = &constraints_.back();
    const auto* scan_matcher = DispatchScanMatcherConstruction(submap_id, submap);
    auto constraint_task = absl::make_unique<common::Task>();
    constraint_task->SetWorkItem([node_id, submap_id, initial_relative_pose, &submap, &constant_data, constraint, scan_matcher, this]() LOCKS_EXCLUDED(mutex_) {
      this->ComputeConstraint(node_id, submap_id, initial_relative_pose, submap, constant_data, false, scan_matcher, constraint);
    });
    constraint_task->AddDependency(scan_matcher->creation_task_handle);
    auto constraint_task_handle = thread_pool_->Schedule(std::move(constraint_task));
    finish_node_task_->AddDependency(constraint_task_handle);
}

void ConstraintBuilder2D::GlobalSearchForConstraint(
    const NodeId node_id,
    const SubmapId submap_id,
    const Submap2D& submap,
    const TrajectoryNode::Data& constant_data)
{
    absl::MutexLock locker(&mutex_);
    CHECK(!when_done_);

    constraints_.emplace_back();
    kQueueLengthMetric->Set(constraints_.size());
    auto* const constraint = &constraints_.back();
    const auto* scan_matcher = DispatchScanMatcherConstruction(submap_id, submap);
    auto constraint_task = absl::make_unique<common::Task>();
    constraint_task->SetWorkItem([node_id, submap_id, &submap, &constant_data, constraint, scan_matcher, this]() LOCKS_EXCLUDED(mutex_) {
      this->ComputeConstraint(node_id, submap_id, transform::Rigid2d::Identity(), submap, constant_data, true, scan_matcher, constraint);
    });
    constraint_task->AddDependency(scan_matcher->creation_task_handle);
    auto constraint_task_handle = thread_pool_->Schedule(std::move(constraint_task));
    finish_node_task_->AddDependency(constraint_task_handle);
}

void ConstraintBuilder2D::NotifyEndOfNode() {
  absl::MutexLock locker(&mutex_);
  CHECK(finish_node_task_ != nullptr);
  finish_node_task_->SetWorkItem([this] {
    absl::MutexLock locker(&mutex_);
    ++num_finished_nodes_;
  });
  auto finish_node_task_handle = thread_pool_->Schedule(std::move(finish_node_task_));
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
ConstraintBuilder2D::DispatchScanMatcherConstruction(const SubmapId& submap_id, const Submap2D& submap)
{
  if (submap_scan_matchers_.count(submap_id) != 0) {
    return &submap_scan_matchers_.at(submap_id);
  }
  auto ret = submap_scan_matchers_.emplace(std::make_pair(submap_id, SubmapScanMatcher{submap, {}, {}}));
  SubmapScanMatcher& submap_scan_matcher = ret.first->second;

  std::unique_ptr<common::Task> scan_matcher_task;

  auto& scan_matcher_options = options_.global_icp_scan_matcher_options_2d();
  scan_matcher_task = absl::make_unique<common::Task>();
  scan_matcher_task->SetWorkItem(
      [&submap_scan_matcher, &scan_matcher_options]() {
        submap_scan_matcher.global_icp_scan_matcher =
            absl::make_unique<scan_matching::GlobalICPScanMatcher2D>(submap_scan_matcher.submap, scan_matcher_options);
  });

  submap_scan_matcher.creation_task_handle = thread_pool_->Schedule(std::move(scan_matcher_task));
  return &submap_scan_matchers_.at(submap_id);
}

void ConstraintBuilder2D::ComputeConstraint(
    const NodeId node_id,
    const SubmapId submap_id,
    const transform::Rigid2d initial_relative_pose,
    const Submap2D& submap,
    const TrajectoryNode::Data& constant_data,
    const bool match_full_submap,
    const SubmapScanMatcher* submap_scan_matcher,
    std::unique_ptr<Constraint>* constraint)
{
  CHECK(submap_scan_matcher);
  CHECK(constraint);

  const transform::Rigid2d initial_pose = ComputeSubmapPose(submap) * initial_relative_pose;

  // The 'constraint_transform' (submap i <- node j) is computed from:
  // - a 'filtered_gravity_aligned_point_cloud' in node j,
  // - the initial guess 'initial_pose' for (map <- node j),
  // - the result 'pose_estimate' of Match() (map <- node j).
  // - the ComputeSubmapPose() (map <- submap i)
  float score = 0.;
  transform::Rigid2d pose_estimate = transform::Rigid2d::Identity();

  // Compute 'pose_estimate' in three stages:
  // 1. Fast estimate using the fast correlative scan matcher.
  // 2. Prune if the score is too low.
  // 3. Refine.

    CHECK(submap_scan_matcher->global_icp_scan_matcher);
    const auto t0 = std::chrono::steady_clock::now();
    auto _t0 = std::chrono::steady_clock::now();

    cartographer::mapping::scan_matching::GlobalICPScanMatcher2D::Result match_result;

    if (match_full_submap) {
      kGlobalConstraintsSearchedMetric->Increment();
      match_result = submap_scan_matcher->global_icp_scan_matcher->Match(
          constant_data.filtered_point_cloud);
    } else {
      kConstraintsSearchedMetric->Increment();
      match_result = submap_scan_matcher->global_icp_scan_matcher->Match(
          initial_pose, constant_data.filtered_point_cloud);
    }

    const auto clusters =
        submap_scan_matcher->global_icp_scan_matcher->DBScanCluster(
            match_result.poses);

    LOG(INFO) << "Found " << match_result.poses.size() << " good proposals"
              << " (took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::steady_clock::now() - _t0)
                     .count()
              << ") with " << clusters.size() << " clusters";

    transform::Rigid2d icp_pose_proposal = transform::Rigid2d::Identity();
    cartographer::mapping::scan_matching::ICPScanMatcher2D::Result icp_result;

    const auto test_tr = transform::Embed3D(pose_estimate.cast<float>());
    const auto test_actual_tpc = sensor::TransformPointCloud(constant_data.filtered_point_cloud, test_tr);

    bool success = false;
    double best_icp_score = 0;
    for (size_t i = 0; i < clusters.size(); ++i) {
      const transform::Rigid2d cluster_estimate({clusters[i].x, clusters[i].y},
                                                clusters[i].rotation);

      auto icp_match =
          submap_scan_matcher->global_icp_scan_matcher->IcpSolver().Match(
              cluster_estimate,
              constant_data.filtered_point_cloud,
              constant_data.circle_features);

//      LOG(INFO) << "ICP: " << cluster_estimate << " -> "
//                << icp_match.pose_estimate
//                << " cost: " << icp_match.summary.initial_cost << " -> "
//                << icp_match.summary.final_cost;

      for (int ii = 0; ii < 100; ++ii) {
        icp_match =
            submap_scan_matcher->global_icp_scan_matcher->IcpSolver()
                .MatchPointPair(
                    icp_match.pose_estimate,
                    constant_data.filtered_point_cloud,
                    constant_data.circle_features);

//        LOG(INFO) << "ICP: " << cluster_estimate << " -> "
//                  << icp_match.pose_estimate
//                  << " cost: " << icp_match.summary.initial_cost << " -> "
//                  << icp_match.summary.final_cost;

        if (icp_match.summary.final_cost < 0.001) break;
      }

      double icp_score =
          std::max(0.01, std::min(1., 1. - icp_match.summary.final_cost));

      // point match check
      float combined_score = 0;
      double agree_fraction = 1.0;
      double miss_fraction = 1.0;
      double hit_fraction = 1.0;
      {
          auto pg = dynamic_cast<const mapping::ProbabilityGrid*>(submap.grid());

          int hit_count = 0;
          const auto mid_value = CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));
          for (const auto& laser_return : constant_data.range_data.returns)
          {
              const auto tr = icp_match.pose_estimate.cast<float>() * laser_return.position.head<2>();
              const auto mp = pg->limits().GetCellIndex(tr);
              if (!pg->limits().Contains(mp))
                  continue;
              const auto cc = pg->correspondence_cost_cells().at(pg->ToFlatIndex(mp));
              if (cc < mid_value || cc == 0)
                  ++hit_count;
          }
          if (!constant_data.range_data.returns.empty())
                hit_fraction = static_cast<double>(hit_count) / static_cast<double>(constant_data.range_data.returns.size());

          int miss_count = 0;
          for (const auto& laser_miss : constant_data.range_data.misses)
          {
              const auto tr = icp_match.pose_estimate.cast<float>() * laser_miss.position.head<2>();
              const auto mp_start = pg->limits().GetCellIndex(icp_match.pose_estimate.translation().cast<float>());
              const auto mp_end = pg->limits().GetCellIndex(tr);
              const unsigned int max_cells = static_cast<unsigned int>(static_cast<double>(laser_miss.position.norm()) / pg->limits().resolution());
              const auto p = mapping::scan_matching::raytraceLine(*pg, mp_start.x(), mp_start.y(), mp_end.x(), mp_end.y(), pg->limits().cell_limits().num_x_cells, max_cells);
              if (p.x == -1 && p.y == -1)
                  ++miss_count;

          }
          if (!constant_data.range_data.misses.empty())
                miss_fraction = static_cast<double>(miss_count) / static_cast<double>(constant_data.range_data.misses.size());

          {
              const int total_count = hit_count + miss_count;
              const size_t total_sum = constant_data.range_data.misses.size() + constant_data.range_data.returns.size();
              if (total_sum > 0)
                  agree_fraction = static_cast<double>(total_count) / static_cast<double>(total_sum);
          }

          combined_score = icp_score * agree_fraction;

          LOG(INFO) << "score: " << (icp_score * agree_fraction) << " icp_score: " << icp_score
                    << " hit_fraction: " << hit_fraction << " miss_fraction: " << miss_fraction << " agree_fraction: " << agree_fraction;
      }

      if (icp_score > best_icp_score) {
        best_icp_score = icp_score;
        score = combined_score;
        pose_estimate = icp_match.pose_estimate;
        icp_pose_proposal = cluster_estimate;
        icp_result = icp_match;
        success = (icp_score > options_.min_icp_score()) && (agree_fraction > options_.min_scan_agreement_fraction());
      }
    }

    //
    // Debug visualisation
    //
    if (!clusters.empty())
    {
        const auto grid = dynamic_cast<const mapping::ProbabilityGrid*>(submap_scan_matcher->submap.grid());

        auto surface = grid->DrawSurface();

        cairo_t* cr = cairo_create(surface.get());

        for (const auto cell : submap_scan_matcher->global_icp_scan_matcher->IcpSolver().kdtree().cells.cells)
        {
            cairo_set_source_rgba(cr, 1.0, 1.0, 0.3, 1);
            cairo_set_line_width(cr, 1.0);
            const auto mp = grid->limits().GetCellIndex({cell.x, cell.y});
            cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
            cairo_fill(cr);
        }

        for (std::size_t i=0; i < match_result.poses.size(); ++i)
        {
            const auto& match = match_result.poses[i];

            const double max_score = 2.0;
            double intensity = std::max(0., max_score - match.score) / max_score;

            auto dir = Eigen::Rotation2Df(match.rotation) * Eigen::Vector2f(10.0, 0);

            cairo_set_source_rgba(cr, 0.1, 1.0, 0, intensity);

            if (i==0)
                cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);

            cairo_set_line_width(cr, 1.0);
            const auto mp = grid->limits().GetCellIndex({match.x, match.y});
            cairo_move_to(cr, mp.x(), mp.y());
            cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2*M_PI);
            cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
            cairo_stroke(cr);

            if (i==0)
            {
                auto match_pose = transform::Rigid2f({match.x, match.y}, match.rotation);
                auto actual_tpc = sensor::TransformPointCloud(constant_data.filtered_point_cloud, transform::Embed3D(match_pose));
                for (const auto& point : actual_tpc)
                {
                  cairo_set_source_rgba(cr, 1, 1, 1, 1);
                  const auto mp = grid->limits().GetCellIndex({point.position.x(), point.position.y()});
                  cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
                  cairo_fill(cr);
                }
            }
        }

        for (const auto& cluster : clusters)
        {
          cairo_set_source_rgba(cr, 1, 0, 0, 1);
          cairo_set_line_width(cr, 1.0);
          const auto mp = grid->limits().GetCellIndex({cluster.x, cluster.y});
          cairo_arc(cr, mp.x(), mp.y(), 16.0, 0, 2*M_PI);
          auto dir = Eigen::Rotation2Df(cluster.poses.front().rotation) * Eigen::Vector2f(10.0, 0);
          cairo_move_to(cr, mp.x(), mp.y());
          cairo_line_to(cr, mp.x() - dir.y(),  mp.y() - dir.x());
          cairo_stroke(cr);

          for (const auto& pose : cluster.poses)
          {
              cairo_set_source_rgba(cr, 1, 0, 0, 0.1);
              cairo_set_line_width(cr, 1.0);
              const auto mp = grid->limits().GetCellIndex({pose.x, pose.y});
              cairo_arc(cr, mp.x(), mp.y(), 14.0, 0, 2*M_PI);
              cairo_stroke(cr);
          }
        }

        {
          cairo_set_source_rgba(cr, 0.5, 1.0, 0, 1);
          cairo_set_line_width(cr, 1.0);
          const auto mp = grid->limits().GetCellIndex({pose_estimate.translation().x(), pose_estimate.translation().y()});
          cairo_arc(cr, mp.x(), mp.y(), 30.0, 0, 2*M_PI);
          const auto dir = pose_estimate.rotation().cast<float>() * Eigen::Vector2f(30.0, 0);
          cairo_move_to(cr, mp.x(), mp.y());
          cairo_line_to(cr, mp.x() - dir.y(),  mp.y() - dir.x());
          cairo_stroke(cr);

          const sensor::PointCloud actual_tpc = sensor::TransformPointCloud(constant_data.filtered_point_cloud, transform::Embed3D(pose_estimate.cast<float>()));
          for (const auto& point : actual_tpc)
          {
            cairo_set_source_rgba(cr, 0, 0, 1, 1);
            const auto mp = grid->limits().GetCellIndex({point.position.x(), point.position.y()});
            cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
            cairo_fill(cr);
          }

          for (const auto& pair : icp_result.pairs)
          {
            cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
            cairo_set_line_width(cr, 1.0);
            const auto src = grid->limits().GetCellIndex(pair.first.cast<float>());
            const auto dst = grid->limits().GetCellIndex(pair.second.cast<float>());
            cairo_move_to(cr, src.x(), src.y());
            cairo_line_to(cr, dst.x(),  dst.y());
            cairo_stroke(cr);
          }

          const auto proposal_tpc = sensor::TransformPointCloud(constant_data.filtered_point_cloud, transform::Embed3D(icp_pose_proposal.cast<float>()));
          for (const auto& point : proposal_tpc)
          {
            cairo_set_source_rgba(cr, 1, 0, 0, 1);
            const auto mp = grid->limits().GetCellIndex({point.position.x(), point.position.y()});
            cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
            cairo_fill(cr);
          }
        }

        for (const auto& f : constant_data.circle_features)
        {
          cairo_set_source_rgba(cr, 1, 0.8, 0.2, 0.5);
          cairo_set_line_width(cr, 1.0);
          const auto tr = pose_estimate.cast<float>() * Eigen::Vector2f(f.keypoint.position.x(), f.keypoint.position.y());
          const auto mp = grid->limits().GetCellIndex(tr);
          cairo_arc(cr, mp.x(), mp.y(), 16.0, 0, 2 * M_PI);
          cairo_stroke(cr);
          cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
          cairo_fill(cr);
        }

        for (const auto& f : submap.CircleFeatures())
        {
          cairo_set_source_rgba(cr, 0.2, 0.2, 0.2, 0.5);
          cairo_set_line_width(cr, 1.0);
          const auto mp = grid->limits().GetCellIndex({f.keypoint.position.x(), f.keypoint.position.y()});
          cairo_arc(cr, mp.x(), mp.y(), 20.0, 0, 2 * M_PI);
          cairo_stroke(cr);
          cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
          cairo_fill(cr);
        }

        cairo_destroy(cr);

        const std::string fname = "cartographer_debug/match_t" + std::to_string(submap_id.trajectory_id) + "_n" + std::to_string(node_id.node_index) + "_s" + std::to_string(submap_id.submap_index) + ".png";
        cairo_surface_write_to_png(surface.get(), fname.c_str());
    }

    if (success && ((match_full_submap && score > options_.min_global_search_score()) || (!match_full_submap && score > options_.min_local_search_score())))
    {
      CHECK_GE(node_id.trajectory_id, 0);
      CHECK_GE(submap_id.trajectory_id, 0);

      if (match_full_submap) {
        kGlobalConstraintsFoundMetric->Increment();
        kGlobalConstraintScoresMetric->Observe(score);
      } else {
        kConstraintsFoundMetric->Increment();
        kConstraintScoresMetric->Observe(score);
      }

      LOG(INFO) << "SUCCESS matching node_id: " << node_id
                << " submap_id: " << submap_id << " score: " << score
                << " took: "
                << std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    } else {
      LOG(INFO) << "FAILED matching node_id: " << node_id
                << " submap_id: " << submap_id << " score: " << score
                << " took: "
                << std::chrono::duration_cast<std::chrono::duration<double>>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
      return;
    }

  {
    absl::MutexLock locker(&mutex_);
    score_histogram_.Add(score);
  }

  // Use the CSM estimate as both the initial and previous pose. This has the
  // effect that, in the absence of better information, we prefer the original
  // CSM estimate.
  ceres::Solver::Summary unused_summary;
  ceres_scan_matcher_.Match(pose_estimate.translation(), pose_estimate,
                            constant_data.filtered_point_cloud,
                            *submap_scan_matcher->submap.grid(), &pose_estimate,
                            &unused_summary);

  const transform::Rigid2d constraint_transform = ComputeSubmapPose(submap).inverse() * pose_estimate;
  constraint->reset(new Constraint{submap_id,
                                   node_id,
                                   {transform::Embed3D(constraint_transform),
                                    score * options_.constraint_translation_weight(),
                                    score * options_.constraint_rotation_weight()},
                                   Constraint::INTER_SUBMAP});

  if (options_.log_matches()) {
    std::ostringstream info;
    info << "Node " << node_id << " with "
         << constant_data.filtered_point_cloud.size()
         << " points on submap " << submap_id << std::fixed;
    if (match_full_submap) {
      info << " matches";
    } else {
      const transform::Rigid2d difference =
          initial_pose.inverse() * pose_estimate;
      info << " differs by translation " << std::setprecision(2)
           << difference.translation().norm() << " rotation "
           << std::setprecision(3) << std::abs(difference.normalized_angle());
    }
    info << " with score " << std::setprecision(1) << 100. * score << "%.";
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
