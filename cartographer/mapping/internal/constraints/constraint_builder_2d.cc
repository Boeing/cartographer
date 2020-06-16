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
#include "cartographer/transform/transform.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace constraints {

using Constraint = PoseGraphInterface::Constraint;

namespace {

double calculateICPscore(const double cost) {
  return std::max(0.01, std::min(1., 1. - cost));
}

transform::Rigid2d ComputeSubmapPose(const Submap2D& submap) {
  return transform::Project2D(submap.local_pose());
}

}  // namespace

ConstraintBuilder2D::ConstraintBuilder2D(
    const constraints::proto::ConstraintBuilderOptions& options)
    : options_(options),
      ceres_scan_matcher_(options.ceres_scan_matcher_options()) {}

ConstraintBuilder2D::~ConstraintBuilder2D() {}

absl::optional<Constraint> ConstraintBuilder2D::LocalSearchForConstraint(
    const NodeId node_id, const SubmapId submap_id,
    const transform::Rigid2d& initial_relative_pose, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data) {
  absl::MutexLock locker(&mutex_);
  const auto* scan_matcher = GetScanMatcher(submap_id, submap);
  auto optional_constraint =
      ComputeConstraint(node_id, submap_id, initial_relative_pose, submap,
                        constant_data, false, *scan_matcher);
  if (optional_constraint)
    return absl::optional<Constraint>(optional_constraint.value().constraint);
  return {};
}

absl::optional<Constraint> ConstraintBuilder2D::GlobalSearchForConstraint(
    const NodeId node_id, const SubmapId submap_id, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data) {
  absl::MutexLock locker(&mutex_);
  const auto* scan_matcher = GetScanMatcher(submap_id, submap);
  auto constraint_task = absl::make_unique<common::Task>();
  auto optional_constraint =
      ComputeConstraint(node_id, submap_id, transform::Rigid2d::Identity(),
                        submap, constant_data, true, *scan_matcher);
  if (optional_constraint)
    return absl::optional<Constraint>(optional_constraint.value().constraint);
  return {};
}

absl::optional<Constraint> ConstraintBuilder2D::GlobalSearchForConstraint(
    const NodeId node_id, const MapById<SubmapId, const Submap2D*>& submaps,
    const TrajectoryNode::Data& constant_data) {
  absl::MutexLock locker(&mutex_);
  std::unique_ptr<FoundConstraint> best_constraint;
  for (const auto item : submaps) {
    const auto* scan_matcher = GetScanMatcher(item.id, *item.data);
    auto optional_constraint = ComputeConstraint(
        node_id, scan_matcher->submap_id, transform::Rigid2d::Identity(),
        scan_matcher->submap, constant_data, true, *scan_matcher);
    if (optional_constraint) {
      const FoundConstraint result = optional_constraint.value();
      if (!best_constraint || result.score > best_constraint->score)
        best_constraint.reset(new FoundConstraint(result));
    }
  }
  if (best_constraint)
    return absl::optional<Constraint>(best_constraint->constraint);
  return {};
}

const ConstraintBuilder2D::SubmapScanMatcher*
ConstraintBuilder2D::GetScanMatcher(const SubmapId& submap_id,
                                    const Submap2D& submap) {
  if (submap_scan_matchers_.count(submap_id) != 0) {
    return &submap_scan_matchers_.at(submap_id);
  }
  auto ret = submap_scan_matchers_.emplace(std::make_pair(
      submap_id,
      SubmapScanMatcher{
          submap_id, submap,
          absl::make_unique<scan_matching::GlobalICPScanMatcher2D>(
              submap, options_.global_icp_scan_matcher_options_2d())}));
  return &ret.first->second;
}

absl::optional<ConstraintBuilder2D::FoundConstraint>
ConstraintBuilder2D::ComputeConstraint(
    const NodeId node_id, const SubmapId submap_id,
    const transform::Rigid2d initial_relative_pose, const Submap2D& submap,
    const TrajectoryNode::Data& constant_data, const bool match_full_submap,
    const SubmapScanMatcher& submap_scan_matcher) {
  CHECK(submap_scan_matcher.global_icp_scan_matcher);

  const auto t0 = std::chrono::steady_clock::now();

  const transform::Rigid2d initial_pose =
      ComputeSubmapPose(submap) * initial_relative_pose;

  // The 'constraint_transform' (submap i <- node j) is computed from:
  // - a 'filtered_gravity_aligned_point_cloud' in node j,
  // - the initial guess 'initial_pose' for (map <- node j),
  // - the result 'pose_estimate' of Match() (map <- node j).
  // - the ComputeSubmapPose() (map <- submap i)

  cartographer::mapping::scan_matching::GlobalICPScanMatcher2D::Result
      match_result;

  if (match_full_submap) {
    match_result = submap_scan_matcher.global_icp_scan_matcher->Match(
        constant_data.filtered_point_cloud, constant_data.circle_features);
  } else {
    match_result = submap_scan_matcher.global_icp_scan_matcher->Match(
        initial_pose, constant_data.filtered_point_cloud,
        constant_data.circle_features);
  }

  const auto clusters =
      submap_scan_matcher.global_icp_scan_matcher->DBScanCluster(
          match_result.poses, constant_data.filtered_point_cloud,
          constant_data.circle_features);

  LOG(INFO) << "Found " << match_result.poses.size() << " good proposals"
            << " (took: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(
                   std::chrono::steady_clock::now() - t0)
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
          submap_scan_matcher.global_icp_scan_matcher->IcpSolver().Match(
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
        submap_scan_matcher.global_icp_scan_matcher->IcpSolver().EvalutateMatch(
            icp_match, constant_data.range_data);

    const bool icp_good = icp_score >= options_.min_icp_score();
    const bool icp_points_inlier_good =
        icp_match.points_inlier_fraction >=
        options_.min_icp_points_inlier_fraction();
    const bool icp_features_inlier_good =
        icp_match.features_inlier_fraction >=
        options_.min_icp_features_inlier_fraction();

    // More features mean we need less actual points to match
    const double require_hit = std::max(
        0.15, options_.min_hit_fraction() - 0.01 * icp_match.features_count -
                  0.4 * std::max(0.0, statistics.ray_trace_fraction - 0.85));

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
              << " hit: " << statistics.hit_fraction << " > " << require_hit
              << " (" << hit_good << ")";

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

  //
  // Debug visualisation
  //
  {
    const std::string prefix = "cartographer_debug/match_t" +
                               std::to_string(submap_id.trajectory_id) + "_n" +
                               std::to_string(node_id.node_index) + "_s" +
                               std::to_string(submap_id.submap_index) + "_";

    const auto grid = dynamic_cast<const mapping::ProbabilityGrid*>(
        submap_scan_matcher.submap.grid());

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
         submap_scan_matcher.global_icp_scan_matcher->IcpSolver()
             .kdtree()
             .cells.cells) {
      cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 1);
      cairo_set_line_width(cr, 1.0);
      const auto mp = grid->limits().GetCellIndex({cell.x, cell.y});
      cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      cairo_fill(cr);
    }

    const std::string fname = prefix + ".png";
    cairo_surface_write_to_png(surface.get(), fname.c_str());

    // draw the proposals
    for (const auto& cluster : clusters) {
      std::uniform_real_distribution<double> dist(0, 1);
      std::random_device rd;
      std::mt19937 gen(rd());

      const double c_b = dist(gen);
      const double c_g = dist(gen);

      for (const auto& pose : cluster.poses) {
        const auto _mp = grid->limits().GetCellIndex({pose.x, pose.y});
        cairo_set_source_rgba(cr, 0.5, c_g, c_b, 1.0);
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

    const std::string proposals_fname = prefix + "proposals.png";
    cairo_surface_write_to_png(surface.get(), proposals_fname.c_str());

    // draw the cluster origins
    for (const auto& cluster : clusters) {
      cairo_new_path(cr);
      cairo_set_source_rgba(cr, 1, 0, 0, 1);
      cairo_set_line_width(cr, 1.0);
      const auto mp = grid->limits().GetCellIndex({cluster.x, cluster.y});
      cairo_arc(cr, mp.x(), mp.y(), 5.0, 0, 2 * M_PI);
      auto dir = Eigen::Rotation2Df(cluster.poses.front().rotation) *
                 Eigen::Vector2f(7.0, 0);
      cairo_stroke(cr);
      cairo_new_path(cr);
      cairo_move_to(cr, mp.x(), mp.y());
      cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
      cairo_stroke(cr);
    }

    const std::string proposals_clusters_fname =
        prefix + "proposals_clusters.png";
    cairo_surface_write_to_png(surface.get(), proposals_clusters_fname.c_str());

    for (std::size_t i = 0; i < cluster_results.size(); ++i) {
      const auto& cluster = clusters[i];
      const auto& match = cluster_results[i];

      cairo_set_line_width(cr, 1.0);

      if (match.success)
        cairo_set_source_rgba(cr, 0.5, 1.0, 0, 1.0);
      else
        cairo_set_source_rgba(cr, 1.0, 0.0, 0, 1.0);

      cairo_new_path(cr);
      const auto mp =
          grid->limits().GetCellIndex({match.pose_estimate.translation().x(),
                                       match.pose_estimate.translation().y()});
      cairo_arc(cr, mp.x(), mp.y(), 8.0, 0, 2 * M_PI);
      const auto dir = match.pose_estimate.rotation().cast<float>() *
                       Eigen::Vector2f(10.0, 0);
      cairo_stroke(cr);
      cairo_new_path(cr);
      cairo_move_to(cr, mp.x(), mp.y());
      cairo_line_to(cr, mp.x() - dir.y(), mp.y() - dir.x());
      cairo_stroke(cr);

      const auto cluster_mp =
          grid->limits().GetCellIndex({cluster.x, cluster.y});

      cairo_new_path(cr);
      cairo_set_line_width(cr, 1.0);
      cairo_set_source_rgba(cr, 0.5, 1.0, 0.5, 1.0);
      cairo_move_to(cr, cluster_mp.x(), cluster_mp.y());
      cairo_line_to(cr, mp.x(), mp.y());
      cairo_stroke(cr);
    }

    const std::string proposals_clusters_opt_fname =
        prefix + "proposals_clusters_opt.png";
    cairo_surface_write_to_png(surface.get(),
                               proposals_clusters_opt_fname.c_str());

    // Visualise the best match
    if (best_match && best_match->success) {
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
          cairo_set_source_rgba(cr, 1, 0, 0, 1);
        cairo_rectangle(cr, mp.x(), mp.y(), 2, 2);
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

      //      const auto proposal_tpc = sensor::TransformPointCloud(
      //          constant_data.range_data.returns,
      //          transform::Embed3D(best_match->icp_pose_proposal.cast<float>()));
      //      for (const auto& point : proposal_tpc) {
      //        cairo_set_source_rgba(cr, 1, 0, 0, 1);
      //        const auto mp = grid->limits().GetCellIndex(
      //            {point.position.x(), point.position.y()});
      //        cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      //        cairo_fill(cr);
      //      }

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

    const std::string best_fname = prefix + "proposals_clusters_opt_best.png";
    cairo_surface_write_to_png(surface.get(), best_fname.c_str());
  }

  if (best_match && best_match->success) {
    CHECK_GE(node_id.trajectory_id, 0);
    CHECK_GE(submap_id.trajectory_id, 0);

    LOG(INFO) << "SUCCESS matching node_id: " << node_id
              << " submap_id: " << submap_id << " score: " << best_match->score
              << " took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::steady_clock::now() - t0)
                     .count();

    // Use the CSM estimate as both the initial and previous pose. This has the
    // effect that, in the absence of better information, we prefer the original
    // CSM estimate.
    ceres::Solver::Summary unused_summary;
    ceres_scan_matcher_.Match(
        best_match->pose_estimate.translation(), best_match->pose_estimate,
        constant_data.filtered_point_cloud, *submap_scan_matcher.submap.grid(),
        &best_match->pose_estimate, &unused_summary);

    const transform::Rigid2d constraint_transform =
        ComputeSubmapPose(submap).inverse() * best_match->pose_estimate;

    Constraint constraint = {
        submap_id,
        node_id,
        {transform::Embed3D(constraint_transform),
         best_match->score * options_.constraint_translation_weight(),
         best_match->score * options_.constraint_rotation_weight()},
        Constraint::INTER_SUBMAP};

    return absl::optional<FoundConstraint>({best_match->score, constraint});

  } else {
    LOG(INFO) << "FAILED matching node_id: " << node_id
              << " submap_id: " << submap_id << " took: "
              << std::chrono::duration_cast<std::chrono::duration<double>>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return {};
  }
}

void ConstraintBuilder2D::DeleteScanMatcher(const SubmapId& submap_id) {
  absl::MutexLock locker(&mutex_);
  submap_scan_matchers_.erase(submap_id);
}

void ConstraintBuilder2D::RegisterMetrics(metrics::FamilyFactory* factory) {}

}  // namespace constraints
}  // namespace mapping
}  // namespace cartographer
