#include "cartographer/mapping/internal/2d/scan_matching/global_icp_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

#include <cartographer/io/submap_painter.h>
#include "cartographer/common/lua_parameter_dictionary_test_helpers.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/transform/rigid_transform_test_helpers.h"
#include "cartographer/transform/transform.h"
#include "gtest/gtest.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

TEST(GlobalICPScanMatcherTest, FullSubmapMatching) {
  std::mt19937 prng(42);
  std::uniform_real_distribution<float> distribution(-0.01f, 0.01f);

  const float resolution = 0.02;
  const float side_length = 40.0;
  const int cells = side_length / resolution;

  ValueConversionTables conversion_tables;

  ProbabilityGrid probability_grid(
      MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                CellLimits(cells, cells)),
      &conversion_tables);

  {
    const int box_size_x = cells * 0.8;
    const int box_size_y = cells * 0.6;

    int box_x = cells * 0.18;
    int box_y = cells * 0.08;

    const float prob = 1.0;

    for (int jj = box_y; jj < box_y + box_size_y; ++jj) {
      if (jj >= cells) break;

      for (int ii = box_x; ii < box_x + box_size_x; ++ii) {
        if (ii >= cells) break;

        if (jj > box_y + 2 && jj < box_y + box_size_y - 2) {
          if (ii > box_x + 2 && ii < box_x + box_size_x - 2) {
            probability_grid.SetProbability({ii, jj}, 0.2);
            continue;
          }
        }

        probability_grid.SetProbability({ii, jj}, prob);
      }
    }

    for (int ii = box_x; ii < box_x + box_size_x; ++ii) {
      if (ii % 100 == 0) {
        ii += 100;
        continue;
      }

      probability_grid.SetProbability({ii, box_y + box_size_y - 9}, prob);
      probability_grid.SetProbability({ii, box_y + box_size_y - 10}, prob);

      probability_grid.SetProbability({ii, box_y + 9}, prob);
      probability_grid.SetProbability({ii, box_y + 10}, prob);
    }

    probability_grid.FinishUpdate();
  }

  // generate a scan
  transform::Rigid2f inserted_pose({side_length / 2.0f, side_length / 2.0f},
                                   -0.1f);

  sensor::PointCloud unperturbed_point_cloud;

  const int scan_points = 200;
  for (int i = 0; i < scan_points; ++i) {
    const float angle = i * 2.f * M_PI / scan_points;
    const float x = std::cos(angle);
    const float y = std::sin(angle);
    const Eigen::Vector2f dir(x, y);
    const Eigen::Vector2f end = inserted_pose.translation() + dir * 30.;

    const auto map_start =
        probability_grid.limits().GetCellIndex(inserted_pose.translation());
    const auto map_end = probability_grid.limits().GetCellIndex(end);

    auto p = raytraceLine(probability_grid, map_start.x(), map_start.y(),
                          map_end.x(), map_end.y(), cells, 30.f / resolution);
    if (p.x > 0 && p.y > 0) {
      const auto real_p = probability_grid.limits().GetCellCenter({p.x, p.y});
      const auto diff = real_p - inserted_pose.translation();
      auto r = Eigen::AngleAxisf(-inserted_pose.rotation().angle(),
                                 Eigen::Vector3f::UnitZ()) *
               Eigen::Vector3f{diff.x(), diff.y(), 0.f};
      unperturbed_point_cloud.push_back({r});
    }
  }

  proto::ICPScanMatcherOptions2D icp_config;
  icp_config.set_nn_huber_loss(0.5);
  icp_config.set_pp_huber_loss(0.5);

  proto::GlobalICPScanMatcherOptions2D global_icp_config;
  global_icp_config.set_num_global_samples(1e3);
  global_icp_config.set_num_global_rotations(16);
  global_icp_config.set_proposal_max_score(1.5);
  global_icp_config.set_min_cluster_size(3);
  global_icp_config.set_min_cluster_distance(1.0);
  *global_icp_config.mutable_icp_options() = icp_config;

  GlobalICPScanMatcher2D global_icp_scan_matcher(probability_grid,
                                                 global_icp_config);

  const auto match_result =
      global_icp_scan_matcher.Match(unperturbed_point_cloud);

  const auto clusters =
      global_icp_scan_matcher.DBScanCluster(match_result.poses);

  double score = 0;
  transform::Rigid2d pose_estimate = transform::Rigid2d::Identity();
  ICPScanMatcher2D::Result icp_result;

  for (size_t i = 0; i < clusters.size(); ++i) {
    const transform::Rigid2d cluster_estimate({clusters[i].x, clusters[i].y},
                                              clusters[i].rotation);
    auto icp_match = global_icp_scan_matcher.IcpSolver().Match(
        cluster_estimate, unperturbed_point_cloud);

    const double icp_score =
        std::max(0.01, std::min(1., 1. - icp_match.summary.final_cost));

    LOG(INFO) << "ICP: " << cluster_estimate << " -> "
              << icp_match.pose_estimate
              << " cost: " << icp_match.summary.final_cost
              << " score: " << icp_score;

    LOG(INFO) << icp_match.summary.FullReport();

    if (icp_score > score) {
      score = icp_score;
      pose_estimate = icp_match.pose_estimate;
      icp_result = icp_match;
    }
  }

  auto surface = probability_grid.DrawSurface();
  cairo_t* cr = cairo_create(surface.get());

  auto actual_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud, transform::Embed3D(pose_estimate.cast<float>()));

  for (const auto& pair : icp_result.pairs) {
    cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto src = probability_grid.limits().GetCellIndex(
        {actual_tpc[pair.first].position.x(),
         actual_tpc[pair.first].position.y()});
    const auto dst =
        probability_grid.limits().GetCellIndex(pair.second.cast<float>());
    cairo_move_to(cr, src.x(), src.y());
    cairo_line_to(cr, dst.x(), dst.y());
    cairo_stroke(cr);
  }

  {
    cairo_set_source_rgba(cr, 0.5, 1.0, 0, 1);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {pose_estimate.translation().x(), pose_estimate.translation().y()});
    cairo_arc(cr, mp.x(), mp.y(), 30.0, 0, 2 * M_PI);
    auto dir =
        pose_estimate.rotation().cast<float>() * Eigen::Vector2f(10.0, 0.0);
    cairo_move_to(cr, mp.x(), mp.y());
    cairo_line_to(cr, mp.x() + dir.x(), mp.y() + dir.y());
    cairo_stroke(cr);

    for (const auto& point : actual_tpc) {
      cairo_set_source_rgba(cr, 0, 0, 1, 1);
      cairo_set_line_width(cr, 8.0);
      const auto mp = probability_grid.limits().GetCellIndex(
          {point.position.x(), point.position.y()});
      cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
      cairo_fill(cr);
      cairo_stroke(cr);
    }
  }

  for (std::size_t i = 0; i < match_result.poses.size(); ++i) {
    const auto& match = match_result.poses[i];

    const double max_score = 2e4;
    double intensity = std::max(0., max_score - match.score) / max_score;

    auto dir = Eigen::Rotation2Df(match.rotation) * Eigen::Vector2f(10.0, 0.0);

    cairo_set_source_rgba(cr, 0, intensity, 0, intensity);

    if (i == 0) cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);

    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex({match.x, match.y});
    cairo_move_to(cr, mp.x(), mp.y());
    cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
    cairo_line_to(cr, mp.x() + dir.x(), mp.y() + dir.y());
    cairo_stroke(cr);

    if (i == 0) {
      auto match_pose = transform::Rigid2f({match.x, match.y}, match.rotation);
      auto actual_tpc = sensor::TransformPointCloud(
          unperturbed_point_cloud, transform::Embed3D(match_pose));
      for (const auto& point : actual_tpc) {
        cairo_set_source_rgba(cr, 1, 1, 1, 1);
        cairo_set_line_width(cr, 8.0);
        const auto mp = probability_grid.limits().GetCellIndex(
            {point.position.x(), point.position.y()});
        cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
        cairo_fill(cr);
        cairo_stroke(cr);
      }
    }
  }

  for (const auto& cluster : clusters) {
    cairo_set_source_rgba(cr, 1, 0, 0, 1);
    cairo_set_line_width(cr, 2.0);
    const auto mp =
        probability_grid.limits().GetCellIndex({cluster.x, cluster.y});
    cairo_arc(cr, mp.x(), mp.y(), 20.0, 0, 2 * M_PI);
    auto dir = Eigen::Rotation2Df(cluster.poses.front().rotation) *
               Eigen::Vector2f(10.0, 0.0);
    cairo_move_to(cr, mp.x(), mp.y());
    cairo_line_to(cr, mp.x() + dir.x(), mp.y() + dir.y());
    cairo_stroke(cr);

    for (const auto& pose : cluster.poses) {
      cairo_set_source_rgba(cr, 1, 0, 0, 0.4);
      cairo_set_line_width(cr, 1.0);
      const auto mp = probability_grid.limits().GetCellIndex({pose.x, pose.y});
      cairo_arc(cr, mp.x(), mp.y(), 12.0, 0, 2 * M_PI);
      cairo_stroke(cr);
    }
  }

  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test.png");

  EXPECT_GT(score, 0.95);
  EXPECT_THAT(inserted_pose,
              transform::IsNearly(pose_estimate.cast<float>(), 0.03f))
      << "Actual: " << transform::ToProto(pose_estimate).DebugString()
      << "\nExpected: " << transform::ToProto(inserted_pose).DebugString();
}
}  // namespace

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer