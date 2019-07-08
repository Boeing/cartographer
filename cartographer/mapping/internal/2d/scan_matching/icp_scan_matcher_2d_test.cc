#include "cartographer/mapping/internal/2d/scan_matching/icp_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

#include <cartographer/io/submap_painter.h>
#include "cartographer/common/lua_parameter_dictionary_test_helpers.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/transform/rigid_transform_test_helpers.h"
#include "cartographer/transform/transform.h"
#include "gtest/gtest.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

TEST(ICPScanMatcherTest, FullSubmapMatching) {
  std::mt19937 prng(42);
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);

  const float resolution = 0.02;
  const float side_length = 60.0;
  const int cells = side_length / resolution;

  ValueConversionTables conversion_tables;

  ProbabilityGrid probability_grid(
      MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                CellLimits(cells, cells)),
      &conversion_tables);

  // insert some random shapes
  std::uniform_real_distribution<float> box_prob(0.7f, 1.0f);
  std::uniform_int_distribution<int> box_dist;
  using param_t = std::uniform_int_distribution<>::param_type;
  for (int ii = 0; ii < 200; ++ii) {
    const int box_size = box_dist(prng, param_t(50, 200));
    const int box_x = box_dist(prng, param_t(0, cells));
    const int box_y = box_dist(prng, param_t(0, cells));

    const float prob = 1.0;  // box_prob(prng);

    for (int jj = box_y; jj < box_y + box_size; ++jj) {
      if (jj >= cells) break;

      for (int ii = box_x; ii < box_x + box_size; ++ii) {
        if (ii >= cells) break;

        if (jj > box_y + 2 && jj < box_y + box_size - 3) {
          if (ii > box_x + 2 && ii < box_x + box_size - 3) {
            continue;
          }
        }

        probability_grid.SetProbability({ii, jj}, prob);
      }
    }
    probability_grid.FinishUpdate();
  }

  // generate a scan
  transform::Rigid2f inserted_pose({side_length / 2.0f, side_length / 2.0f},
                                   0.234f);

  sensor::PointCloud unperturbed_point_cloud;
  inserted_pose =
      transform::Rigid2f({side_length / 2.0f + distribution(prng) * 1.0,
                          side_length / 2.0f + distribution(prng) * 1.0},
                         distribution(prng));
  unperturbed_point_cloud.clear();

  const int scan_points = 400;
  for (int i = 0; i < scan_points; ++i) {
    const float angle = i * 2.f * M_PI / scan_points;
    const float x = std::cos(angle);
    const float y = std::sin(angle);
    const Eigen::Vector2f dir(x, y);
    const Eigen::Vector2f end = inserted_pose.translation() + dir * 10.;

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
  icp_config.set_nn_huber_loss(0.05);
  icp_config.set_pp_huber_loss(0.01);

  ICPScanMatcher2D icp_scan_matcher(probability_grid, icp_config);

  transform::Rigid2d pose_estimate({inserted_pose.translation().x() + 0.2,
                                    inserted_pose.translation().y() + 0.2},
                                   inserted_pose.rotation().angle() + 0.1);

  auto surface = probability_grid.DrawSurface();

  auto result = icp_scan_matcher.Match(pose_estimate, unperturbed_point_cloud);

  cairo_t* cr = cairo_create(surface.get());

  auto guess_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud, transform::Embed3D(pose_estimate.cast<float>()));
  for (const auto& point : guess_tpc) {
    cairo_set_source_rgba(cr, 0, 0, 1, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_stroke(cr);
  }

  auto actual_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud, transform::Embed3D(inserted_pose));
  for (const auto& point : actual_tpc) {
    cairo_set_source_rgba(cr, 1, 1, 1, 1);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_stroke(cr);
  }

  auto icp_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud,
      transform::Embed3D(result.pose_estimate.cast<float>()));
  for (const auto& point : icp_tpc) {
    cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_stroke(cr);
  }

  for (const auto& pair : result.pairs) {
    cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto src = probability_grid.limits().GetCellIndex(
        {icp_tpc[pair.first].position.x(), icp_tpc[pair.first].position.y()});
    const auto dst =
        probability_grid.limits().GetCellIndex(pair.second.cast<float>());
    cairo_move_to(cr, src.x(), src.y());
    cairo_line_to(cr, dst.x(), dst.y());
    cairo_stroke(cr);
  }

  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test.png");

  LOG(INFO) << result.summary.FullReport();

  EXPECT_LT(result.summary.final_cost, 0.05);

  EXPECT_THAT(inserted_pose,
              transform::IsNearly(pose_estimate.cast<float>(), 0.03f))
      << "Actual: " << transform::ToProto(pose_estimate).DebugString()
      << "\nExpected: " << transform::ToProto(inserted_pose).DebugString();
}
}  // namespace

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
