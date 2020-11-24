#include "cartographer/mapping/internal/2d/scan_matching/icp_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"

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
  const float side_length = 40.0;
  const int cells = side_length / resolution;

  ValueConversionTables conversion_tables;
  auto grid = absl::make_unique<ProbabilityGrid>(
      MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                CellLimits(cells, cells)),
      &conversion_tables);
  ProbabilityGrid& probability_grid = *grid;

  for (int ii = 0; ii < cells; ++ii)
    for (int jj = 0; jj < cells; ++jj)
      probability_grid.SetProbability({ii, jj}, 0.5);

  // insert some random box shapes
  std::uniform_int_distribution<int> box_dist;
  using param_t = std::uniform_int_distribution<>::param_type;
  for (int i = 0; i < 20; ++i) {
    const int box_size = box_dist(prng, param_t(50, 200));
    const int box_x = box_dist(prng, param_t(0, cells));
    const int box_y = box_dist(prng, param_t(0, cells));

    const float prob = 1.0;

    for (int jj = box_y; jj < box_y + box_size; ++jj) {
      if (jj >= cells) break;

      for (int ii = box_x; ii < box_x + box_size; ++ii) {
        if (ii >= cells) break;

        if (jj > box_y && jj < box_y + box_size - 1) {
          if (ii > box_x && ii < box_x + box_size - 1) {
            continue;
          }
        }

        probability_grid.SetProbability({ii, jj}, prob);
      }
    }
    probability_grid.FinishUpdate();
  }

  // insert some random retro reflective poles
  const float pole_radius = 0.07f;
  std::vector<std::pair<int, int>> pole_centers;
  std::set<std::pair<int, int>> reflective_cells;
  std::uniform_int_distribution<int> pole_dist;
  using param_t = std::uniform_int_distribution<>::param_type;
  for (int i = 0; i < 60; ++i) {
    const int pole_size = static_cast<int>(pole_radius / resolution);
    const int pole_x = pole_dist(prng, param_t(0, cells));
    const int pole_y = pole_dist(prng, param_t(0, cells));

    const float prob = 1.0;
    const double r2 =
        static_cast<double>(pole_size) * static_cast<double>(pole_size);

    pole_centers.push_back({pole_x, pole_y});

    for (int jj = pole_y - pole_size; jj <= pole_y + pole_size; ++jj) {
      if (jj >= cells || jj < 0) break;

      for (int ii = pole_x - pole_size; ii <= pole_x + pole_size; ++ii) {
        if (ii >= cells || ii < 0) break;

        const double w = static_cast<double>(ii - pole_x);
        const double h = static_cast<double>(jj - pole_y);
        const double _r2 = w * w + h * h;
        if (_r2 <= r2) {
          probability_grid.SetProbability({ii, jj}, prob);
          reflective_cells.insert({ii, jj});
        }
      }
    }
    probability_grid.FinishUpdate();
  }

  // generate a scan
  transform::Rigid2f inserted_pose({side_length / 2.0f, side_length / 2.0f},
                                   0.234f);

  sensor::PointCloud unperturbed_point_cloud;
  inserted_pose =
      transform::Rigid2f({side_length / 2.0f + distribution(prng) * 1.f,
                          side_length / 2.0f + distribution(prng) * 1.f},
                         distribution(prng));
  unperturbed_point_cloud.clear();

  const int scan_points = 1000;
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
      float intensity = 0;
      if (reflective_cells.find({p.x, p.y}) != reflective_cells.end())
        intensity = 9000;
      unperturbed_point_cloud.push_back({r, intensity});
    }
  }

  // find circle features in the scan
  const auto poles =
      DetectReflectivePoles(unperturbed_point_cloud, pole_radius);
  std::vector<CircleFeature> scan_circle_features;
  for (const auto& c : poles) {
    const auto p = FitCircle(c);
    scan_circle_features.push_back(
        CircleFeature{Keypoint{{p.position.x(), p.position.y(), 0.0f}},
                      CircleDescriptor{p.mse, p.radius}});
  }

  std::vector<CircleFeature> map_circle_features;
  for (const auto& c : pole_centers) {
    const auto real_p =
        probability_grid.limits().GetCellCenter({c.first, c.second});
    map_circle_features.push_back(
        CircleFeature{Keypoint{{real_p.x(), real_p.y(), 0.0f}},
                      CircleDescriptor{0, pole_radius}});
  }

  proto::ICPScanMatcherOptions2D icp_config;
  icp_config.set_nearest_neighbour_point_huber_loss(0.01);
  icp_config.set_nearest_neighbour_feature_huber_loss(0.01);
  icp_config.set_point_pair_point_huber_loss(0.01);
  icp_config.set_point_pair_feature_huber_loss(0.01);
  icp_config.set_point_weight(1.0);
  icp_config.set_feature_weight(2.0);
  icp_config.set_point_inlier_threshold(1.0);
  icp_config.set_feature_inlier_threshold(1.0);
  icp_config.set_raytrace_threshold(0.3);
  icp_config.set_hit_threshold(0.3);
  icp_config.set_feature_match_threshold(0.2);

  Eigen::Vector2f origin = {0.f, 0.f};

  cartographer::mapping::proto::SubmapsOptions2D options;
  options.set_min_feature_observations(2);
  options.set_max_feature_score(0.5);

  Submap2D submap(origin, std::move(grid), &conversion_tables, options);
  submap.SetCircleFeatures(map_circle_features);

  ICPScanMatcher2D icp_scan_matcher(submap, icp_config);

  transform::Rigid2d pose_estimate({inserted_pose.translation().x() + 0.2,
                                    inserted_pose.translation().y() + 0.2},
                                   inserted_pose.rotation().angle() + 0.1);

  auto surface = probability_grid.DrawSurface();

  auto result = icp_scan_matcher.Match(pose_estimate, unperturbed_point_cloud,
                                       scan_circle_features);

  cairo_t* cr = cairo_create(surface.get());

  auto guess_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud, transform::Embed3D(pose_estimate.cast<float>()));
  for (const auto& point : guess_tpc) {
    cairo_set_source_rgba(cr, 0, 0, 1, 1.0);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
    cairo_fill(cr);
  }

  auto icp_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud,
      transform::Embed3D(result.pose_estimate.cast<float>()));
  for (const auto& point : icp_tpc) {
    cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
    cairo_fill(cr);
  }

  for (const auto& pair : result.pairs) {
    cairo_set_source_rgba(cr, 0, 1, 0, 0.5);
    cairo_set_line_width(cr, 1.0);
    const auto src =
        probability_grid.limits().GetCellIndex(pair.first.cast<float>());
    const auto dst =
        probability_grid.limits().GetCellIndex(pair.second.cast<float>());
    cairo_move_to(cr, src.x(), src.y());
    cairo_line_to(cr, dst.x(), dst.y());
    cairo_stroke(cr);
  }

  for (const auto& c : poles) {
    cairo_set_source_rgba(cr, 1, 1, 0, 1.0);
    cairo_set_line_width(cr, 1.0);
    const auto tr = inserted_pose * c.position;
    const auto mp = probability_grid.limits().GetCellIndex(tr);
    cairo_arc(cr, mp.x(), mp.y(), c.radius / resolution, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  for (const auto& c : submap.CircleFeatures()) {
    cairo_set_source_rgba(cr, 1, 0.2, 0.2, 1.0);
    cairo_set_line_width(cr, 1.0);
    const auto mp =
        probability_grid.limits().GetCellIndex(c.keypoint.position.head<2>());
    cairo_arc(cr, mp.x(), mp.y(), 2 * c.fdescriptor.radius / resolution, 0,
              2 * M_PI);
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
