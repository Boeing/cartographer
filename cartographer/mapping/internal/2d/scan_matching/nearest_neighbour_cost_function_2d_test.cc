#include "cartographer/mapping/internal/2d/scan_matching/nearest_neighbour_cost_function_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"

#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/probability_values.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace {

using ::testing::DoubleEq;
using ::testing::ElementsAre;

TEST(NearestNeighbourCostFunction2DTest, SmokeTest) {
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
  std::uniform_int_distribution<int> box_dist;
  using param_t = std::uniform_int_distribution<>::param_type;
  for (int ii = 0; ii < 100; ++ii) {
    const int box_size = box_dist(prng, param_t(50, 200));
    const int box_x = box_dist(prng, param_t(0, cells));
    const int box_y = box_dist(prng, param_t(0, cells));

    const float prob = 1.0;

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

  auto tpc = sensor::TransformPointCloud(unperturbed_point_cloud,
                                         transform::Embed3D(inserted_pose));

  auto surface = probability_grid.DrawSurface();

  cairo_t* cr = cairo_create(surface.get());
  for (const auto& point : tpc) {
    cairo_set_source_rgba(cr, 0, 1, 0, 1);
    cairo_set_line_width(cr, 1.0);
    const auto mp = probability_grid.limits().GetCellIndex(
        {point.position.x(), point.position.y()});
    cairo_arc(cr, mp.x(), mp.y(), 1.0, 0, 2 * M_PI);
    cairo_fill(cr);
    cairo_stroke(cr);
  }
  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test.png");

  auto kdtree = CreateRealIndexForGrid(probability_grid);

  std::unique_ptr<ceres::CostFunction> cost_function(
      CreateNearestNeighbourCostFunction2D(1.f, tpc, probability_grid.limits(),
                                           *kdtree.kdtree));

  const std::array<double, 3> pose_estimate{{0., 0., 0.}};
  const std::array<const double*, 1> parameter_blocks{{pose_estimate.data()}};

  std::vector<double> residuals(unperturbed_point_cloud.size());
  std::vector<std::array<double, 3>> jacobians(unperturbed_point_cloud.size());
  std::vector<double*> jacobians_ptrs(unperturbed_point_cloud.size());
  for (int i = 0; i < 1; ++i) jacobians_ptrs[i] = jacobians[i].data();

  cost_function->Evaluate(parameter_blocks.data(), residuals.data(),
                          jacobians_ptrs.data());

  for (std::size_t i = 0; i < unperturbed_point_cloud.size(); ++i)
    std::cout << residuals[i] << std::endl;

  //  EXPECT_THAT(residuals, ElementsAre(DoubleEq(kMaxProbability)));
}

}  // namespace
}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
