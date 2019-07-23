#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"

#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"

#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/lua_parameter_dictionary_test_helpers.h"
#include "cartographer/common/math.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"

#include <vector>

#include "cartographer/common/time.h"
#include "cartographer/mapping/2d/submap_2d.h"

namespace cartographer {
namespace mapping {
namespace {

TEST(CirclDetector, real_data) {
  const std::vector<double> ranges = {0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.8199999928474426,
                                      0.8100000023841858,
                                      0.800000011920929,
                                      0.7799999713897705,
                                      0.7900000214576721,
                                      0.800000011920929,
                                      0.7799999713897705,
                                      0.7699999809265137,
                                      0.7799999713897705,
                                      0.7799999713897705,
                                      0.7799999713897705,
                                      0.7099999785423279,
                                      0.6299999952316284,
                                      0.6800000071525574,
                                      0.75,
                                      0.7400000095367432,
                                      0.7400000095367432,
                                      0.7300000190734863,
                                      0.7400000095367432,
                                      0.7400000095367432,
                                      0.7300000190734863,
                                      0.7400000095367432,
                                      0.7400000095367432,
                                      0.7300000190734863,
                                      0.7300000190734863,
                                      0.7300000190734863,
                                      0.7200000286102295,
                                      0.7200000286102295,
                                      0.7300000190734863,
                                      0.7300000190734863,
                                      0.7099999785423279,
                                      0.7099999785423279,
                                      0.699999988079071,
                                      0.7200000286102295,
                                      0.7099999785423279,
                                      0.7099999785423279,
                                      0.699999988079071,
                                      0.7099999785423279,
                                      0.6899999976158142,
                                      0.6499999761581421,
                                      0.6499999761581421,
                                      0.6399999856948853,
                                      0.6399999856948853,
                                      0.6299999952316284,
                                      0.6299999952316284,
                                      0.6200000047683716,
                                      0.6200000047683716,
                                      0.6499999761581421,
                                      0.6899999976158142,
                                      0.7200000286102295,
                                      0.7300000190734863,
                                      0.75,
                                      0.7799999713897705,
                                      0.7900000214576721,
                                      0.800000011920929,
                                      0.8299999833106995,
                                      0.8500000238418579,
                                      0.8700000047683716,
                                      0.9100000262260437,
                                      0.9300000071525574,
                                      0.9700000286102295,
                                      1.0199999809265137,
                                      1.059999942779541,
                                      1.100000023841858,
                                      1.149999976158142,
                                      1.190000057220459,
                                      1.2799999713897705,
                                      1.3700000047683716,
                                      1.4299999475479126,
                                      1.5299999713897705,
                                      1.7000000476837158,
                                      1.6699999570846558,
                                      1.75,
                                      2.2300000190734863,
                                      2.2899999618530273,
                                      2.740000009536743,
                                      2.6600000858306885,
                                      2.5199999809265137,
                                      2.5399999618530273,
                                      2.6600000858306885,
                                      2.890000104904175,
                                      2.9200000762939453,
                                      2.9000000953674316,
                                      2.9100000858306885,
                                      2.9100000858306885,
                                      2.890000104904175,
                                      2.880000114440918,
                                      2.890000104904175,
                                      2.890000104904175,
                                      2.9000000953674316,
                                      2.9000000953674316,
                                      2.9000000953674316,
                                      2.9200000762939453,
                                      2.890000104904175,
                                      29.979999542236328,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      26.0,
                                      29.959999084472656,
                                      17.280000686645508,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      13.270000457763672,
                                      12.579999923706055,
                                      12.470000267028809,
                                      29.959999084472656,
                                      29.959999084472656,
                                      11.569999694824219,
                                      11.380000114440918,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      11.25,
                                      13.270000457763672,
                                      29.959999084472656,
                                      29.959999084472656,
                                      11.479999542236328,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      29.959999084472656,
                                      12.569999694824219,
                                      12.649999618530273,
                                      12.609999656677246,
                                      6.519999980926514,
                                      29.959999084472656,
                                      6.329999923706055,
                                      6.309999942779541,
                                      11.649999618530273,
                                      11.479999542236328,
                                      7.059999942779541,
                                      7.119999885559082,
                                      10.949999809265137,
                                      29.959999084472656,
                                      29.959999084472656,
                                      5.449999809265137,
                                      5.380000114440918,
                                      5.449999809265137,
                                      5.440000057220459,
                                      5.409999847412109,
                                      5.380000114440918,
                                      5.260000228881836,
                                      5.369999885559082,
                                      5.340000152587891,
                                      5.179999828338623,
                                      5.179999828338623,
                                      5.119999885559082,
                                      5.070000171661377,
                                      4.820000171661377,
                                      4.78000020980835,
                                      4.75,
                                      4.690000057220459,
                                      4.639999866485596,
                                      4.610000133514404,
                                      4.579999923706055,
                                      4.550000190734863,
                                      4.480000019073486,
                                      4.420000076293945,
                                      4.369999885559082,
                                      4.210000038146973,
                                      4.170000076293945,
                                      4.050000190734863,
                                      4.090000152587891,
                                      4.21999979019165,
                                      4.190000057220459,
                                      4.150000095367432,
                                      4.070000171661377,
                                      4.03000020980835,
                                      3.9700000286102295,
                                      4.019999980926514,
                                      4.130000114440918,
                                      4.090000152587891,
                                      4.070000171661377,
                                      4.039999961853027,
                                      4.010000228881836,
                                      3.9700000286102295,
                                      3.9600000381469727,
                                      3.930000066757202,
                                      3.890000104904175,
                                      3.8499999046325684,
                                      3.259999990463257,
                                      3.2200000286102295,
                                      3.3399999141693115,
                                      3.9700000286102295,
                                      3.7200000286102295,
                                      3.640000104904175,
                                      3.619999885559082,
                                      3.5899999141693115,
                                      3.5999999046325684,
                                      3.5899999141693115,
                                      3.569999933242798,
                                      3.5399999618530273,
                                      3.5199999809265137,
                                      3.490000009536743,
                                      3.4800000190734863,
                                      3.4600000381469727,
                                      3.440000057220459,
                                      3.4000000953674316,
                                      3.3499999046325684,
                                      3.3399999141693115,
                                      3.319999933242798,
                                      3.319999933242798,
                                      3.319999933242798,
                                      3.309999942779541,
                                      3.2799999713897705,
                                      3.2699999809265137,
                                      3.259999990463257,
                                      3.240000009536743,
                                      3.2300000190734863,
                                      3.2100000381469727,
                                      3.2100000381469727,
                                      3.180000066757202,
                                      3.1600000858306885,
                                      3.140000104904175,
                                      3.130000114440918,
                                      3.119999885559082,
                                      3.0899999141693115,
                                      2.609999895095825,
                                      2.569999933242798,
                                      3.1500000953674316,
                                      3.25,
                                      3.3399999141693115,
                                      3.4200000762939453,
                                      3.0899999141693115,
                                      3.0,
                                      2.559999942779541,
                                      2.5199999809265137,
                                      2.2799999713897705,
                                      2.140000104904175,
                                      2.130000114440918,
                                      2.119999885559082,
                                      2.119999885559082,
                                      2.130000114440918,
                                      2.130000114440918,
                                      2.25,
                                      2.4800000190734863,
                                      2.4700000286102295,
                                      2.4600000381469727,
                                      2.4700000286102295,
                                      2.4700000286102295,
                                      2.4600000381469727,
                                      2.4600000381469727,
                                      2.450000047683716,
                                      2.4600000381469727,
                                      2.4600000381469727,
                                      4.460000038146973,
                                      4.460000038146973,
                                      4.460000038146973,
                                      2.369999885559082,
                                      2.390000104904175,
                                      2.430000066757202,
                                      2.880000114440918,
                                      3.259999990463257,
                                      3.359999895095825,
                                      3.2799999713897705,
                                      3.2799999713897705,
                                      3.2799999713897705,
                                      3.2899999618530273,
                                      3.319999933242798,
                                      3.359999895095825,
                                      3.5399999618530273,
                                      4.179999828338623,
                                      3.5799999237060547,
                                      3.380000114440918,
                                      3.0799999237060547,
                                      3.0899999141693115,
                                      3.25,
                                      4.380000114440918,
                                      4.320000171661377,
                                      4.400000095367432,
                                      3.9200000762939453,
                                      3.259999990463257,
                                      3.25,
                                      3.240000009536743,
                                      3.2699999809265137,
                                      3.299999952316284,
                                      12.59000015258789,
                                      12.600000381469727,
                                      12.630000114440918,
                                      12.619999885559082,
                                      12.680000305175781,
                                      12.680000305175781,
                                      12.699999809265137,
                                      12.729999542236328,
                                      12.75,
                                      12.779999732971191,
                                      12.8100004196167,
                                      12.829999923706055,
                                      12.869999885559082,
                                      12.880000114440918,
                                      12.90999984741211,
                                      12.9399995803833,
                                      12.960000038146973,
                                      13.0,
                                      13.029999732971191,
                                      12.6899995803833,
                                      12.359999656677246,
                                      9.90999984741211,
                                      9.6899995803833,
                                      9.640000343322754,
                                      13.270000457763672,
                                      9.319999694824219,
                                      9.149999618530273,
                                      9.770000457763672,
                                      9.680000305175781,
                                      9.479999542236328,
                                      9.109999656677246,
                                      8.970000267028809,
                                      8.680000305175781,
                                      8.510000228881836,
                                      8.350000381469727,
                                      8.199999809265137,
                                      8.0600004196167,
                                      7.909999847412109,
                                      7.909999847412109,
                                      7.449999809265137,
                                      7.5,
                                      7.659999847412109,
                                      7.53000020980835,
                                      7.420000076293945,
                                      7.309999942779541,
                                      7.210000038146973,
                                      7.090000152587891,
                                      7.0,
                                      6.900000095367432,
                                      6.820000171661377,
                                      6.71999979019165,
                                      6.639999866485596,
                                      6.550000190734863,
                                      6.46999979019165,
                                      6.389999866485596,
                                      6.309999942779541,
                                      6.239999771118164,
                                      6.159999847412109,
                                      6.090000152587891,
                                      6.03000020980835,
                                      5.949999809265137,
                                      5.889999866485596,
                                      5.829999923706055,
                                      5.760000228881836,
                                      5.699999809265137,
                                      5.690000057220459,
                                      5.730000019073486,
                                      5.78000020980835,
                                      5.829999923706055,
                                      5.869999885559082,
                                      5.920000076293945,
                                      5.96999979019165,
                                      4.929999828338623,
                                      4.840000152587891,
                                      4.949999809265137,
                                      5.070000171661377,
                                      5.28000020980835,
                                      6.28000020980835,
                                      6.340000152587891,
                                      6.400000095367432,
                                      6.46999979019165,
                                      6.53000020980835,
                                      6.590000152587891,
                                      6.650000095367432,
                                      6.730000019073486,
                                      4.360000133514404,
                                      4.340000152587891,
                                      4.380000114440918,
                                      5.860000133514404,
                                      4.389999866485596,
                                      4.110000133514404,
                                      4.150000095367432,
                                      4.139999866485596,
                                      4.110000133514404,
                                      4.079999923706055,
                                      4.099999904632568,
                                      4.059999942779541,
                                      4.070000171661377,
                                      4.050000190734863,
                                      3.9600000381469727,
                                      4.03000020980835,
                                      3.930000066757202,
                                      4.03000020980835,
                                      4.010000228881836,
                                      3.869999885559082,
                                      3.8399999141693115,
                                      3.859999895095825,
                                      3.740000009536743,
                                      3.7799999713897705,
                                      3.7899999618530273,
                                      3.7799999713897705,
                                      3.7699999809265137,
                                      8.050000190734863,
                                      8.020000457763672,
                                      7.980000019073486,
                                      4.210000038146973,
                                      4.199999809265137,
                                      4.199999809265137,
                                      4.179999828338623,
                                      4.130000114440918,
                                      3.9600000381469727,
                                      3.940000057220459,
                                      3.940000057220459,
                                      3.9200000762939453,
                                      3.9200000762939453,
                                      4.090000152587891,
                                      4.389999866485596,
                                      4.539999961853027,
                                      4.579999923706055,
                                      4.710000038146973,
                                      4.920000076293945,
                                      4.949999809265137,
                                      5.21999979019165,
                                      5.360000133514404,
                                      5.380000114440918,
                                      5.400000095367432,
                                      5.389999866485596,
                                      7.440000057220459,
                                      7.539999961853027,
                                      7.53000020980835,
                                      7.880000114440918,
                                      9.319999694824219,
                                      9.359999656677246,
                                      7.539999961853027,
                                      7.5,
                                      10.609999656677246,
                                      10.609999656677246,
                                      10.579999923706055,
                                      10.5600004196167,
                                      10.4399995803833,
                                      10.520000457763672,
                                      10.539999961853027,
                                      10.529999732971191,
                                      10.529999732971191,
                                      10.520000457763672,
                                      10.510000228881836,
                                      10.489999771118164,
                                      10.449999809265137,
                                      10.180000305175781,
                                      10.170000076293945,
                                      10.170000076293945,
                                      10.149999618530273,
                                      10.15999984741211,
                                      10.149999618530273,
                                      10.140000343322754,
                                      10.15999984741211,
                                      10.15999984741211,
                                      29.959999084472656,
                                      1.659999966621399,
                                      1.649999976158142,
                                      1.6299999952316284,
                                      1.6299999952316284,
                                      1.6299999952316284,
                                      1.6200000047683716,
                                      1.6299999952316284,
                                      1.6399999856948853,
                                      1.649999976158142,
                                      1.7000000476837158,
                                      6.670000076293945,
                                      6.570000171661377,
                                      6.489999771118164,
                                      6.480000019073486,
                                      4.559999942779541,
                                      29.959999084472656,
                                      2.049999952316284,
                                      2.0199999809265137,
                                      1.9600000381469727,
                                      1.940000057220459,
                                      1.8899999856948853,
                                      1.840000033378601,
                                      1.7300000190734863,
                                      1.3200000524520874,
                                      1.3200000524520874,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.3200000524520874,
                                      1.5099999904632568,
                                      1.6799999475479126,
                                      1.7000000476837158,
                                      1.7100000381469727,
                                      1.7100000381469727,
                                      1.7100000381469727,
                                      1.7300000190734863,
                                      1.7400000095367432,
                                      1.7400000095367432,
                                      1.75,
                                      1.690000057220459,
                                      1.5499999523162842,
                                      1.4800000190734863,
                                      1.3899999856948853,
                                      1.309999942779541,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2899999618530273,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.2999999523162842,
                                      1.309999942779541,
                                      1.309999942779541,
                                      1.2999999523162842,
                                      1.309999942779541,
                                      1.3300000429153442,
                                      1.3200000524520874,
                                      1.3300000429153442,
                                      1.3200000524520874,
                                      1.340000033378601,
                                      1.350000023841858,
                                      1.350000023841858,
                                      1.3600000143051147,
                                      1.3600000143051147,
                                      1.3600000143051147,
                                      1.3799999952316284,
                                      1.3899999856948853,
                                      1.399999976158142,
                                      1.5399999618530273,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0,
                                      0.0};

  sensor::PointCloud unperturbed_point_cloud;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const float angle = i * 0.00872664619237f;
    const float x = std::cos(angle);
    const float y = std::sin(angle);
    const Eigen::Vector2f end = Eigen::Vector2f{x, y} * ranges[i];
    unperturbed_point_cloud.push_back({{end.x(), end.y(), 0.f}, 9000.0f});
  }

  const float resolution = 0.01;
  const float side_length = 20.0;
  const int cells = side_length / resolution;

  ValueConversionTables conversion_tables;

  ProbabilityGrid probability_grid(
      MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                CellLimits(cells, cells)),
      &conversion_tables);

  transform::Rigid2f inserted_pose({side_length / 2.0f, side_length / 2.0f},
                                   0.0f);
  auto actual_tpc = sensor::TransformPointCloud(
      unperturbed_point_cloud, transform::Embed3D(inserted_pose.cast<float>()));

  const std::vector<Circle<float>> circles =
      DetectReflectivePoles(unperturbed_point_cloud, {0.060});

  auto surface = probability_grid.DrawSurface();
  cairo_t* cr = cairo_create(surface.get());

  cairo_set_source_rgba(cr, 1, 1, 1, 1);
  cairo_set_line_width(cr, 2.0);
  const auto mp =
      probability_grid.limits().GetCellIndex(inserted_pose.translation());
  cairo_arc(cr, mp.x(), mp.y(), 10.0, 0, 2 * M_PI);
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

  std::vector<Circle<float>> opt;
  for (const auto& c : circles) {
    opt.push_back(FitCircle(c));
  }

  for (const auto& c : circles) {
    cairo_set_source_rgba(cr, 1, 1, 0, 0.1);
    cairo_set_line_width(cr, 2.0);
    const auto tr = inserted_pose * c.position;
    const auto mp = probability_grid.limits().GetCellIndex(tr);
    cairo_arc(cr, mp.x(), mp.y(), c.radius / resolution, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  for (const auto& c : opt) {
    if (c.mse > 1e-4) continue;

    cairo_set_source_rgba(cr, 0.2, 1, 0.2, 1.0);
    cairo_set_line_width(cr, 2.0);
    const auto tr = inserted_pose * c.position;
    const auto mp = probability_grid.limits().GetCellIndex(tr);
    cairo_arc(cr, mp.x(), mp.y(), c.radius / resolution, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test.png");
}

TEST(CirclDetector, test_hough) {
  const float resolution = 0.01;
  const float side_length = 20.0;
  const int cells = side_length / resolution;

  ValueConversionTables conversion_tables;

  ProbabilityGrid probability_grid(
      MapLimits(resolution, Eigen::Vector2d(side_length, side_length),
                CellLimits(cells, cells)),
      &conversion_tables);

  for (int jj = 0; jj < cells; ++jj) {
    for (int ii = 0; ii < cells / 8.0; ++ii) {
      if (ii < jj) {
      } else if (ii == jj) {
        probability_grid.SetProbability({ii + 1, jj}, 1.0);
        probability_grid.SetProbability({ii, jj}, 1.0);
      } else
        probability_grid.SetProbability({ii, jj}, 0.1);
    }
  }
  probability_grid.FinishUpdate();

  // insert some random retro reflective poles
  std::mt19937 prng(42);
  std::uniform_int_distribution<int> pole_dist;
  using param_t = std::uniform_int_distribution<>::param_type;
  for (int ii = 0; ii < 60; ++ii) {
    const int pole_size = static_cast<int>(0.07f / resolution);
    const int pole_x = pole_dist(prng, param_t(0, cells));
    const int pole_y = pole_dist(prng, param_t(0, cells));

    const float prob = 1.0;
    const double r2 =
        static_cast<double>(pole_size) * static_cast<double>(pole_size);

    for (int jj = pole_y - pole_size; jj <= pole_y + pole_size; ++jj) {
      if (jj >= cells || jj < 0) break;

      for (int ii = pole_x - pole_size; ii <= pole_x + pole_size; ++ii) {
        if (ii >= cells || ii < 0) break;

        const double w = static_cast<double>(ii - pole_x);
        const double h = static_cast<double>(jj - pole_y);
        const double _r2 = w * w + h * h;
        if (std::abs(_r2 - r2) <= 1) {
          probability_grid.SetProbability({ii, jj}, prob);
        }
      }
    }
    probability_grid.FinishUpdate();
  }

  LOG(INFO) << "build grid";

  const auto circles = DetectCircles(probability_grid, 0.07f);

  auto surface = probability_grid.DrawSurface();
  cairo_t* cr = cairo_create(surface.get());

  const float radius = 0.07f;
  const int circle_radius_cells = static_cast<int>(0.07f / resolution + 0.5f);

  scan_matching::proto::CeresScanMatcherOptions2D options;
  options.set_occupied_space_weight(1.0);
  options.set_translation_weight(1.0);
  options.set_rotation_weight(1.0);
  options.mutable_ceres_solver_options()->set_max_num_iterations(20);
  options.mutable_ceres_solver_options()->set_num_threads(1);
  scan_matching::CeresScanMatcher2D scan_matcher(options);

  for (const auto& c : circles) {
    const auto real_p = probability_grid.limits().GetCellCenter({c.x(), c.y()});

    const transform::Rigid2d initial_pose_estimate({real_p.x(), real_p.y()},
                                                   0.0);

    sensor::PointCloud pc;
    const int num_scans = 16;
    for (int i = 0; i < num_scans; ++i) {
      const double angle = -M_PI + (static_cast<double>(i) * 2.0 * M_PI) /
                                       static_cast<double>(num_scans);
      const auto dir = Eigen::Rotation2Df(static_cast<float>(angle)) *
                       Eigen::Vector2f(radius, 0.f);
      pc.push_back(sensor::RangefinderPoint{{dir.x(), dir.y(), 0.0}, 0.0f});

      const auto tr = initial_pose_estimate.cast<float>() * dir;
      const auto mp = probability_grid.limits().GetCellIndex(tr);
      cairo_set_source_rgba(cr, 1, 1, 1, 0.5);
      cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      cairo_fill(cr);
    }

    transform::Rigid2d pose_estimate;
    ceres::Solver::Summary summary;

    scan_matcher.Match(initial_pose_estimate.translation(),
                       initial_pose_estimate, pc, probability_grid,
                       &pose_estimate, &summary);

    LOG(INFO) << initial_pose_estimate.translation().transpose() << " -> "
              << pose_estimate.translation().transpose();
    //        LOG(INFO) << summary.FullReport();

    const auto mp = probability_grid.limits().GetCellIndex(
        pose_estimate.translation().cast<float>());

    cairo_set_source_rgba(cr, 1, 1, 0, 0.1);
    cairo_set_line_width(cr, 1.0);
    cairo_arc(cr, mp.x(), mp.y(), circle_radius_cells, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test2.png");
}

TEST(CirclDetector, test_hough_from_file) {
  const float resolution = 0.02f;
  const int width = 1566;
  const int height = 917;

  auto input_surface = cairo_image_surface_create_from_png(
      "/home/boeing/ros/robotics_ws/cartographer_debug/raw_t0_n0_s0.png");

  ValueConversionTables conversion_tables;
  ProbabilityGrid probability_grid(
      MapLimits(resolution,
                Eigen::Vector2d(width * resolution, height * resolution),
                CellLimits(width, height)),
      &conversion_tables);

  uint32_t* pixel_data =
      reinterpret_cast<uint32_t*>(cairo_image_surface_get_data(input_surface));
  for (const Eigen::Array2i& xy_index :
       XYIndexRangeIterator(probability_grid.limits().cell_limits())) {
    const int i = probability_grid.ToFlatIndex(xy_index);

    const uint8_t o_intensity = (pixel_data[i] >> 16) & 0xFF;
    const uint8_t m_intensity = (pixel_data[i] >> 8) & 0xFF;

    float prob = 0.0;
    if (o_intensity > 0)
      prob = (static_cast<float>(o_intensity) / 255.f) * 0.5f + 0.5f;
    else
      prob = 0.5f - (static_cast<float>(m_intensity) / 255.f) * 0.5f;

    probability_grid.SetProbability(xy_index, prob);
  }
  probability_grid.FinishUpdate();

  const auto circles = DetectCircles(probability_grid, 0.07f);

  auto surface = probability_grid.DrawSurface();
  cairo_t* cr = cairo_create(surface.get());

  const float radius = 0.07f;
  const int circle_radius_cells = static_cast<int>(0.07f / resolution + 0.5f);

  scan_matching::proto::CeresScanMatcherOptions2D options;
  options.set_occupied_space_weight(1.0);
  options.set_translation_weight(1.0);
  options.set_rotation_weight(1.0);
  options.mutable_ceres_solver_options()->set_max_num_iterations(20);
  options.mutable_ceres_solver_options()->set_num_threads(1);
  scan_matching::CeresScanMatcher2D scan_matcher(options);

  for (const auto& c : circles) {
    const auto real_p = probability_grid.limits().GetCellCenter({c.x(), c.y()});

    const transform::Rigid2d initial_pose_estimate({real_p.x(), real_p.y()},
                                                   0.0);

    sensor::PointCloud pc;
    const int num_scans = 16;
    for (int i = 0; i < num_scans; ++i) {
      const double angle = -M_PI + (static_cast<double>(i) * 2.0 * M_PI) /
                                       static_cast<double>(num_scans);
      const auto dir = Eigen::Rotation2Df(static_cast<float>(angle)) *
                       Eigen::Vector2f(radius, 0.f);
      pc.push_back(sensor::RangefinderPoint{{dir.x(), dir.y(), 0.0}, 0.0f});

      //            const auto tr = initial_pose_estimate.cast<float>() * dir;
      //            const auto mp = probability_grid.limits().GetCellIndex(tr);
      //            cairo_set_source_rgba(cr, 1, 1, 1, 0.5);
      //            cairo_rectangle(cr, mp.x(), mp.y(), 1, 1);
      //            cairo_fill(cr);
    }

    transform::Rigid2d pose_estimate;
    ceres::Solver::Summary summary;

    scan_matcher.Match(initial_pose_estimate.translation(),
                       initial_pose_estimate, pc, probability_grid,
                       &pose_estimate, &summary);

    //        LOG(INFO) << initial_pose_estimate.translation().transpose() << "
    //        -> " << pose_estimate.translation().transpose(); LOG(INFO) <<
    //        summary.FullReport();

    const auto mp = probability_grid.limits().GetCellIndex(
        pose_estimate.translation().cast<float>());

    cairo_set_source_rgba(cr, 1, 1, 0, 1.0);
    cairo_set_line_width(cr, 1.0);
    cairo_arc(cr, mp.x(), mp.y(), circle_radius_cells, 0, 2 * M_PI);
    cairo_stroke(cr);
  }

  cairo_destroy(cr);

  cairo_surface_write_to_png(surface.get(), "test3.png");
}

}  // namespace
}  // namespace mapping
}  // namespace cartographer
