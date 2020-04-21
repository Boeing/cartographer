#include "cartographer/mapping/internal/2d/scan_matching/icp_scan_matcher_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ceres_scan_matcher_2d.h"

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/ceres_solver_options.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/ray_trace.h"
#include "cartographer/mapping/internal/2d/scan_matching/rotation_delta_cost_functor_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/translation_delta_cost_functor_2d.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

namespace {

CircleFeatureIndex CreateIndexForCircleFeatures(
    const std::vector<CircleFeature>& circle_features) {
  CircleFeatureIndex index;
  index.feature_set.data = circle_features;
  index.kdtree.reset(new CircleFeatureKDTree(
      3, index.feature_set, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  index.kdtree->buildIndex();
  return index;
}

}  // namespace

proto::ICPScanMatcherOptions2D CreateICPScanMatcherOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::ICPScanMatcherOptions2D options;
  options.set_nearest_neighbour_point_huber_loss(
      parameter_dictionary->GetDouble("nearest_neighbour_point_huber_loss"));
  options.set_nearest_neighbour_feature_huber_loss(
      parameter_dictionary->GetDouble("nearest_neighbour_feature_huber_loss"));
  options.set_point_pair_point_huber_loss(
      parameter_dictionary->GetDouble("point_pair_point_huber_loss"));
  options.set_point_pair_feature_huber_loss(
      parameter_dictionary->GetDouble("point_pair_feature_huber_loss"));
  options.set_point_weight(parameter_dictionary->GetDouble("point_weight"));
  options.set_feature_weight(parameter_dictionary->GetDouble("feature_weight"));
  options.set_point_inlier_threshold(
      parameter_dictionary->GetDouble("point_inlier_threshold"));
  options.set_feature_inlier_threshold(
      parameter_dictionary->GetDouble("feature_inlier_threshold"));
  return options;
}

ceres::Solver::Options CreateCeresSolverOptions() {
  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.max_num_iterations = 100;
  options.num_threads = 1;
  options.linear_solver_type = ceres::DENSE_QR;
  return options;
}

ICPScanMatcher2D::ICPScanMatcher2D(
    const Submap2D& submap, const proto::ICPScanMatcherOptions2D& options)
    : submap_(submap),
      options_(options),
      ceres_solver_options_(CreateCeresSolverOptions()),
      limits_(submap.grid()->limits()),
      kdtree_(CreateRealIndexForGrid(*submap.grid())),
      circle_feature_index_(
          CreateIndexForCircleFeatures(submap.CircleFeatures())) {
  CHECK(options_.feature_weight() >= 0);
  CHECK(options_.point_weight() >= 0);
  CHECK(options_.feature_weight() + options_.point_weight() > 0);
  CHECK(options_.point_inlier_threshold() > 0);
  CHECK(options_.feature_inlier_threshold() > 0);
}

ICPScanMatcher2D::~ICPScanMatcher2D() {}

ICPScanMatcher2D::Result ICPScanMatcher2D::Match(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud,
    const std::vector<CircleFeature>& features) const {
  double ceres_pose_estimate[3] = {
      initial_pose_estimate.translation().x(),
      initial_pose_estimate.translation().y(),
      initial_pose_estimate.rotation().smallestAngle()};
  ceres::Problem problem;

  const double point_inlier_d2 =
      options_.point_inlier_threshold() * options_.point_inlier_threshold();
  const double feature_inlier_d2 =
      options_.feature_inlier_threshold() * options_.feature_inlier_threshold();

  std::vector<size_t> included_features;
  if (!circle_feature_index_.feature_set.data.empty()) {
    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(num_results);
    for (size_t i = 0; i < features.size(); ++i) {
      const Eigen::Vector2d fp(features[i].keypoint.position.x(),
                               features[i].keypoint.position.y());
      const auto world = initial_pose_estimate * fp;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = static_cast<float>(world[0]);
      query_pt[1] = static_cast<float>(world[1]);
      query_pt[2] = features[i].fdescriptor.radius;

      circle_feature_index_.kdtree->findNeighbors(result_set, &query_pt[0],
                                                  nanoflann::SearchParams(10));

      if (static_cast<double>(out_dist_sqr) < feature_inlier_d2) {
        auto cost_fn = new NearestFeatureCostFunction2D(
            options_.feature_weight() /
                std::sqrt(static_cast<double>(features.size())),
            features[i], *(circle_feature_index_.kdtree));
        auto numeric_diff =
            new ceres::NumericDiffCostFunction<NearestFeatureCostFunction2D,
                                               ceres::CENTRAL, 2, 3>(cost_fn);
        problem.AddResidualBlock(
            numeric_diff,
            new ceres::HuberLoss(
                options_.nearest_neighbour_feature_huber_loss()),
            ceres_pose_estimate);

        included_features.push_back(i);
      }
    }
  }

  CHECK(!kdtree_.cells.cells.empty());

  std::vector<size_t> included_points;
  {
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);
    for (size_t i = 0; i < point_cloud.size(); ++i) {
      const Eigen::Vector2d point(point_cloud[i].position.x(),
                                  point_cloud[i].position.y());

      const auto world = initial_pose_estimate * point;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = world[0];
      query_pt[1] = world[1];

      kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                    nanoflann::SearchParams(10));

      if (out_dist_sqr < point_inlier_d2) {
        auto cost_fn = new SingleNearestNeighbourCostFunction2D(
            options_.point_weight() /
                std::sqrt(static_cast<double>(point_cloud.size())),
            point, *kdtree_.kdtree);
        auto numeric_diff = new ceres::NumericDiffCostFunction<
            SingleNearestNeighbourCostFunction2D, ceres::CENTRAL, 2, 3>(
            cost_fn);

        problem.AddResidualBlock(
            numeric_diff,
            new ceres::HuberLoss(options_.nearest_neighbour_point_huber_loss()),
            ceres_pose_estimate);

        included_points.push_back(i);
      }
    }
  }

  Result result;
  result.points_count = included_points.size();
  result.points_inlier_fraction =
      !point_cloud.empty()
          ? static_cast<double>(included_points.size()) / point_cloud.size()
          : 1.0;
  result.features_count = included_features.size();
  result.features_inlier_fraction =
      !features.empty()
          ? static_cast<double>(included_features.size()) / features.size()
          : 1.0;

  ceres::Solve(ceres_solver_options_, &problem, &result.summary);

  result.pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

  {
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);
    for (size_t i = 0; i < included_points.size(); ++i) {
      const Eigen::Vector2d point(point_cloud[included_points[i]].position.x(),
                                  point_cloud[included_points[i]].position.y());
      const auto world = result.pose_estimate * point;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = world[0];
      query_pt[1] = world[1];

      kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                    nanoflann::SearchParams(10));

      result.pairs.push_back({world,
                              {kdtree_.cells.cells[ret_index].x,
                               kdtree_.cells.cells[ret_index].y}});
    }
  }

  {
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(1);
    for (size_t i = 0; i < included_features.size(); ++i) {
      const Eigen::Vector2d fp(
          features[included_features[i]].keypoint.position.x(),
          features[included_features[i]].keypoint.position.y());
      const auto world = result.pose_estimate * fp;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = static_cast<float>(world[0]);
      query_pt[1] = static_cast<float>(world[1]);
      query_pt[2] = features[i].fdescriptor.radius;

      circle_feature_index_.kdtree->findNeighbors(result_set, &query_pt[0],
                                                  nanoflann::SearchParams(10));

      result.pairs.push_back({world,
                              {circle_feature_index_.feature_set.data[ret_index]
                                   .keypoint.position.x(),
                               circle_feature_index_.feature_set.data[ret_index]
                                   .keypoint.position.y()}});
    }
  }

  return result;
}

ICPScanMatcher2D::Statistics ICPScanMatcher2D::EvalutateMatch(
    const Result& result, const sensor::RangeData& range_data) const {
  // point match check
  Statistics statistics;
  statistics.agree_fraction = 1.0;
  statistics.miss_fraction = 1.0;
  statistics.hit_fraction = 1.0;
  statistics.ray_trace_fraction = 1.0;

  auto pg = dynamic_cast<const mapping::ProbabilityGrid*>(submap_.grid());

  int hit_count = 0;
  int ray_trace_success_count = 0;

  const double max_distance = 1.0 * submap_.grid()->limits().resolution();
  const double max_squared_distance = max_distance * max_distance;

  const size_t num_results = 1;
  size_t ret_index;
  double out_dist_sqr;
  nanoflann::KNNResultSet<double> result_set(num_results);
  double query_pt[2];

  const auto mp_start = pg->limits().GetCellIndex(
      result.pose_estimate.translation().cast<float>());

  for (const auto& laser_return : range_data.returns) {
    result_set.init(&ret_index, &out_dist_sqr);
    const auto tr =
        result.pose_estimate.cast<float>() * laser_return.position.head<2>();
    query_pt[0] = static_cast<double>(tr.x());
    query_pt[1] = static_cast<double>(tr.y());
    kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                  nanoflann::SearchParams(10));
    if (out_dist_sqr < max_squared_distance) {
      ++hit_count;
    }
    {
      const auto mp_end = pg->limits().GetCellIndex(tr);
      const auto p = mapping::scan_matching::raytraceLine(
          *pg, mp_start.x(), mp_start.y(), mp_end.x(), mp_end.y(),
          pg->limits().cell_limits().num_x_cells);

      // Consider point as valid if it is not behind occupied space
      // this will return the first cell which is occupied
      if (p.x != -1 && p.y != -1) {
        const Eigen::Vector2f inter =
            pg->limits().GetCellCenter(Eigen::Array2i(p.x, p.y));
        const float dist_from_end =
            Eigen::Vector2f(query_pt[0] - inter[0], query_pt[1] - inter[1])
                .norm();
        if (dist_from_end < 0.1) {
          ++ray_trace_success_count;
        }
      } else {
        ++ray_trace_success_count;
      }
    }
  }

  if (!range_data.returns.empty()) {
    statistics.hit_fraction = static_cast<double>(hit_count) /
                              static_cast<double>(range_data.returns.size());
    statistics.ray_trace_fraction =
        static_cast<double>(ray_trace_success_count) /
        static_cast<double>(range_data.returns.size());
  }

  int miss_count = 0;
  for (const auto& laser_miss : range_data.misses) {
    const auto tr =
        result.pose_estimate.cast<float>() * laser_miss.position.head<2>();
    const auto mp_end = pg->limits().GetCellIndex(tr);
    const unsigned int max_cells = static_cast<unsigned int>(
        static_cast<double>(laser_miss.position.norm()) /
        pg->limits().resolution());
    const auto p = mapping::scan_matching::raytraceLine(
        *pg, mp_start.x(), mp_start.y(), mp_end.x(), mp_end.y(),
        pg->limits().cell_limits().num_x_cells, max_cells);
    if (p.x == -1 && p.y == -1) ++miss_count;
  }
  if (!range_data.misses.empty())
    statistics.miss_fraction = static_cast<double>(miss_count) /
                               static_cast<double>(range_data.misses.size());

  {
    const int total_count = hit_count + miss_count;
    const size_t total_sum =
        range_data.misses.size() + range_data.returns.size();
    if (total_sum > 0)
      statistics.agree_fraction =
          static_cast<double>(total_count) / static_cast<double>(total_sum);
  }

  return statistics;
}

ICPScanMatcher2D::Result ICPScanMatcher2D::MatchPointPair(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud,
    const std::vector<CircleFeature>& features) const {
  double ceres_pose_estimate[3] = {
      initial_pose_estimate.translation().x(),
      initial_pose_estimate.translation().y(),
      initial_pose_estimate.rotation().smallestAngle()};
  ceres::Problem problem;

  Result result;

  const double point_inlier_d2 =
      options_.point_inlier_threshold() * options_.point_inlier_threshold();
  const double feature_inlier_d2 =
      options_.feature_inlier_threshold() * options_.feature_inlier_threshold();

  std::size_t features_count = 0;
  if (!circle_feature_index_.feature_set.data.empty()) {
    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(num_results);

    for (size_t i = 0; i < features.size(); ++i) {
      const Eigen::Vector2d fp(features[i].keypoint.position.x(),
                               features[i].keypoint.position.y());
      const auto world = initial_pose_estimate * fp;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = static_cast<float>(world[0]);
      query_pt[1] = static_cast<float>(world[1]);
      query_pt[2] = features[i].fdescriptor.radius;

      circle_feature_index_.kdtree->findNeighbors(result_set, &query_pt[0],
                                                  nanoflann::SearchParams(10));

      if (static_cast<double>(out_dist_sqr) < feature_inlier_d2) {
        auto cost_fn = new PointPairCostFunction2D(
            options_.feature_weight() /
                std::sqrt(static_cast<double>(features.size())),
            fp,
            {circle_feature_index_.feature_set.data[ret_index]
                 .keypoint.position.x(),
             circle_feature_index_.feature_set.data[ret_index]
                 .keypoint.position.y()});
        auto auto_diff =
            new ceres::AutoDiffCostFunction<PointPairCostFunction2D, 2, 3>(
                cost_fn);

        problem.AddResidualBlock(
            auto_diff,
            new ceres::HuberLoss(options_.point_pair_feature_huber_loss()),
            ceres_pose_estimate);

        result.pairs.push_back(
            {world,
             {circle_feature_index_.feature_set.data[ret_index]
                  .keypoint.position.x(),
              circle_feature_index_.feature_set.data[ret_index]
                  .keypoint.position.y()}});

        features_count++;
      }
    }
  }

  CHECK(!kdtree_.cells.cells.empty());

  std::size_t points_count = 0;
  {
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);

    for (size_t i = 0; i < point_cloud.size(); ++i) {
      const Eigen::Vector2d point(point_cloud[i].position.x(),
                                  point_cloud[i].position.y());
      const auto world = initial_pose_estimate * point;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = world[0];
      query_pt[1] = world[1];

      kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                    nanoflann::SearchParams(10));

      if (out_dist_sqr < point_inlier_d2) {
        auto cost_fn = new PointPairCostFunction2D(
            options_.point_weight() /
                std::sqrt(static_cast<double>(point_cloud.size())),
            point,
            {kdtree_.cells.cells[ret_index].x,
             kdtree_.cells.cells[ret_index].y});
        auto auto_diff =
            new ceres::AutoDiffCostFunction<PointPairCostFunction2D, 2, 3>(
                cost_fn);

        problem.AddResidualBlock(
            auto_diff,
            new ceres::HuberLoss(options_.point_pair_point_huber_loss()),
            ceres_pose_estimate);

        result.pairs.push_back({world,
                                {kdtree_.cells.cells[ret_index].x,
                                 kdtree_.cells.cells[ret_index].y}});

        points_count++;
      }
    }
  }

  ceres::Solve(ceres_solver_options_, &problem, &result.summary);

  result.pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

  result.points_count = points_count;
  result.points_inlier_fraction =
      !point_cloud.empty()
          ? static_cast<double>(points_count) / point_cloud.size()
          : 1.0;
  result.features_count = features_count;
  result.features_inlier_fraction =
      !features.empty() ? static_cast<double>(features_count) / features.size()
                        : 1.0;

  return result;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
