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
#include "cartographer/mapping/internal/2d/scan_matching/rotation_delta_cost_functor_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/translation_delta_cost_functor_2d.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

namespace {

class StaticCost {
 public:
  explicit StaticCost(const double cost) : cost_(cost) {}

  template <typename T>
  bool operator()(const T* const, T* residual) const {
    residual[0] = T(cost_);
    residual[1] = T(cost_);
    return true;
  }

 private:
  StaticCost(const StaticCost&) = delete;
  StaticCost& operator=(const StaticCost&) = delete;

  const double cost_;
};

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
  options.set_unmatched_feature_cost(
      parameter_dictionary->GetDouble("unmatched_feature_cost"));
  options.set_point_weight(parameter_dictionary->GetDouble("point_weight"));
  options.set_feature_weight(parameter_dictionary->GetDouble("feature_weight"));
  options.set_inlier_distance_threshold(
      parameter_dictionary->GetDouble("inlier_distance_threshold"));
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
    : options_(options),
      ceres_solver_options_(CreateCeresSolverOptions()),
      limits_(submap.grid()->limits()),
      kdtree_(CreateRealIndexForGrid(*submap.grid())),
      circle_feature_index_(
          CreateIndexForCircleFeatures(submap.CircleFeatures())) {
  CHECK(options_.feature_weight() >= 0);
  CHECK(options_.point_weight() >= 0);
  CHECK(options_.feature_weight() + options_.point_weight() > 0);
  CHECK(options_.inlier_distance_threshold() > 0);
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

  const double inlier_d2 = options_.inlier_distance_threshold() *
                           options_.inlier_distance_threshold();

  {
    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(num_results);

    for (size_t i = 0; i < features.size(); ++i) {
      if (circle_feature_index_.feature_set.data.empty()) {
        // Add a fixed cost for un-matched features
        auto cost_fn = new StaticCost(
            options_.unmatched_feature_cost() * options_.feature_weight() /
            std::sqrt(static_cast<double>(features.size())));
        auto auto_diff =
            new ceres::AutoDiffCostFunction<StaticCost, 2, 3>(cost_fn);
        problem.AddResidualBlock(
            auto_diff,
            new ceres::HuberLoss(
                options_.nearest_neighbour_feature_huber_loss()),
            ceres_pose_estimate);
      } else {
        const Eigen::Vector2d fp(features[i].keypoint.position.x(),
                                 features[i].keypoint.position.y());
        const auto world = initial_pose_estimate * fp;

        result_set.init(&ret_index, &out_dist_sqr);

        query_pt[0] = static_cast<float>(world[0]);
        query_pt[1] = static_cast<float>(world[1]);
        query_pt[2] = features[i].fdescriptor.radius;

        circle_feature_index_.kdtree->findNeighbors(
            result_set, &query_pt[0], nanoflann::SearchParams(10));

        if (static_cast<double>(out_dist_sqr) < inlier_d2) {
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
        }
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

      if (out_dist_sqr < inlier_d2) {
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
  result.num_inlier_points = included_points.size();

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

  if (!circle_feature_index_.feature_set.data.empty()) {
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(1);
    for (size_t i = 0; i < features.size(); ++i) {
      const Eigen::Vector2d fp(features[i].keypoint.position.x(),
                               features[i].keypoint.position.y());
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

  const double inlier_d2 = options_.inlier_distance_threshold() *
                           options_.inlier_distance_threshold();

  {
    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    float query_pt[3];
    nanoflann::KNNResultSet<float> result_set(num_results);

    for (size_t i = 0; i < features.size(); ++i) {
      if (circle_feature_index_.feature_set.data.empty()) {
        // Add a fixed cost for un-matched features
        auto cost_fn = new StaticCost(
            options_.unmatched_feature_cost() * options_.feature_weight() /
            std::sqrt(static_cast<double>(features.size())));
        auto auto_diff =
            new ceres::AutoDiffCostFunction<StaticCost, 2, 3>(cost_fn);
        problem.AddResidualBlock(
            auto_diff,
            new ceres::HuberLoss(options_.point_pair_feature_huber_loss()),
            ceres_pose_estimate);
      } else {
        const Eigen::Vector2d fp(features[i].keypoint.position.x(),
                                 features[i].keypoint.position.y());
        const auto world = initial_pose_estimate * fp;

        result_set.init(&ret_index, &out_dist_sqr);

        query_pt[0] = static_cast<float>(world[0]);
        query_pt[1] = static_cast<float>(world[1]);
        query_pt[2] = features[i].fdescriptor.radius;

        circle_feature_index_.kdtree->findNeighbors(
            result_set, &query_pt[0], nanoflann::SearchParams(10));

        if (static_cast<double>(out_dist_sqr) < inlier_d2) {
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
        }
      }
    }
  }

  CHECK(!kdtree_.cells.cells.empty());

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

      if (out_dist_sqr < inlier_d2) {
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
      }
    }
  }

  ceres::Solve(ceres_solver_options_, &problem, &result.summary);

  result.pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

  return result;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
