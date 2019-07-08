#include "cartographer/mapping/internal/2d/scan_matching/icp_scan_matcher_2d.h"

#include <utility>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/ceres_solver_options.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/rotation_delta_cost_functor_2d.h"
#include "cartographer/mapping/internal/2d/scan_matching/translation_delta_cost_functor_2d.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::ICPScanMatcherOptions2D CreateICPScanMatcherOptions2D(
    common::LuaParameterDictionary* const parameter_dictionary) {
  proto::ICPScanMatcherOptions2D options;
  options.set_nn_huber_loss(parameter_dictionary->GetDouble("nn_huber_loss"));
  options.set_pp_huber_loss(parameter_dictionary->GetDouble("pp_huber_loss"));
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
    const Grid2D& grid, const proto::ICPScanMatcherOptions2D& options)
    : options_(options),
      ceres_solver_options_(CreateCeresSolverOptions()),
      limits_(grid.limits()),
      kdtree_(CreateRealIndexForGrid(grid)) {}

ICPScanMatcher2D::~ICPScanMatcher2D() {}

ICPScanMatcher2D::Result ICPScanMatcher2D::Match(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud) const {
  double ceres_pose_estimate[3] = {
      initial_pose_estimate.translation().x(),
      initial_pose_estimate.translation().y(),
      initial_pose_estimate.rotation().smallestAngle()};
  ceres::Problem problem;

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const Eigen::Vector2d point(point_cloud[i].position.x(),
                                point_cloud[i].position.y());

    auto cost_fn =
        new SingleNearestNeighbourCostFunction2D(point, *kdtree_.kdtree);
    auto numeric_diff =
        new ceres::NumericDiffCostFunction<SingleNearestNeighbourCostFunction2D,
                                           ceres::CENTRAL, 2, 3>(cost_fn);

    problem.AddResidualBlock(numeric_diff,
                             new ceres::HuberLoss(options_.nn_huber_loss()),
                             ceres_pose_estimate);
  }

  Result result;

  ceres::Solve(ceres_solver_options_, &problem, &result.summary);

  result.pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

  {
    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    double query_pt[2];
    nanoflann::KNNResultSet<double> result_set(num_results);
    for (size_t i = 0; i < point_cloud.size(); ++i) {
      const Eigen::Vector2d point(point_cloud[i].position.x(),
                                  point_cloud[i].position.y());
      const auto world = result.pose_estimate * point;

      result_set.init(&ret_index, &out_dist_sqr);

      query_pt[0] = world[0];
      query_pt[1] = world[1];

      kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                    nanoflann::SearchParams(10));

      result.pairs.push_back({i,
                              {kdtree_.cells.cells[ret_index].x,
                               kdtree_.cells.cells[ret_index].y}});
    }
  }

  return result;
}

ICPScanMatcher2D::Result ICPScanMatcher2D::MatchPointPair(
    const transform::Rigid2d& initial_pose_estimate,
    const sensor::PointCloud& point_cloud) const {
  double ceres_pose_estimate[3] = {
      initial_pose_estimate.translation().x(),
      initial_pose_estimate.translation().y(),
      initial_pose_estimate.rotation().smallestAngle()};
  ceres::Problem problem;

  double occupied_space_weight = 1.0;

  const size_t num_results = 1;
  size_t ret_index;
  double out_dist_sqr;
  double query_pt[2];
  nanoflann::KNNResultSet<double> result_set(num_results);

  Result result;

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const Eigen::Vector2d point(point_cloud[i].position.x(),
                                point_cloud[i].position.y());
    const auto world = initial_pose_estimate * point;

    result_set.init(&ret_index, &out_dist_sqr);

    query_pt[0] = world[0];
    query_pt[1] = world[1];

    kdtree_.kdtree->findNeighbors(result_set, &query_pt[0],
                                  nanoflann::SearchParams(10));

    auto cost_fn = new PointPairCostFunction2D(
        occupied_space_weight, point,
        {kdtree_.cells.cells[ret_index].x, kdtree_.cells.cells[ret_index].y});
    auto auto_diff =
        new ceres::AutoDiffCostFunction<PointPairCostFunction2D, 2, 3>(cost_fn);

    problem.AddResidualBlock(auto_diff,
                             new ceres::HuberLoss(options_.pp_huber_loss()),
                             ceres_pose_estimate);

    result.pairs.push_back(
        {i,
         {kdtree_.cells.cells[ret_index].x, kdtree_.cells.cells[ret_index].y}});
  }

  ceres::Solve(ceres_solver_options_, &problem, &result.summary);

  result.pose_estimate = transform::Rigid2d(
      {ceres_pose_estimate[0], ceres_pose_estimate[1]}, ceres_pose_estimate[2]);

  return result;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
