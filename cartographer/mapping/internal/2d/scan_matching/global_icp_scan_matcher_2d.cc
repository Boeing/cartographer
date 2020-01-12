#include "cartographer/mapping/internal/2d/scan_matching/global_icp_scan_matcher_2d.h"

#include <queue>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/ceres_solver_options.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/transform/transform.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

proto::GlobalICPScanMatcherOptions2D CreateGlobalICPScanMatcherOptions2D(
    common::LuaParameterDictionary* parameter_dictionary) {
  proto::GlobalICPScanMatcherOptions2D options;
  options.set_num_global_samples(
      parameter_dictionary->GetInt("num_global_samples"));
  options.set_num_global_rotations(
      parameter_dictionary->GetInt("num_global_rotations"));

  options.set_proposal_max_score(
      parameter_dictionary->GetDouble("proposal_max_score"));

  options.set_proposal_min_inlier_fraction(
      parameter_dictionary->GetDouble("proposal_min_inlier_fraction"));

  options.set_min_cluster_size(
      parameter_dictionary->GetInt("min_cluster_size"));
  options.set_min_cluster_distance(
      parameter_dictionary->GetDouble("min_cluster_distance"));

  options.set_num_local_samples(
      parameter_dictionary->GetInt("num_local_samples"));

  options.set_local_sample_linear_distance(
      parameter_dictionary->GetDouble("local_sample_linear_distance"));
  options.set_local_sample_angular_distance(
      parameter_dictionary->GetDouble("local_sample_angular_distance"));

  options.set_proposal_features_weight(
      parameter_dictionary->GetDouble("proposal_features_weight"));

  options.set_proposal_points_weight(
      parameter_dictionary->GetDouble("proposal_points_weight"));

  *options.mutable_icp_options() = scan_matching::CreateICPScanMatcherOptions2D(
      parameter_dictionary->GetDictionary("icp_options").get());
  return options;
}

namespace {

std::vector<GlobalICPScanMatcher2D::RotatedScan> GenerateRotatedScans(
    const sensor::PointCloud& point_cloud, const std::size_t samples) {
  std::vector<GlobalICPScanMatcher2D::RotatedScan> rotated_scans(samples);

  double theta = -M_PI;
  const double delta_theta = 2.0 * M_PI / samples;
  for (std::size_t scan_index = 0; scan_index < samples; ++scan_index) {
    const auto roation = transform::Rigid3f::Rotation(
        Eigen::AngleAxisf(static_cast<float>(theta), Eigen::Vector3f::UnitZ()));
    rotated_scans[scan_index].rotation = theta;
    rotated_scans[scan_index].scan_data =
        sensor::TransformPointCloud(point_cloud, roation);
    theta += delta_theta;
  }
  return rotated_scans;
}

struct SamplePoseMinHeapOperator {
  bool operator()(const GlobalICPScanMatcher2D::SamplePose& lhs,
                  const GlobalICPScanMatcher2D::SamplePose& rhs) {
    return lhs.score > rhs.score;
  }
};

}  // namespace

std::vector<Eigen::Array2i> FreeCells(const Grid2D& grid) {
  std::vector<Eigen::Array2i> result;
  const uint16 occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));
  const auto limits = grid.limits();

  for (int y = 0; y < limits.cell_limits().num_y_cells; ++y) {
    for (int x = 0; x < limits.cell_limits().num_x_cells; ++x) {
      const uint16 cc = grid.correspondence_cost_cells()[static_cast<size_t>(
          grid.ToFlatIndex({x, y}))];
      if (cc != kUnknownCorrespondenceValue && cc > occupied_value) {
        result.push_back({x, y});
      }
    }
  }

  return result;
}

GlobalICPScanMatcher2D::GlobalICPScanMatcher2D(
    const Submap2D& submap, const proto::GlobalICPScanMatcherOptions2D& options)
    : options_(options),
      limits_(submap.grid()->limits()),
      sampler_(*submap.grid()),
      icp_solver_(submap, options_.icp_options()) {
  CHECK(options_.num_global_samples() > 0);
  CHECK(options_.num_global_rotations() > 0);
  CHECK(options_.proposal_max_score() > 0);
  CHECK(options_.proposal_min_inlier_fraction() > 0);
  CHECK(options_.min_cluster_size() > 0);
  CHECK(options_.min_cluster_distance() > 0);
  CHECK(options_.num_local_samples() > 0);
  CHECK(options_.local_sample_linear_distance() > 0);
  CHECK(options_.local_sample_angular_distance() > 0);
}

GlobalICPScanMatcher2D::~GlobalICPScanMatcher2D() {}

GlobalICPScanMatcher2D::Result GlobalICPScanMatcher2D::Match(
    const transform::Rigid2d pose_estimate,
    const sensor::PointCloud& point_cloud,
    const std::vector<CircleFeature>& features) {
  std::mt19937 gen(42);

  const double initial_theta = pose_estimate.rotation().smallestAngle();

  std::normal_distribution<double> linear_dist(
      0, options_.local_sample_linear_distance());
  std::normal_distribution<double> angular_dist(
      initial_theta, options_.local_sample_angular_distance());

  Result result;

  int num_samples = options_.num_global_samples();
  if (!features.empty()) num_samples *= 10;

  for (int i = 0; i < num_samples; ++i) {
    const double theta = angular_dist(gen);
    const Eigen::AngleAxisf rotation =
        Eigen::AngleAxisf(static_cast<float>(theta), Eigen::Vector3f::UnitZ());
    const transform::Rigid3f transform(transform::Rigid3f::Vector(0, 0, 0),
                                       rotation);
    const auto scan_data = sensor::TransformPointCloud(point_cloud, transform);

    SamplePose sample_pose;
    sample_pose.rotation = theta;
    sample_pose.x = pose_estimate.translation().x() + linear_dist(gen);
    sample_pose.y = pose_estimate.translation().y() + linear_dist(gen);

    if (evaluateSample(sample_pose, scan_data, features))
      result.poses.push_back(sample_pose);
  }

  return result;
}

GlobalICPScanMatcher2D::Result GlobalICPScanMatcher2D::Match(
    const sensor::PointCloud& point_cloud,
    const std::vector<CircleFeature>& features) {
  Result result;

  if (sampler_.size() == 0) return result;

  const std::vector<RotatedScan> rotated_scans = GenerateRotatedScans(
      point_cloud, static_cast<size_t>(options_.num_global_rotations()));

  int num_samples = options_.num_global_samples();
  if (!features.empty()) num_samples *= 10;

  for (int i = 0; i < num_samples; ++i) {
    const auto mp = sampler_.sample();
    const auto real_p = limits_.GetCellCenter(mp);

    for (size_t r = 0; r < rotated_scans.size(); ++r) {
      const RotatedScan& scan = rotated_scans[r];

      SamplePose sample_pose;
      sample_pose.rotation = scan.rotation;
      sample_pose.x = static_cast<double>(real_p.x());
      sample_pose.y = static_cast<double>(real_p.y());

      if (evaluateSample(sample_pose, scan.scan_data, features))
        result.poses.push_back(sample_pose);
    }
  }

  return result;
}

bool GlobalICPScanMatcher2D::evaluateSample(
    SamplePose& sample_pose, const sensor::PointCloud& rotated_scan,
    const std::vector<CircleFeature>& features) {
  sample_pose.score = std::numeric_limits<double>::max();
  sample_pose.feature_match_cost = 0.0;
  sample_pose.point_match_cost = 0.0;

  // filter based on features
  if (!features.empty()) {
    const Eigen::AngleAxisf rotation = Eigen::AngleAxisf(
        static_cast<float>(sample_pose.rotation), Eigen::Vector3f::UnitZ());

    const size_t num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    nanoflann::KNNResultSet<float> result_set(num_results);
    float query_pt[3];

    for (std::size_t p = 0; p < features.size(); ++p) {
      result_set.init(&ret_index, &out_dist_sqr);
      const Eigen::Vector3f rotated_point =
          rotation * features[p].keypoint.position;

      query_pt[0] = static_cast<float>(sample_pose.x) + rotated_point.x();
      query_pt[1] = static_cast<float>(sample_pose.y) + rotated_point.y();
      query_pt[2] = features[p].fdescriptor.radius;
      icp_solver_.circle_feature_index().kdtree->findNeighbors(
          result_set, &query_pt[0], nanoflann::SearchParams(10));
      sample_pose.feature_match_cost +=
          static_cast<double>(std::sqrt(out_dist_sqr));
    }
    sample_pose.feature_match_cost /= features.size();
  }

  // don't bother considering if the features are a bad match
  if (sample_pose.feature_match_cost > options_.proposal_max_score())
    return false;

  // filter based on scan points
  {
    const double proposal_max_squared =
        options_.proposal_max_score() * options_.proposal_max_score();

    const size_t num_results = 1;
    size_t ret_index;
    double out_dist_sqr;
    nanoflann::KNNResultSet<double> result_set(num_results);
    double query_pt[2];

    size_t count = 0;
    for (std::size_t p = 0; p < rotated_scan.size(); ++p) {
      result_set.init(&ret_index, &out_dist_sqr);
      query_pt[0] = static_cast<double>(sample_pose.x) +
                    static_cast<double>(rotated_scan[p].position.x());
      query_pt[1] = static_cast<double>(sample_pose.y) +
                    static_cast<double>(rotated_scan[p].position.y());
      icp_solver_.kdtree().kdtree->findNeighbors(result_set, &query_pt[0],
                                                 nanoflann::SearchParams(10));
      if (out_dist_sqr < proposal_max_squared) {
        ++count;
        sample_pose.point_match_cost += std::sqrt(out_dist_sqr);
      }
    }

    // reject sample if there are no inliers
    if (count == 0) return false;

    sample_pose.inlier_fraction =
        static_cast<double>(count) / static_cast<double>(rotated_scan.size());
    sample_pose.point_match_cost /= static_cast<double>(count);
    sample_pose.point_match_cost /=
        (sample_pose.inlier_fraction * sample_pose.inlier_fraction);
  }

  // don't bother considering if the inlier fraction is low
  if (sample_pose.inlier_fraction < options_.proposal_min_inlier_fraction())
    return false;

  // the final score is the average of features + scan
  if (sample_pose.feature_match_cost > 0) {
    sample_pose.score =
        (options_.proposal_features_weight() * sample_pose.feature_match_cost +
         options_.proposal_points_weight() * sample_pose.point_match_cost) /
        (options_.proposal_features_weight() +
         options_.proposal_points_weight());
  } else
    sample_pose.score = sample_pose.point_match_cost;

  // don't bother considering if the score is bad
  if (sample_pose.score > options_.proposal_max_score()) return false;

  return true;
}

std::vector<GlobalICPScanMatcher2D::PoseCluster>
GlobalICPScanMatcher2D::DBScanCluster(const std::vector<SamplePose>& poses) {
  int cluster_counter = 0;
  static constexpr int UNDEFINED = -1;
  static constexpr int NOISE = -2;

  ClusterData sample_pose_data;
  sample_pose_data.data = poses;

  ClusterTree kdtree(3, sample_pose_data,
                     nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  std::vector<PoseCluster> clusters;

  std::vector<int> labels(poses.size(), -1);
  for (std::size_t i = 0; i < sample_pose_data.data.size(); ++i) {
    const auto& pose = sample_pose_data.data[i];

    if (labels[i] != UNDEFINED) continue;

    double query_pt[3] = {pose.x, pose.y, pose.rotation};
    std::vector<std::pair<size_t, double>> ret_matches;
    const size_t num_matches =
        kdtree.radiusSearch(&query_pt[0], options_.min_cluster_distance(),
                            ret_matches, nanoflann::SearchParams());

    if (num_matches + 1 < static_cast<size_t>(options_.min_cluster_size())) {
      labels[i] = NOISE;
      continue;
    }

    cluster_counter++;

    PoseCluster cluster;
    cluster.poses.push_back(pose);

    labels[i] = cluster_counter;

    for (std::size_t n = 0; n < ret_matches.size(); ++n) {
      const auto& match = ret_matches[n];

      if (labels[match.first] == NOISE) {
        labels[match.first] = cluster_counter;
      } else if (labels[match.first] != UNDEFINED) {
        continue;
      } else {
        labels[match.first] = cluster_counter;
      }

      cluster.poses.push_back(sample_pose_data.data[match.first]);

      {
        double sub_query_pt[3] = {kdtree.dataset.kdtree_get_pt(match.first, 0),
                                  kdtree.dataset.kdtree_get_pt(match.first, 1),
                                  kdtree.dataset.kdtree_get_pt(match.first, 2)};
        std::vector<std::pair<size_t, double>> sub_ret_matches;
        const size_t sub_num_matches = kdtree.radiusSearch(
            &sub_query_pt[0], options_.min_cluster_distance(), sub_ret_matches,
            nanoflann::SearchParams());

        if (static_cast<int>(sub_num_matches) + 1 >
            options_.min_cluster_size()) {
          for (const auto& match : sub_ret_matches)
            ret_matches.push_back(match);
        }
      }
    }

    // determine the weighted cluster center
    cluster.x = 0;
    cluster.y = 0;
    double rotation_x = 0;
    double rotation_y = 0;
    double weights_sum = 0;
    for (std::size_t p = 0; p < cluster.poses.size(); ++p) {
      const double weight = 1.0 / std::max(0.01, cluster.poses[p].score);
      cluster.x += weight * cluster.poses[p].x;
      cluster.y += weight * cluster.poses[p].y;
      rotation_x += weight * std::cos(cluster.poses[p].rotation);
      rotation_y += weight * std::sin(cluster.poses[p].rotation);
      weights_sum += weight;
    }
    cluster.x /= weights_sum;
    cluster.y /= weights_sum;
    rotation_x /= weights_sum;
    rotation_y /= weights_sum;
    cluster.rotation = std::atan2(rotation_y, rotation_x);

    clusters.push_back(cluster);
  }

  return clusters;
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
