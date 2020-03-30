#include "cartographer/mapping/internal/2d/scan_features/circle_detector_2d.h"

#include "cartographer/common/nanoflann.h"

#include "ceres/ceres.h"

#include <queue>

namespace cartographer {
namespace mapping {
namespace {

struct PointCloudSorter {
  PointCloudSorter() {}
  bool operator()(const sensor::RangefinderPoint& lhs,
                  const sensor::RangefinderPoint& rhs) {
    const Eigen::Vector2f delta_lhs = lhs.position.head<2>().normalized();
    const Eigen::Vector2f delta_rhs = rhs.position.head<2>().normalized();
    if ((delta_lhs[1] < 0.f) != (delta_rhs[1] < 0.f)) {
      return delta_lhs[1] < 0.f;
    } else if (delta_lhs[1] < 0.f) {
      return delta_lhs[0] < delta_rhs[0];
    } else {
      return delta_lhs[0] > delta_rhs[0];
    }
  }
};

template <typename T>
struct CircleSorter {
  bool operator()(const Circle<T>& lhs, const Circle<T>& rhs) {
    return lhs.mse < rhs.mse;
  }
};

template <typename Itr>
Itr advance_and_wrap(Itr i, Itr begin, Itr end) {
  i++;
  if (i == end)
    return begin;
  else
    return i;
}

float angle_between(const Eigen::Vector2f& v1, const Eigen::Vector2f& v2) {
  const float dot = v1.dot(v2);
  const float mag = v1.norm() * v2.norm();
  return std::acos(dot / mag);
}

template <typename T>
constexpr T calc_a(const T B, const T A) {
  return -B / (2.0 * A);
}

template <typename T>
constexpr T calc_b(const T C, const T A) {
  return -C / (2.0 * A);
}

template <typename T>
constexpr T calc_R(const T A) {
  return 1 / (2.0 * std::abs(A));
}

template <typename T>
constexpr T calc_A(const T R) {
  return 1 / (2.0 * R);
}

template <typename T>
constexpr T calc_B(const T A, const T a) {
  return -2.0 * A * a;
}

template <typename T>
constexpr T calc_C(const T A, const T b) {
  return -2.0 * A * b;
}

template <typename T>
constexpr T calc_D(const T A, const T B, const T C) {
  return (B * B + C * C - 1) / (4.0 * A);
}

template <typename T>
constexpr T calc_squared_sum(const T x, const T y) {
  return x * x + y * y;
}

template <typename T>
constexpr T calc_P(const T A, const T B, const T C, const T D, const T x,
                   const T y, const T squared_sum) {
  return A * squared_sum + B * x + C * y + D;
}

template <typename T>
constexpr T calc_P(const T A, const T B, const T C, const T D, const T x,
                   const T y) {
  return calc_P(A, B, C, D, x, y, calc_squared_sum(x, y));
}

template <typename T>
constexpr T calc_d(const T P, const T A) {
  return 2.0 * P / (1.0 + ceres::sqrt(1.0 + 4.0 * A * P));
}

class CircleCostFunction {
 public:
  CircleCostFunction(const Eigen::Vector2f& point)
      : point_(point),
        squared_sum_(point_[0] * point_[0] + point_[1] * point_[1]) {}

  template <typename T>
  bool operator()(const T* const A, const T* const B, const T* const C,
                  const T* const D, T* residual) const {
    T P = calc_P(A[0], B[0], C[0], D[0], T(point_[0]), T(point_[1]),
                 T(squared_sum_));
    if (T(1.0) + T(4.0) * A[0] * P < T(0.0)) return false;
    residual[0] = calc_d(P, A[0]);
    return true;
  }

 private:
  CircleCostFunction(const CircleCostFunction&) = delete;
  CircleCostFunction& operator=(const CircleCostFunction&) = delete;

  const Eigen::Vector2f point_;
  const float squared_sum_;
};

struct DataSet {
  struct Cell {
    int x;
    int y;
    float cost;
  };

  std::vector<Cell> cells;

  inline size_t kdtree_get_point_count() const { return cells.size(); }

  inline int kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return cells[idx].x;
    else
      return cells[idx].y;
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<int, DataSet, double>, DataSet, 2>
    KDTree;

}  // namespace

std::vector<Circle<float>> DetectReflectivePoles(
    const sensor::PointCloud& point_cloud, const float radius) {
  // Assume the point cloud is centered about the sensor origin

  // Sort the point cloud data radially for quick neigbour searching
  auto sorted_range_data = point_cloud;
  std::sort(sorted_range_data.begin(), sorted_range_data.end(),
            PointCloudSorter());

  std::vector<Circle<float>> circles;
  std::vector<std::vector<size_t>> circles_scan_points;

  const float excl_radius = radius * 1.5f;
  const float max_error = radius * 0.25f;

  for (size_t i = 0; i < sorted_range_data.size(); ++i) {
    // assess every point
    // assume it is exactly aligned to the center of the circle

    // skip points which are not retro reflective
    if (sorted_range_data[i].intensity <= 0) continue;

    const Eigen::Vector2f range = sorted_range_data[i].position.head<2>();
    const Eigen::Vector2f dir = range;
    const Eigen::Vector2f dir_norm = dir.normalized();

    // position hypothesis
    const Eigen::Vector2f position = range + dir_norm * radius;

    float mse = 0.f;
    int count = 1;
    int intense_count = 1;

    // max angle +/- from centreline to pole
    const float max_angle = std::asin(excl_radius / (excl_radius + dir.norm()));

    Circle<float> circle;
    circle.radius = radius;
    circle.position = position;
    std::vector<size_t> circle_scan_points;

    circle_scan_points.push_back(i);

    // walk left
    auto fw_it = sorted_range_data.begin() + static_cast<int>(i);
    fw_it = advance_and_wrap(fw_it, sorted_range_data.begin(),
                             sorted_range_data.end());
    float angle = std::abs(angle_between(dir, fw_it->position.head<2>()));
    while (angle <= max_angle) {
      // assess candidate point
      const float distance_to_circumference =
          std::abs(radius - (fw_it->position.head<2>() - position).norm());

        if (distance_to_circumference < excl_radius) {  // consider tightening

          mse += distance_to_circumference;
          count++;
          if (fw_it->intensity > 0) intense_count++;
          circle_scan_points.push_back(static_cast<size_t>(
              std::distance(sorted_range_data.begin(), fw_it)));
        }

      fw_it = advance_and_wrap(fw_it, sorted_range_data.begin(),
                               sorted_range_data.end());
      angle = std::abs(angle_between(dir, fw_it->position.head<2>()));
    }

    // walk right
    auto bw_it = sorted_range_data.rbegin() +
                 static_cast<int>(sorted_range_data.size()) - 1 -
                 static_cast<int>(i);
    bw_it = advance_and_wrap(bw_it, sorted_range_data.rbegin(),
                             sorted_range_data.rend());
    angle = std::abs(angle_between(dir, bw_it->position.head<2>()));
    while (angle <= max_angle) {
      // assess candidate point
      const float distance_to_circumference =
          std::abs(radius - (bw_it->position.head<2>() - position).norm());

        if (distance_to_circumference < excl_radius) {  // consider tightening

          mse += distance_to_circumference;
          count++;
          if (bw_it->intensity > 0) intense_count++;
          circle_scan_points.push_back(static_cast<size_t>(
              std::distance(sorted_range_data.begin(), bw_it.base()) - 1));
        }

      bw_it = advance_and_wrap(bw_it, sorted_range_data.rbegin(),
                               sorted_range_data.rend());
      angle = std::abs(angle_between(dir, bw_it->position.head<2>()));
    }

    mse /= count;

    if (count > 2 && intense_count > 1 && mse < max_error) {
      circle.mse = mse;
      circle.count = count;
      circles.push_back(circle);
      circles_scan_points.push_back(circle_scan_points);
    }
  }

  if (circles.empty()) return {};

  std::vector<Circle<float>> clusters;

  auto ComputeCluster = [circles, circles_scan_points, radius,
                         sorted_range_data, excl_radius](const size_t start_i,
                                            const size_t end_i) {
    Circle<float> circle;
    circle.mse = 0;
    circle.radius = radius;
    circle.position = Eigen::Matrix<float, 2, 1>::Zero();
    std::set<size_t> included_points;
    for (size_t j = start_i; j <= end_i; ++j) {
      for (const size_t idx : circles_scan_points[j]) {
        included_points.insert(idx);
      }
    }
    if (start_i != end_i) {
      float weights_sum = 0.0;
      for (size_t j = start_i; j <= end_i; ++j) {
        const float weight = (1.f / circles[j].mse);
        circle.position += weight * circles[j].position;
        weights_sum += weight;
      }
      circle.position /= weights_sum;
    } else {
      circle.position = circles[start_i].position;
    }
    circle.count = 0;
    for (const size_t idx : included_points) {
      const auto scan_point = sorted_range_data[idx].position.head<2>();
      circle.points.push_back(scan_point);
      const float distance_to_circumference = std::abs(radius - (scan_point - circle.position).norm());
      if (distance_to_circumference < excl_radius) {
          circle.count++;
          circle.mse += distance_to_circumference;
      }
    }
    if (circle.count > 0)
        circle.mse /= static_cast<float>(included_points.size());
    else
        circle.mse = std::numeric_limits<float>::max();
    return circle;
  };

  // cluster groups of circles
  size_t start_i = 0;
  size_t end_i = 0;
  for (size_t i = 1; i < circles.size(); ++i) {
    // distance to start
    const float d = (circles[start_i].position - circles[i].position).norm();
    if (d < radius * 3.f) {
      end_i = i;
    } else {
      const auto c = ComputeCluster(start_i, end_i);
      if (c.mse < max_error)
          clusters.push_back(c);
      start_i = i;
      end_i = i;
    }
  }
  const auto c = ComputeCluster(start_i, end_i);
  if (c.mse < max_error)
      clusters.push_back(c);

  std::sort(clusters.begin(), clusters.end(), CircleSorter<float>());

  return clusters;
}

std::map<std::pair<int, int>, size_t> HoughCircles(const ProbabilityGrid& grid,
                                                   const double radius,
                                                   const size_t num_rotations) {
  std::map<std::pair<int, int>, size_t> accumulator;
  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));
  for (int j = 0; j < grid.limits().cell_limits().num_y_cells; ++j) {
    for (int i = 0; i < grid.limits().cell_limits().num_x_cells; ++i) {
      const auto cc =
          grid.correspondence_cost_cells().at(grid.ToFlatIndex({i, j}));
      if (cc < occupied_value && cc != 0) {
        for (size_t rot = 0; rot < num_rotations; ++rot) {
          const double theta = 2.0 * M_PI * static_cast<double>(rot) /
                               static_cast<double>(num_rotations);
          const int x = i - static_cast<int>(radius * std::cos(theta) /
                                                 grid.limits().resolution() +
                                             0.5);
          const int y = j - static_cast<int>(radius * std::sin(theta) /
                                                 grid.limits().resolution() +
                                             0.5);

          if (x > 0 && x < grid.limits().cell_limits().num_x_cells && y > 0 &&
              y < grid.limits().cell_limits().num_y_cells) {
            const std::pair<int, int> key = {x, y};
            const auto it = accumulator.find(key);
            if (it != accumulator.end()) {
              it->second++;
            } else {
              accumulator.insert({key, 0});
            }
          }
        }
      }
    }
  }
  return accumulator;
}

std::vector<DataSet::Cell> DBScan(const std::vector<DataSet::Cell>& cells,
                                  const double min_cluster_distance,
                                  const size_t min_cluster_size) {
  int cluster_counter = 0;
  static constexpr int UNDEFINED = -1;
  static constexpr int NOISE = -2;

  DataSet data_set;
  data_set.cells = cells;

  KDTree kdtree(2, data_set, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  kdtree.buildIndex();

  std::vector<DataSet::Cell> clusters;

  std::vector<int> labels(data_set.cells.size(), UNDEFINED);

  for (std::size_t i = 0; i < data_set.cells.size(); ++i) {
    const auto& cell = data_set.cells[i];

    if (labels[i] != UNDEFINED) {
      continue;
    }

    int query_pt[2] = {cell.x, cell.y};
    std::vector<std::pair<size_t, double>> ret_matches;
    const size_t num_matches =
        kdtree.radiusSearch(&query_pt[0], min_cluster_distance, ret_matches,
                            nanoflann::SearchParams());

    if (num_matches + 1 < static_cast<size_t>(min_cluster_size)) {
      LOG(INFO) << i << " is noise";
      labels[i] = NOISE;
      continue;
    }

    cluster_counter++;

    std::vector<DataSet::Cell> cluster;
    cluster.push_back(cell);

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

      cluster.push_back(data_set.cells[match.first]);

      {
        int sub_query_pt[2] = {kdtree.dataset.kdtree_get_pt(match.first, 0),
                               kdtree.dataset.kdtree_get_pt(match.first, 1)};
        std::vector<std::pair<size_t, double>> sub_ret_matches;
        const size_t sub_num_matches =
            kdtree.radiusSearch(&sub_query_pt[0], min_cluster_distance,
                                sub_ret_matches, nanoflann::SearchParams());

        if (sub_num_matches + 1 > min_cluster_size) {
          for (const auto& match : sub_ret_matches)
            ret_matches.push_back(match);
        }
      }
    }

    // determine the best scoring pose
    float max_score = 0;
    size_t max_p = 0;
    for (std::size_t p = 0; p < cluster.size(); ++p) {
      if (cluster[p].cost > max_score) {
        max_score = cluster[p].cost;
        max_p = p;
      }
    }

    clusters.push_back(cluster[max_p]);
  }

  return clusters;
}

std::vector<Eigen::Array2i> FilterByComponentSize(
    const std::vector<Eigen::Array2i>& cells, const ProbabilityGrid& grid,
    const int radius_in_cells, const size_t min_cell_count,
    const size_t max_cell_count) {
  static const std::vector<std::pair<int, int>> neighbours = {
      {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

  std::vector<Eigen::Array2i> components;

  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));
  for (size_t i = 0; i < cells.size(); ++i) {
    const auto& cell = cells[i];

    std::set<std::pair<int, int>> connected;

    std::queue<std::pair<int, int>> to_visit;

    // queue all cells inside the circle
    {
      const double r2 = static_cast<double>(radius_in_cells) *
                        static_cast<double>(radius_in_cells);

      const int x_min = std::max(0, cell.x() - radius_in_cells);
      const int y_min = std::max(0, cell.y() - radius_in_cells);

      const int x_limit = std::min(cell.x() + radius_in_cells,
                                   grid.limits().cell_limits().num_x_cells - 1);
      const int y_limit = std::min(cell.y() + radius_in_cells,
                                   grid.limits().cell_limits().num_y_cells - 1);

      for (int jj = y_min; jj <= y_limit; ++jj) {
        for (int ii = x_min; ii <= x_limit; ++ii) {
          const double _r2 = (ii - cell.x()) * (ii - cell.x()) +
                             (jj - cell.y()) * (jj - cell.y());
          if (_r2 <= r2) {
            const auto cc =
                grid.correspondence_cost_cells().at(grid.ToFlatIndex({ii, jj}));
            if (cc < occupied_value && cc != 0) {
              connected.insert({ii, jj});
              to_visit.push({ii, jj});
            }
          }
        }
      }
    }

    while (!to_visit.empty() && connected.size() <= max_cell_count) {
      for (const auto& n : neighbours) {
        const std::pair<int, int> c(to_visit.front().first + n.first,
                                    to_visit.front().second + n.second);
        if (connected.find(c) == connected.end()) {
          if (!grid.limits().Contains({c.first, c.second})) continue;

          const auto cc = grid.correspondence_cost_cells().at(
              grid.ToFlatIndex({c.first, c.second}));
          if (cc < occupied_value && cc != 0) {
            connected.insert(c);
            to_visit.push(c);
          }
        }
      }

      to_visit.pop();
    }

    if (connected.size() >= min_cell_count &&
        connected.size() < max_cell_count) {
      components.push_back({cell.x(), cell.y()});
    }
  }

  return components;
}

float CircumferenceCoverage(const int x, const int y,
                            const ProbabilityGrid& grid,
                            const int radius_in_cells) {
  const double r2 = static_cast<double>(radius_in_cells) *
                    static_cast<double>(radius_in_cells);
  int n = 0;
  float coverage = 0;
  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));

  const int x_min = std::max(0, x - radius_in_cells);
  const int y_min = std::max(0, y - radius_in_cells);

  const int x_limit = std::min(x + radius_in_cells,
                               grid.limits().cell_limits().num_x_cells - 1);
  const int y_limit = std::min(y + radius_in_cells,
                               grid.limits().cell_limits().num_y_cells - 1);

  for (int jj = y_min; jj <= y_limit; ++jj) {
    for (int ii = x_min; ii <= x_limit; ++ii) {
      const double _r2 = (ii - x) * (ii - x) + (jj - y) * (jj - y);
      if (std::abs(_r2 - r2) < 1.5) {
        const auto cc =
            grid.correspondence_cost_cells().at(grid.ToFlatIndex({ii, jj}));
        if (cc < occupied_value && cc != 0) coverage += 1.0;
        n++;
      }
    }
  }
  coverage /= n;
  return coverage;
}

std::vector<DataSet::Cell> ComputeKernel(const Eigen::MatrixXd& kernel,
                                         const ProbabilityGrid& grid,
                                         const float threshold) {
  CHECK(kernel.rows() == kernel.cols());

  const int k_size = static_cast<int>(kernel.rows());

  // must be an odd number
  CHECK(k_size % 2 == 1);

  const int half_k = k_size / 2;

  const int x_limit = grid.limits().cell_limits().num_x_cells - half_k;
  const int y_limit = grid.limits().cell_limits().num_y_cells - half_k;

  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));

  std::vector<DataSet::Cell> cells;
  for (int jj = half_k; jj < y_limit; ++jj) {
    for (int ii = half_k; ii < x_limit; ++ii) {
      float cost = 0;

      // evaluate kernel centered on (ii, jj)
      for (int k_jj = 0; k_jj < k_size; ++k_jj) {
        for (int k_ii = 0; k_ii < k_size; ++k_ii) {
          const int e_ii = ii - half_k + k_ii;
          const int e_jj = jj - half_k + k_jj;
          auto cc = grid.correspondence_cost_cells().at(
              grid.ToFlatIndex({e_ii, e_jj}));
          if (cc < occupied_value && cc != 0) {
            cost += kernel(k_jj, k_ii);
          }
        }
      }

      if (cost > threshold) {
        cells.push_back({ii, jj, cost});
      }
    }
  }

  return cells;
}

std::vector<Eigen::Array2i> DetectCircles(const ProbabilityGrid& grid,
                                          const double radius) {
  const int radius_in_cells =
      static_cast<int>(radius / grid.limits().resolution() + 0.5);

  int k_size = static_cast<int>(radius_in_cells * 2.5);
  if (k_size % 2 == 0) k_size++;

  Eigen::MatrixXd kernel(k_size, k_size);

  {
    const double k_centre = static_cast<double>(k_size) / 2.0 - 0.5;
    const double r2 = (radius / grid.limits().resolution()) *
                      (radius / grid.limits().resolution());
    for (int jj = 0; jj < k_size; ++jj) {
      for (int ii = 0; ii < k_size; ++ii) {
        const double w = static_cast<double>(ii) - k_centre;
        const double h = static_cast<double>(jj) - k_centre;
        const double _r2 = w * w + h * h;
        const double error = _r2 - r2;
        if (std::abs(error) <= 2.0) {
          kernel(jj, ii) = 1.0;
        } else if (std::abs(error) <= 3.5) {
          kernel(jj, ii) = 0.8;
        } else if (std::abs(error) <= 4.0) {
          kernel(jj, ii) = 0.5;
        } else if (error > 0) {
          kernel(jj, ii) = -0.1;
        } else {
          kernel(jj, ii) = -0.1;
        }
      }
    }
  }

  /*
  {
      auto surface =
  cairo_image_surface_create(cairo_format_t::CAIRO_FORMAT_ARGB32, k_size,
                                                k_size);

      uint32_t* pixel_data =
  reinterpret_cast<uint32_t*>(cairo_image_surface_get_data(surface)); for
  (size_t i=0; i<kernel.size(); ++i)
      {
          int prob;
          int miss = 0;
          if (kernel(i) > 0)
          {
              prob = static_cast<int>(kernel(i) * 255.0);
          }
          else
          {
              prob = 0;
              miss = 255;
          }
          pixel_data[i] = (255 << 24) | (prob << 16) | (miss << 8) | (0);
      }

      cairo_surface_write_to_png(surface, "test4.png");

      delete surface;
  }
  */

  const float cost_threshold = 0.25 * M_PI * 2.0 * radius_in_cells;
  const auto cells = ComputeKernel(kernel, grid, cost_threshold);

  {
    const double k_centre = static_cast<double>(k_size) / 2.0 - 0.5;
    const double r2 = (radius / grid.limits().resolution()) *
                      (radius / grid.limits().resolution());
    for (int jj = 0; jj < k_size; ++jj) {
      for (int ii = 0; ii < k_size; ++ii) {
        const double w = static_cast<double>(ii) - k_centre;
        const double h = static_cast<double>(jj) - k_centre;
        const double _r2 = w * w + h * h;
        const double error = _r2 - r2;
        if (error > 0) {
          kernel(jj, ii) = -0.1;
        } else {
          kernel(jj, ii) = 1.0;
        }
      }
    }
  }

  std::vector<DataSet::Cell> filtered_cells;

  const float free_space_cost_threshold =
      0.30 * M_PI * radius_in_cells * radius_in_cells;

  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));

  for (const auto& cell : cells) {
    // evaluate kernel centered on cell
    const int half_k = k_size / 2;
    float cost = 0;
    for (int k_jj = 0; k_jj < k_size; ++k_jj) {
      for (int k_ii = 0; k_ii < k_size; ++k_ii) {
        const int e_ii = cell.x - half_k + k_ii;
        const int e_jj = cell.y - half_k + k_jj;
        auto cc =
            grid.correspondence_cost_cells().at(grid.ToFlatIndex({e_ii, e_jj}));
        if (cc == 0 || cc == occupied_value) {
          cost += kernel(k_jj, k_ii);
        }
      }
    }

    if (cost > free_space_cost_threshold) {
      filtered_cells.push_back(cell);
    }
  }

  const double min_cluster_distance = radius_in_cells;
  const size_t min_cluster_size = 1;
  const auto clusters =
      DBScan(filtered_cells, min_cluster_distance, min_cluster_size);

  std::vector<Eigen::Array2i> cluster_cells;
  for (const auto& c : clusters) cluster_cells.push_back({c.x, c.y});

  const size_t max_cell_count =
      static_cast<size_t>(2 * radius_in_cells * radius_in_cells);
  const size_t min_cell_count = static_cast<size_t>(radius_in_cells);

  const auto components = FilterByComponentSize(
      cluster_cells, grid, radius_in_cells, min_cell_count, max_cell_count);

  return components;
}

/*
Circle FitCircleKasa(const Circle& circle)
{
    float x_hat = 0;
    float y_hat = 0;
    float z_hat = 0;

    float xx_hat = 0;
    float xy_hat = 0;
    float xz_hat = 0;

    float yy_hat = 0;
    float yz_hat = 0;

    for (size_t i = 0; i < circle.points.size(); ++i)
    {
        const float x = circle.points[i].x();//  - circle.position.x();
        const float y = circle.points[i].y(); // - circle.position.y();
        const float z = (x * x) + (y * y);

        x_hat += x;
        y_hat += y;
        z_hat += z;

        xx_hat += x * x;
        xy_hat += x * y;
        xz_hat += x * z;

        yy_hat += y * y;
        yz_hat += y * z;
    }

    x_hat /= circle.points.size();
    y_hat /= circle.points.size();
    z_hat /= circle.points.size();

    xx_hat /= circle.points.size();
    xy_hat /= circle.points.size();
    xz_hat /= circle.points.size();

    yy_hat /= circle.points.size();
    yz_hat /= circle.points.size();

    // let
    // z = (xi + yi)^2
    // B = -2a
    // C = -2b
    // D = a^2 + b^2 - R^2

    // solving linear equations
    // xx_hat * B + xy_hat * C + x_hat * D = -xz_hat
    // xy_hat * B + yy_hat * C + y_hat * D = -yz_hat
    // x_hat * B + y_hat * C + D = -z_hat

    // the radius is fixed therefore D can be solved up front
//    const float D = circle.position.x() * circle.position.x() +
circle.position.y() * circle.position.y() - circle.radius * circle.radius;

    Eigen::Matrix<float, 3, 3> A;
    A << xx_hat, xy_hat, x_hat,
         xy_hat, yy_hat, y_hat,
         x_hat, y_hat, 1.f;

    const float r2 = circle.radius * circle.radius;

    const Eigen::Matrix<float, 3, 1> b(-xz_hat + r2 * x_hat,
                                       -yz_hat + r2 * y_hat,
                                       -z_hat + r2);

    std::cout << "A: " << std::endl << A << std::endl;
    std::cout << "b: " << std::endl << b << std::endl;

    //    const Eigen::Matrix<float, 3, 1> b(-xz_hat - x_hat * D, -yz_hat -
y_hat * D, -z_hat - D);

    const Eigen::Matrix<float, 3, 1> x = A.lu().solve(b);
//A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    const auto b_check = A * x;

    std::cout << "b_check: " << std::endl << b_check << std::endl;

    const float a_fit = - x[0] / 2.f; // + circle.position.x();
    const float b_fit = - x[1] / 2.f; // + circle.position.y();
    const float D_check = std::sqrt(a_fit*a_fit + b_fit*b_fit);

    std::cout << "FIT: " << x.transpose() << std::endl;
    std::cout << "circle.x: " << circle.position.x() << " -> " << a_fit <<
std::endl; std::cout << "circle.y: " << circle.position.y() << " -> " << b_fit
<< std::endl; std::cout << "check D: " << x[2] << " -> " << D_check <<
std::endl;

    return Circle{0.f, static_cast<int>(circle.points.size()), circle.radius,
Eigen::Vector2f{a_fit, b_fit}, circle.points};
}

Circle CircleFitByKasa(const Circle& data)
{
    float Xi,Yi,Zi;
    float Mxy,Mxx,Myy,Mxz,Myz;
    float B,C,G11,G12,G22,D1,D2;

//     computing moments

    Mxx=0.f;
    Myy=0.f;
    Mxy=0.f;
    Mxz=0.f;
    Myz=0.f;

    for (int i=0; i<data.points.size(); i++)
    {
        Xi = data.points[i].x() - data.position.x();   //  centered
x-coordinates Yi = data.points[i].y() - data.position.y();   //  centered
y-coordinates Zi = Xi*Xi + Yi*Yi;

        Mxx += Xi*Xi;
        Myy += Yi*Yi;
        Mxy += Xi*Yi;
        Mxz += Xi*Zi;
        Myz += Yi*Zi;
    }
    Mxx /= data.points.size();
    Myy /= data.points.size();
    Mxy /= data.points.size();
    Mxz /= data.points.size();
    Myz /= data.points.size();

//    solving system of equations by Cholesky factorization

    G11 = std::sqrt(Mxx);
    G12 = Mxy/G11;
    G22 = std::sqrt(Myy - G12*G12);

    D1 = Mxz/G11;
    D2 = (Myz - D1*G12)/G22;

//    computing paramters of the fitting circle

    C = D2/G22/2.f;
    B = (D1 - G12*C)/G11/2.f;

//       assembling the output

    Circle circle;

    circle.position.x() = B + data.position.x();
    circle.position.y() = C + data.position.y();
    circle.radius = std::sqrt(B*B + C*C + Mxx + Myy);
//    circle.s = Sigma(data,circle);

    return circle;
}
*/

Circle<float> FitCircle(const Circle<float>& circle) {
  // Algorithm from https://people.cas.uab.edu/~mosya/papers/cl1.pdf
  // Algebraic optimisation of the circle position

  CHECK(!circle.points.empty());

  double a = static_cast<double>(circle.position.x());
  double b = static_cast<double>(circle.position.y());

  double A = calc_A(static_cast<double>(circle.radius));
  double B = calc_B(A, a);
  double C = calc_C(A, b);
  double D = calc_D(A, B, C);

  ceres::Problem problem;

  double d_max = 0;
  for (size_t i = 0; i < circle.points.size(); ++i) {
    auto cost_fn = new CircleCostFunction(circle.points[i]);
    auto auto_diff =
        new ceres::AutoDiffCostFunction<CircleCostFunction, 1, 1, 1, 1, 1>(
            cost_fn);
    problem.AddResidualBlock(auto_diff, nullptr, &A, &B, &C, &D);

    const double P =
        calc_P(A, B, C, D, static_cast<double>(circle.points[i][0]),
               static_cast<double>(circle.points[i][1]));
    const double d = std::abs(calc_d(P, A));
    if (d > d_max) {
      d_max = d;
    }
  }

  const double B_bound_0 = calc_B(A, a + d_max);
  const double B_bound_1 = calc_B(A, a - d_max);
  const double B_min = std::min(B_bound_0, B_bound_1);
  const double B_max = std::max(B_bound_0, B_bound_1);

  const double C_bound_0 = calc_C(A, b + d_max);
  const double C_bound_1 = calc_C(A, b - d_max);
  const double C_min = std::min(C_bound_0, C_bound_1);
  const double C_max = std::max(C_bound_0, C_bound_1);

  const double D_min = calc_D(A, std::min(std::abs(B_min), std::abs(B_max)),
                              std::min(std::abs(C_min), std::abs(C_max)));
  const double D_max = calc_D(A, std::max(std::abs(B_min), std::abs(B_max)),
                              std::max(std::abs(C_min), std::abs(C_max)));

  // don't optimise the radius
  problem.SetParameterBlockConstant(&A);

  problem.SetParameterLowerBound(&B, 0, B_min);
  problem.SetParameterUpperBound(&B, 0, B_max);

  problem.SetParameterLowerBound(&C, 0, C_min);
  problem.SetParameterUpperBound(&C, 0, C_max);

  problem.SetParameterLowerBound(&D, 0, D_min);
  problem.SetParameterUpperBound(&D, 0, D_max);

  ceres::Solver::Options options;
  options.use_nonmonotonic_steps = true;
  options.max_num_iterations = 100;
  options.num_threads = 1;
  options.linear_solver_type = ceres::DENSE_QR;

  ceres::Solver::Summary summary;

  ceres::Solve(options, &problem, &summary);

  a = calc_a(B, A);
  b = calc_b(C, A);

  return Circle<float>{static_cast<float>(summary.final_cost),
                       static_cast<int>(circle.points.size()), circle.radius,
                       Eigen::Vector2f{a, b}, circle.points};
}

}  // namespace mapping
}  // namespace cartographer
