#include "cartographer/mapping/internal/2d/scan_matching/nearest_neighbour_cost_function_2d.h"

#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/probability_values.h"
#include "ceres/cubic_interpolation.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

CellIndex CreateCellIndexForGrid(const Grid2D& grid) {
  CellIndex res;
  const auto occupied_value = ProbabilityToCorrespondenceCost(0.8f);

  auto pg = dynamic_cast<const ProbabilityGrid&>(grid);

  const auto limits = grid.limits();
  for (int y = 0; y < limits.cell_limits().num_y_cells; ++y) {
    for (int x = 0; x < limits.cell_limits().num_x_cells; ++x) {
      if (grid.GetCorrespondenceCost({x, y}) < occupied_value) {
        res.cells.cells.emplace_back(DataSet<int>::Cell{x, y});
      }
    }
  }

  std::cout << "kdtree size: " << res.cells.cells.size() << " of "
            << grid.limits().cell_limits().num_x_cells *
                   grid.limits().cell_limits().num_y_cells
            << std::endl;

  res.kdtree.reset(new CellKDTree(
      2, res.cells, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  res.kdtree->buildIndex();
  return res;
}

RealIndex CreateRealIndexForGrid(const Grid2D& grid) {
  RealIndex res;
  const auto occupied_value = ProbabilityToCorrespondenceCost(0.5f);

  auto pg = dynamic_cast<const ProbabilityGrid&>(grid);

  const auto limits = grid.limits();
  for (int y = 0; y < limits.cell_limits().num_y_cells; ++y) {
    for (int x = 0; x < limits.cell_limits().num_x_cells; ++x) {
      if (grid.GetCorrespondenceCost({x, y}) < occupied_value) {
        const auto real_p = grid.limits().GetCellCenter({x, y});
        res.cells.cells.emplace_back(
            DataSet<double>::Cell{real_p.x(), real_p.y()});
      }
    }
  }

  res.kdtree.reset(new RealKDTree(
      2, res.cells, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
  res.kdtree->buildIndex();
  return res;
}

ceres::CostFunction* CreateNearestNeighbourCostFunction2D(
    const double scaling_factor, const sensor::PointCloud& point_cloud,
    const MapLimits& limits, const RealKDTree& kdtree) {
  auto cost_fn = new NearestNeighbourCostFunction2D(scaling_factor, point_cloud,
                                                    limits, kdtree);
  return new ceres::NumericDiffCostFunction<NearestNeighbourCostFunction2D,
                                            ceres::RIDDERS, ceres::DYNAMIC, 3>(
      cost_fn, ceres::TAKE_OWNERSHIP, point_cloud.size());
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
