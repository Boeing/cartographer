#include "cartographer/mapping/internal/2d/scan_matching/nearest_neighbour_cost_function_2d.h"

#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/probability_values.h"
#include "ceres/cubic_interpolation.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

RealIndex CreateRealIndexForGrid(const Grid2D& grid) {
  RealIndex res;
  const auto occupied_value = ProbabilityToCorrespondenceCost(0.5f);

  // auto pg = dynamic_cast<const ProbabilityGrid&>(grid);

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

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
