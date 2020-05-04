#ifndef CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_RAY_TRACE_H_
#define CARTOGRAPHER_MAPPING_INTERNAL_2D_SCAN_MATCHING_RAY_TRACE_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/mapping/2d/probability_grid.h"

namespace cartographer {
namespace mapping {
namespace scan_matching {

struct Point {
  int x;
  int y;
};

inline int sign(int x) { return x > 0 ? 1.0 : -1.0; }

inline Point bresenham2D(const ProbabilityGrid& grid, unsigned int abs_da,
                         unsigned int abs_db, int error_b, int offset_a,
                         int offset_b, unsigned int offset,
                         const unsigned int size_x, unsigned int max_length) {
  const auto occupied_value =
      CorrespondenceCostToValue(ProbabilityToCorrespondenceCost(0.5f));
  unsigned int end = std::min(max_length, abs_da);
  for (unsigned int i = 0; i < end; ++i) {
    const int my = offset / size_x;
    const int mx = offset - (my * size_x);

    const Eigen::Array2i cell{mx, my};
    if (!grid.limits().Contains(cell)) return {-1, -1};
    auto cc = grid.correspondence_cost_cells().at(grid.ToFlatIndex(cell));
    if (cc != kUnknownCorrespondenceValue && cc < occupied_value)
      return {mx, my};

    offset += offset_a;
    error_b += abs_db;
    if ((unsigned int)error_b >= abs_da) {
      offset += offset_b;
      error_b -= abs_da;
    }
  }
  return {-1, -1};
}

inline Point raytraceLine(const ProbabilityGrid& grid, const unsigned int x0,
                          const unsigned int y0, const unsigned int x1,
                          const unsigned int y1, const unsigned int size_x,
                          const unsigned int max_length = UINT_MAX) {
  int dx = x1 - x0;
  int dy = y1 - y0;

  unsigned int abs_dx = abs(dx);
  unsigned int abs_dy = abs(dy);

  int offset_dx = sign(dx);
  int offset_dy = sign(dy) * size_x;

  unsigned int offset = y0 * size_x + x0;

  // we need to chose how much to scale our dominant dimension, based on the
  // maximum length of the line
  double dist = hypot(dx, dy);
  double scale = (dist == 0.0) ? 1.0 : std::min(1.0, max_length / dist);

  // if x is dominant
  if (abs_dx >= abs_dy) {
    int error_y = abs_dx / 2;
    return bresenham2D(grid, abs_dx, abs_dy, error_y, offset_dx, offset_dy,
                       offset, size_x, (unsigned int)(scale * abs_dx));
  }

  // otherwise y is dominant
  int error_x = abs_dy / 2;
  return bresenham2D(grid, abs_dy, abs_dx, error_x, offset_dy, offset_dx,
                     offset, size_x, (unsigned int)(scale * abs_dy));
}

}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

#endif
