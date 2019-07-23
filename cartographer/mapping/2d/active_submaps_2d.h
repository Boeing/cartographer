#ifndef CARTOGRAPHER_MAPPING_2D_ACTIVE_SUBMAPS_2D_H_
#define CARTOGRAPHER_MAPPING_2D_ACTIVE_SUBMAPS_2D_H_

#include <memory>
#include <vector>

#include "Eigen/Core"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/map_limits.h"
#include "cartographer/mapping/proto/2d/submaps_options_2d.pb.h"
#include "cartographer/mapping/proto/serialization.pb.h"
#include "cartographer/mapping/proto/submap_visualization.pb.h"
#include "cartographer/mapping/range_data_inserter_interface.h"
#include "cartographer/mapping/submaps.h"
#include "cartographer/mapping/trajectory_node.h"
#include "cartographer/mapping/value_conversion_tables.h"
#include "cartographer/sensor/range_data.h"
#include "cartographer/transform/rigid_transform.h"

namespace cartographer {
namespace mapping {

// The first active submap will be created on the insertion of the first range
// data. Except during this initialization when no or only one single submap
// exists, there are always two submaps into which range data is inserted: an
// old submap that is used for matching, and a new one, which will be used for
// matching next, that is being initialized.
//
// Once a certain number of range data have been inserted, the new submap is
// considered initialized: the old submap is no longer changed, the "new" submap
// is now the "old" submap and is used for scan-to-map matching. Moreover, a
// "new" submap gets created. The "old" submap is forgotten by this object.
class ActiveSubmaps2D {
 public:
  explicit ActiveSubmaps2D(const proto::SubmapsOptions2D& options);

  ActiveSubmaps2D(const ActiveSubmaps2D&) = delete;
  ActiveSubmaps2D& operator=(const ActiveSubmaps2D&) = delete;

  // Inserts 'range_data' into the Submap collection.
  std::vector<std::shared_ptr<const Submap2D>> InsertRangeData(const sensor::RangeData& range_data);

  const std::vector<std::shared_ptr<Submap2D>>& submaps();
  std::vector<std::shared_ptr<const Submap2D>> submaps() const;

 private:
  std::unique_ptr<RangeDataInserterInterface> CreateRangeDataInserter();
  std::unique_ptr<GridInterface> CreateGrid(const Eigen::Vector2f& origin);
  void AddSubmap(const Eigen::Vector2f& origin);

  const proto::SubmapsOptions2D options_;
  std::vector<std::shared_ptr<Submap2D>> submaps_;
  std::unique_ptr<RangeDataInserterInterface> range_data_inserter_;
  ValueConversionTables conversion_tables_;
};

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_2D_SUBMAP_2D_H_
