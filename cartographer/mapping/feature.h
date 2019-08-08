#ifndef CARTOGRAPHER_MAPPING_FEATURE_H_
#define CARTOGRAPHER_MAPPING_FEATURE_H_

#include <memory>
#include <vector>

#include "Eigen/Geometry"
#include "cartographer/common/math.h"
#include "cartographer/mapping/proto/feature.pb.h"

namespace cartographer {
namespace mapping {

struct Keypoint {
  Eigen::Vector3f position;
  Eigen::Vector3f covariance;
};

struct CircleDescriptor {
  float score;
  float radius;
};

struct CircleFeature {
  Keypoint keypoint;
  CircleDescriptor fdescriptor;
};

proto::Keypoint ToProto(const Keypoint& feature);
Keypoint FromProto(const proto::Keypoint& proto);

proto::CircleDescriptor ToProto(const CircleDescriptor& feature);
CircleDescriptor FromProto(const proto::CircleDescriptor& proto);

proto::CircleFeature ToProto(const CircleFeature& feature);
CircleFeature FromProto(const proto::CircleFeature& proto);

}  // namespace mapping
}  // namespace cartographer

#endif  // CARTOGRAPHER_MAPPING_FEATURE_H_
