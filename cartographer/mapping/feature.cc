#include "cartographer/mapping/feature.h"
#include "cartographer/transform/transform.h"

namespace cartographer {
namespace mapping {

proto::Keypoint ToProto(const Keypoint& feature)
{
    proto::Keypoint proto;
    *proto.mutable_position() = transform::ToProto(feature.position);
    return proto;
}

Keypoint FromProto(const proto::Keypoint& proto)
{
    return Keypoint{
        transform::ToEigen(proto.position())
    };
}

proto::CircleDescriptor ToProto(const CircleDescriptor& fdescriptor)
{
    proto::CircleDescriptor proto;
    proto.set_radius(fdescriptor.radius);
    proto.set_score(fdescriptor.score);
    return proto;
}

CircleDescriptor FromProto(const proto::CircleDescriptor& proto)
{
    CircleDescriptor fdescriptor;
    fdescriptor.radius = proto.radius();
    fdescriptor.score = proto.score();
    return fdescriptor;
}

proto::CircleFeature ToProto(const CircleFeature& feature)
{
    proto::CircleFeature proto;
    *proto.mutable_keypoint() = ToProto(feature.keypoint);
    *proto.mutable_fdescriptor() = ToProto(feature.fdescriptor);
    return proto;
}

CircleFeature FromProto(const proto::CircleFeature& proto)
{
    CircleFeature feature;
    feature.keypoint = FromProto(proto.keypoint());
    feature.fdescriptor = FromProto(proto.fdescriptor());
    return feature;
}

}  // namespace mapping
}  // namespace cartographer
