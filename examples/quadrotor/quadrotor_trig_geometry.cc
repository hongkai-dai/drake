#include "drake/examples/quadrotor/quadrotor_trig_geometry.h"

#include <memory>

#include "drake/common/find_resource.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

using Eigen::Matrix3d;

namespace drake {
namespace examples {
namespace quadrotor {

const QuadrotorTrigGeometry* QuadrotorTrigGeometry::AddToBuilder(
    systems::DiagramBuilder<double>* builder,
    const systems::OutputPort<double>& quadrotor_state_port,
    const std::optional<std::string>& name,
    geometry::SceneGraph<double>* scene_graph) {
  DRAKE_THROW_UNLESS(builder != nullptr);
  DRAKE_THROW_UNLESS(scene_graph != nullptr);

  auto quadrotor_geometry =
      builder->AddSystem(std::unique_ptr<QuadrotorTrigGeometry>(
          new QuadrotorTrigGeometry(scene_graph, name)));
  builder->Connect(quadrotor_state_port, quadrotor_geometry->get_input_port(0));
  builder->Connect(
      quadrotor_geometry->get_output_port(0),
      scene_graph->get_source_pose_port(quadrotor_geometry->source_id_));

  return quadrotor_geometry;
}

QuadrotorTrigGeometry::QuadrotorTrigGeometry(
    geometry::SceneGraph<double>* scene_graph,
    const std::optional<std::string>& name) {
  DRAKE_THROW_UNLESS(scene_graph != nullptr);

  // Use (temporary) MultibodyPlant to parse the urdf and setup the
  // scene_graph.
  // TODO(SeanCurtis-TRI): Update this on resolution of #10775.
  multibody::MultibodyPlant<double> mbp(0.0);
  multibody::Parser parser(&mbp, scene_graph);

  auto model_id = parser.AddModelFromFile(
      FindResourceOrThrow("drake/examples/quadrotor/quadrotor.urdf"),
      name.value_or("quadrotor"));
  mbp.Finalize();

  source_id_ = *mbp.get_source_id();
  frame_id_ = mbp.GetBodyFrameIdOrThrow(mbp.GetBodyIndices(model_id)[0]);

  this->DeclareVectorInputPort("state", 13);
  this->DeclareAbstractOutputPort("geometry_pose",
                                  &QuadrotorTrigGeometry::OutputGeometryPose);
}

QuadrotorTrigGeometry::~QuadrotorTrigGeometry() = default;

void QuadrotorTrigGeometry::OutputGeometryPose(
    const systems::Context<double>& context,
    geometry::FramePoseVector<double>* poses) const {
  DRAKE_DEMAND(frame_id_.is_valid());

  const auto& state = get_input_port(0).Eval(context);
  const Eigen::Quaterniond quat(state(0) + 1, state(1), state(2), state(3));
  math::RigidTransformd pose(quat, state.segment<3>(4));

  *poses = {{frame_id_, pose}};
}

}  // namespace quadrotor
}  // namespace examples
}  // namespace drake
