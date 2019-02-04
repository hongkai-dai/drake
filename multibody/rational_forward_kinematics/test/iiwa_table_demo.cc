#include <memory>

#include "drake/common/find_resource.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram.h"
#include "drake/systems/framework/vector_system.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/rendering/multibody_position_to_geometry_pose.h"

namespace drake {
namespace multibody {

int DoMain() {
  const std::string file_path =
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf";
  auto plant = std::make_unique<MultibodyPlant<double>>();
  auto scene_graph = std::make_unique<geometry::SceneGraph<double>>();
  plant->RegisterAsSourceForSceneGraph(scene_graph.get());
  Parser(plant.get()).AddModelFromFile(FindResourceOrThrow(file_path));
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("iiwa_link_0"));

  plant->Finalize(scene_graph.get());

  MultibodyPlantVisualizer visualizer(*plant, std::move(scene_graph));
  Eigen::Matrix<double, 7, 1> q;
  q << 0.4, 0.4, 0.4, -0.4, 0.4, 0.4, 0.4;
  visualizer.VisualizePosture(q);
  return 0;
}

}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
