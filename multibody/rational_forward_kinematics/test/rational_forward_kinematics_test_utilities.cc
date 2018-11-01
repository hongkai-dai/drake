#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

#include "drake/common/find_resource.h"
#include "drake/multibody/benchmarks/kuka_iiwa_robot/make_kuka_iiwa_model.h"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"

namespace drake {
namespace multibody {
std::unique_ptr<multibody_plant::MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name) {
  const std::string file_path =
      "drake/manipulation/models/iiwa_description/sdf/" + iiwa_sdf_name;
  auto plant = std::make_unique<multibody_plant::MultibodyPlant<double>>(0);
  parsing::AddModelFromSdfFile(FindResourceOrThrow(file_path), plant.get());
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("iiwa_link_0"));
  plant->Finalize();
  return plant;
}

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(const Eigen::Vector3d& size,
                                                const Eigen::Isometry3d& pose) {
  Eigen::Matrix<double, 3, 8> vertices;
  // clang-format off
  vertices << 1, 1, 1, 1, -1, -1, -1, -1,
              1, 1, -1, -1, 1, 1, -1, -1,
              1, -1, 1, -1, 1, -1, 1, -1;
  // clang-format on
  for (int i = 0; i < 3; ++i) {
    DRAKE_ASSERT(size(i) > 0);
    vertices.row(i) *= size(i) / 2;
  }
  vertices = pose.linear() * vertices +
             pose.translation() * Eigen::Matrix<double, 1, 8>::Ones();

  return vertices;
}

}  // namespace multibody
}  // namespace drake
