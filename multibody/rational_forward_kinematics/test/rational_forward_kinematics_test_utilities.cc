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

std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>
GenerateIiwaLinkPolytopes(const multibody_plant::MultibodyPlant<double>& iiwa) {
  std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope> link_polytopes;
  const int link7_idx = iiwa.GetBodyByName("iiwa_link_7").node_index();
  Eigen::Isometry3d link7_box_pose = Eigen::Isometry3d::Identity();
  link7_box_pose.translation() << 0, 0, 0.05;
  Eigen::Matrix<double, 3, 8> link7_pts =
      GenerateBoxVertices(Eigen::Vector3d(0.04, 0.14, 0.1), link7_box_pose);
  link_polytopes.emplace_back(link7_idx, link7_pts);
  return link_polytopes;
}

/*
void AddBoxToTree(RigidBodyTreed* tree,
                  const Eigen::Ref<const Eigen::Vector3d>& box_size,
                  const Eigen::Isometry3d& box_pose, const std::string& name,
                  const Eigen::Vector4d& color) {
  auto body = std::make_unique<RigidBody<double>>();
  body->set_name(name);
  body->set_mass(1.0);
  body->set_spatial_inertia(Matrix6<double>::Identity());

  const DrakeShapes::Box shape(box_size);
  const DrakeShapes::VisualElement visual_element(
      shape, Eigen::Isometry3d::Identity(), color);
  body->AddVisualElement(visual_element);

  auto joint = std::make_unique<FixedJoint>(name + "joint", box_pose);
  body->add_joint(&tree->world(), std::move(joint));

  tree->add_rigid_body(std::move(body));
}

Eigen::Matrix<double, 3, 8> BoxVertices(
    const Eigen::Ref<const Eigen::Vector3d>& box_size,
    const Eigen::Isometry3d& box_pose) {
  Eigen::Matrix<double, 3, 8> box_vertices;
  box_vertices.row(0) << 1, 1, 1, 1, -1, -1, -1, -1;
  box_vertices.row(1) << 1, 1, -1, -1, 1, 1, -1, -1;
  box_vertices.row(2) << 1, -1, 1, -1, 1, -1, 1, -1;
  for (int i = 0; i < 3; ++i) {
    box_vertices.row(i) *= box_size(i) / 2;
  }
  box_vertices = box_pose.linear() * box_vertices;
  for (int i = 0; i < 8; ++i) {
    box_vertices.col(i) += box_pose.translation();
  }
  return box_vertices;
}

void AddSphereToBody(RigidBodyTreed* tree, int link_idx,
                     const Eigen::Vector3d& pt, const std::string& name,
                     double radius) {
  auto sphere_body = std::make_unique<RigidBody<double>>();
  sphere_body->set_name(name);
  sphere_body->set_mass(0.001);
  sphere_body->set_spatial_inertia(Matrix6<double>::Identity() * 1E-8);

  const DrakeShapes::Sphere shape(radius);
  const Eigen::Vector4d material(1, 0, 0, 1);

  const DrakeShapes::VisualElement visual_element(
      shape, Eigen::Isometry3d::Identity(), material);

  sphere_body->AddVisualElement(visual_element);

  Eigen::Isometry3d joint_transform;
  joint_transform.linear() = Eigen::Matrix3d::Identity();
  joint_transform.translation() = pt;

  auto joint = std::make_unique<FixedJoint>(name + "_joint", joint_transform);
  auto link = tree->get_mutable_body(link_idx);
  sphere_body->add_joint(link, std::move(joint));
  tree->add_rigid_body(std::move(sphere_body));
}

std::unique_ptr<RigidBodyTreed> ConstructKukaRBT() {
  std::unique_ptr<RigidBodyTreed> rigid_body_tree =
      std::make_unique<RigidBodyTreed>();

  const std::string model_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_polytope_collision.urdf");

  const std::string table_path = FindResourceOrThrow(
      "drake/examples/kuka_iiwa_arm/models/table/"
      "extra_heavy_duty_table_surface_only_collision.sdf");

  const double kRobotBaseShiftX = -0.243716;
  const double kRobotBaseShiftY = -0.625087;
  auto table1_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_table", rigid_body_tree->get_mutable_body(0),
      Eigen::Vector3d(kRobotBaseShiftX, kRobotBaseShiftY, 0),
      Eigen::Vector3d::Zero());

  auto table2_frame = std::make_shared<RigidBodyFrame<double>>(
      "object_table", rigid_body_tree->get_mutable_body(0),
      Eigen::Vector3d(0.8 + kRobotBaseShiftX, kRobotBaseShiftY, 0),
      Eigen::Vector3d::Zero());

  parsers::sdf::AddModelInstancesFromSdfFile(
      table_path, drake::multibody::joints::kFixed, table1_frame,
      rigid_body_tree.get());

  parsers::sdf::AddModelInstancesFromSdfFile(
      table_path, drake::multibody::joints::kFixed, table2_frame,
      rigid_body_tree.get());

  const double kTableTopZInWorld = 0.736 + 0.057 / 2;
  const Eigen::Vector3d kRobotBase(kRobotBaseShiftX, kRobotBaseShiftY,
                                   kTableTopZInWorld);

  auto robot_base_frame = std::make_shared<RigidBodyFrame<double>>(
      "iiwa_base", rigid_body_tree->get_mutable_body(0), kRobotBase,
      Eigen::Vector3d::Zero());
  rigid_body_tree->addFrame(robot_base_frame);

  parsers::urdf::AddModelInstanceFromUrdfFile(
      model_path, drake::multibody::joints::kFixed, robot_base_frame,
      rigid_body_tree.get());

  auto iiwa_frame_ee = rigid_body_tree->findFrame("iiwa_frame_ee");
  const std::string schunk_path = FindResourceOrThrow(
      "drake/examples/schunk_wsg/models/schunk_wsg_50_fixed_joint.sdf");
  parsers::sdf::AddModelInstancesFromSdfFile(
      schunk_path, drake::multibody::joints::kFixed, iiwa_frame_ee,
      rigid_body_tree.get());
  return rigid_body_tree;
}

std::unique_ptr<RigidBodyTreed> ConstructSchunkGripperRBT() {
  std::unique_ptr<RigidBodyTreed> rigid_body_tree =
      std::make_unique<RigidBodyTreed>();
  const std::string schunk_path = FindResourceOrThrow(
      "drake/examples/schunk_wsg/models/schunk_wsg_50_fixed_joint.sdf");
  parsers::sdf::AddModelInstancesFromSdfFile(
      schunk_path, drake::multibody::joints::kQuaternion, nullptr,
      rigid_body_tree.get());
  return rigid_body_tree;
}*/

}  // namespace multibody
}  // namespace drake
