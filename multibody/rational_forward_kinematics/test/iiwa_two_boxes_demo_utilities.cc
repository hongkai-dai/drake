#include "drake/multibody/rational_forward_kinematics/test/iiwa_two_boxes_demo_utilities.h"

namespace drake {
namespace multibody {
IiwaTwoBoxesDemo::IiwaTwoBoxesDemo()
    : plant(ConstructIiwaPlant("iiwa14_no_collision.sdf")) {
  for (int i = 0; i <= 7; ++i) {
    iiwa_link[i] =
        plant->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
  }
  world = plant->world_body().index();
  link_polytopes = GenerateIiwaLinkPolytopes(*plant);
  // Add obstacles (two boxes) to the world.
  Eigen::Isometry3d box0_pose = Eigen::Isometry3d::Identity();
  box0_pose.translation() << -0.5, 0, 0.5;
  Eigen::Isometry3d box1_pose = Eigen::Isometry3d::Identity();
  box1_pose.translation() << 0.5, 0, 0.5;
  obstacle_boxes.emplace_back(std::make_shared<const ConvexPolytope>(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box0_pose)));
  obstacle_boxes.emplace_back(std::make_shared<const ConvexPolytope>(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box1_pose)));
}

void IiwaTwoBoxesDemo::ComputeBoxPosition(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    Eigen::Vector3d* p_4C0_star, Eigen::Vector3d* p_4C1_star) const {
  auto context = plant->CreateDefaultContext();
  plant->SetPositions(context.get(), q_star);
  // The position of box1's center C1 in link4's frame at q_star
  plant->CalcPointsPositions(
      *context, plant->get_body(world).body_frame(), obstacle_boxes[0]->p_BC(),
      plant->get_body(iiwa_link[4]).body_frame(), p_4C0_star);
  plant->CalcPointsPositions(
      *context, plant->get_body(world).body_frame(), obstacle_boxes[1]->p_BC(),
      plant->get_body(iiwa_link[4]).body_frame(), p_4C1_star);
}
}  // namespace multibody
}  // namespace drake
