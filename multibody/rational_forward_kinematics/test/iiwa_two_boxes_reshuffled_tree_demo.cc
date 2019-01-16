#include <memory>

#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
int DoMain() {
  auto plant = ConstructIiwaPlant("iiwa14_no_collision.sdf");

  const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>
      link_polytopes = GenerateIiwaLinkPolytopes(*plant);

  Eigen::Isometry3d box1_pose = Eigen::Isometry3d::Identity();
  box1_pose.translation() << -0.5, 0, 0.5;
  Eigen::Isometry3d box2_pose = Eigen::Isometry3d::Identity();
  box2_pose.translation() << 0.5, 0, 0.5;
  std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope> obstacle_boxes;
  obstacle_boxes.emplace_back(
      0, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box1_pose));
  obstacle_boxes.emplace_back(
      0, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box2_pose));

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  solvers::MathematicalProgram prog;
  auto t = prog.NewIndeterminates<7>("t");
  // Hyperplane a0.dot(x - c0) = 1 separates link 7 from obstacle_boxes[0]
  // Hyperplane a1.dot(x - c1) = 1 separates link 7 from obstacle_boxes[1]
  auto a0 = prog.NewContinuousVariables<3>("a0");
  auto a1 = prog.NewContinuousVariables<3>("a1");

  RationalForwardKinematics rational_forward_kin(*plant);

  const BodyIndex iiwa_link_3 = plant->GetBodyByName("iiwa_link_3").index();

  const std::vector<RationalForwardKinematics::Pose<symbolic::RationalFunction>>
      X_3B = rational_forward_kin.CalcLinkPoses(q_star, iiwa_link_3);
  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
