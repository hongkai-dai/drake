#include <memory>

#include <Eigen/Geometry>
#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sos_basis_generator.h"

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

  ConfigurationSpaceCollisionFreeRegion dut(*plant, link_polytopes,
                                            obstacle_boxes);

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  solvers::MathematicalProgram prog;
  //const double rho = 0.15;

  ConfigurationSpaceCollisionFreeRegion::VerificationOptions
      verification_options{};
  verification_options.link_polynomial_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSdsos;

  //dut.ConstructProgramToVerifyEllipsoidalFreeRegionAroundPosture(
  //    q_star, Eigen::VectorXd::Ones(7), rho, verification_options, &prog);

  verification_options.link_polynomial_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSos;
  verification_options.lagrangian_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSdsos;
  Eigen::VectorXd t_lower(7);
  t_lower << -0.07, -0.08, -0.09, -0.1, -0.11, -0.12, -0.13;
  Eigen::VectorXd t_upper = -t_lower;
  dut.ConstructProgramToVerifyBoxFreeRegionAroundPosture(
      q_star, t_lower, t_upper, verification_options, &prog);

  solvers::MosekSolver mosek_solver;
  mosek_solver.set_stream_logging(true, "");
  // solvers::ScsSolver scs_solver;
  // scs_solver.SetVerbose(true);
  std::cout << "Call Solve.\n";
  const auto result = mosek_solver.Solve(prog);
  std::cout << "Solution result: " << result << "\n";

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::DoMain();
}
