#include <memory>

#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/test/iiwa_two_boxes_demo_utilities.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
int DoMain() {
  IiwaTwoBoxesDemo demo{};

  //------------------------------------------------------
  // Now run this optimization using ConfigurationSpaceCollisionFreeRegion
  ConfigurationSpaceCollisionFreeRegion dut(*(demo.plant), demo.link_polytopes,
                                            demo.obstacle_boxes,
                                            SeparatingPlaneOrder::kConstant);
  const ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{};
  Eigen::VectorXd q_star(7);
  q_star << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  const auto link_vertex_rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs);

  auto prog = dut.ConstructProgramToVerifyCollisionFreeBox(
      link_vertex_rationals, Eigen::Matrix<double, 7, 1>::Constant(-0.15),
      Eigen::Matrix<double, 7, 1>::Constant(0.15), filtered_collision_pairs);

  solvers::MosekSolver mosek_solver;
  solvers::MathematicalProgramResult result;
  mosek_solver.Solve(*prog, {}, {}, &result);
  std::cout << result.get_solution_result() << "\n";
  auto a0_val =
      result.GetSolution(dut.separation_planes()[0].decision_variables);
  auto a1_val =
      result.GetSolution(dut.separation_planes()[1].decision_variables);
  std::cout << "a0: " << a0_val.transpose() << "\n";
  std::cout << "a1: " << a1_val.transpose() << "\n";

  const double best_rho = dut.FindLargestBoxThroughBinarySearch(
      q_star, filtered_collision_pairs, -Eigen::VectorXd::Ones(7),
      Eigen::VectorXd::Ones(7), 0.15, 0.5, 0.01);
  std::cout << "best rho is " << best_rho << ", corresponding to angle(deg) "
            << best_rho / M_PI * 180.0 << "\n";

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
