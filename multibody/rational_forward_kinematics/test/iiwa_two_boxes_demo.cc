#include <memory>

#include <Eigen/Geometry>
#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/manipulation/util/simple_tree_visualizer.h"
#include "drake/multibody/multibody_tree/parsing/multibody_plant_sdf_parser.h"
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

  ConfigurationSpaceCollisionFreeRegion dut(plant->tree(), link_polytopes,
                                            obstacle_boxes);

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(dut.rational_forward_kinematics().t());
  const auto& a_hyperplane = dut.a_hyperplane();
  for (int i = 1; i < static_cast<int>(a_hyperplane.size()); ++i) {
    for (int j = 0; j < static_cast<int>(a_hyperplane[i].size()); ++j) {
      for (int k = 0; k < static_cast<int>(a_hyperplane[i][j].size()); ++k) {
        prog.AddDecisionVariables(a_hyperplane[i][j][k]);
      }
    }
  }

  const auto& obstacles_inside_halfspace =
      dut.GenerateObstacleInsideHalfspaceExpression();
  for (const auto& expr : obstacles_inside_halfspace) {
    prog.AddLinearConstraint(expr <= 0);
  }

  const auto& links_outside_halfspace =
      dut.GenerateLinkOutsideHalfspacePolynomials(q_star);
  const double rho = 0.001;
  const symbolic::Polynomial indeterminate_bound(
      rho - prog.indeterminates().cast<symbolic::Expression>().dot(
                prog.indeterminates()));
  const symbolic::Monomial monomial_one{};
  for (const auto& link_outside_halfspace : links_outside_halfspace) {
    // Create the Lagrangian multiplier
    const auto link_outside_halfspace_monomial_basis =
        solvers::ConstructMonomialBasis(link_outside_halfspace);
    const auto lagrangian_hessian = prog.NewSymmetricContinuousVariables(
        link_outside_halfspace_monomial_basis.rows());
    prog.AddPositiveSemidefiniteConstraint(lagrangian_hessian);
    MatrixX<symbolic::Polynomial> lagrangian_hessian_poly(
        lagrangian_hessian.rows(), lagrangian_hessian.cols());
    for (int i = 0; i < lagrangian_hessian.rows(); ++i) {
      for (int j = 0; j < lagrangian_hessian.cols(); ++j) {
        lagrangian_hessian_poly(i, j) =
            symbolic::Polynomial({{monomial_one, lagrangian_hessian(i, j)}});
      }
    }
    const VectorX<symbolic::Polynomial>
        link_outside_halfspace_monomial_basis_poly =
            link_outside_halfspace_monomial_basis.cast<symbolic::Polynomial>();
    const symbolic::Polynomial lagrangian =
        link_outside_halfspace_monomial_basis_poly.dot(
            lagrangian_hessian_poly *
            link_outside_halfspace_monomial_basis_poly);

    const symbolic::Polynomial p =
        link_outside_halfspace - lagrangian * indeterminate_bound;
    const auto monomial_basis = solvers::ConstructMonomialBasis(p);
    std::cout << "monomial_basis size: " << monomial_basis.size() << "\n";
    prog.AddSosConstraint(p, monomial_basis);
  }

  // solvers::MosekSolver mosek_solver;
  // mosek_solver.set_stream_logging(true, "");
  solvers::ScsSolver scs_solver;
  scs_solver.SetVerbose(true);
  std::cout << "Call Solve.\n";
  const auto result = scs_solver.Solve(prog);
  std::cout << "Solution result: " << result << "\n";

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::DoMain();
}
