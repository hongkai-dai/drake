// A demo that we can verify box region in configuration space being collison
// free. This demo runs slow since we express the link pose in the world frame.
#include <memory>

#include <Eigen/Geometry>
#include <gflags/gflags.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sos_basis_generator.h"

namespace drake {
namespace multibody {
int DoMain() {
  auto plant = ConstructIiwaPlant("iiwa14_no_collision.sdf");

  const std::vector<std::shared_ptr<const ConvexPolytope>> link_polytopes =
      GenerateIiwaLinkPolytopes(*plant);
  DRAKE_DEMAND(link_polytopes.size() == 1);

  Eigen::Isometry3d box1_pose = Eigen::Isometry3d::Identity();
  box1_pose.translation() << -0.5, 0, 0.5;
  Eigen::Isometry3d box2_pose = Eigen::Isometry3d::Identity();
  box2_pose.translation() << 0.5, 0, 0.5;
  std::vector<ConvexPolytope> obstacle_boxes;
  const BodyIndex world = plant->world_body().index();
  obstacle_boxes.emplace_back(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box1_pose));
  obstacle_boxes.emplace_back(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box2_pose));

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  solvers::MathematicalProgram prog;
  // Hyperplane a0.dot(x - c0) = 1 separates link 7 from obstacle_boxes[0]
  // Hyperplane a1.dot(x - c1) = 1 separates link 7 from obstalce_boxes[1]
  auto a0 = prog.NewContinuousVariables<3>("a0");
  auto a1 = prog.NewContinuousVariables<3>("a1");

  RationalForwardKinematics rational_forward_kinematics(*plant);
  prog.AddIndeterminates(rational_forward_kinematics.t());

  Eigen::VectorXd t_lower(7);
  t_lower << -0.07, -0.08, -0.09, -0.1, -0.11, -0.12, -0.13;
  Eigen::VectorXd t_upper = -t_lower;
  Eigen::Matrix<symbolic::Polynomial, 7, 1> t_minus_t_lower, t_upper_minus_t;
  const symbolic::Monomial monomial_one{};
  for (int i = 0; i < 7; ++i) {
    const symbolic::Monomial ti_monomial(rational_forward_kinematics.t()(i));
    t_minus_t_lower(i) =
        symbolic::Polynomial({{ti_monomial, 1}, {monomial_one, -t_lower(i)}});
    t_upper_minus_t(i) =
        symbolic::Polynomial({{monomial_one, t_upper(i)}, {ti_monomial, -1}});
  }
  const VectorX<symbolic::Monomial> world_to_link7_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(
          symbolic::Variables{rational_forward_kinematics.t()});

  const BodyIndex expressed_body_index = world;
  VerificationOption verification_option;
  verification_option.lagrangian_type =
      solvers::MathematicalProgram::NonnegativePolynomial::kSdsos;
  // First compute the pose of link 7 in the world frame as a multilinear
  // polynomial.
  std::cout << "compute X_W7\n";
  const auto X_W7 =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, link_polytopes[0]->body_index(), expressed_body_index);
  std::cout << "compute link7_on_positive_side_a0_rational\n";
  const std::vector<LinkVertexOnPlaneSideRational>
      link7_on_positive_side_a0_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, link_polytopes[0], X_W7, a0,
              obstacle_boxes[0].p_BC(), PlaneSide::kPositive);
  std::cout << "Add nonnegative polynomial constraint that link 7 is on the "
               "positive side of the plane between obstacle[0]\n";
  for (const auto& rational : link7_on_positive_side_a0_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        world_to_link7_monomial_basis, verification_option);
  }
  std::cout << "compute link7_on_positive_side_a1_rational\n";
  const std::vector<LinkVertexOnPlaneSideRational>
      link7_on_positive_side_a1_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, link_polytopes[0], X_W7, a1,
              obstacle_boxes[1].p_BC(), PlaneSide::kPositive);
  std::cout << "Add nonnegative polynomial constraint that link 7 is on the "
               "positive side of the plane between obstacle[1]\n";
  for (const auto& rational : link7_on_positive_side_a1_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        world_to_link7_monomial_basis, verification_option);
  }
  // Add constraint that the obstacle boxes are on the negative side of the
  // plane.
  obstacle_boxes[0].AddInsideHalfspaceConstraint(obstacle_boxes[0].p_BC(), a0,
                                                 &prog);
  obstacle_boxes[1].AddInsideHalfspaceConstraint(obstacle_boxes[1].p_BC(), a0,
                                                 &prog);

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
