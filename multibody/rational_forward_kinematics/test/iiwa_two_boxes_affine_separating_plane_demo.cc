#include <memory>

#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/test/iiwa_two_boxes_demo_utilities.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
int DoMain() {
  IiwaTwoBoxesDemo demo{};

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  // The position of box1's center C1 in link4's frame at q_star
  Eigen::Vector3d p_4C0_star, p_4C1_star;
  demo.ComputeBoxPosition(q_star, &p_4C0_star, &p_4C1_star);

  solvers::MathematicalProgram prog;
  RationalForwardKinematics rational_forward_kinematics(*(demo.plant));
  prog.AddIndeterminates(rational_forward_kinematics.t());
  const symbolic::Variables t_variables{rational_forward_kinematics.t()};
  // Hyperplane a0.dot(x - c0) = 1 separates link 7 from obstacle_boxes[0]
  // Hyperplane a1.dot(x - c1) = 1 separates link 7 from obstacle_boxes[1]
  auto A0 = prog.NewContinuousVariables<3, 7>("A0");
  auto A1 = prog.NewContinuousVariables<3, 7>("A1");
  auto b0 = prog.NewContinuousVariables<3>("b0");
  auto b1 = prog.NewContinuousVariables<3>("b1");
  Vector3<symbolic::Expression> a0, a1;
  const symbolic::Monomial monomial_one{};
  for (int i = 0; i < 3; ++i) {
    a0(i) = (A0.row(i) * rational_forward_kinematics.t())(0) + b0(i);
    a1(i) = (A1.row(i) * rational_forward_kinematics.t())(0) + b1(i);
  }

  Eigen::Matrix<double, 7, 1> t_upper, t_lower;
  t_upper = Eigen::Matrix<double, 7, 1>::Constant(0.2);
  t_lower = -t_upper;
  Eigen::Matrix<symbolic::Polynomial, 7, 1> t_minus_t_lower, t_upper_minus_t;
  for (int i = 0; i < 7; ++i) {
    const symbolic::Monomial ti_monomial(rational_forward_kinematics.t()(i));
    t_minus_t_lower(i) =
        symbolic::Polynomial({{ti_monomial, 1}, {monomial_one, -t_lower(i)}});
    t_upper_minus_t(i) =
        symbolic::Polynomial({{monomial_one, t_upper(i)}, {ti_monomial, -1}});
  }

  const BodyIndex expressed_body_index = demo.iiwa_link[4];
  const VectorX<symbolic::Monomial> link4_to_7_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(
          symbolic::Variables{rational_forward_kinematics.FindTOnPath(
              demo.iiwa_link[4], demo.iiwa_link[7])});
  const VectorX<symbolic::Monomial> world_to_link4_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(
          symbolic::Variables{rational_forward_kinematics.FindTOnPath(
              demo.world, demo.iiwa_link[4])});

  SeparatingPlaneOrder a_order = SeparatingPlaneOrder::kAffine;
  const auto X_47 =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, demo.link_polytopes[0]->body_index(), expressed_body_index);
  const std::vector<LinkVertexOnPlaneSideRational>
      link7_on_positive_side_a0_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, demo.link_polytopes[0],
              demo.obstacle_boxes[0], X_47, a0, p_4C0_star,
              PlaneSide::kPositive, a_order);
  for (const auto& rational : link7_on_positive_side_a0_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        link4_to_7_monomial_basis);
  }

  const auto X_4W =
      rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
          q_star, demo.obstacle_boxes[0]->body_index(), expressed_body_index);
  const std::vector<LinkVertexOnPlaneSideRational>
      box0_on_negative_side_a0_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, demo.obstacle_boxes[0],
              demo.link_polytopes[0], X_4W, a0, p_4C0_star,
              PlaneSide::kNegative, a_order);
  for (const auto& rational : box0_on_negative_side_a0_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        world_to_link4_monomial_basis);
  }

  const std::vector<LinkVertexOnPlaneSideRational>
      link7_on_positive_side_a1_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, demo.link_polytopes[0],
              demo.obstacle_boxes[1], X_47, a1, p_4C1_star,
              PlaneSide::kPositive, a_order);
  for (const auto& rational : link7_on_positive_side_a1_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        link4_to_7_monomial_basis);
  }

  const std::vector<LinkVertexOnPlaneSideRational>
      box1_on_negative_side_a1_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, demo.obstacle_boxes[1],
              demo.link_polytopes[0], X_4W, a1, p_4C1_star,
              PlaneSide::kNegative, a_order);
  for (const auto& rational : box1_on_negative_side_a1_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational.rational, t_minus_t_lower, t_upper_minus_t,
        world_to_link4_monomial_basis);
  }

  solvers::MosekSolver mosek_solver;
  mosek_solver.set_stream_logging(true, "");
  solvers::MathematicalProgramResult result;
  mosek_solver.Solve(prog, {}, {}, &result);
  std::cout << result.get_solution_result() << "\n";
  Eigen::MatrixXd A0_val = result.GetSolution(A0);
  Eigen::MatrixXd A1_val = result.GetSolution(A1);
  Eigen::Vector3d b0_val = result.GetSolution(b0);
  Eigen::Vector3d b1_val = result.GetSolution(b1);
  std::cout << "A0:\n " << A0_val << "\n";
  std::cout << "b0: " << b0_val.transpose() << "\n";
  std::cout << "A1:\n " << A1_val << "\n";
  std::cout << "b1: " << b1_val.transpose() << "\n";

  //------------------------------------------------------
  // Now run this optimization using ConfigurationSpaceCollisionFreeRegion
  ConfigurationSpaceCollisionFreeRegion dut(*(demo.plant), demo.link_polytopes,
                                            demo.obstacle_boxes,
                                            SeparatingPlaneOrder::kAffine);
  const ConfigurationSpaceCollisionFreeRegion::FilteredCollisionPairs
      filtered_collision_pairs{};
  const auto link_vertex_rationals = dut.GenerateLinkOnOneSideOfPlaneRationals(
      q_star, filtered_collision_pairs);

  auto prog2 = dut.ConstructProgramToVerifyCollisionFreeBox(
      link_vertex_rationals, Eigen::Matrix<double, 7, 1>::Constant(-0.2),
      Eigen::Matrix<double, 7, 1>::Constant(0.2), filtered_collision_pairs);

  mosek_solver.Solve(*prog2, {}, {}, &result);
  std::cout << result.get_solution_result() << "\n";
  A0_val = result.GetSolution(
      dut.separation_planes()[0].decision_variables.head<21>());
  A0_val.resize(3, 7);
  A1_val = result.GetSolution(
      dut.separation_planes()[1].decision_variables.head<21>());
  A1_val.resize(3, 7);
  std::cout << A0_val << "\n";
  std::cout << A1_val << "\n";
  b0_val = result.GetSolution(
      dut.separation_planes()[0].decision_variables.tail<3>());
  b1_val = result.GetSolution(
      dut.separation_planes()[1].decision_variables.tail<3>());
  std::cout << b0_val.transpose() << "\n";
  std::cout << b1_val.transpose() << "\n";

  const double best_rho = dut.FindLargestBoxThroughBinarySearch(
      q_star, filtered_collision_pairs, -Eigen::VectorXd::Ones(7),
      Eigen::VectorXd::Ones(7), 0.2, 0.5, 0.01);
  std::cout << "best rho is " << best_rho << ", corresponding to angle(deg) "
            << best_rho / M_PI * 180.0 << "\n";
  return 1;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
