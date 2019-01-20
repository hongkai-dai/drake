#include <memory>

#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace multibody {
int DoMain() {
  auto plant = ConstructIiwaPlant("iiwa14_no_collision.sdf");

  const std::vector<ConvexPolytope> link_polytopes =
      GenerateIiwaLinkPolytopes(*plant);
  DRAKE_DEMAND(link_polytopes[0].body_index() ==
               plant->GetBodyByName("iiwa_link_7").index());

  // Add obstacles (two boxes) to the world.
  Eigen::Isometry3d box0_pose = Eigen::Isometry3d::Identity();
  box0_pose.translation() << -0.5, 0, 0.5;
  Eigen::Isometry3d box1_pose = Eigen::Isometry3d::Identity();
  box1_pose.translation() << 0.5, 0, 0.5;
  std::vector<ConvexPolytope> obstacle_boxes;
  const BodyIndex world = plant->world_body().index();
  obstacle_boxes.emplace_back(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box0_pose));
  obstacle_boxes.emplace_back(
      world, GenerateBoxVertices(Eigen::Vector3d(0.4, 0.6, 1), box1_pose));

  Eigen::VectorXd q_star(7);
  q_star.setZero();

  auto context = plant->CreateDefaultContext();
  plant->SetPositions(context.get(), q_star);
  const BodyIndex iiwa_link_3 = plant->GetBodyByName("iiwa_link_3").index();
  const BodyIndex iiwa_link_7 = plant->GetBodyByName("iiwa_link_7").index();
  // The position of box1's center C1 in link3's frame at q_star
  Eigen::Vector3d p_3C0_star, p_3C1_star;
  plant->CalcPointsPositions(
      *context, plant->get_body(world).body_frame(), box0_pose.translation(),
      plant->get_body(iiwa_link_3).body_frame(), &p_3C0_star);
  plant->CalcPointsPositions(
      *context, plant->get_body(world).body_frame(), box1_pose.translation(),
      plant->get_body(iiwa_link_3).body_frame(), &p_3C1_star);

  solvers::MathematicalProgram prog;
  // Hyperplane a0.dot(x - c0) = 1 separates link 7 from obstacle_boxes[0]
  // Hyperplane a1.dot(x - c1) = 1 separates link 7 from obstacle_boxes[1]
  auto a0 = prog.NewContinuousVariables<3>("a0");
  auto a1 = prog.NewContinuousVariables<3>("a1");

  RationalForwardKinematics rational_forward_kinematics(*plant);
  prog.AddIndeterminates(rational_forward_kinematics.t());

  Eigen::Matrix<double, 7, 1> t_upper, t_lower;
  t_upper << 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15;
  t_lower = -t_upper;
  Eigen::Matrix<symbolic::Polynomial, 7, 1> t_minus_t_lower, t_upper_minus_t;
  const symbolic::Monomial monomial_one{};
  for (int i = 0; i < 7; ++i) {
    const symbolic::Monomial ti_monomial(rational_forward_kinematics.t()(i));
    t_minus_t_lower(i) =
        symbolic::Polynomial({{ti_monomial, 1}, {monomial_one, -t_lower(i)}});
    t_upper_minus_t(i) =
        symbolic::Polynomial({{monomial_one, t_upper(i)}, {ti_monomial, -1}});
  }

  const BodyIndex expressed_body_index = iiwa_link_3;
  const VectorX<symbolic::Monomial> link3_to_7_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables{
          rational_forward_kinematics.FindTOnPath(iiwa_link_3, iiwa_link_7)});
  const VectorX<symbolic::Monomial> world_to_link3_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(symbolic::Variables{
          rational_forward_kinematics.FindTOnPath(world, iiwa_link_3)});

  const std::vector<symbolic::RationalFunction>
      link7_on_positive_side_a0_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, link_polytopes[0], q_star,
              expressed_body_index, a0, p_3C0_star, PlaneSide::kPositive);
  for (const auto& rational : link7_on_positive_side_a0_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational, t_minus_t_lower.tail<4>(), t_upper_minus_t.tail<4>(),
        link3_to_7_monomial_basis);
  }

  const std::vector<symbolic::RationalFunction>
      box0_on_negative_side_a0_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, obstacle_boxes[0], q_star,
              expressed_body_index, a0, p_3C0_star, PlaneSide::kNegative);
  for (const auto& rational : box0_on_negative_side_a0_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational, t_minus_t_lower.head<3>(), t_upper_minus_t.head<3>(),
        world_to_link3_monomial_basis);
  }

  const std::vector<symbolic::RationalFunction>
      link7_on_positive_side_a1_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, link_polytopes[0], q_star,
              expressed_body_index, a1, p_3C1_star, PlaneSide::kPositive);
  for (const auto& rational : link7_on_positive_side_a1_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational, t_minus_t_lower.tail<4>(), t_upper_minus_t.tail<4>(),
        link3_to_7_monomial_basis);
  }

  const std::vector<symbolic::RationalFunction>
      box1_on_negative_side_a1_rational =
          GenerateLinkOnOneSideOfPlaneRationalFunction(
              rational_forward_kinematics, obstacle_boxes[1], q_star,
              expressed_body_index, a1, p_3C1_star, PlaneSide::kNegative);
  for (const auto& rational : box1_on_negative_side_a1_rational) {
    AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
        &prog, rational, t_minus_t_lower.head<3>(), t_upper_minus_t.head<3>(),
        world_to_link3_monomial_basis);
  }

  solvers::MosekSolver mosek_solver;
  mosek_solver.set_stream_logging(true, "");
  solvers::MathematicalProgramResult result;
  mosek_solver.Solve(prog, {}, {}, &result);
  const Eigen::Vector3d a0_val = prog.GetSolution(a0, result);
  const Eigen::Vector3d a1_val = prog.GetSolution(a1, result);
  std::cout << "a0: " << a0_val.transpose() << "\n";
  std::cout << "a1: " << a1_val.transpose() << "\n";

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
