#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/multibody/inverse_kinematics/inverse_kinematics.h"
#include "drake/multibody/inverse_kinematics/position_constraint.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/snopt_solver.h"

namespace drake {
namespace multibody {
// For the halfplane condition a.dot(r(t)) - s(t), generate the polynomial
// a.dot(r(t)) - s(t) - l_upper(t) * (t_upper - t) - l_lower(t) * (t - t_lower)
// which is non-negative.
// Also l_upper(t) and l_lower(t) are nonnegative polynomials.
void GenerateVerifiedPolynomialForSeparatingHalfplane(
    const VectorX<symbolic::Monomial>& monomial_basis,
    const Eigen::Matrix<symbolic::Polynomial, 7, 1>& t_upper_minus_t,
    const Eigen::Matrix<symbolic::Polynomial, 7, 1>& t_minus_t_lower,
    symbolic::Polynomial* verified_polynomial,
    solvers::MathematicalProgram* prog) {
  for (int i = 0; i < 7; ++i) {
    const symbolic::Polynomial lagrangian_ti_upper =
        prog->NewNonnegativePolynomial(
                monomial_basis,
                solvers::MathematicalProgram::NonnegativePolynomial::kSdsos)
            .first;
    const symbolic::Polynomial lagrangian_ti_lower =
        prog->NewNonnegativePolynomial(
                monomial_basis,
                solvers::MathematicalProgram::NonnegativePolynomial::kSdsos)
            .first;
    *verified_polynomial -= lagrangian_ti_lower * (t_minus_t_lower(i));
    *verified_polynomial -= lagrangian_ti_upper * (t_upper_minus_t(i));
  }
  const symbolic::Polynomial verified_polynomial_expected =
      prog->NewNonnegativePolynomial(
              monomial_basis,
              solvers::MathematicalProgram::NonnegativePolynomial::kSos)
          .first;
  const symbolic::Polynomial diff_poly{*verified_polynomial -
                                       verified_polynomial_expected};
  for (const auto& diff_poly_item : diff_poly.monomial_to_coefficient_map()) {
    prog->AddLinearEqualityConstraint(diff_poly_item.second == 0);
  }
}

int DoMain() {
  // Left IIWA base in the world frame.
  Eigen::Isometry3d X_WL = Eigen::Isometry3d::Identity();
  // Right IIWA base in the world frame.
  Eigen::Isometry3d X_WR = Eigen::Isometry3d::Identity();
  X_WR.translation() << 0.5, 0, 0;

  ModelInstanceIndex left_iiwa_instance, right_iiwa_instance;
  auto plant =
      ConstructDualArmIiwaPlant("iiwa14_no_collision.sdf", X_WL, X_WR,
                                &left_iiwa_instance, &right_iiwa_instance);
  EXPECT_EQ(plant->num_positions(), 14);

  RationalForwardKinematics rational_forward_kinematics(*plant);

  // Now register a polytope on the left iiwa link 7, and a polytope on the
  // right iiwa link 7.
  constexpr int kNumVerticesOnLink7Box = 8;
  Eigen::Matrix<double, 3, kNumVerticesOnLink7Box> link7_box;
  // clang-format off
  link7_box << 1, 1, 1, 1, -1, -1, -1, -1,
               1, 1, -1, -1, 1, 1, -1, -1,
               1, -1, 1, -1, 1, -1, 1, -1;
  // clang-format on
  link7_box.row(0) *= 0.05;
  link7_box.row(1) *= 0.05;
  link7_box.row(2) *= 0.05;

  //-------------------------------------------------------------
  // First find a collision free posture.
  InverseKinematics ik(*plant);

  const Frame<double>& left_link7 =
      plant->GetFrameByName("iiwa_link_7", left_iiwa_instance);
  const Frame<double>& right_link7 =
      plant->GetFrameByName("iiwa_link_7", right_iiwa_instance);
  const double kInf = std::numeric_limits<double>::infinity();
  for (int i = 0; i < link7_box.cols(); ++i) {
    ik.AddPositionConstraint(left_link7, link7_box.col(i), plant->world_frame(),
                             Eigen::Vector3d(-kInf, 0, -kInf),
                             Eigen::Vector3d(0.2, kInf, kInf));
    ik.AddPositionConstraint(
        right_link7, link7_box.col(i), plant->world_frame(),
        Eigen::Vector3d(0.3, 0, -kInf), Eigen::Vector3d::Constant(kInf));
  }
  ik.AddPositionConstraint(
      left_link7, Eigen::Vector3d::Zero(), plant->world_frame(),
      Eigen::Vector3d(0.05, -kInf, -kInf), Eigen::Vector3d::Constant(kInf));
  ik.AddPositionConstraint(
      right_link7, Eigen::Vector3d::Zero(), plant->world_frame(),
      Eigen::Vector3d::Constant(-kInf), Eigen::Vector3d(0.45, kInf, kInf));

  solvers::SnoptSolver snopt_solver;
  solvers::MathematicalProgramResult ik_result;
  Eigen::VectorXd ik_q_init = Eigen::VectorXd::Zero(14);
  snopt_solver.Solve(ik.prog(), ik_q_init, {}, &ik_result);

  Eigen::VectorXd q0 = ik.prog().GetSolution(ik.q(), ik_result);
  std::cout << "IK result: " << ik_result.get_solution_result() << "\n";
  //----------------------------------------------------------------

  const std::vector<RationalForwardKinematics::Pose<symbolic::Polynomial>>
      link_poses_poly =
          rational_forward_kinematics.CalcLinkPosesAsMultilinearPolynomials(
              q0, BodyIndex(0));

  auto context = plant->CreateDefaultContext();
  plant->SetPositions(context.get(), q0);
  Eigen::Vector3d p_W7r;
  plant->CalcPointsPositions(*context, right_link7, Eigen::Vector3d::Zero(),
                             plant->world_frame(), &p_W7r);

  // Compute the position of left IIWA link 7 vertices (Vl) position in the
  // world frame as multilinear polynomial on sin(theta) and cos(theta), same
  // for the right IIWA link 7 vertices (Vr)
  Eigen::Matrix<symbolic::Polynomial, 3, kNumVerticesOnLink7Box> p_WVl, p_WVr;
  for (int i = 0; i < kNumVerticesOnLink7Box; ++i) {
    p_WVl.col(i) =
        link_poses_poly[left_link7.body().index()].p_AB +
        link_poses_poly[left_link7.body().index()].R_AB * link7_box.col(i);
    p_WVr.col(i) =
        link_poses_poly[right_link7.body().index()].p_AB +
        link_poses_poly[right_link7.body().index()].R_AB * link7_box.col(i);
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < kNumVerticesOnLink7Box; ++j) {
      p_WVl(i, j) = p_WVl(i, j).RemoveTermsWithSmallCoefficients(1E-12);
      p_WVr(i, j) = p_WVr(i, j).RemoveTermsWithSmallCoefficients(1E-12);
    }
  }

  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(rational_forward_kinematics.t());
  // Now add the separating hyperplane normal a
  const auto a = prog.NewContinuousVariables<3>();

  const symbolic::Monomial monomial_one{};
  Vector3<symbolic::Polynomial> a_poly;
  for (int i = 0; i < 3; ++i) {
    a_poly(i) = symbolic::Polynomial({{monomial_one, a(i)}});
  }

  // Both left_iiwa_outside_halfplane_poly[i] and
  // right_iiwa_outside_halfplane_poly[i] should be non-negative when t is
  // inside the box.
  std::array<symbolic::Polynomial, kNumVerticesOnLink7Box>
      left_iiwa_outside_halfplane_poly;
  std::array<symbolic::Polynomial, kNumVerticesOnLink7Box>
      right_iiwa_inside_halfplane_poly;

  // left_iiwa_t contains the indeterminates t on the left IIWA. right_iiwa_t
  // contains the indeterminates t on the right IIWA.
  symbolic::Variables left_iiwa_t, right_iiwa_t;
  std::vector<int> left_iiwa_t_index, right_iiwa_t_index;
  left_iiwa_t_index.reserve(7);
  right_iiwa_t_index.reserve(7);
  for (int i = 0; i < 14; ++i) {
    const auto ti_model_instance =
        rational_forward_kinematics.map_t_to_mobilizer()
            .at(rational_forward_kinematics.t()(i).get_id())
            ->model_instance();
    if (ti_model_instance == left_iiwa_instance) {
      left_iiwa_t.insert(rational_forward_kinematics.t()(i));
      left_iiwa_t_index.emplace_back(i);
    } else {
      right_iiwa_t.insert(rational_forward_kinematics.t()(i));
      right_iiwa_t_index.emplace_back(i);
    }
  }

  Eigen::Matrix<double, 7, 1> t_right_iiwa_upper, t_right_iiwa_lower,
      t_left_iiwa_upper, t_left_iiwa_lower;
  t_right_iiwa_upper << 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04;
  t_right_iiwa_lower = -t_right_iiwa_upper;
  t_left_iiwa_upper << 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03;
  t_left_iiwa_lower = -t_left_iiwa_upper;

  Eigen::Matrix<symbolic::Polynomial, 7, 1> t_minus_t_lower_left_iiwa,
      t_minus_t_lower_right_iiwa, t_upper_minus_t_left_iiwa,
      t_upper_minus_t_right_iiwa;
  for (int i = 0; i < 7; ++i) {
    const symbolic::Variable& ti_left =
        rational_forward_kinematics.t()[left_iiwa_t_index[i]];
    const symbolic::Variable& ti_right =
        rational_forward_kinematics.t()[right_iiwa_t_index[i]];
    t_minus_t_lower_left_iiwa[i] =
        symbolic::Polynomial({{symbolic::Monomial(ti_left), 1.0},
                              {monomial_one, -t_left_iiwa_lower(i)}});
    t_minus_t_lower_right_iiwa[i] =
        symbolic::Polynomial({{symbolic::Monomial(ti_right), 1.0},
                              {monomial_one, -t_right_iiwa_lower(i)}});
    t_upper_minus_t_left_iiwa[i] =
        symbolic::Polynomial({{symbolic::Monomial(ti_left), -1.0},
                              {monomial_one, t_left_iiwa_upper(i)}});
    t_upper_minus_t_right_iiwa[i] =
        symbolic::Polynomial({{symbolic::Monomial(ti_right), -1.0},
                              {monomial_one, t_right_iiwa_upper(i)}});
  }

  const VectorX<symbolic::Monomial> left_iiwa_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(left_iiwa_t);
  const VectorX<symbolic::Monomial> right_iiwa_monomial_basis =
      GenerateMonomialBasisWithOrderUpToOne(right_iiwa_t);
  for (int i = 0; i < kNumVerticesOnLink7Box; ++i) {
    const symbolic::Polynomial left_iiwa_outside_halfplane_multilinear =
        a_poly.dot(p_WVl.col(i) - p_W7r) - 1;
    const symbolic::RationalFunction left_iiwa_outside_halfplane_rational =
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                left_iiwa_outside_halfplane_multilinear
                    .RemoveTermsWithSmallCoefficients(1E-12));
    left_iiwa_outside_halfplane_poly[i] =
        left_iiwa_outside_halfplane_rational.numerator();

    GenerateVerifiedPolynomialForSeparatingHalfplane(
        left_iiwa_monomial_basis, t_upper_minus_t_left_iiwa,
        t_minus_t_lower_left_iiwa, &(left_iiwa_outside_halfplane_poly[i]),
        &prog);
    std::cout << "left iiwa vertex " << i << "\n";

    const symbolic::Polynomial right_iiwa_inside_halfplane_multilinear =
        1 - a_poly.dot(p_WVr.col(i) - p_W7r);
    const symbolic::RationalFunction right_iiwa_inside_halfplane_rational =
        rational_forward_kinematics
            .ConvertMultilinearPolynomialToRationalFunction(
                right_iiwa_inside_halfplane_multilinear
                    .RemoveTermsWithSmallCoefficients(1e-12));
    right_iiwa_inside_halfplane_poly[i] =
        right_iiwa_inside_halfplane_rational.numerator();
    GenerateVerifiedPolynomialForSeparatingHalfplane(
        right_iiwa_monomial_basis, t_upper_minus_t_right_iiwa,
        t_minus_t_lower_right_iiwa, &(right_iiwa_inside_halfplane_poly[i]),
        &prog);
    std::cout << "right iiwa vertex " << i << "\n";
  }

  solvers::MosekSolver mosek_solver;
  mosek_solver.set_stream_logging(true, "");
  solvers::MathematicalProgramResult result;
  mosek_solver.Solve(prog, {}, {}, &result);
  std::cout << "verification result: " << result.get_solution_result() << "\n";
  const auto mosek_solver_details =
      result.get_solver_details().GetValue<solvers::MosekSolverDetails>();
  std::cout << "rescode: " << mosek_solver_details.rescode << "\n";
  std::cout << "solution status: " << mosek_solver_details.solution_status
            << "\n";
  if (result.get_solution_result() == solvers::SolutionResult::kSolutionFound) {
    std::cout << "a: " << prog.GetSolution(a, result).transpose();
  }

  return 0;
}
}  // namespace multibody
}  // namespace drake

int main() { return drake::multibody::DoMain(); }
