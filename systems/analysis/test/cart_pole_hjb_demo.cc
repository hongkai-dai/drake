#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/hjb.h"
#include "drake/systems/analysis/test/cart_pole.h"

namespace drake {
namespace systems {
namespace analysis {

int DoMain() {
  const CartPoleParams params;
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }

  // First synthesize an LQR controller.
  Eigen::Matrix<double, 5, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10;
  const Eigen::Matrix<double, 5, 5> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = SynthesizeCartpoleTrigLqr(params, lqr_Q, 10);
  const Vector1<symbolic::Polynomial> u_lqr(symbolic::Polynomial(
      -lqr_result.K.row(0).dot(x), symbolic::Variables(x)));

  // Now construct HJB upper bound
  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial dynamics_numerator;
  TrigPolyDynamics(params, x, &f, &G, &dynamics_numerator);
  Eigen::Matrix<double, 1, 1> R(10);
  const symbolic::Polynomial l(x.cast<symbolic::Expression>().dot(lqr_Q * x));
  const Vector1<symbolic::Polynomial> state_constraints(CartpoleStateEqConstraint(x));
  const HjbUpper dut(x, l, R, f, G, dynamics_numerator, state_constraints);
  Eigen::MatrixXd state_samples =
      Eigen::MatrixXd::Random(4, 1000) * 0.1 + Eigen::Vector4d(0, M_PI, 0, 0);
  Eigen::Matrix<double, 5, Eigen::Dynamic> x_samples(5, state_samples.cols());
  for (int i = 0; i < state_samples.cols(); ++i) {
    x_samples.col(i) = ToCartpoleTrigState<double>(state_samples.col(i));
  }
  int J_degree = 2;
  Vector1<symbolic::Polynomial> cin(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.01));
  const std::vector<int> r_degrees{{6}};
  const std::vector<int> state_constraints_lagrangian_degrees{{6}};
  symbolic::Polynomial J;
  VectorX<symbolic::Polynomial> r;
  VectorX<symbolic::Polynomial> state_constraints_lagrangian;

  Eigen::Matrix<symbolic::Polynomial, 1, 1> policy_numerator =
      u_lqr * dynamics_numerator;
  int iter_count = 0;
  const int iter_max = 10;
  symbolic::Polynomial J_sol;
  while (iter_count < iter_max) {
    auto prog = dut.ConstructJupperProgram(
        J_degree, x_samples, policy_numerator, cin, r_degrees,
        state_constraints_lagrangian_degrees, &J, &r,
        &state_constraints_lagrangian);

    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
    DRAKE_DEMAND(result.is_success());
    J_sol = result.GetSolution(J);
    policy_numerator = dut.ComputePolicyNumerator(J_sol);
    iter_count++;
  }

  Save(J_sol, "cart_pole_trig_hjb.txt");

  return 0;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() { return drake::systems::analysis::DoMain(); }
