#include <math.h>

#include <iostream>
#include <limits>

#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/test/quadrotor2d.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
symbolic::Polynomial FindCbfInit(
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, int h_degree) {
  // I first synthesize an LQR controller, and find a Lyapunov function V for
  // this LQR controller, then -V + eps should satisfy the CBF condition within
  // a small neighbourhood.
  Eigen::VectorXd lqr_Q_diag(7);
  lqr_Q_diag << 1, 1, 1, 1, 10, 10, 10;
  const Eigen::MatrixXd lqr_Q = lqr_Q_diag.asDiagonal();
  const Eigen::Matrix2d lqr_R = 10 * Eigen::Matrix2d::Identity();
  const auto lqr_result = SynthesizeTrigLqr(lqr_Q, lqr_R);
  QuadrotorPlant<double> quadrotor2d;
  const double thrust_equilibrium = EquilibriumThrust(quadrotor2d);
  const Vector2<symbolic::Expression> u_lqr =
      -lqr_result.K * x +
      Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium);
  const Eigen::Matrix<symbolic::Expression, 7, 1> dynamics_expr =
      TrigDynamics<symbolic::Expression>(quadrotor2d,
                                         x.cast<symbolic::Expression>(), u_lqr);
  Eigen::Matrix<symbolic::Polynomial, 7, 1> dynamics;
  for (int i = 0; i < 7; ++i) {
    dynamics(i) = symbolic::Polynomial(dynamics_expr(i));
  }

  const double positivity_eps = 0.01;
  const int d = h_degree / 2;
  const double deriv_eps = 0.0001;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      StateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{h_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{
      {static_cast<int>(std::ceil((h_degree + 1) / 2.) * 2 - 2)}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{h_degree - 2};
  const std::vector<int> derivative_cin_lagrangian_degrees =
      derivative_ceq_lagrangian_degrees;
  symbolic::Polynomial V;
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian;
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian;
  VectorX<symbolic::Polynomial> derivative_cin_lagrangian;
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian;
  symbolic::Polynomial positivity_sos_condition;
  symbolic::Polynomial derivative_sos_condition;
  auto prog = FindCandidateRegionalLyapunov(
      x, dynamics, std::nullopt /* dynamics_denominator*/, h_degree,
      positivity_eps, d, deriv_eps, state_eq_constraints,
      positivity_ceq_lagrangian_degrees, derivative_ceq_lagrangian_degrees,
      state_ineq_constraints, positivity_cin_lagrangian_degrees,
      derivative_cin_lagrangian_degrees, &V, &positivity_cin_lagrangian,
      &positivity_ceq_lagrangian, &derivative_cin_lagrangian,
      &derivative_ceq_lagrangian, &positivity_sos_condition,
      &derivative_sos_condition);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(V);
  return -V_sol + 0.1;
}

symbolic::Polynomial Search(const QuadrotorPlant<double>& quadrotor,
                            const Eigen::Matrix<symbolic::Variable, 7, 1>& x) {
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const double thrust_max = 3 * thrust_equilibrium;
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));

  // The unsafe region is the ground and the ceiling.
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions(2);
  unsafe_regions[0].resize(1);
  unsafe_regions[0](0) = symbolic::Polynomial(x(1) + 0.3);
  unsafe_regions[1].resize(1);
  unsafe_regions[1](0) = symbolic::Polynomial(0.5 - x(1));

  const ControlBarrier dut(f, G, x, unsafe_regions, u_vertices,
                           state_constraints);

  const int h_degree = 2;
  symbolic::Polynomial h_init = FindCbfInit(x, h_degree);
  const double deriv_eps = 0.1;
  const int lambda0_degree = 4;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_state_constraints_lagrangian_degrees{h_degree -
                                                                   1};
  const std::vector<int> t_degree = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{h_degree - 2},
                                                   {h_degree - 2}};
  const std::vector<std::vector<int>>
      unsafe_state_constraints_lagrangian_degrees{{h_degree - 1},
                                                  {h_degree - 1}};
  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  ellipsoids.emplace_back(Eigen::Matrix<double, 7, 1>::Zero(),
                          Eigen::Matrix<double, 7, 7>::Identity(), 0, 0, 2,
                          0.01, h_degree - 2, std::vector<int>{{h_degree - 1}});

  const Eigen::Matrix<double, 7, 1> x_anchor =
      Eigen::Matrix<double, 7, 1>::Zero();
  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.barrier_step_solver_options = solvers::SolverOptions();
  search_options.barrier_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  symbolic::Polynomial h_sol;
  symbolic::Polynomial lambda0_sol;
  VectorX<symbolic::Polynomial> l_sol;
  VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian_sol;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  std::vector<VectorX<symbolic::Polynomial>>
      unsafe_state_constraints_lagrangian_sol;
  dut.Search(h_init, h_degree, deriv_eps, lambda0_degree, l_degrees,
             hdot_state_constraints_lagrangian_degrees, t_degree, s_degrees,
             unsafe_state_constraints_lagrangian_degrees, ellipsoids, x_anchor,
             search_options, &h_sol, &lambda0_sol, &l_sol,
             &hdot_state_constraints_lagrangian_sol, &t_sol, &s_sol,
             &unsafe_state_constraints_lagrangian_sol);
  std::cout << "h_sol: " << h_sol << "\n";

  Eigen::Matrix<double, 6, Eigen::Dynamic> state_samples(6, 5);
  state_samples.col(0) << 0.2, 0.2, 0, 0, 0, 0;
  state_samples.col(1) << 0.2, -0.3, 0, 0, 0, 0;
  state_samples.col(2) << 0, 0.4, 0.1 * M_PI, 0, 0, 0;
  state_samples.col(3) << 0, -0.3, 0.2 * M_PI, 0, 0, 0;
  state_samples.col(4) << 0, 0.2, 0.3 * M_PI, 0.1, 0.2, 0;
  Eigen::Matrix<double, 7, Eigen::Dynamic> x_samples(7, state_samples.cols());
  for (int i = 0; i < state_samples.cols(); ++i) {
    x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
  }
  std::cout << "h at samples: "
            << h_sol.EvaluateIndeterminates(x, x_samples).transpose() << "\n";

  return h_sol;
}

int DoMain() {
  const QuadrotorPlant<double> plant{};
  Eigen::Matrix<symbolic::Variable, 7, 1> x;
  for (int i = 0; i < 7; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const symbolic::Polynomial h_sol = Search(plant, x);
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
