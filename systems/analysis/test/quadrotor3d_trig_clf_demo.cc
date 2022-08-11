#include <iostream>

#include "drake/math/gray_code.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/test/quadrotor.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
controllers::LinearQuadraticRegulatorResult SynthesizeTrigLqr() {
  const QuadrotorTrigPlant<double> quadrotor;
  auto context = quadrotor.CreateDefaultContext();
  context->SetContinuousState(Eigen::Matrix<double, 13, 1>::Zero());
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  quadrotor.get_input_port().FixValue(
      context.get(), Eigen::Vector4d::Ones() * thrust_equilibrium);
  auto linearized_quadrotor = Linearize(quadrotor, *context);
  Eigen::Matrix<double, 1, 13> F = Eigen::Matrix<double, 1, 13>::Zero();
  F(0) = 1;
  Eigen::Matrix<double, 13, 1> lqr_Q_diag;
  lqr_Q_diag.head<7>() = Eigen::VectorXd::Ones(7);
  lqr_Q_diag.tail<6>() = 10 * Eigen::VectorXd::Ones(6);
  const Eigen::Matrix<double, 13, 13> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      linearized_quadrotor->A(), linearized_quadrotor->B(), lqr_Q,
      10 * Eigen::Matrix4d::Identity(), Eigen::MatrixXd(0, 4), F);
  return lqr_result;
}

symbolic::Polynomial FindClfInit(
    int V_degree,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 13, 1>>& x) {
  QuadrotorTrigPlant<double> quadrotor;
  QuadrotorTrigPlant<symbolic::Expression> quadrotor_sym;
  const auto lqr_result = SynthesizeTrigLqr();
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const Vector4<symbolic::Expression> u_lqr =
      -lqr_result.K * x + Eigen::Vector4d::Ones() * thrust_equilibrium;
  auto context_sym = quadrotor_sym.CreateDefaultContext();
  context_sym->SetContinuousState(x.cast<symbolic::Expression>());
  quadrotor_sym.get_input_port().FixValue(context_sym.get(), u_lqr);
  const Eigen::Matrix<symbolic::Expression, 13, 1> dynamics_expr =
      quadrotor_sym.EvalTimeDerivatives(*context_sym).CopyToVector();
  Eigen::Matrix<symbolic::Polynomial, 13, 1> dynamics;
  for (int i = 0; i < 13; ++i) {
    dynamics(i) = symbolic::Polynomial(dynamics_expr(i));
    dynamics(i) = dynamics(i).RemoveTermsWithSmallCoefficients(1E-8);
  }

  const double positivity_eps = 0.0001;
  const int d = V_degree / 2;
  const double deriv_eps = 0.1;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      StateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{
      {static_cast<int>(std::ceil((V_degree + 3) / 2.) * 2 - 2)}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.01));
  const std::vector<int> positivity_cin_lagrangian_degrees{V_degree - 2};
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
      x, dynamics, std::nullopt /*dynamics_denominator*/, V_degree,
      positivity_eps, d, deriv_eps, state_eq_constraints,
      positivity_ceq_lagrangian_degrees, derivative_ceq_lagrangian_degrees,
      state_ineq_constraints, positivity_cin_lagrangian_degrees,
      derivative_cin_lagrangian_degrees, &V, &positivity_cin_lagrangian,
      &positivity_ceq_lagrangian, &derivative_cin_lagrangian,
      &derivative_ceq_lagrangian, &positivity_sos_condition,
      &derivative_sos_condition);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(V);
  return V_sol;
}

void SearchWTrigDynamics() {
  QuadrotorTrigPlant<double> quadrotor;
  Eigen::Matrix<symbolic::Variable, 13, 1> x;
  for (int i = 0; i < 13; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const symbolic::Variables x_set{x};
  Eigen::Matrix<symbolic::Polynomial, 13, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 13, 4> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const double thrust_max = 3 * thrust_equilibrium;
  const Eigen::Matrix<double, 4, 16> u_vertices =
      math::CalculateReflectedGrayCodes<4>().transpose().cast<double>() *
      thrust_max;
  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
  symbolic::Polynomial V_init;
  const int V_degree = 2;

  bool search_init = false;
  if (search_init) {
    drake::log()->info("Find initial Clf with LQR controller.");
    V_init = FindClfInit(V_degree, x);
    V_init = V_init.RemoveTermsWithSmallCoefficients(1E-6);
    Save(V_init, "quadrotor3d_trig_clf_regional.txt");
  } else {
    drake::log()->info("Load initial Clf with LQR controller.");
    V_init = Load(x_set, "quadrotor3d_trig_clf_regional.txt");
  }

  const ControlLyapunov dut(x, f, G, std::nullopt /* dynamics denominator */,
                            u_vertices, state_constraints);
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees(16, 2);
  const std::vector<int> p_degrees{{4}};
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p;
  const double deriv_eps = 0.1;
  bool maximize_init_rho = false;
  if (maximize_init_rho) {
    // Maximize rho such that V(x) <= rho defines a valid ROA.
    const int d_degree = lambda0_degree / 2 + 1;
    auto lagrangian_ret =
        dut.ConstructLagrangianProgram(V_init, symbolic::Polynomial(), d_degree,
                                       l_degrees, p_degrees, deriv_eps);
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    drake::log()->info("Maximize rho for the initial Clf");
    const auto result =
        solvers::Solve(*(lagrangian_ret.prog), std::nullopt, solver_options);
    DRAKE_DEMAND(result.is_success());
    const double rho_sol = result.GetSolution(lagrangian_ret.rho);
    std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
    V_init = V_init / rho_sol;
    Save(V_init, "quadrotor3d_trig_clf_max_rho.txt");
  } else {
    drake::log()->info("Load initial Clf.");
    V_init = Load(x_set, "quadrotor3d_trig_clf_max_rho.txt");
  }

  {
    ControlLyapunov::SearchOptions search_options;
    search_options.d_converge_tol = 0.;
    search_options.bilinear_iterations = 25;
    search_options.lyap_step_backoff_scale = 0.01;
    search_options.lsol_tiny_coeff_tol = 1E-8;
    search_options.lyap_tiny_coeff_tol = 1E-8;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    Eigen::MatrixXd state_samples(12, 4);
    state_samples.col(0) << 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    state_samples.col(1) << 1, 0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    state_samples.col(2) << 1, 0., 0, 0, M_PI / 2, 0, 0, 0, 0, 0, 0, 0;
    state_samples.col(3) << 1, 0., 0, M_PI / 2, 0, 0, 0, 0, 0, 0, 0, 0;
    Eigen::MatrixXd x_samples(13, state_samples.cols());
    for (int i = 0; i < state_samples.cols(); ++i) {
      x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
    }

    const double positivity_eps = 0.0001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    const bool minimize_max = true;
    SearchResultDetails search_result_details;
    const auto search_result =
        dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                   positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                   deriv_eps, x_samples, std::nullopt /* in_roa_samples */,
                   minimize_max, search_options);
    std::cout
        << "V(x_samples): "
        << search_result.V.EvaluateIndeterminates(x, x_samples).transpose()
        << "\n";
    Save(search_result.V, "quadrotor3d_trig_clf_sol.txt");
  }
}

int DoMain() {
  SearchWTrigDynamics();
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() { return drake::systems::analysis::DoMain(); }
