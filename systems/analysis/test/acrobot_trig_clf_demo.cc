#include "examples/acrobot/gen/acrobot_params.h"

#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/test/acrobot.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
controllers::LinearQuadraticRegulatorResult SynthesizeTrigLqr(
    const examples::acrobot::AcrobotParams<double>& p) {
  const Eigen::Matrix<double, 7, 1> xu_des =
      Eigen::Matrix<double, 7, 1>::Zero();
  const auto xu_des_ad = math::InitializeAutoDiff(xu_des);
  Vector6<AutoDiffXd> n;
  AutoDiffXd d;
  TrigDynamics<AutoDiffXd>(p, xu_des_ad.head<6>(), xu_des_ad(6), &n, &d);
  const Vector6<AutoDiffXd> xdot_des_ad = n / d;
  const auto xdot_des_grad = math::ExtractGradient(xdot_des_ad);
  // The constraints are x(0) * x(0) + (x(1) + 1) * (x(1) + 1) = 1
  // and x(2) * x(2) + (x(3) + 1) * (x(3) + 1) = 1
  Eigen::Matrix<double, 2, 6> F = Eigen::Matrix<double, 2, 6>::Zero();
  F(0, 1) = 1;
  F(1, 3) = 1;
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 1, 10, 10;
  const Matrix6<double> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      xdot_des_grad.leftCols<6>(), xdot_des_grad.rightCols<1>(), lqr_Q,
      10 * Vector1d::Ones(), Eigen::MatrixXd(0, 1), F);
  return lqr_result;
}

symbolic::Polynomial FindClfInit(
    const examples::acrobot::AcrobotParams<double>& p, int V_degree,
    const Eigen::Ref<const Vector6<symbolic::Variable>>& x) {
  const auto lqr_result = SynthesizeTrigLqr(p);
  const symbolic::Expression u_lqr = -lqr_result.K.row(0).dot(x);
  Vector6<symbolic::Expression> n_expr;
  symbolic::Expression d_expr;
  TrigDynamics<symbolic::Expression>(p, x.cast<symbolic::Expression>(), u_lqr,
                                     &n_expr, &d_expr);
  Vector6<symbolic::Polynomial> dynamics_numerator;
  for (int i = 0; i < 6; ++i) {
    dynamics_numerator(i) = symbolic::Polynomial(n_expr(i));
  }
  const symbolic::Polynomial dynamics_denominator =
      symbolic::Polynomial(d_expr);

  const double positivity_eps = 0.001;
  const int d = V_degree / 2;
  const double deriv_eps = 0.01;
  const Vector2<symbolic::Polynomial> state_eq_constraints =
      StateEqConstraints(x);
  const std::vector<int> positivity_ceq_lagrangian_degrees{
      {V_degree - 2, V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{{2, 2}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{V_degree - 2};
  const std::vector<int> derivative_cin_lagrangian_degrees{{2}};

  symbolic::Polynomial V;
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian;
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian;
  VectorX<symbolic::Polynomial> derivative_cin_lagrangian;
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian;
  symbolic::Polynomial positivity_sos_condition;
  symbolic::Polynomial derivative_sos_condition;
  auto prog = FindCandidateRegionalLyapunov(
      x, dynamics_numerator, dynamics_denominator, V_degree, positivity_eps, d,
      deriv_eps, state_eq_constraints, positivity_ceq_lagrangian_degrees,
      derivative_ceq_lagrangian_degrees, state_ineq_constraints,
      positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees, &V,
      &positivity_cin_lagrangian, &positivity_ceq_lagrangian,
      &derivative_cin_lagrangian, &derivative_ceq_lagrangian,
      &positivity_sos_condition, &derivative_sos_condition);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(V);
  return V_sol;
}

void SearchWTrigDynamics() {
  examples::acrobot::AcrobotPlant<double> acrobot;
  auto context = acrobot.CreateDefaultContext();
  // examples::acrobot::AcrobotParams<double>& mutable_parameters =
  //    acrobot.get_mutable_parameters(context.get());
  const auto& parameters = acrobot.get_parameters(*context);
  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const int V_degree = 4;
  symbolic::Polynomial V_init = FindClfInit(parameters, V_degree, x);
  V_init = V_init.RemoveTermsWithSmallCoefficients(1E-5);

  Vector6<symbolic::Polynomial> f;
  Vector6<symbolic::Polynomial> G;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(parameters, x, &f, &G, &dynamics_denominator);
  const Vector2<symbolic::Polynomial> state_constraints = StateEqConstraints(x);

  // Arbitrary maximal joint torque.
  const double u_max = 5;
  const Eigen::RowVector2d u_vertices(-u_max, u_max);
  const ControlLyapunov dut(x, f, G, dynamics_denominator, u_vertices,
                            state_constraints);

  const int lambda0_degree = 0;
  const std::vector<int> l_degrees{{2, 2}};
  const std::vector<int> p_degrees{{6, 6}};
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p;
  const double deriv_eps = 0.01;
  {
    // Now maximize rho to prove V(x)<=rho is an ROA
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    symbolic::Variable rho_var;
    symbolic::Polynomial vdot_sos;
    VectorX<symbolic::Monomial> vdot_monomials;
    MatrixX<symbolic::Variable> vdot_gram;
    const int d_degree = lambda0_degree / 2 + 1;
    auto prog = dut.ConstructLagrangianProgram(
        V_init, symbolic::Polynomial(), d_degree, l_degrees, p_degrees,
        deriv_eps, &l, &l_grams, &p, &rho_var, &vdot_sos, &vdot_monomials,
        &vdot_gram);
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    drake::log()->info("Maximize rho for the initial Clf");
    const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
    DRAKE_DEMAND(result.is_success());
    const double rho_sol = result.GetSolution(rho_var);
    std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
    V_init = V_init / rho_sol;
  }
  symbolic::Polynomial V_sol;
  {
    ControlLyapunov::SearchOptions search_options;
    search_options.rho_converge_tol = 0.;
    search_options.bilinear_iterations = 10;
    search_options.backoff_scale = 0.01;
    search_options.lsol_tiny_coeff_tol = 1E-8;
    search_options.lyap_tiny_coeff_tol = 1E-8;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    Eigen::MatrixXd state_samples(4, 1);
    state_samples.col(0) << 0, 0, 0, 0;
    Eigen::MatrixXd x_samples(6, state_samples.cols());
    for (int i = 0; i < state_samples.cols(); ++i) {
      x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
    }
    std::cout << "x_samples:\n" << x_samples.transpose() << "\n";

    const double positivity_eps = 0.0001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    symbolic::Polynomial lambda0_sol;
    VectorX<symbolic::Polynomial> l_sol;
    VectorX<symbolic::Polynomial> p_sol;
    const bool minimize_max = true;
    std::cout << "V_init(x_samples): "
              << V_init.EvaluateIndeterminates(x, x_samples).transpose()
              << "\n";
    dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
               positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
               deriv_eps, x_samples, minimize_max, search_options, &V_sol,
               &positivity_eq_lagrangian, &lambda0_sol, &l_sol, &p_sol);
    std::cout << "V(x_samples): "
              << V_sol.EvaluateIndeterminates(x, x_samples).transpose() << "\n";
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
