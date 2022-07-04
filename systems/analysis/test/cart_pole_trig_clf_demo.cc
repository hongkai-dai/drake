#include <iostream>

#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/test/cart_pole.h"

namespace drake {
namespace systems {
namespace analysis {

symbolic::Polynomial FindClfInit(
    const CartPoleParams& params, int V_degree,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x) {
  Eigen::Matrix<double, 5, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10;
  const Eigen::Matrix<double, 5, 5> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = SynthesizeTrigLqr(params, lqr_Q, 10);

  const symbolic::Expression u_lqr = -lqr_result.K.row(0).dot(x);
  Eigen::Matrix<symbolic::Expression, 5, 1> n_expr;
  symbolic::Expression d_expr;
  TrigDynamics<symbolic::Expression>(params, x.cast<symbolic::Expression>(),
                                     u_lqr, &n_expr, &d_expr);
  Eigen::Matrix<symbolic::Polynomial, 5, 1> dynamics_numerator;
  for (int i = 0; i < 5; ++i) {
    dynamics_numerator(i) = symbolic::Polynomial(n_expr(i));
  }
  const symbolic::Polynomial dynamics_denominator =
      symbolic::Polynomial(d_expr);

  const double positivity_eps = 0.001;
  const int d = V_degree / 2;
  const double deriv_eps = 0.01;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      StateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{{4}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.04));
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
  // VerifyLyapunovInit(x, V_sol, dynamics_numerator, dynamics_denominator);
  // VerifyLyapunovInitPablo(x, V_sol, dynamics_numerator,
  // dynamics_denominator);
  return V_sol;
}

symbolic::Polynomial SearchWTrigDynamics(
    const CartPoleParams& params,
    const Eigen::Matrix<symbolic::Variable, 5, 1>& x, double u_max,
    double deriv_eps, const std::optional<std::string>& load_V_init) {
  const int V_degree = 2;
  symbolic::Polynomial V_init = FindClfInit(params, V_degree, x);

  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(params, x, &f, &G, &dynamics_denominator);

  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
  // Arbitrary maximal joint torque.
  const Eigen::RowVector2d u_vertices(-u_max, u_max);
  const ControlLyapunov dut(x, f, G, dynamics_denominator, u_vertices,
                            state_constraints);

  const int lambda0_degree = 4;
  const std::vector<int> l_degrees{{4, 4}};
  const std::vector<int> p_degrees{{8}};
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p;
  if (load_V_init.has_value()) {
    V_init = Load(symbolic::Variables(x), load_V_init.value());
  } else {
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
    double rho_sol = result.GetSolution(rho_var);
    std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
    V_init = V_init / rho_sol;
    V_init = V_init.RemoveTermsWithSmallCoefficients(1E-8);
  }
  symbolic::Polynomial V_sol;
  {
    ControlLyapunov::SearchOptions search_options;
    search_options.rho_converge_tol = 0.;
    search_options.bilinear_iterations = 30;
    search_options.backoff_scale = 0.01;
    search_options.lsol_tiny_coeff_tol = 1E-6;
    search_options.lyap_tiny_coeff_tol = 1E-6;
    search_options.Vsol_tiny_coeff_tol = 1E-8;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    // search_options.lyap_step_solver_options->SetOption(
    //    solvers::MosekSolver::id(), "writedata",
    //    "cart_pole_trig_clf_lyapunov.task.gz");
    const double positivity_eps = 0.0001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    symbolic::Polynomial lambda0_sol;
    VectorX<symbolic::Polynomial> l_sol;
    VectorX<symbolic::Polynomial> p_sol;

    const bool search_inner_ellipsoid = true;
    if (search_inner_ellipsoid) {
      const double rho_min = 0.0001;
      const double rho_max = 3;
      const double rho_tol = 0.01;
      const std::vector<int> ellipsoid_c_lagrangian_degrees{{0}};
      Eigen::Matrix<double, 5, 1> x_star;
      x_star << 0, 0, 0.2, 0, 0;
      Eigen::Matrix<double, 5, 5> S;
      S.setZero();
      S(0, 0) = 1;
      S(1, 1) = 10;
      S(2, 2) = 10;
      S(3, 3) = 10;
      S(4, 4) = 10;
      const int r_degree = 0;
      const ControlLyapunov::RhoBisectionOption rho_bisection_option(
          rho_min, rho_max, rho_tol);
      symbolic::Polynomial r_sol;
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
      double rho_sol;
      VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S, r_degree,
                 search_options, rho_bisection_option, &V_sol, &lambda0_sol,
                 &l_sol, &r_sol, &p_sol, &positivity_eq_lagrangian_sol,
                 &rho_sol, &ellipsoid_c_lagrangian_sol);
    } else {
      Eigen::MatrixXd state_samples(4, 4);
      state_samples.col(0) << 0, 1.1 * M_PI, 0, 0;
      state_samples.col(1) << 0, 0.9 * M_PI, 0, 0;
      state_samples.col(2) << 0.2, 1.1 * M_PI, 0, 0;
      state_samples.col(3) << 0.2, 0.9 * M_PI, 0, 0;
      Eigen::MatrixXd x_samples(5, state_samples.cols());
      for (int i = 0; i < state_samples.cols(); ++i) {
        x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
      }

      const bool minimize_max = true;
      symbolic::Environment env;
      env.insert(x, x_samples.col(0));
      std::cout << "V_init: " << V_init << "\n";
      std::cout << "V_init(x_samples): "
                << V_init.EvaluateIndeterminates(x, x_samples).transpose()
                << "\n";
      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 deriv_eps, x_samples, minimize_max, search_options, &V_sol,
                 &positivity_eq_lagrangian, &lambda0_sol, &l_sol, &p_sol);
      std::cout << "V(x_samples): "
                << V_sol.EvaluateIndeterminates(x, x_samples).transpose()
                << "\n";
    }
    Save(V_sol, "cart_pole_trig_clf2.txt");
  }
  return V_sol;
}

int DoMain() {
  const CartPoleParams params;
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const double u_max = 32;
  const double deriv_eps = 0.1;
  const symbolic::Polynomial V_sol = SearchWTrigDynamics(
      params, x, u_max, deriv_eps, "cart_pole_trig_clf1.txt");
  // const symbolic::Polynomial V_sol = Load(symbolic::Variables(x),
  // "cart_pole_trig_clf10.txt");
  const double duration = 200;
  Eigen::Matrix<double, 4, 10> state_samples;
  state_samples.col(0) << 0.1, 0.5, 0.2, 0.4;
  state_samples.col(1) << 0.8, 1.2 * M_PI, 0.4, -0.5;
  state_samples.col(2) << 0.4, 1.05 * M_PI, 0, 0.1;
  state_samples.col(3) << 0.2, 0.5 * M_PI, -0.2, 1.5;
  state_samples.col(4) << 0.1, 0.2 * M_PI, -0.5, 1.5;
  state_samples.col(5) << 0, 1.05 * M_PI, 0, 0;
  state_samples.col(6) << 0.2, 1.05 * M_PI, 0, 0;
  state_samples.col(7) << 0.2, 1.1 * M_PI, 0, 0;
  state_samples.col(8) << 0.2, 1.2 * M_PI, 0, 0;
  Eigen::Matrix<double, 5, Eigen::Dynamic> x_samples(5, state_samples.cols());
  for (int i = 0; i < state_samples.cols(); ++i) {
    x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
  }
  std::cout << V_sol.EvaluateIndeterminates(x, x_samples) << "\n";
  Simulate(params, x, V_sol, u_max, deriv_eps, state_samples.col(6), duration);
  // SearchWTrigDynamics("cart_pole_trig_clf.txt");
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
