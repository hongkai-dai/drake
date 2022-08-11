#include <iostream>

#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/sdpa_free_format.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/test/acrobot.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
namespace systems {
namespace analysis {

[[maybe_unused]] void SwingUpTrajectoryOptimization(
    Eigen::MatrixXd* state_traj, Eigen::MatrixXd* control_traj) {
  // Swing up acrobot.
  examples::acrobot::AcrobotPlant<double> acrobot;
  auto context = acrobot.CreateDefaultContext();
  const int num_time_samples = 30;
  const double minimum_timestep = 0.02;
  const double maximum_timestep = 0.08;
  trajectory_optimization::DirectCollocation dircol(
      &acrobot, *context, num_time_samples, minimum_timestep, maximum_timestep,
      acrobot.get_input_port().get_index());
  dircol.prog().AddBoundingBoxConstraint(
      Eigen::Vector4d::Zero(), Eigen::Vector4d::Zero(), dircol.state(0));
  dircol.prog().AddBoundingBoxConstraint(Eigen::Vector4d(M_PI, 0, 0, 0),
                                         Eigen::Vector4d(M_PI, 0, 0, 0),
                                         dircol.state(num_time_samples - 1));
  for (int i = 0; i < num_time_samples; ++i) {
    dircol.prog().AddBoundingBoxConstraint(-30, 30, dircol.input(i)(0));
  }
  dircol.AddRunningCost(
      dircol.input().cast<symbolic::Expression>().dot(dircol.input()));
  const auto result = solvers::Solve(dircol.prog());
  DRAKE_DEMAND(result.is_success());
  *state_traj = dircol.GetStateSamples(result);
  *control_traj = dircol.GetInputSamples(result);
  std::cout << "swingup control traj: " << *control_traj << "\n";
}

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
      1000 * Vector1d::Ones(), Eigen::MatrixXd(0, 1), F);
  return lqr_result;
}

[[maybe_unused]] void VerifyLyapunovInit(
    const Eigen::Ref<const Vector6<symbolic::Variable>>& x,
    const symbolic::Polynomial& V_init,
    const Eigen::Ref<const Vector6<symbolic::Polynomial>>& dynamics_numerator,
    const symbolic::Polynomial& dynamics_denominator) {
  // Verify the condition
  // −∂V/∂x*n(x) − l(x)(ρ−V(x))*d(x)−p(x)c(x) is sos.
  // l(x) is sos.
  // Namely c(x) = 0 and V(x) <= ρ implies ∂V/∂x*n(x)/d(x) <= 0
  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(x);
  symbolic::Polynomial vdot_sos = -V_init.Jacobian(x).dot(dynamics_numerator);
  const int l_degree = 4;
  symbolic::Polynomial l;
  const symbolic::Variables x_set{x};
  std::tie(l, std::ignore) = prog.NewSosPolynomial(x_set, l_degree);
  const double rho = 1E-5;
  vdot_sos -= l * (rho - V_init) * dynamics_denominator;

  const std::vector<int> p_degrees{{6, 6}};
  const Vector2<symbolic::Polynomial> state_constraints = StateEqConstraints(x);
  Vector2<symbolic::Polynomial> p;
  for (int i = 0; i < 2; ++i) {
    p(i) = prog.NewFreePolynomial(x_set, p_degrees[i]);
  }
  vdot_sos -= p.dot(state_constraints);
  prog.AddSosConstraint(vdot_sos);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
}

[[maybe_unused]] void VerifyLyapunovInitPablo(
    const Eigen::Ref<const Vector6<symbolic::Variable>>& x,
    const symbolic::Polynomial& V_init,
    const Eigen::Ref<const Vector6<symbolic::Polynomial>>& dynamics_numerator,
    const symbolic::Polynomial& dynamics_denominator) {
  // Use Pablo's formulation to find max rho such that V(x) <= rho is a region
  // of attraction.
  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(x);
  // Add the constraint (xᵀx)ᵈ(V(x)−ρ)d(x) − l(x)*∂V/∂x*n(x) − p(x)ᵀc(x) is sos
  // where the closed loop dynamics (with u = -Kx) is xdot = n(x)/d(x).

  const int d_degree = 1;
  const int l_degree = 4;
  const symbolic::Variable rho = prog.NewContinuousVariables<1>()(0);
  const symbolic::Variables x_set{x};
  symbolic::Polynomial vdot_sos =
      symbolic::Polynomial(
          pow(x.cast<symbolic::Expression>().dot(x), d_degree)) *
      (V_init - symbolic::Polynomial(rho, x_set)) * dynamics_denominator;
  symbolic::Polynomial l;
  std::tie(l, std::ignore) = prog.NewSosPolynomial(x_set, l_degree);
  vdot_sos -= l * V_init.Jacobian(x).dot(dynamics_numerator);
  const std::vector<int> p_degrees{{6, 6}};
  Vector2<symbolic::Polynomial> p;
  for (int i = 0; i < 2; ++i) {
    p(i) = prog.NewFreePolynomial(x_set, p_degrees[i]);
  }
  const Vector2<symbolic::Polynomial> state_eq_constraints =
      StateEqConstraints(x);
  vdot_sos -= p.dot(state_eq_constraints);
  prog.AddSosConstraint(vdot_sos);
  prog.AddLinearCost(Vector1d(-1), 0, Vector1<symbolic::Variable>(rho));
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  std::cout << "rho: " << result.GetSolution(rho) << "\n";
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
  const std::vector<int> derivative_ceq_lagrangian_degrees{{4, 4}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.01));
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

void SearchWTrigDynamics(double u_max,
                         const std::optional<std::string>& load_clf,
                         const std::optional<std::string>& save_clf) {
  examples::acrobot::AcrobotPlant<double> acrobot;
  auto context = acrobot.CreateDefaultContext();
  // examples::acrobot::AcrobotParams<double>& mutable_parameters =
  //    acrobot.get_mutable_parameters(context.get());
  const auto& parameters = acrobot.get_parameters(*context);
  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const int V_degree = 2;
  Vector6<symbolic::Polynomial> f;
  Vector6<symbolic::Polynomial> G;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(parameters, x, &f, &G, &dynamics_denominator);
  const Vector2<symbolic::Polynomial> state_constraints = StateEqConstraints(x);

  // Arbitrary maximal joint torque.
  const Eigen::RowVector2d u_vertices(-u_max, u_max);
  const ControlLyapunov dut(x, f, G, dynamics_denominator, u_vertices,
                            state_constraints);

  const int lambda0_degree = 2;
  const std::vector<int> l_degrees{{2, 2}};
  const std::vector<int> p_degrees{{6, 6}};
  symbolic::Polynomial lambda0;
  const double deriv_eps = 0.1;
  symbolic::Polynomial V_init;
  double rho_val = 0.1;
  if (load_clf.has_value()) {
    V_init = Load(symbolic::Variables(x), load_clf.value());
  } else {
    V_init = FindClfInit(parameters, V_degree, x);
    const bool binary_search_rho = true;
    if (binary_search_rho) {
      symbolic::Polynomial lambda0_sol;
      VectorX<symbolic::Polynomial> l_sol;
      VectorX<symbolic::Polynomial> p_sol;
      ControlLyapunov::SearchOptions search_options{};
      search_options.lagrangian_step_solver_options = solvers::SolverOptions();
      search_options.lagrangian_step_solver_options->SetOption(
          solvers::CommonSolverOption::kPrintToConsole, 1);
      search_options.lagrangian_tiny_coeff_tol = 1E-10;
      double rho_sol;
      bool found_rho = dut.FindRhoBinarySearch(
          V_init, 0, 0.00125, 5E-4, lambda0_degree, l_degrees, p_degrees,
          deriv_eps, search_options, &rho_sol, &lambda0_sol, &l_sol, &p_sol);
      if (found_rho) {
        std::cout << "Binary search rho_sol: " << rho_sol << "\n";
        V_init =
            V_init.RemoveTermsWithSmallCoefficients(1E-8) * (rho_val / rho_sol);
      } else {
        abort();
      }
    } else {
      // Now maximize rho to prove V(x)<=rho is an ROA
      std::vector<MatrixX<symbolic::Variable>> l_grams;
      symbolic::Variable rho_var;
      symbolic::Polynomial vdot_sos;
      VectorX<symbolic::Monomial> vdot_monomials;
      MatrixX<symbolic::Variable> vdot_gram;
      const int d_degree = lambda0_degree / 2 + 1;
      auto lagrangian_ret = dut.ConstructLagrangianProgram(
          V_init, symbolic::Polynomial(), d_degree, l_degrees, p_degrees,
          deriv_eps);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      drake::log()->info("Maximize rho for the initial Clf");
      const auto result =
          solvers::Solve(*(lagrangian_ret.prog), std::nullopt, solver_options);
      DRAKE_DEMAND(result.is_success());
      const double rho_sol = result.GetSolution(lagrangian_ret.rho);
      std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
      // V_init = V_init / rho_sol;
      V_init =
          V_init.RemoveTermsWithSmallCoefficients(1E-8) * rho_val / rho_sol;
      DRAKE_DEMAND(rho_sol > 0);
    }
  }

  {
    ControlLyapunov::SearchOptions search_options;
    search_options.d_converge_tol = 0.;
    search_options.bilinear_iterations = 15;
    search_options.lyap_step_backoff_scale = 0.04;
    search_options.lagrangian_tiny_coeff_tol = 1E-6;
    search_options.lsol_tiny_coeff_tol = 1E-5;
    search_options.lyap_tiny_coeff_tol = 1E-5;
    search_options.Vsol_tiny_coeff_tol = 1E-6;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    // search_options.lyap_step_solver_options->SetOption(
    //    solvers::MosekSolver::id(), "writedata", "lyapunov.task.gz");
    search_options.rho = rho_val;

    Eigen::MatrixXd state_swingup;
    Eigen::MatrixXd control_swingup;
    SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
    Eigen::Matrix<double, 6, Eigen::Dynamic> x_swingup(6, state_swingup.cols());
    for (int i = 0; i < x_swingup.cols(); ++i) {
      x_swingup.col(i) = ToTrigState<double>(state_swingup.col(i));
    }
    std::cout << "V_init at x_swingup "
              << V_init.EvaluateIndeterminates(x, x_swingup).transpose()
              << "\n";
    std::vector<int> x_indices = {28};
    Eigen::Matrix4Xd state_samples(4, x_indices.size());
    for (int i = 0; i < static_cast<int>(x_indices.size()); ++i) {
      state_samples.col(i) = state_swingup.col(x_indices[i]);
    }
    std::cout << "state samples:\n" << state_samples.transpose() << "\n";
    Eigen::MatrixXd x_samples(6, state_samples.cols());
    for (int i = 0; i < state_samples.cols(); ++i) {
      x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
    }

    const double positivity_eps = 0.0001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{
        {V_degree - 2, V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    const bool minimize_max = true;
    symbolic::Environment env;
    env.insert(x, x_samples.col(0));
    std::cout << "V_init: " << V_init << "\n";
    std::cout << "V_init(x_samples): "
              << V_init.EvaluateIndeterminates(x, x_samples).transpose()
              << "\n";
    const auto search_result =
        dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                   positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                   deriv_eps, x_samples, std::nullopt /* in_roa_samples */,
                   minimize_max, search_options);
    std::cout
        << "V(x_samples): "
        << search_result.V.EvaluateIndeterminates(x, x_samples).transpose()
        << "\n";
    std::cout << "rho=" << search_options.rho << "\n";
    if (save_clf.has_value()) {
      Save(search_result.V, save_clf.value());
    }
  }
}

int DoMain() {
  const double u_max = 40;
  std::optional<std::string> load_clf = "acrobot_trig_clf1.txt";
  SearchWTrigDynamics(u_max, load_clf, "acrobot_trig_clf2.txt");
  // SearchWTrigDynamics("acrobot_trig_clf.txt");
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() { return drake::systems::analysis::DoMain(); }
