#include <iostream>

#include <limits>

#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/hjb.h"
#include "drake/systems/analysis/test/cart_pole.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

[[maybe_unused]] void SwingUpTrajectoryOptimization(Eigen::MatrixXd* x_traj,
                                                    Eigen::MatrixXd* u_traj) {
  // Swing up cart pole.
  multibody::MultibodyPlant<double> cart_pole(0.);
  multibody::Parser(&cart_pole)
      .AddModelFromFile(FindResourceOrThrow(
          "drake/examples/multibody/cart_pole/cart_pole.sdf"));
  cart_pole.Finalize();
  auto context = cart_pole.CreateDefaultContext();
  const int num_time_samples = 30;
  const double minimum_timestep = 0.02;
  const double maximum_timestep = 0.06;
  trajectory_optimization::DirectCollocation dircol(
      &cart_pole, *context, num_time_samples, minimum_timestep,
      maximum_timestep, cart_pole.get_actuation_input_port().get_index());
  dircol.prog().AddBoundingBoxConstraint(
      Eigen::Vector4d::Zero(), Eigen::Vector4d::Zero(), dircol.state(0));
  dircol.prog().AddBoundingBoxConstraint(Eigen::Vector4d(0, M_PI, 0, 0),
                                         Eigen::Vector4d(0, M_PI, 0, 0),
                                         dircol.state(num_time_samples - 1));
  dircol.prog().AddBoundingBoxConstraint(0, 0, dircol.input(num_time_samples-1)(0));
  // for (int i = 0; i < num_time_samples; ++i) {
  //  dircol.prog().AddBoundingBoxConstraint(-110, 110, dircol.input(i)(0));
  //}
  dircol.AddRunningCost(
      dircol.input().cast<symbolic::Expression>().dot(dircol.input()));
  const auto result = solvers::Solve(dircol.prog());
  DRAKE_DEMAND(result.is_success());
  *x_traj = dircol.GetStateSamples(result);
  *u_traj = dircol.GetInputSamples(result);
  std::cout << "swingup u: " << *u_traj << "\n";
}

[[maybe_unused]] symbolic::Polynomial FindClfInit(
    const CartPoleParams& params, int V_degree,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x) {
  Eigen::Matrix<double, 5, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10;
  const Eigen::Matrix<double, 5, 5> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = SynthesizeTrigLqr(params, lqr_Q, 20);
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

[[maybe_unused]] symbolic::Polynomial FindHjb(
    const CartPoleParams& params,
    const Eigen::Matrix<symbolic::Variable, 5, 1>& x) {
  // First synthesize an LQR controller.
  Eigen::Matrix<double, 5, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10;
  const Eigen::Matrix<double, 5, 5> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = SynthesizeTrigLqr(params, lqr_Q, 10);
  const Vector1<symbolic::Polynomial> u_lqr(symbolic::Polynomial(
      -lqr_result.K.row(0).dot(x), symbolic::Variables(x)));

  // Now construct HJB upper bound
  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial dynamics_numerator;
  TrigPolyDynamics(params, x, &f, &G, &dynamics_numerator);
  Eigen::Matrix<double, 1, 1> R(10);
  const symbolic::Polynomial l(x.cast<symbolic::Expression>().dot(lqr_Q * x));
  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
  const HjbUpper dut(x, l, R, f, G, dynamics_numerator, state_constraints);
  Eigen::MatrixXd state_samples =
      Eigen::MatrixXd::Random(4, 1000) * 0.1 + Eigen::Vector4d(0, M_PI, 0, 0);
  Eigen::Matrix<double, 5, Eigen::Dynamic> x_samples(5, state_samples.cols());
  for (int i = 0; i < state_samples.cols(); ++i) {
    x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
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
  return J_sol;
}

symbolic::Polynomial SearchWTrigDynamics(
    const CartPoleParams& params,
    const Eigen::Matrix<symbolic::Variable, 5, 1>& x, double u_max,
    double deriv_eps, const std::optional<std::string>& load_V_init,
    const std::optional<std::string>& save_V,
    SearchResultDetails* search_result_details) {
  const int V_degree = 2;

  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(params, x, &f, &G, &dynamics_denominator);

  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
  const Eigen::RowVector2d u_vertices(-u_max, u_max);
  const ControlLyapunov dut(x, f, G, dynamics_denominator, u_vertices,
                            state_constraints);

  const int lambda0_degree = 4;
  const std::vector<int> l_degrees{{4, 4}};
  const std::vector<int> p_degrees{{8}};
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p;
  symbolic::Polynomial V_init;
  double rho_sol;
  if (load_V_init.has_value()) {
    V_init = Load(symbolic::Variables(x), load_V_init.value());
    rho_sol = 1;
    std::cout << "V_init(x_bottom): "
              << V_init.EvaluateIndeterminates(
                     x, ToTrigState<double>(Eigen::Vector4d::Zero()))
              << "\n";
  } else {
    V_init = FindClfInit(params, V_degree, x);
    // Now maximize rho to prove V(x)<=rho is an ROA
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
      bool found_rho = dut.FindRhoBinarySearch(
          V_init, 0, 0.025, 1E-3, lambda0_degree, l_degrees, p_degrees,
          deriv_eps, search_options, &rho_sol, &lambda0_sol, &l_sol, &p_sol);
      if (found_rho) {
        std::cout << "Binary search rho_sol: " << rho_sol << "\n";
        V_init = V_init / rho_sol;
        rho_sol = 1;
        V_init = V_init.RemoveTermsWithSmallCoefficients(1E-10);
      } else {
        abort();
      }
    } else {
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
      rho_sol = result.GetSolution(rho_var);
      std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
      V_init = V_init / rho_sol;
      rho_sol = 1;
      V_init = V_init.RemoveTermsWithSmallCoefficients(1E-8);
    }
  }
  symbolic::Polynomial V_sol;
  {
    ControlLyapunov::SearchOptions search_options;
    search_options.rho_converge_tol = 0.0;
    search_options.bilinear_iterations = 20;
    search_options.backoff_scale = 0.008;
    search_options.lagrangian_tiny_coeff_tol = 1E-5;
    search_options.lsol_tiny_coeff_tol = 1E-5;
    search_options.lyap_tiny_coeff_tol = 1E-5;
    search_options.Vsol_tiny_coeff_tol = 1E-7;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 1);
    // search_options.lyap_step_solver_options->SetOption(
    //    solvers::MosekSolver::id(), "writedata",
    //    "cart_pole_trig_clf_lyapunov.task.gz");
    search_options.rho = rho_sol;
    if (save_V.has_value()) {
      search_options.save_clf_file = save_V.value();
    }
    const double positivity_eps = 0.00001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    symbolic::Polynomial lambda0_sol;
    VectorX<symbolic::Polynomial> l_sol;
    VectorX<symbolic::Polynomial> p_sol;

    const bool search_inner_ellipsoid = false;
    if (search_inner_ellipsoid) {
      const double rho_min = 0.32;
      const double rho_max = 0.5;
      const double rho_tol = 0.001;
      const std::vector<int> ellipsoid_c_lagrangian_degrees{{0}};
      const Eigen::Matrix<double, 5, 1> x_star =
          ToTrigState<double>(Eigen::Vector4d(0., 1. * M_PI, 0, 0));
      Eigen::Matrix<double, 5, 5> S;
      S.setZero();
      S(0, 0) = 1;
      S(1, 1) = 1;
      S(2, 2) = 1;
      S(3, 3) = 1;
      S(4, 4) = 1;
      const int r_degree = 0;
      const ControlLyapunov::RhoBisectionOption rho_bisection_option(
          rho_min, rho_max, rho_tol);
      symbolic::Polynomial r_sol;
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
      double ellipsoid_rho_sol;
      VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S, r_degree,
                 search_options, rho_bisection_option, &V_sol, &lambda0_sol,
                 &l_sol, &r_sol, &p_sol, &positivity_eq_lagrangian_sol,
                 &ellipsoid_rho_sol, &ellipsoid_c_lagrangian_sol);
    } else {
      Eigen::MatrixXd state_swingup;
      Eigen::MatrixXd control_swingup;
      SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
      Eigen::Matrix<double, 5, Eigen::Dynamic> x_swingup(5,
                                                         state_swingup.cols());
      for (int i = 0; i < x_swingup.cols(); ++i) {
        x_swingup.col(i) = ToTrigState<double>(state_swingup.col(i));
      }
      std::cout << "V_init at x_swingup "
                << V_init.EvaluateIndeterminates(x, x_swingup).transpose()
                << "\n";
      std::vector<int> x_indices = {12, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28};
      Eigen::Matrix4Xd state_samples(4, x_indices.size());
      for (int i = 0; i < static_cast<int>(x_indices.size()); ++i) {
        state_samples.col(i) = state_swingup.col(x_indices[i]);
      }
      std::cout << "state samples:\n" << state_samples.transpose() << "\n";
      Eigen::MatrixXd x_samples(5, state_samples.cols());
      for (int i = 0; i < state_samples.cols(); ++i) {
        x_samples.col(i) = ToTrigState<double>(state_samples.col(i));
      }

      std::optional<Eigen::MatrixXd> in_roa_samples;
      in_roa_samples.emplace(Eigen::Matrix<double, 5, 3>());
      in_roa_samples->col(0) = x_samples.col(1);
      in_roa_samples->col(1) = x_samples.col(2);
      in_roa_samples->col(2) = x_samples.col(3);
      // in_roa_samples = std::nullopt;

      const bool minimize_max = true;
      symbolic::Environment env;
      env.insert(x, x_samples.col(0));
      std::cout << "V_init: " << V_init << "\n";
      std::cout << "V_init(x_samples): "
                << V_init.EvaluateIndeterminates(x, x_samples).transpose()
                << "\n";
      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 deriv_eps, x_samples, in_roa_samples, minimize_max,
                 search_options, &V_sol, &positivity_eq_lagrangian,
                 &lambda0_sol, &l_sol, &p_sol, search_result_details);
      std::cout << "V(x_swingup): "
                << V_sol.EvaluateIndeterminates(x, x_swingup).transpose()
                << "\n";
      std::cout << "V(x_samples): "
                << V_sol.EvaluateIndeterminates(x, x_samples).transpose()
                << "\n";
    }
    if (save_V.has_value()) {
      Save(V_sol / search_options.rho, save_V.value());
    }
  }
  return V_sol;
}

int DoMain() {
  const CartPoleParams params;
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }

  double u_max = 185.1;
  const double deriv_eps = 0.1;
  symbolic::Polynomial V_sol;
  bool found_u_max = false;
  int u_max_trial = 0;
  while (u_max < 250) {
    std::cout << "Try u_max = " << u_max << "\n";
    //std::optional<std::string> load_file =
    //    "/home/hongkaidai/Dropbox/sos_clf_cbf/cart_pole_trig_clf_last16_1.txt";
    std::optional<std::string> load_file = "cart_pole_trig_clf27.txt";
    SearchResultDetails search_result_details;
    V_sol =
        SearchWTrigDynamics(params, x, u_max, deriv_eps, load_file,
                            "cart_pole_trig_clf28.txt", &search_result_details);
    if (search_result_details.num_bilinear_iterations >= 1 ||
        search_result_details.bilinear_iteration_status !=
            BilinearIterationStatus::kFailLagrangian) {
      found_u_max = true;
      std::cout << fmt::format("Found u_max = {} after {} trials\n", u_max,
                               u_max_trial);
      const double duration = 20;
      Simulate(params, x, V_sol, u_max, deriv_eps, Eigen::Vector4d(0, 0, 0, 7),
               duration);
      return 0;
    } else {
      u_max += 0.005;
      u_max_trial++;
    }
  }
  if (!found_u_max) {
    std::cout << "Cannot find u_max\n";
  }
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
