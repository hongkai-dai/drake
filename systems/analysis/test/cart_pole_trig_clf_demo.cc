#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/common_solver_option.h"
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
  dircol.prog().AddBoundingBoxConstraint(0, 0,
                                         dircol.input(num_time_samples - 1)(0));
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

// symbolic::Polynomial SearchClfNoInputBound(
//    const CartPoleParams& params,
//    const Eigen::Matrix<symbolic::Variable, 5, 1>& x, double deriv_eps) {
//  const std::optional<std::string> load_file =
//      "cart_pole_trig_clf_no_limit6.txt.iter5";
//  const std::string save_file_name = "cart_pole_trig_clf_no_limit7.txt";
//  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
//  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
//  symbolic::Polynomial dynamics_denominator;
//  TrigPolyDynamics(params, x, &f, &G, &dynamics_denominator);
//  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
//  const ControlLyapunovNoInputBound dut(x, f, G, dynamics_denominator,
//                                        state_constraints);
//  const symbolic::Variables x_set(x);
//
//  const int V_degree = 4;
//  symbolic::Polynomial V_init;
//  if (load_file.has_value()) {
//    V_init = Load(x_set, load_file.value());
//  } else {
//    V_init = FindClfInit(params, V_degree, x);
//    V_init = V_init / 1.3;
//  }
//
//  Eigen::Matrix<double, 5, 1> x_lb;
//  Eigen::Matrix<double, 5, 1> x_ub;
//  // -0.3 <= x(0) <= 0.3
//  // sin(1.1pi) <= sin(theta) = x(1) <= sin(0.9pi)
//  // cos(theta) = x(2)-1 <= cos(0.9*pi)
//  x_lb << -0.3, std::sin(1.1 * M_PI), -1, -1, -2;
//  x_ub << 0.3, std::sin(0.9 * M_PI), std::cos(0.9 * M_PI) + 1, 1, 2;
//
//  const int lambda0_degree = 2;
//  const std::vector<int> l_degrees = {4};
//  const std::vector<int> eq_lagrangian_degrees = {6};
//  const std::vector<int> ineq_lagrangian_degrees = {
//      6};  //{6, 6, 6, 6, 6, 6, 6, 6, 6};
//
//  symbolic::Polynomial V_sol = V_init;
//  symbolic::Polynomial lambda0_sol;
//  VectorX<symbolic::Polynomial> l_sol;
//  VectorX<symbolic::Polynomial> eq_lagrangian_sol;
//  VectorX<symbolic::Polynomial> ineq_lagrangian_sol;
//  int iter_count = 0;
//  const int iter_max = 30;
//  Eigen::MatrixXd state_swingup;
//  Eigen::MatrixXd control_swingup;
//  SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
//  Eigen::Matrix<double, 5, Eigen::Dynamic> x_swingup(5, state_swingup.cols());
//  for (int i = 0; i < state_swingup.cols(); ++i) {
//    x_swingup.col(i) = ToTrigState<double>(state_swingup.col(i));
//  }
//  drake::log()->info("Before bilinear alternation, V(x_swingup)={}",
//                     V_sol.EvaluateIndeterminates(x, x_swingup).transpose());
//  while (iter_count < iter_max) {
//    drake::log()->info("Iter {}", iter_count);
//    {
//      // for (int i = 0; i < 5; ++i) {
//      //  ineq_constraints(i) = symbolic::Polynomial(x(i) - x_ub(i));
//      //}
//      // ineq_constraints(5) = symbolic::Polynomial(-x(0) + x_lb(0));
//      // ineq_constraints(6) = symbolic::Polynomial(-x(1) + x_lb(1));
//      // ineq_constraints(7) = symbolic::Polynomial(-x(3) + x_lb(3));
//      // ineq_constraints(8) = symbolic::Polynomial(-x(4) + x_lb(4));
//
//      VectorX<symbolic::Polynomial> ineq_constraints(1);
//      ineq_constraints(0) = V_sol - 1;
//      solvers::SolverOptions lagrangian_solver_options;
//      lagrangian_solver_options.SetOption(
//          solvers::CommonSolverOption::kPrintToConsole, 1);
//      const double lsol_tiny_coeff_tol = 0;
//      const bool found_lagrangian = dut.SearchLagrangian(
//          V_sol, lambda0_degree, l_degrees, deriv_eps, eq_lagrangian_degrees,
//          ineq_constraints, ineq_lagrangian_degrees,
//          lagrangian_solver_options, lsol_tiny_coeff_tol, &lambda0_sol,
//          &l_sol, &eq_lagrangian_sol, &ineq_lagrangian_sol);
//      if (found_lagrangian) {
//        drake::log()->info("Find lagrangian");
//      } else {
//        drake::log()->error("Failed to find Lagrangian");
//        return V_sol;
//      }
//    }
//    {
//      // Now fix Lagrangian multiplier and search for V.
//      solvers::MathematicalProgram prog;
//      prog.AddIndeterminates(x);
//      // Add the constraint that V >= 0 on state_constraints;
//      const int d_degree = V_degree / 2;
//      symbolic::Polynomial V = NewFreePolynomialPassOrigin(
//          &prog, x_set, V_degree, "V", symbolic::internal::DegreeType::kAny,
//          symbolic::Variables({x(0), x(1), x(3), x(4)}));
//      const double positivity_eps = 0.0001;
//      symbolic::Polynomial positivity_sos =
//          V - positivity_eps *
//                  symbolic::Polynomial(
//                      pow(x.cast<symbolic::Expression>().dot(x), d_degree));
//      VectorX<symbolic::Polynomial> positivity_eq_lagrangian(
//          state_constraints.rows());
//      for (int i = 0; i < positivity_eq_lagrangian.rows(); ++i) {
//        positivity_eq_lagrangian(i) =
//            prog.NewFreePolynomial(x_set, V_degree - 2);
//      }
//      positivity_sos -= positivity_eq_lagrangian.dot(state_constraints);
//      prog.AddSosConstraint(positivity_sos);
//
//      VectorX<symbolic::Polynomial> eq_lagrangian(state_constraints.rows());
//      for (int i = 0; i < eq_lagrangian.rows(); ++i) {
//        eq_lagrangian(i) =
//            prog.NewFreePolynomial(x_set, eq_lagrangian_degrees[i]);
//      }
//      VectorX<symbolic::Polynomial> ineq_constraints(1);
//      ineq_constraints(0) = V - 1;
//      symbolic::Polynomial vdot_sos;
//      VectorX<symbolic::Monomial> vdot_sos_monomials;
//      MatrixX<symbolic::Variable> vdot_sos_gram;
//      dut.AddControlLyapunovConstraint(&prog, lambda0_sol, l_sol, V,
//      deriv_eps,
//                                       eq_lagrangian, ineq_constraints,
//                                       ineq_lagrangian_sol, &vdot_sos,
//                                       &vdot_sos_monomials, &vdot_sos_gram);
//      // The cost is to minimize max(V(x_samples));
//      std::vector<int> x_indices = {14, 15, 16, 17, 18, 19, 20, 21,
//                                    22, 23, 24, 25, 26, 27, 28};
//      Eigen::MatrixXd x_samples(5, x_indices.size());
//      for (int i = 0; i < x_samples.cols(); ++i) {
//        x_samples.col(i) = x_swingup.col(x_indices[i]);
//      }
//      OptimizePolynomialAtSamples(&prog, V, x, x_samples,
//                                  OptimizePolynomialMode::kMinimizeMaximal);
//
//      std::optional<Eigen::MatrixXd> in_roa_samples;
//      in_roa_samples.emplace(Eigen::MatrixXd(5, 1));
//      in_roa_samples->col(0) = x_samples.col(1);
//      if (in_roa_samples.has_value()) {
//        Eigen::MatrixXd A_in_roa;
//        Eigen::VectorXd b_in_roa;
//        VectorX<symbolic::Variable> var_in_roa;
//        V.EvaluateWithAffineCoefficients(x, in_roa_samples.value(), &A_in_roa,
//                                         &var_in_roa, &b_in_roa);
//        prog.AddLinearConstraint(
//            A_in_roa, Eigen::VectorXd::Constant(b_in_roa.rows(), -kInf),
//            Eigen::VectorXd::Ones(b_in_roa.rows()) - b_in_roa, var_in_roa);
//      }
//      solvers::SolverOptions solver_options;
//      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole,
//      1); const double backoff_scale = 0.00;
//
//      const double prev_cost =
//          V_sol.EvaluateIndeterminates(x, x_samples).maxCoeff();
//      const auto result = SearchWithBackoff(&prog, solvers::MosekSolver::id(),
//                                            solver_options, backoff_scale);
//      if (result.is_success()) {
//        V_sol = result.GetSolution(V);
//        Save(V_sol, save_file_name + ".iter" + std::to_string(iter_count));
//        const double curr_cost =
//            V_sol.EvaluateIndeterminates(x, x_samples).maxCoeff();
//        drake::log()->info("Found V, cost={}, cost decreases by {}",
//        curr_cost,
//                           prev_cost - curr_cost);
//        drake::log()->info(
//            "After search for V, V(x_swingup)={}",
//            V_sol.EvaluateIndeterminates(x, x_swingup).transpose());
//        if (prev_cost - curr_cost < 0) {
//          return V_sol;
//        }
//      } else {
//        drake::log()->error("Failed to find V");
//        abort();
//      }
//    }
//    iter_count++;
//  }
//  drake::log()->info("V(x_swingup)={}",
//                     V_sol.EvaluateIndeterminates(x, x_swingup).transpose());
//  Save(V_sol, save_file_name);
//  return V_sol;
//}
//
// symbolic::Polynomial SearchWithSlackA(
//    const CartPoleParams& params,
//    const Eigen::Matrix<symbolic::Variable, 5, 1>& x, double u_max,
//    double deriv_eps, const std::string& load_V_init,
//    const std::string& save_V) {
//  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));
//  const Eigen::RowVector2d u_vertices(-u_max, u_max);
//  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
//  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
//  symbolic::Polynomial dynamics_denominator;
//  TrigPolyDynamics(params, x, &f, &G, &dynamics_denominator);
//  const ControlLyapunov dut(x, f, G, dynamics_denominator, u_vertices,
//                            state_constraints);
//
//  Eigen::MatrixXd state_swingup;
//  Eigen::MatrixXd control_swingup;
//  SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
//  Eigen::MatrixXd x_swingup(5, state_swingup.cols());
//  for (int i = 0; i < x_swingup.cols(); ++i) {
//    x_swingup.col(i) = ToTrigState<double>(state_swingup.col(i));
//  }
//  Eigen::MatrixXd in_roa_state(4, 2);
//  in_roa_state.col(0) = state_swingup.col(12);
//  in_roa_state.col(1) = state_swingup.col(13);
//  Eigen::MatrixXd in_roa_x(5, in_roa_state.cols());
//  for (int i = 0; i < in_roa_state.cols(); ++i) {
//    in_roa_x.col(i) = ToTrigState<double>(in_roa_state.col(i));
//  }
//
//  const symbolic::Variables x_set{x};
//  symbolic::Polynomial V_init = Load(x_set, load_V_init);
//  std::cout << "V_init(x_swingup): "
//            << V_init.EvaluateIndeterminates(x, x_swingup).transpose() <<
//            "\n";
//  const Eigen::VectorXd V_init_in_roa_x =
//      V_init.EvaluateIndeterminates(x, in_roa_x);
//  const double rho = 1;
//  if (V_init_in_roa_x.maxCoeff() > rho) {
//    V_init *= rho / V_init_in_roa_x.maxCoeff() * 0.99;
//  }
//
//  const int V_degree = 2;
//  const int lambda0_degree = 4;
//  const std::vector<int> l_degrees = {4, 4};
//  const std::vector<int> p_degrees = {6};
//  const int a_degree = 8;
//
//  const double converge_tol = 1E-5;
//  symbolic::Polynomial V_sol = V_init;
//
//  symbolic::Polynomial lambda0_sol;
//  VectorX<symbolic::Polynomial> l_sol;
//
//  auto minimize_a = [&x](solvers::MathematicalProgram* prog,
//                         const symbolic::Polynomial& a) {
//    // Minimize a(x).
//    const symbolic::Polynomial a_int_034 =
//        a.Integrate(x(0), -1, 1).Integrate(x(3), -1, 1).Integrate(x(4), -1,
//        1);
//    // Now sample many theta and do a Monte-Carlo integration.
//    Eigen::MatrixXd theta_monte_carlo_state(4, 100);
//    theta_monte_carlo_state.row(1) = Eigen::RowVectorXd::LinSpaced(
//        theta_monte_carlo_state.cols(), 0.5 * M_PI, 1.5 * M_PI);
//    Eigen::MatrixXd theta_monte_carlo_x(5, theta_monte_carlo_state.cols());
//    for (int i = 0; i < theta_monte_carlo_x.cols(); ++i) {
//      theta_monte_carlo_x.col(i) =
//          ToTrigState<double>(theta_monte_carlo_state.col(i));
//    }
//
//    auto cost = OptimizePolynomialAtSamples(
//        prog, a_int_034, x.segment<2>(1),
//        theta_monte_carlo_x.middleRows<2>(1),
//        OptimizePolynomialMode::kMinimizeMaximal);
//    return cost;
//  };
//  unused(minimize_a);
//
//  std::cout << "V_init(in_roa_x): "
//            << V_init.EvaluateIndeterminates(x, in_roa_x).transpose() << "\n";
//  int iter_count = 0;
//  int iter_max = 10;
//  bool converged = false;
//  while (iter_count < iter_max && !converged) {
//    {
//      // Construct the program to search for Lagrangian.
//      auto construct_lagrangian_prog =
//          [&dut, &V_sol, &u_vertices, deriv_eps, &x, &x_set, &lambda0_degree,
//           &l_degrees, &p_degrees](bool with_a, double rho_val,
//                                   symbolic::Polynomial* lambda0,
//                                   VectorX<symbolic::Polynomial>* l,
//                                   MatrixX<symbolic::Variable>* a_gram,
//                                   symbolic::Polynomial* vdot_sos) {
//            auto prog = std::make_unique<solvers::MathematicalProgram>();
//            prog->AddIndeterminates(x);
//            std::tie(*lambda0, std::ignore) =
//                prog->NewSosPolynomial(x_set, lambda0_degree);
//            l->resize(2);
//            for (int i = 0; i < 2; ++i) {
//              std::tie((*l)(i), std::ignore) =
//                  prog->NewSosPolynomial(x_set, l_degrees[i]);
//            }
//            Vector1<symbolic::Polynomial> p;
//            p(0) = prog->NewFreePolynomial(x_set, p_degrees[0]);
//            const int d_degree = 1;
//            std::optional<symbolic::Polynomial> a = symbolic::Polynomial();
//            if (with_a) {
//              std::tie(*a, *a_gram) = prog->NewSosPolynomial(x_set, a_degree);
//              prog->AddLinearCost(a_gram->cast<symbolic::Expression>().trace());
//            }
//            VectorX<symbolic::Monomial> vdot_monomials;
//            MatrixX<symbolic::Variable> vdot_gram;
//            dut.AddControlLyapunovConstraint(prog.get(), x, *lambda0,
//            d_degree,
//                                             *l, V_sol, rho_val, u_vertices,
//                                             deriv_eps, p, a, vdot_sos,
//                                             &vdot_monomials, &vdot_gram);
//            return prog;
//          };
//      symbolic::Polynomial lambda0;
//      VectorX<symbolic::Polynomial> l;
//      MatrixX<symbolic::Variable> a_gram;
//      symbolic::Polynomial vdot_sos;
//      auto prog_no_a = construct_lagrangian_prog(false, rho, &lambda0, &l,
//                                                 &a_gram, &vdot_sos);
//      solvers::SolverOptions solver_options;
//      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole,
//      1); const auto result_no_a =
//          solvers::Solve(*prog_no_a, std::nullopt, solver_options);
//      if (result_no_a.is_success()) {
//        converged = true;
//        break;
//      }
//      auto prog_with_a = construct_lagrangian_prog(true, rho, &lambda0, &l,
//                                                   &a_gram, &vdot_sos);
//      const double backoff_scale = 0.0;
//      const auto result_with_a =
//          SearchWithBackoff(prog_with_a.get(), solvers::MosekSolver::id(),
//                            solver_options, backoff_scale);
//      lambda0_sol =
//          result_with_a.GetSolution(lambda0).RemoveTermsWithSmallCoefficients(
//              1E-6);
//      GetPolynomialSolutions(result_with_a, l, 1E-6, &l_sol);
//      if (result_with_a.is_success()) {
//        const Eigen::MatrixXd a_gram_sol = result_with_a.GetSolution(a_gram);
//        drake::log()->info("Found Lagrangian, trace(a_gram) is {}",
//                           a_gram_sol.trace());
//        if (a_gram_sol.trace() < converge_tol) {
//          drake::log()->info("a converge to almost 0.");
//        }
//      } else {
//        drake::log()->info("Failed to find Lagrangian");
//        return V_sol;
//      }
//    }
//
//    {
//      // Given lambda0_sol and V_sol, search for V.
//      solvers::MathematicalProgram prog;
//      prog.AddIndeterminates(x);
//      symbolic::Polynomial V = NewFreePolynomialPassOrigin(
//          &prog, x_set, V_degree, "V", symbolic::internal::DegreeType::kAny,
//          symbolic::Variables({x(0), x(1), x(3), x(4)}));
//      symbolic::Polynomial positivity_sos = V;
//      const double positivity_eps = 1E-4;
//      positivity_sos -=
//          positivity_eps *
//          symbolic::Polynomial(
//              pow(x.cast<symbolic::Expression>().dot(x), V_degree / 2));
//      symbolic::Polynomial positivity_lagrangian =
//          prog.NewFreePolynomial(x_set, V_degree - 2);
//      positivity_sos -= positivity_lagrangian * state_constraints(0);
//      prog.AddSosConstraint(positivity_sos);
//
//      const int d_degree = 1;
//      Vector1<symbolic::Polynomial> p;
//      p(0) = prog.NewFreePolynomial(x_set, p_degrees[0]);
//      symbolic::Polynomial a;
//      MatrixX<symbolic::Variable> a_gram;
//      std::tie(a, a_gram) = prog.NewSosPolynomial(x_set, a_degree);
//      symbolic::Polynomial vdot_sos;
//      VectorX<symbolic::Monomial> vdot_monomials;
//      MatrixX<symbolic::Variable> vdot_gram;
//      dut.AddControlLyapunovConstraint(&prog, x, lambda0_sol, d_degree, l_sol,
//                                       V, rho, u_vertices, deriv_eps, p, a,
//                                       &vdot_sos, &vdot_monomials,
//                                       &vdot_gram);
//      // Now add the constraint V(in_roa_x) <= rho
//      {
//        Eigen::MatrixXd A_in_roa;
//        Eigen::VectorXd b_in_roa;
//        VectorX<symbolic::Variable> var_in_roa;
//        V.EvaluateWithAffineCoefficients(x, in_roa_x, &A_in_roa, &var_in_roa,
//                                         &b_in_roa);
//        prog.AddLinearConstraint(
//            A_in_roa, Eigen::VectorXd::Constant(b_in_roa.rows(), -kInf),
//            Eigen::VectorXd::Constant(b_in_roa.rows(), rho) - b_in_roa,
//            var_in_roa);
//      }
//      // Minimize a
//      auto cost =
//          prog.AddLinearCost(a_gram.cast<symbolic::Expression>().trace());
//      const double backoff_scale = 0.;
//      solvers::SolverOptions solver_options;
//      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole,
//      1); const auto result = SearchWithBackoff(&prog,
//      solvers::MosekSolver::id(),
//                                            solver_options, backoff_scale);
//      if (result.is_success()) {
//        V_sol = result.GetSolution(V);
//        const double cost_val = result.EvalBinding(cost)(0);
//        drake::log()->info("Find Lyapunov, cost={}", cost_val);
//        if (cost_val < converge_tol) {
//          drake::log()->info("Converged with a almost 0.");
//        }
//      } else {
//        drake::log()->error("Failed to find Lyapunov");
//        return V_sol;
//      }
//    }
//    iter_count++;
//  }
//  if (converged) {
//    ControlLyapunov::SearchOptions search_options;
//    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
//    search_options.lagrangian_step_solver_options->SetOption(
//        solvers::CommonSolverOption::kPrintToConsole, 1);
//    VectorX<symbolic::Polynomial> p_sol;
//    const bool found_lagrangian = dut.SearchLagrangian(
//        V_sol, rho, lambda0_degree, l_degrees, p_degrees, deriv_eps,
//        search_options, &lambda0_sol, &l_sol, &p_sol);
//    std::cout << "found Lagrangian in the end: " << found_lagrangian << "\n";
//    Save(V_sol, save_V);
//  }
//  std::cout << "V_sol(x_swingup): "
//            << V_sol.EvaluateIndeterminates(x, x_swingup).transpose() << "\n";
//  const double duration = 20;
//  Simulate(params, x, V_sol, u_max, deriv_eps, state_swingup.col(12),
//  duration); return V_sol;
//}

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

  // V_init = SearchClfNoInputBound(params, x, deriv_eps);

  Eigen::MatrixXd state_swingup;
  Eigen::MatrixXd control_swingup;
  SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
  std::cout << "state_swingup\n" << state_swingup.transpose() << "\n";
  Eigen::Matrix<double, 5, Eigen::Dynamic> x_swingup(5, state_swingup.cols());
  for (int i = 0; i < x_swingup.cols(); ++i) {
    x_swingup.col(i) = ToTrigState<double>(state_swingup.col(i));
  }
  symbolic::Polynomial V_sol;
  {
    ControlLyapunov::SearchOptions search_options;
    search_options.d_converge_tol = 0.0;
    search_options.bilinear_iterations = 20;
    search_options.backoff_scale = 0.013;
    // search_options.lagrangian_tiny_coeff_tol = 1E-5;
    search_options.lsol_tiny_coeff_tol = 1E-5;
    search_options.lyap_tiny_coeff_tol = 1E-7;
    // search_options.Vsol_tiny_coeff_tol = 1E-7;
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

    std::cout << "V_init at x_swingup "
              << V_init.EvaluateIndeterminates(x, x_swingup).transpose()
              << "\n";
    const bool search_inner_ellipsoid = true;
    if (search_inner_ellipsoid) {
      const double size_min = 0.;
      const double size_max = 0.5;
      const double size_tol = 0.001;
      const std::vector<int> ellipsoid_c_lagrangian_degrees{{V_degree - 2}};
      const Eigen::Matrix<double, 5, 1> x_star = ToTrigState<double>(
          0.55 * state_swingup.col(12) +
          0.45 * state_swingup.col(13));  // x_swingup.col(16);
      Eigen::Matrix<double, 5, 5> S;
      S.setZero();
      S(0, 0) = 100;
      S(1, 1) = 100;
      S(2, 2) = 100;
      S(3, 3) = 1;
      S(4, 4) = 1;
      const int r_degree = 0;
      const ControlLyapunov::EllipsoidBisectionOption
          ellipsoid_bisection_option(size_min, size_max, size_tol);
      const double ellipsoid_backoff_scale = 0.02;
      const ControlLyapunov::EllipsoidMaximizeOption ellipsoid_maximize_option(
          symbolic::Polynomial(), 0, ellipsoid_backoff_scale);
      symbolic::Polynomial r_sol;
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
      double ellipsoid_d_sol;
      VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S, r_degree,
                 search_options, ellipsoid_maximize_option, &V_sol,
                 &lambda0_sol, &l_sol, &r_sol, &p_sol,
                 &positivity_eq_lagrangian_sol, &ellipsoid_d_sol,
                 &ellipsoid_c_lagrangian_sol);
      std::cout << "V(x_swingup): "
                << V_sol.EvaluateIndeterminates(x, x_swingup).transpose()
                << "\n";
    } else {
      std::vector<int> x_indices = {14, 15, 16, 17, 18, 19, 20, 21,
                                    22, 23, 24, 25, 26, 27, 28};
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

  double u_max = 200;
  const double deriv_eps = 0.075;
  symbolic::Polynomial V_sol;
  bool found_u_max = false;
  // SearchWithSlackA(
  //    params, x, u_max, deriv_eps,
  //    //"/home/hongkaidai/Dropbox/sos_clf_cbf/cart_pole_trig_clf_last16_1.txt",
  //    "cart_pole_trig_clf0.txt", "cart_pole_trig_clf1.txt");
  // abort();
  int u_max_trial = 0;
  while (u_max < 250) {
    std::cout << "Try u_max = " << u_max << "\n";
    // std::optional<std::string> load_file =
    //    "/home/hongkaidai/Dropbox/sos_clf_cbf/cart_pole_trig_clf_last17_21.txt";
    std::optional<std::string> load_file =
        "sos_data/cart_pole_trig_clf52.txt.iter1";
    SearchResultDetails search_result_details;
    V_sol = SearchWTrigDynamics(params, x, u_max, deriv_eps, load_file,
                                "sos_data/cart_pole_trig_clf53.txt",
                                &search_result_details);
    if (search_result_details.num_bilinear_iterations >= 1 ||
        search_result_details.bilinear_iteration_status !=
            BilinearIterationStatus::kFailLagrangian) {
      found_u_max = true;
      std::cout << fmt::format("Found u_max = {} after {} trials\n", u_max,
                               u_max_trial);
      const double duration = 20;
      Simulate(params, x, V_sol, u_max, deriv_eps, Eigen::Vector4d(0, 0, 0, 0),
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
