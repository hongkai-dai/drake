#include <math.h>

#include <iostream>
#include <limits>

#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/test/quadrotor2d.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

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

[[maybe_unused]] symbolic::Polynomial SearchWithSlackA(
    const QuadrotorPlant<double>& quadrotor,
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, double thrust_max,
    double deriv_eps,
    const std::vector<VectorX<symbolic::Polynomial>>& unsafe_regions,
    const Eigen::MatrixXd& x_safe,
    const std::optional<std::string>& load_cbf_file) {
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));

  const std::optional<symbolic::Polynomial> dynamics_denominator = std::nullopt;

  const double beta = 0.1;
  const ControlBarrier dut(f, G, dynamics_denominator, x, beta, unsafe_regions,
                           u_vertices, state_constraints);

  // Set h_init to some values, it should satisfy h_init(x_safe) >= 0.
  const int h_degree = 4;
  const symbolic::Variables x_set(x);
  // symbolic::Polynomial h_init = FindCbfInit(x, h_degree);
  symbolic::Polynomial h_init;
  if (load_cbf_file.has_value()) {
    h_init = Load(x_set, load_cbf_file.value());
  } else {
    const VectorX<symbolic::Monomial> h_monomials =
        ComputeMonomialBasisNoConstant(x_set, h_degree / 2,
                                       symbolic::internal::DegreeType::kAny);
    h_init = symbolic::Polynomial(1 - h_monomials.dot(h_monomials));
  }
  const Eigen::VectorXd h_init_x_safe =
      h_init.EvaluateIndeterminates(x, x_safe);
  std::cout << "h_init(x_safe): " << h_init_x_safe.transpose() << "\n";
  if ((h_init_x_safe.array() < 0).any()) {
    h_init -= h_init_x_safe.minCoeff();
    h_init += 0.1;
  }

  const int lambda0_degree = 4;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_eq_lagrangian_degrees = {6};
  const int hdot_a_degree = 8;

  const std::vector<int> t_degrees = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{h_degree - 2},
                                                   {h_degree - 2}};
  const std::vector<std::vector<int>> unsafe_eq_lagrangian_degrees = {
      {h_degree - 2}, {h_degree - 2}};
  const std::vector<int> unsafe_a_degrees = {h_degree, h_degree};

  symbolic::Polynomial h_sol = h_init;

  symbolic::Polynomial lambda0_sol;
  VectorX<symbolic::Polynomial> l_sol;
  int iter_count = 0;
  const int iter_max = 40;
  const double a_is_zero_tol = 1E-8;
  bool hdot_a_is_zero = false;
  std::vector<bool> unsafe_a_is_zero(unsafe_regions.size(), false);
  bool converged = false;
  while (iter_count < iter_max && !converged) {
    {
      // Search for hdot Lagrangian and a(x)
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      symbolic::Polynomial lambda0;
      std::tie(lambda0, std::ignore) =
          prog.NewSosPolynomial(x_set, lambda0_degree);
      VectorX<symbolic::Polynomial> l(4);
      for (int i = 0; i < 4; ++i) {
        std::tie(l(i), std::ignore) =
            prog.NewSosPolynomial(x_set, l_degrees[i]);
      }
      VectorX<symbolic::Polynomial> hdot_eq_lagrangians(1);
      hdot_eq_lagrangians(0) =
          prog.NewFreePolynomial(x_set, hdot_eq_lagrangian_degrees[0]);
      std::optional<symbolic::Polynomial> a = std::nullopt;
      std::optional<MatrixX<symbolic::Variable>> a_gram = std::nullopt;
      if (!hdot_a_is_zero) {
        a.emplace(symbolic::Polynomial());
        a_gram.emplace(MatrixX<symbolic::Variable>());
        std::tie(*a, *a_gram) = prog.NewSosPolynomial(x_set, hdot_a_degree);
      }
      symbolic::Polynomial hdot_sos;
      VectorX<symbolic::Monomial> hdot_monomials;
      MatrixX<symbolic::Variable> hdot_gram;
      dut.AddControlBarrierConstraint(&prog, lambda0, l, hdot_eq_lagrangians,
                                      h_sol, deriv_eps, a, &hdot_sos,
                                      &hdot_monomials, &hdot_gram);
      if (a_gram.has_value()) {
        // Add the cost to minimize the trace of a_gram;
        prog.AddLinearCost(a_gram->cast<symbolic::Expression>().trace());
      }
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      const auto result = solvers::Solve(prog, std::nullopt, solver_options);
      if (result.is_success()) {
        lambda0_sol = result.GetSolution(lambda0);
        if (a_gram.has_value()) {
          const auto a_gram_sol = result.GetSolution(*a_gram);
          drake::log()->info("hdot_a_gram.trace()={}", a_gram_sol.trace());
          if (a_gram_sol.trace() <= a_is_zero_tol) {
            hdot_a_is_zero = true;
          }
        }
        GetPolynomialSolutions(result, l, 0., &l_sol);
      } else {
        drake::log()->error("Failed to find Lagrangian for hdot condition");
        return h_sol;
      }
    }
    VectorX<symbolic::Polynomial> t_sol(unsafe_regions.size());
    {
      // Search Lagrangian for unsafe regions.
      for (int i = 0; i < static_cast<int>(unsafe_regions.size()); ++i) {
        solvers::MathematicalProgram prog;
        prog.AddIndeterminates(x);
        symbolic::Polynomial t;
        std::tie(t, std::ignore) = prog.NewSosPolynomial(x_set, t_degrees[i]);
        VectorX<symbolic::Polynomial> s(unsafe_regions[i].rows());
        for (int j = 0; j < s.rows(); ++j) {
          std::tie(s(j), std::ignore) =
              prog.NewSosPolynomial(x_set, s_degrees[i][j]);
        }
        VectorX<symbolic::Polynomial> unsafe_eq_lagrangian(1);
        unsafe_eq_lagrangian(0) =
            prog.NewFreePolynomial(x_set, unsafe_eq_lagrangian_degrees[i][0]);
        std::optional<symbolic::Polynomial> a = std::nullopt;
        std::optional<MatrixX<symbolic::Variable>> a_gram = std::nullopt;
        if (!unsafe_a_is_zero[i]) {
          a.emplace(symbolic::Polynomial());
          a_gram.emplace(MatrixX<symbolic::Variable>());
          std::tie(*a, *a_gram) =
              prog.NewSosPolynomial(x_set, unsafe_a_degrees[i]);
        }
        symbolic::Polynomial unsafe_sos_poly =
            (1 + t) * -h_sol + s.dot(unsafe_regions[i]) -
            unsafe_eq_lagrangian.dot(state_constraints) +
            a.value_or(symbolic::Polynomial());
        prog.AddSosConstraint(unsafe_sos_poly);
        if (a_gram.has_value()) {
          prog.AddLinearCost(a_gram->cast<symbolic::Expression>().trace());
        }
        solvers::SolverOptions solver_options;
        solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole,
                                 0);
        const auto result = solvers::Solve(prog, std::nullopt, solver_options);
        if (result.is_success()) {
          t_sol(i) = result.GetSolution(t);
          if (a_gram.has_value()) {
            const auto a_gram_sol = result.GetSolution(*a_gram);
            drake::log()->info("unsafe region {} a_gram.trace()={}", i,
                               a_gram_sol.trace());
            if (a_gram_sol.trace() <= a_is_zero_tol) {
              unsafe_a_is_zero[i] = true;
            }
          }
        } else {
          drake::log()->error("Cannot find Lagrangian for unsafe region {}", i);
          return h_sol;
        }
      }
    }
    if (hdot_a_is_zero &&
        std::all_of(unsafe_a_is_zero.begin(), unsafe_a_is_zero.end(),
                    [](int flag) { return flag; })) {
      converged = true;
      break;
    }

    {
      // Find h_sol
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      symbolic::Polynomial h = prog.NewFreePolynomial(x_set, h_degree);
      VectorX<symbolic::Polynomial> hdot_eq_lagrangian(1);
      hdot_eq_lagrangian(0) =
          prog.NewFreePolynomial(x_set, hdot_eq_lagrangian_degrees[0]);
      std::optional<symbolic::Polynomial> hdot_a = std::nullopt;
      std::optional<MatrixX<symbolic::Variable>> hdot_a_gram = std::nullopt;
      if (!hdot_a_is_zero) {
        hdot_a.emplace(symbolic::Polynomial());
        hdot_a_gram.emplace(MatrixX<symbolic::Variable>());
        std::tie(*hdot_a, *hdot_a_gram) =
            prog.NewSosPolynomial(x_set, hdot_a_degree);
      }
      symbolic::Polynomial hdot_sos;
      VectorX<symbolic::Monomial> hdot_monomials;
      MatrixX<symbolic::Variable> hdot_sos_gram;
      dut.AddControlBarrierConstraint(
          &prog, lambda0_sol, l_sol, hdot_eq_lagrangian, h, deriv_eps, hdot_a,
          &hdot_sos, &hdot_monomials, &hdot_sos_gram);
      if (hdot_a_gram.has_value()) {
        prog.AddLinearCost(hdot_a_gram->cast<symbolic::Expression>().trace());
      }

      // Add constraint for each unsafe region.
      std::vector<std::optional<symbolic::Polynomial>> unsafe_a(
          unsafe_regions.size(), std::nullopt);
      std::vector<std::optional<MatrixX<symbolic::Variable>>> unsafe_a_gram(
          unsafe_regions.size(), std::nullopt);
      for (int i = 0; i < static_cast<int>(unsafe_regions.size()); ++i) {
        VectorX<symbolic::Polynomial> s(unsafe_regions[i].rows());
        for (int j = 0; j < unsafe_regions[i].rows(); ++j) {
          std::tie(s(j), std::ignore) =
              prog.NewSosPolynomial(x_set, s_degrees[i][j]);
        }
        VectorX<symbolic::Polynomial> unsafe_eq_lagrangian(1);
        unsafe_eq_lagrangian(0) =
            prog.NewFreePolynomial(x_set, unsafe_eq_lagrangian_degrees[i][0]);
        if (!unsafe_a_is_zero[i]) {
          unsafe_a[i].emplace(symbolic::Polynomial());
          unsafe_a_gram[i].emplace(MatrixX<symbolic::Variable>());
          std::tie(*unsafe_a[i], *unsafe_a_gram[i]) =
              prog.NewSosPolynomial(x_set, unsafe_a_degrees[i]);
        }
        const symbolic::Polynomial unsafe_sos_poly =
            (1 + t_sol(i)) * -h + s.dot(unsafe_regions[i]) -
            unsafe_eq_lagrangian.dot(state_constraints) +
            unsafe_a[i].value_or(symbolic::Polynomial());
        prog.AddSosConstraint(unsafe_sos_poly);
        if (unsafe_a_gram[i].has_value()) {
          prog.AddLinearCost(
              unsafe_a_gram[i]->cast<symbolic::Expression>().trace());
        }
      }
      // Add constraint h(x_safe) >= 0.
      {
        Eigen::MatrixXd A_safe;
        Eigen::VectorXd b_safe;
        VectorX<symbolic::Variable> var_safe;
        h.EvaluateWithAffineCoefficients(x, x_safe, &A_safe, &var_safe,
                                         &b_safe);
        const double h_safe_states_min = 0.01;
        prog.AddLinearConstraint(
            A_safe,
            Eigen::VectorXd::Constant(b_safe.rows(), h_safe_states_min) -
                b_safe,
            Eigen::VectorXd::Constant(b_safe.rows(), kInf), var_safe);
      }

      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
      const auto result = solvers::Solve(prog, std::nullopt, solver_options);
      if (result.is_success()) {
        h_sol = result.GetSolution(h);
        drake::log()->info("barrier step in iter {}", iter_count);
        if (hdot_a.has_value()) {
          const auto hdot_a_gram_sol = result.GetSolution(*hdot_a_gram);
          drake::log()->info("hdot_a_gram.trace()={}", hdot_a_gram_sol.trace());
          if (hdot_a_gram_sol.trace() <= a_is_zero_tol) {
            hdot_a_is_zero = true;
          }
        }
        for (int i = 0; i < static_cast<int>(unsafe_regions.size()); ++i) {
          if (unsafe_a_gram[i].has_value()) {
            const auto unsafe_a_gram_sol =
                result.GetSolution(*(unsafe_a_gram[i]));
            drake::log()->info("unsafe region {} a_gram.trace()={}", i,
                               unsafe_a_gram_sol.trace());
            if (unsafe_a_gram_sol.trace() <= a_is_zero_tol) {
              unsafe_a_is_zero[i] = true;
            }
          }
        }
      } else {
        drake::log()->error("Cannot find h");
        return h_sol;
      }
    }
    if (hdot_a_is_zero &&
        std::all_of(unsafe_a_is_zero.begin(), unsafe_a_is_zero.end(),
                    [](int flag) { return flag; })) {
      converged = true;
      break;
    }
    iter_count++;
  }

  ControlBarrier::SearchOptions search_options;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  VectorX<symbolic::Polynomial> hdot_eq_lagrangian_sol;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  std::vector<VectorX<symbolic::Polynomial>> unsafe_eq_lagrangian_sol;
  bool is_h_valid = dut.SearchLagrangian(
      h_sol, deriv_eps, lambda0_degree, l_degrees, hdot_eq_lagrangian_degrees,
      t_degrees, s_degrees, unsafe_eq_lagrangian_degrees, search_options,
      &lambda0_sol, &l_sol, &hdot_eq_lagrangian_sol, &t_sol, &s_sol,
      &unsafe_eq_lagrangian_sol);
  drake::log()->info("h_sol is valid? {}", is_h_valid);
  return h_sol;
}

[[maybe_unused]] symbolic::Polynomial Search(
    const QuadrotorPlant<double>& quadrotor,
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, double thrust_max,
    double deriv_eps,
    const std::vector<VectorX<symbolic::Polynomial>>& unsafe_regions) {
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(StateEqConstraint(x));

  const std::optional<symbolic::Polynomial> dynamics_denominator = std::nullopt;

  const double beta = 1;
  const ControlBarrier dut(f, G, dynamics_denominator, x, beta, unsafe_regions,
                           u_vertices, state_constraints);

  const int h_degree = 2;
  symbolic::Polynomial h_init = FindCbfInit(x, h_degree);
  const int lambda0_degree = 4;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_eq_lagrangian_degrees{h_degree - 1};
  const std::vector<int> t_degree = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{h_degree - 2},
                                                   {h_degree - 2}};
  const std::vector<std::vector<int>>
      unsafe_state_constraints_lagrangian_degrees{{h_degree - 1},
                                                  {h_degree - 1}};
  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  std::vector<ControlBarrier::EllipsoidBisectionOption>
      ellipsoid_bisection_options;
  ellipsoids.emplace_back(Eigen::Matrix<double, 7, 1>::Zero(),
                          Eigen::Matrix<double, 7, 7>::Identity(), 0,
                          h_degree - 2, std::vector<int>{{h_degree - 1}});
  ellipsoid_bisection_options.emplace_back(0, 2, 0.01);

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
             hdot_eq_lagrangian_degrees, t_degree, s_degrees,
             unsafe_state_constraints_lagrangian_degrees, x_anchor,
             search_options, &ellipsoids, &ellipsoid_bisection_options, &h_sol,
             &lambda0_sol, &l_sol, &hdot_state_constraints_lagrangian_sol,
             &t_sol, &s_sol, &unsafe_state_constraints_lagrangian_sol);
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
  const double thrust_equilibrium = EquilibriumThrust(plant);
  const double thrust_max = 3 * thrust_equilibrium;
  const double deriv_eps = 0.5;
  // The unsafe region is the ground and the ceiling.
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions(2);
  unsafe_regions[0].resize(1);
  unsafe_regions[0](0) = symbolic::Polynomial(x(1) + 0.3);
  unsafe_regions[1].resize(1);
  unsafe_regions[1](0) = symbolic::Polynomial(0.5 - x(1));

  Eigen::MatrixXd safe_states(6, 4);
  safe_states.col(0) << 0, 0, 0, 0, 0, 0;
  safe_states.col(1) << 0, 0.2, 0, 0, 0, 0;
  safe_states.col(2) << 0, -0.2, 0, 0, 0, 0;
  safe_states.col(3) << 0, 0.45, 0.1, 0, 0, 0;
  Eigen::MatrixXd x_safe(7, safe_states.cols());
  for (int i = 0; i < safe_states.cols(); ++i) {
    x_safe.col(i) = ToTrigState<double>(safe_states.col(i));
  }

  std::optional<std::string> load_cbf_file = std::nullopt;
  load_cbf_file = "sos_data/quadrotor2d_trig_cbf4.txt";
  const symbolic::Polynomial h_sol = SearchWithSlackA(
      plant, x, thrust_max, deriv_eps, unsafe_regions, x_safe, load_cbf_file);
  Save(h_sol, "sos_data/quadrotor2d_trig_cbf5.txt");

  // const symbolic::Polynomial h_sol = Search(plant, x, thrust_max, deriv_eps,
  // unsafe_regions);
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
