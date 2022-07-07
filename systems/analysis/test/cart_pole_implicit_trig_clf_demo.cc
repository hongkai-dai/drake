#include <iostream>
#include <limits>

#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/test/cart_pole.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

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
  // VerifyLyapunovInit(x, V_sol, dynamics_numerator, dynamics_denominator);
  // VerifyLyapunovInitPablo(x, V_sol, dynamics_numerator,
  // dynamics_denominator);
  return V_sol;
}

// Add the constraint (1+λ₀(x, z))xᵀx(V(x)−ρ) − l₁(x, z) * (∂V/∂q*q̇+∂V/∂v*z¹+εV)
// + l₂(x, z)(∂V/∂q*q̇+∂V/∂v*z²+εV)−p(x, z)c(x, z) is sos
symbolic::Polynomial AddControlLyapunovConstraint(
    solvers::MathematicalProgram* prog,
    const Eigen::Matrix<symbolic::Variable, 5, 1>& x,
    const Vector2<symbolic::Polynomial>& z1_poly,
    const Vector2<symbolic::Polynomial>& z2_poly,
    const symbolic::Polynomial& lambda0, const symbolic::Polynomial& V,
    double rho, const Vector2<symbolic::Polynomial>& l,
    const Vector3<symbolic::Polynomial>& qdot, double deriv_eps,
    const Eigen::Matrix<symbolic::Polynomial, 5, 1>& p,
    const Eigen::Matrix<symbolic::Polynomial, 5, 1>& state_constraints) {
  symbolic::Polynomial vdot_sos =
      (1 + lambda0) *
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x)) * (V - rho);
  const RowVector3<symbolic::Polynomial> dVdq = V.Jacobian(x.head<3>());
  vdot_sos -= l.sum() * (dVdq.dot(qdot) + deriv_eps * V);
  const RowVector2<symbolic::Polynomial> dVdv = V.Jacobian(x.tail<2>());
  vdot_sos -= l(0) * dVdv.dot(z1_poly);
  vdot_sos -= l(1) * dVdv.dot(z2_poly);
  vdot_sos -= p.dot(state_constraints);
  prog->AddSosConstraint(vdot_sos);
  return vdot_sos;
}

bool FindLagrangian(const Eigen::Matrix<symbolic::Variable, 5, 1>& x,
                    const Vector2<symbolic::Variable>& z1,
                    const Vector2<symbolic::Variable>& z2, int lambda0_degree,
                    const symbolic::Polynomial& V, double rho,
                    const std::vector<int>& l_degrees,
                    const std::vector<int>& p_degrees,
                    const Vector3<symbolic::Polynomial>& qdot, double deriv_eps,
                    Eigen::Matrix<symbolic::Polynomial, 5, 1> state_constraints,
                    symbolic::Polynomial* lambda0_sol,
                    Vector2<symbolic::Polynomial>* l_sol) {
  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(x);
  prog.AddIndeterminates(z1);
  prog.AddIndeterminates(z2);
  symbolic::Polynomial lambda0;
  symbolic::Variables xz_set{x};
  xz_set.insert(symbolic::Variables(z1));
  xz_set.insert(symbolic::Variables(z2));
  std::tie(lambda0, std::ignore) = prog.NewSosPolynomial(
      xz_set, lambda0_degree,
      solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Lambda");
  Vector2<symbolic::Polynomial> l;
  for (int i = 0; i < 2; ++i) {
    std::tie(l(i), std::ignore) = prog.NewSosPolynomial(
        xz_set, l_degrees[i],
        solvers::MathematicalProgram::NonnegativePolynomial::kSos,
        "l" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 5, 1> p;
  for (int i = 0; i < 5; ++i) {
    p(i) =
        prog.NewFreePolynomial(xz_set, p_degrees[i], "p" + std::to_string(i));
  }
  Vector2<symbolic::Polynomial> z1_poly;
  Vector2<symbolic::Polynomial> z2_poly;
  for (int i = 0; i < 2; ++i) {
    z1_poly(i) = symbolic::Polynomial(z1(i));
    z2_poly(i) = symbolic::Polynomial(z2(i));
  }
  AddControlLyapunovConstraint(&prog, x, z1_poly, z2_poly, lambda0, V, rho, l,
                               qdot, deriv_eps, p, state_constraints);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  RemoveTinyCoeff(&prog, 1E-9);
  std::cout << "Smallest coeff in Lagrangian program: " << SmallestCoeff(prog)
            << "\n";
  std::cout << "Largest coeff in Lagrangian program: " << LargestCoeff(prog)
            << "\n";
  // solver_options.SetOption(solvers::MosekSolver::id(), "writedata",
  // "cart_pole_implicit_trig_lagrangian.task.gz");
  const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  if (result.is_success()) {
    *lambda0_sol = result.GetSolution(lambda0);
    const double lsol_tol = 1E-5;
    *lambda0_sol = lambda0_sol->RemoveTermsWithSmallCoefficients(lsol_tol);
    for (int i = 0; i < 2; ++i) {
      (*l_sol)(i) = result.GetSolution(l(i));
      (*l_sol)(i) = (*l_sol)(i).RemoveTermsWithSmallCoefficients(lsol_tol);
    }
    return true;
  } else {
    drake::log()->error("Failed to find Lagrangian");
    return false;
  }
}

void Search(const std::optional<std::string>& load_V_init) {
  const CartPoleParams params;
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const int V_degree = 2;
  symbolic::Polynomial V_init = FindClfInit(params, V_degree, x);
  const Vector2<symbolic::Variable> z1(symbolic::Variable("z1(0)"),
                                       symbolic::Variable("z1(1)"));
  const Vector2<symbolic::Variable> z2(symbolic::Variable("z2(0)"),
                                       symbolic::Variable("z2(1)"));
  const double u_max = 40;
  const Eigen::Matrix<symbolic::Expression, 5, 1> x_expr =
      x.cast<symbolic::Expression>();
  const Matrix2<symbolic::Expression> M_expr =
      MassMatrix<symbolic::Expression>(params, x_expr);
  const Vector2<symbolic::Expression> bias_expr =
      CalcBiasTerm<symbolic::Expression>(params, x_expr) -
      CalcGravityVector<symbolic::Expression>(params, x_expr);
  Eigen::Matrix<symbolic::Polynomial, 5, 1> state_constraints;
  state_constraints(0) = StateEqConstraint(x);
  const Vector2<symbolic::Expression> constraint_expr1 =
      M_expr * z1 - Eigen::Vector2d(u_max, 0) + bias_expr;
  const Vector2<symbolic::Expression> constraint_expr2 =
      M_expr * z2 - Eigen::Vector2d(-u_max, 0) + bias_expr;
  for (int i = 0; i < 2; ++i) {
    state_constraints(1 + i) = symbolic::Polynomial(constraint_expr1(i));
    state_constraints(3 + i) = symbolic::Polynomial(constraint_expr2(i));
  }
  symbolic::Variables xz_set{x};
  xz_set.insert(symbolic::Variables(z1));
  xz_set.insert(symbolic::Variables(z2));
  const Vector3<symbolic::Expression> qdot_expr =
      CalcQdot<symbolic::Expression>(x_expr);
  Vector3<symbolic::Polynomial> qdot;
  for (int i = 0; i < 3; ++i) {
    qdot(i) = symbolic::Polynomial(qdot_expr(i));
  }
  Vector2<symbolic::Polynomial> z1_poly;
  Vector2<symbolic::Polynomial> z2_poly;
  for (int i = 0; i < 2; ++i) {
    z1_poly(i) = symbolic::Polynomial(z1(i));
    z2_poly(i) = symbolic::Polynomial(z2(i));
  }
  const symbolic::Variables x_set{x};

  const double deriv_eps = 0.01;
  int lambda0_degree = 4;
  const std::vector<int> l_degrees{{4, 4}};
  const std::vector<int> p_degrees{{6, 6, 6, 6, 6}};

  double rho_sol;
  if (load_V_init.has_value()) {
    V_init = Load(symbolic::Variables(x), load_V_init.value());
    rho_sol = 1;
  } else {
    const bool binary_search_rho = true;
    // Maximize rho
    if (binary_search_rho) {
      double rho_max = 0.01;
      double rho_min = 1E-4;
      double rho_tol = 1E-5;

      auto is_rho_feasible = [&x, &z1, &z2, lambda0_degree, &V_init, &l_degrees,
                              &p_degrees, &qdot, deriv_eps,
                              &state_constraints](double rho) {
        symbolic::Polynomial lambda0_sol;
        Vector2<symbolic::Polynomial> l_sol;
        return FindLagrangian(x, z1, z2, lambda0_degree, V_init / rho, 1,
                              l_degrees, p_degrees, qdot, deriv_eps,
                              state_constraints, &lambda0_sol, &l_sol);
      };
      if (is_rho_feasible(rho_max)) {
        rho_sol = rho_max;
        V_init = V_init / rho_sol;
        rho_sol = 1;
      } else if (!is_rho_feasible(rho_min)) {
        rho_sol = -kInf;
        return;
      } else {
        while (rho_max - rho_min > rho_tol) {
          const double rho_mid = (rho_max + rho_min) / 2;
          std::cout << fmt::format("rho_max={}, rho_min={}, rho_mid={}\n",
                                   rho_max, rho_min, rho_mid);
          if (is_rho_feasible(rho_mid)) {
            rho_min = rho_mid;
          } else {
            rho_max = rho_mid;
          }
        }
        rho_sol = rho_min;
      }
    } else {
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      prog.AddIndeterminates(z1);
      prog.AddIndeterminates(z2);
      const int d_degree = lambda0_degree / 2 + 1;
      const symbolic::Variable rho = prog.NewContinuousVariables<1>("rho")(0);
      symbolic::Polynomial vdot_sos =
          symbolic::Polynomial(pow(x_expr.dot(x), d_degree)) * (V_init - rho);
      Vector2<symbolic::Polynomial> l;
      const RowVector3<symbolic::Polynomial> dVdq =
          V_init.Jacobian(x.head<3>());
      for (int i = 0; i < 2; ++i) {
        std::tie(l(i), std::ignore) = prog.NewSosPolynomial(
            xz_set, l_degrees[i],
            solvers::MathematicalProgram::NonnegativePolynomial::kSos, "l");
      }
      const RowVector2<symbolic::Polynomial> dVdv =
          V_init.Jacobian(x.tail<2>());
      vdot_sos -= l.sum() * (dVdq.dot(qdot) + deriv_eps * V_init);
      vdot_sos -= l(0) * dVdv.dot(z1_poly) + l(1) * dVdv.dot(z2_poly);
      Eigen::Matrix<symbolic::Polynomial, 5, 1> p;
      for (int i = 0; i < 5; ++i) {
        p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i],
                                      "p" + std::to_string(i));
      }
      vdot_sos -= p.dot(state_constraints);
      prog.AddSosConstraint(vdot_sos);
      prog.AddLinearCost(Vector1d(-1), 0, Vector1<symbolic::Variable>(rho));
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      // solver_options.SetOption(solvers::MosekSolver::id(), "writedata",
      //                         "cart_pole_implicit_trig_max_rho.task.gz");
      RemoveTinyCoeff(&prog, 1E-9);
      std::cout << "Smallest coeff: " << SmallestCoeff(prog) << "\n";
      std::cout << "Largest coeff: " << LargestCoeff(prog) << "\n";
      const auto result = solvers::Solve(prog, std::nullopt, solver_options);
      DRAKE_DEMAND(result.is_success());
      rho_sol = result.GetSolution(rho);
      std::cout << "V_init <= " << rho_sol << "\n";
      DRAKE_DEMAND(rho_sol > 0);
    }
  }

  const int max_iters = 10;
  int iter_count = 0;
  symbolic::Polynomial V_sol = V_init;
  const double rho = rho_sol;
  double prev_cost = kInf;
  std::cout << "start bilinear alternation\n";
  while (iter_count < max_iters) {
    symbolic::Polynomial lambda0_sol;
    Vector2<symbolic::Polynomial> l_sol;
    {
      const bool found_lagrangian = FindLagrangian(
          x, z1, z2, lambda0_degree, V_sol, rho, l_degrees, p_degrees, qdot,
          deriv_eps, state_constraints, &lambda0_sol, &l_sol);
      if (!found_lagrangian) {
        std::cout << "rho=" << rho_sol << "\n";
        return;
      }
    }

    // Search for V.
    {
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      prog.AddIndeterminates(z1);
      prog.AddIndeterminates(z2);
      symbolic::Polynomial V = NewFreePolynomialPassOrigin(
          &prog, x_set, V_degree, "V", symbolic::internal::DegreeType::kAny,
          symbolic::Variables{});
      // First add the constraint that V −ε₁(xᵀx)ᵈ − p₁(x)c₁(x) is sos.
      const double positivity_eps = 0.0001;
      const int d = V_degree / 2;
      symbolic::Polynomial positivity_sos =
          V - positivity_eps * symbolic::Polynomial(pow(
                                   x.cast<symbolic::Expression>().dot(x), d));
      const int positivity_lagrangian_degree{V_degree - 2};
      symbolic::Polynomial positivity_lagrangian =
          prog.NewFreePolynomial(x_set, positivity_lagrangian_degree);
      positivity_sos -= positivity_lagrangian * StateEqConstraint(x);
      prog.AddSosConstraint(positivity_sos);
      // Now add the constraint on Vdot.
      Eigen::Matrix<symbolic::Polynomial, 5, 1> p;
      for (int i = 0; i < 5; ++i) {
        p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i],
                                      "p" + std::to_string(i));
      }
      const symbolic::Polynomial vdot_sos = AddControlLyapunovConstraint(
          &prog, x, z1_poly, z2_poly, lambda0_sol, V, rho, l_sol, qdot,
          deriv_eps, p, state_constraints);

      // Now minimize V on x_samples.
      Eigen::Matrix<double, 5, 4> x_samples;
      x_samples.col(0) = ToTrigState<double>(Eigen::Vector4d(0, 0, 0, 0));
      x_samples.col(1) =
          ToTrigState<double>(Eigen::Vector4d(0, M_PI * 1.05, 0, 0));
      x_samples.col(2) =
          ToTrigState<double>(Eigen::Vector4d(0, M_PI * 1.1, 0, 0));
      x_samples.col(3) =
          ToTrigState<double>(Eigen::Vector4d(0, M_PI * 0.1, 0, 0));
      OptimizePolynomialAtSamples(&prog, V, x, x_samples,
                                  OptimizePolynomialMode::kMinimizeMaximal);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      const double backoff_scale = 0.;
      std::cout << "Smallest coeff: " << SmallestCoeff(prog) << "\n";
      const auto result_lyapunov = SearchWithBackoff(
          &prog, solvers::MosekSolver::id(), solver_options, backoff_scale);
      if (result_lyapunov.is_success()) {
        V_sol = result_lyapunov.GetSolution(V);
        drake::log()->info(
            "Optimal cost = {}",
            V_sol.EvaluateIndeterminates(x, x_samples).maxCoeff());

        Save(V_sol, "cart_pole_implicit_trig_clf.txt");
        const Eigen::VectorXd V_at_samples =
            V_sol.EvaluateIndeterminates(x, x_samples);
        std::cout << "V at samples: " << V_at_samples.transpose() << "\n";
        const double curr_cost = V_at_samples.maxCoeff();
        if (curr_cost < prev_cost) {
          prev_cost = curr_cost;
        } else {
          std::cout << "rho: " << rho << "\n";
          return;
        }
      } else {
        drake::log()->error("Failed to find Lyapunov");
        std::cout << "rho: " << rho << "\n";
        return;
      }
    }
    iter_count++;
  }
}

int DoMain() {
  Search(std::nullopt);
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  drake::systems::analysis::DoMain();
}
