#include <iostream>
#include <limits>

#include "drake/common/find_resource.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
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

symbolic::Polynomial FindClfInit(
    const CartPoleParams& params, int V_degree,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x) {
  Eigen::Matrix<double, 5, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10;
  const Eigen::Matrix<double, 5, 5> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = SynthesizeCartpoleTrigLqr(params, lqr_Q, 20);

  const symbolic::Expression u_lqr = -lqr_result.K.row(0).dot(x);
  Eigen::Matrix<symbolic::Expression, 5, 1> n_expr;
  symbolic::Expression d_expr;
  CartpoleTrigDynamics<symbolic::Expression>(params, x.cast<symbolic::Expression>(),
                                     u_lqr, &n_expr, &d_expr);
  Eigen::Matrix<symbolic::Polynomial, 5, 1> dynamics_numerator;
  for (int i = 0; i < 5; ++i) {
    dynamics_numerator(i) = symbolic::Polynomial(n_expr(i));
  }
  const symbolic::Polynomial dynamics_denominator =
      symbolic::Polynomial(d_expr);

  const double positivity_eps = 0.001;
  const int d = V_degree / 2;
  const double kappa = 0.01;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      CartpoleStateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{{4}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{V_degree - 2};
  const std::vector<int> derivative_cin_lagrangian_degrees{{2}};

  auto ret = FindCandidateRegionalLyapunov(
      x, dynamics_numerator, dynamics_denominator, V_degree, positivity_eps, d,
      kappa, state_eq_constraints, positivity_ceq_lagrangian_degrees,
      derivative_ceq_lagrangian_degrees, state_ineq_constraints,
      positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*(ret.prog), std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(ret.V);
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
    const Vector2<symbolic::Polynomial>& z2_poly, double z_factor,
    const symbolic::Polynomial& lambda0, const symbolic::Polynomial& V,
    double rho, const Vector2<symbolic::Polynomial>& l,
    const Vector3<symbolic::Polynomial>& qdot, double kappa,
    const Eigen::Matrix<symbolic::Polynomial, 5, 1>& p,
    const Eigen::Matrix<symbolic::Polynomial, 5, 1>& state_constraints,
    MatrixX<symbolic::Variable>* vdot_sos_gram,
    VectorX<symbolic::Monomial>* vdot_sos_monomials) {
  symbolic::Polynomial vdot_sos =
      (1 + lambda0) *
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x)) * (V - rho);
  const RowVector3<symbolic::Polynomial> dVdq = V.Jacobian(x.head<3>());
  vdot_sos -= l.sum() * (dVdq.dot(qdot) + kappa * V);
  const RowVector2<symbolic::Polynomial> dVdv = V.Jacobian(x.tail<2>());
  vdot_sos -= l(0) * dVdv.dot(z1_poly) * z_factor;
  vdot_sos -= l(1) * dVdv.dot(z2_poly) * z_factor;
  vdot_sos -= p.dot(state_constraints);
  std::tie(*vdot_sos_gram, *vdot_sos_monomials) =
      prog->AddSosConstraint(vdot_sos);
  return vdot_sos;
}

bool FindLagrangian(const Eigen::Matrix<symbolic::Variable, 5, 1>& x,
                    const Vector2<symbolic::Variable>& z1,
                    const Vector2<symbolic::Variable>& z2, double z_factor,
                    int lambda0_degree, const symbolic::Polynomial& V,
                    double rho, const std::vector<int>& l_degrees,
                    const std::vector<int>& p_degrees,
                    const Vector3<symbolic::Polynomial>& qdot, double kappa,
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
  MatrixX<symbolic::Variable> vdot_sos_gram;
  VectorX<symbolic::Monomial> vdot_sos_monomials;
  AddControlLyapunovConstraint(&prog, x, z1_poly, z2_poly, z_factor, lambda0, V,
                               rho, l, qdot, kappa, p, state_constraints,
                               &vdot_sos_gram, &vdot_sos_monomials);
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

void Search(const std::optional<std::string>& load_V_init,
            const std::optional<std::string>& save_clf_file) {
  const CartPoleParams params;
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const int V_degree = 2;
  const Vector2<symbolic::Variable> z1(symbolic::Variable("z1(0)"),
                                       symbolic::Variable("z1(1)"));
  const Vector2<symbolic::Variable> z2(symbolic::Variable("z2(0)"),
                                       symbolic::Variable("z2(1)"));
  const double u_max = 170;
  const double z_factor = 10;
  const Eigen::Matrix<symbolic::Expression, 5, 1> x_expr =
      x.cast<symbolic::Expression>();
  const Matrix2<symbolic::Expression> M_expr =
      CartpoleMassMatrix<symbolic::Expression>(params, x_expr);
  const Vector2<symbolic::Expression> bias_expr =
      CalcCartpoleBiasTerm<symbolic::Expression>(params, x_expr) -
      CalcCartpoleGravityVector<symbolic::Expression>(params, x_expr);
  Eigen::Matrix<symbolic::Polynomial, 5, 1> state_constraints;
  state_constraints(0) = CartpoleStateEqConstraint(x);
  const Vector2<symbolic::Expression> constraint_expr1 =
      M_expr * z1 - Eigen::Vector2d(u_max, 0) / z_factor + bias_expr / z_factor;
  const Vector2<symbolic::Expression> constraint_expr2 =
      M_expr * z2 - Eigen::Vector2d(-u_max, 0) / z_factor +
      bias_expr / z_factor;
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

  const double kappa = 0.01;
  int lambda0_degree = 2;
  const std::vector<int> l_degrees{{2, 2}};
  const std::vector<int> p_degrees{{6, 6, 6, 6, 6}};

  symbolic::Polynomial V_init;
  double rho_sol;
  if (load_V_init.has_value()) {
    V_init = Load(symbolic::Variables(x), load_V_init.value());
    rho_sol = 1;
  } else {
    V_init = FindClfInit(params, V_degree, x);
    const bool binary_search_rho = false;
    // Maximize rho
    if (binary_search_rho) {
      double rho_max = 0.01;
      double rho_min = 1E-5;
      double rho_tol = 3E-4;

      auto is_rho_feasible = [&x, &z1, &z2, z_factor, lambda0_degree, &V_init,
                              &l_degrees, &p_degrees, &qdot, kappa,
                              &state_constraints](double rho) {
        symbolic::Polynomial lambda0_sol;
        Vector2<symbolic::Polynomial> l_sol;
        return FindLagrangian(x, z1, z2, z_factor, lambda0_degree, V_init, rho,
                              l_degrees, p_degrees, qdot, kappa,
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
        V_init = V_init / rho_sol;
        rho_sol = 1;
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
      vdot_sos -= l.sum() * (dVdq.dot(qdot) + kappa * V_init);
      vdot_sos -= l(0) * dVdv.dot(z1_poly) * z_factor +
                  l(1) * dVdv.dot(z2_poly) * z_factor;
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
  Eigen::MatrixXd state_swingup;
  Eigen::MatrixXd control_swingup;
  SwingUpTrajectoryOptimization(&state_swingup, &control_swingup);
  std::cout << "start bilinear alternation\n";
  symbolic::Polynomial lambda0_sol;
  Vector2<symbolic::Polynomial> l_sol;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> p_sol;
  MatrixX<symbolic::Variable> vdot_sos_gram;
  VectorX<symbolic::Monomial> vdot_sos_monomials;
  symbolic::Polynomial vdot_sos;
  Eigen::MatrixXd vdot_sos_gram_val;
  symbolic::Polynomial vdot_sos_sol;
  while (iter_count < max_iters) {
    {
      const bool found_lagrangian = FindLagrangian(
          x, z1, z2, z_factor, lambda0_degree, V_sol, rho, l_degrees, p_degrees,
          qdot, kappa, state_constraints, &lambda0_sol, &l_sol);
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
      positivity_sos -= positivity_lagrangian * CartpoleStateEqConstraint(x);
      prog.AddSosConstraint(positivity_sos);
      // Now add the constraint on Vdot.
      Eigen::Matrix<symbolic::Polynomial, 5, 1> p;
      for (int i = 0; i < 5; ++i) {
        p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i],
                                      "p" + std::to_string(i));
      }
      vdot_sos = AddControlLyapunovConstraint(
          &prog, x, z1_poly, z2_poly, z_factor, lambda0_sol, V, rho, l_sol,
          qdot, kappa, p, state_constraints, &vdot_sos_gram,
          &vdot_sos_monomials);

      // Now minimize V on x_samples.
      Eigen::Matrix<double, 5, Eigen::Dynamic> x_swingup(5,
                                                         state_swingup.cols());
      for (int i = 0; i < x_swingup.cols(); ++i) {
        x_swingup.col(i) = ToCartpoleTrigState<double>(state_swingup.col(i));
      }
      std::cout << "V_init at x_swingup "
                << V_init.EvaluateIndeterminates(x, x_swingup).transpose()
                << "\n";
      std::vector<int> x_indices = {13, 14, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28};
      Eigen::Matrix4Xd state_samples(4, x_indices.size());
      for (int i = 0; i < static_cast<int>(x_indices.size()); ++i) {
        state_samples.col(i) = state_swingup.col(x_indices[i]);
      }
      std::cout << "state samples:\n" << state_samples.transpose() << "\n";
      Eigen::MatrixXd x_samples(5, state_samples.cols());
      for (int i = 0; i < state_samples.cols(); ++i) {
        x_samples.col(i) = ToCartpoleTrigState<double>(state_samples.col(i));
      }

      std::optional<Eigen::MatrixXd> in_roa_samples;
      in_roa_samples.emplace(Eigen::Matrix<double, 5, 3>());
      in_roa_samples->col(0) = x_samples.col(1);
      in_roa_samples->col(1) = x_samples.col(2);
      in_roa_samples->col(2) = x_samples.col(3);
      if (in_roa_samples.has_value()) {
        // Add the constraint V(in_roa_samples) <= rho
        Eigen::MatrixXd A_in_roa_samples;
        VectorX<symbolic::Variable> variables_in_roa_samples;
        Eigen::VectorXd b_in_roa_samples;
        V.EvaluateWithAffineCoefficients(
            x, in_roa_samples.value(), &A_in_roa_samples,
            &variables_in_roa_samples, &b_in_roa_samples);
        prog.AddLinearConstraint(
            A_in_roa_samples,
            Eigen::VectorXd::Constant(b_in_roa_samples.rows(), -kInf),
            Eigen::VectorXd::Constant(b_in_roa_samples.rows(), rho) -
                b_in_roa_samples,
            variables_in_roa_samples);
      }
      OptimizePolynomialAtSamples(&prog, V, x, x_samples,
                                  OptimizePolynomialMode::kMinimizeMaximal);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      const double backoff_scale = 0.1;
      std::cout << "Smallest coeff: " << SmallestCoeff(prog) << "\n";
      const auto result_lyapunov = SearchWithBackoff(
          &prog, solvers::MosekSolver::id(), solver_options, backoff_scale);
      if (result_lyapunov.is_success()) {
        V_sol = result_lyapunov.GetSolution(V);
        drake::log()->info(
            "Optimal cost = {}",
            V_sol.EvaluateIndeterminates(x, x_samples).maxCoeff());

        if (save_clf_file.has_value()) {
          Save(V_sol,
               save_clf_file.value() + ".iter" + std::to_string(iter_count));
        }
        for (int i = 0; i < p_sol.rows(); ++i) {
          p_sol(i) = result_lyapunov.GetSolution(p(i));
        }
        vdot_sos_gram_val = result_lyapunov.GetSolution(vdot_sos_gram);
        vdot_sos_sol = result_lyapunov.GetSolution(vdot_sos);
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
  if (save_clf_file.has_value()) {
    Save(V_sol, save_clf_file.value());
  }

  Eigen::Vector4d state_val = state_swingup.col(13);
  const Eigen::Matrix<double, 5, 1> x_val = ToCartpoleTrigState<double>(state_val);
  symbolic::Environment env;
  env.insert(x, x_val);
  const double V_val = V_sol.Evaluate(env);
  Eigen::Matrix<symbolic::Polynomial, 5, 1> f_numerator;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G_numerator;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(params, x, &f_numerator, &G_numerator,
                   &dynamics_denominator);
  const VdotCalculator vdot_calculator(x, V_sol, f_numerator, G_numerator,
                                       dynamics_denominator,
                                       Eigen::RowVector2d(-u_max, u_max));
  const double Vdot_val = vdot_calculator.CalcMin(x_val)(0);
  std::cout << "V_val: " << V_val << "\n";
  std::cout << "Vdot_val: " << Vdot_val << "\n";
  std::cout << "Vdot_val + kappa * V_val" << Vdot_val + kappa * V_val << "\n";
  Eigen::Matrix<double, 5, 1> n_val1;
  Eigen::Matrix<double, 5, 1> n_val2;
  double d_val1, d_val2;
  CartpoleTrigDynamics<double>(params, x_val, u_max, &n_val1, &d_val1);
  CartpoleTrigDynamics<double>(params, x_val, -u_max, &n_val2, &d_val2);
  const Eigen::Matrix<double, 5, 1> xdot_val1 = n_val1 / d_val1;
  const Eigen::Matrix<double, 5, 1> xdot_val2 = n_val2 / d_val2;
  const Eigen::Vector2d z1_val = xdot_val1.tail<2>() / z_factor;
  const Eigen::Vector2d z2_val = xdot_val2.tail<2>() / z_factor;
  std::cout << "z1_val: " << z1_val.transpose() << "\n";
  std::cout << "z2_val: " << z2_val.transpose() << "\n";
  env.insert(z1, z1_val);
  env.insert(z2, z2_val);
  Eigen::Matrix<double, 5, 1> state_constraints_val;
  for (int i = 0; i < state_constraints.rows(); ++i) {
    state_constraints_val(i) = state_constraints(i).Evaluate(env);
  }
  std::cout << "state_constraints_val: " << state_constraints_val.transpose()
            << "\n";
  Eigen::Matrix<double, 5, 1> p_sol_val;
  for (int i = 0; i < p_sol.rows(); ++i) {
    p_sol_val(i) = p_sol(i).Evaluate(env);
  }
  std::cout << "p_sol_val: " << p_sol_val.transpose() << "\n";
  std::cout << "vdot_sos_sol_val: " << vdot_sos_sol.Evaluate(env) << "\n";
  Eigen::VectorXd vdot_sos_monomials_val(vdot_sos_monomials.rows());
  for (int i = 0; i < vdot_sos_monomials.rows(); ++i) {
    vdot_sos_monomials_val(i) = vdot_sos_monomials(i).Evaluate(env);
  }
  std::cout << "vdot_sos evaluate with monomial and gram: "
            << vdot_sos_monomials_val.dot(vdot_sos_gram_val *
                                          vdot_sos_monomials_val)
            << "\n";

  const double duration = 20;
  Simulate(params, x, V_sol, u_max, kappa, state_swingup.col(12), duration);
}

int DoMain() {
  Search("/home/hongkaidai/Dropbox/sos_clf_cbf/cart_pole_trig_clf_last16_1.txt",
         "cart_pole_implicit_trig_clf0.txt");
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  drake::systems::analysis::DoMain();
}
