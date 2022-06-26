#include "examples/acrobot/gen/acrobot_params.h"

#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sdpa_free_format.h"
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
      1000 * Vector1d::Ones(), Eigen::MatrixXd(0, 1), F);
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
  const std::vector<int> derivative_ceq_lagrangian_degrees{{4, 4}};
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

void AddControlLyapunovConstraint(
    solvers::MathematicalProgram* prog, const Vector6<symbolic::Variable>& x,
    const Vector4<symbolic::Polynomial>& z_poly,
    const symbolic::Polynomial& lambda0, const symbolic::Polynomial& V,
    const Vector2<symbolic::Polynomial>& l,
    const Vector4<symbolic::Polynomial>& qdot, double deriv_eps,
    const Vector6<symbolic::Polynomial>& p,
    const Vector6<symbolic::Polynomial>& state_constraints) {
  symbolic::Polynomial vdot_sos =
      (1 + lambda0) *
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x)) * (V - 1);
  const RowVector4<symbolic::Polynomial> dVdq = V.Jacobian(x.head<4>());
  vdot_sos -= l.sum() * (dVdq.dot(qdot) + deriv_eps * V);
  const RowVector2<symbolic::Polynomial> dVdv = V.Jacobian(x.tail<2>());
  vdot_sos -= l(0) * dVdv.dot(z_poly.head<2>());
  vdot_sos -= l(1) * dVdv.dot(z_poly.tail<2>());
  vdot_sos -= p.dot(state_constraints);
  prog->AddSosConstraint(vdot_sos);
}

void SearchWImplicitTrigDynamics() {
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
  symbolic::Polynomial V_init = FindClfInit(parameters, V_degree, x);
  std::cout << "V_init(x_bottom): "
            << V_init.EvaluateIndeterminates(
                   x, ToTrigState<double>(Eigen::Vector4d::Zero()))
            << "\n";

  Vector4<symbolic::Variable> z;
  for (int i = 0; i < 4; ++i) {
    z(i) = symbolic::Variable("z" + std::to_string(i));
  }
  const double u_max = 10;
  const Matrix2<symbolic::Expression> M_expr = MassMatrix<symbolic::Expression>(
      parameters, x.cast<symbolic::Expression>());
  const Vector2<symbolic::Expression> bias_expr =
      DynamicsBiasTerm<symbolic::Expression>(parameters,
                                             x.cast<symbolic::Expression>());
  Matrix2<symbolic::Polynomial> M;
  Vector2<symbolic::Polynomial> bias;
  for (int i = 0; i < 2; ++i) {
    bias(i) = symbolic::Polynomial(bias_expr(i));
    for (int j = 0; j < 2; ++j) {
      M(i, j) = symbolic::Polynomial(M_expr(i, j));
    }
  }
  Vector6<symbolic::Polynomial> state_constraints;
  state_constraints.head<2>() = StateEqConstraints(x);
  const Vector2<symbolic::Expression> constraint_expr1 =
      M_expr * z.head<2>() - Eigen::Vector2d(0, u_max) + bias_expr;
  const Vector2<symbolic::Expression> constraint_expr2 =
      M_expr * z.tail<2>() - Eigen::Vector2d(0, -u_max) + bias_expr;
  for (int i = 0; i < 2; ++i) {
    state_constraints(2 + i) = symbolic::Polynomial(constraint_expr1(i));
    state_constraints(4 + i) = symbolic::Polynomial(constraint_expr2(i));
  }
  symbolic::Variables xz_set{x};
  xz_set.insert(symbolic::Variables(z));
  const Vector4<symbolic::Expression> qdot_expr =
      CalcQdot<symbolic::Expression>(x.cast<symbolic::Expression>());
  Vector4<symbolic::Polynomial> qdot;
  for (int i = 0; i < 4; ++i) {
    qdot(i) = symbolic::Polynomial(qdot_expr(i));
  }
  Vector4<symbolic::Polynomial> z_poly;
  for (int i = 0; i < 4; ++i) {
    z_poly(i) = symbolic::Polynomial(z(i));
  }
  const symbolic::Variables x_set{x};
  symbolic::Variables xz1{x};
  xz1.insert(symbolic::Variables(z.head<2>()));
  symbolic::Variables xz2{x};
  xz2.insert(symbolic::Variables(z.tail<2>()));

  const double deriv_eps = 0.1;
  const int lambda0_degree = 0;
  const std::vector<int> l_degrees{{2, 2}};
  const std::vector<int> p_degrees{{3, 3, 3, 3, 3, 3}};
  double rho_sol;
  {
    // Maximize rho
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    prog.AddIndeterminates(z);
    const int d_degree = lambda0_degree / 2 + 1;
    const symbolic::Variable rho = prog.NewContinuousVariables<1>("rho")(0);
    symbolic::Polynomial vdot_sos =
        symbolic::Polynomial(
            pow(x.cast<symbolic::Expression>().dot(x), d_degree)) *
        (V_init - rho);
    Vector2<symbolic::Polynomial> l;
    const RowVector4<symbolic::Polynomial> dVdq = V_init.Jacobian(x.head<4>());
    for (int i = 0; i < 2; ++i) {
      std::tie(l(i), std::ignore) = prog.NewSosPolynomial(
          xz_set, l_degrees[i],
          solvers::MathematicalProgram::NonnegativePolynomial::kSos, "l");
    }
    const RowVector2<symbolic::Polynomial> dVdv = V_init.Jacobian(x.tail<2>());
    vdot_sos -= l.sum() * (dVdq.dot(qdot) + deriv_eps * V_init);

    vdot_sos -= l(0) * (dVdv.dot(z_poly.head<2>())) +
                l(1) * (dVdv.dot(z_poly.tail<2>()));
    Vector6<symbolic::Polynomial> p;
    for (int i = 0; i < 6; ++i) {
      p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i], "p");
    }
    vdot_sos -= p.dot(state_constraints);
    prog.AddSosConstraint(vdot_sos);
    prog.AddLinearCost(Vector1d(-1), 0, Vector1<symbolic::Variable>(rho));
    RemoveTinyCoeff(&prog, 1E-8);
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    std::cout << "Smallest coeff: " << SmallestCoeff(prog) << "\n";
    const auto result = solvers::Solve(prog, std::nullopt, solver_options);
    DRAKE_DEMAND(result.is_success());
    rho_sol = result.GetSolution(rho);
    std::cout << "V_init <= " << rho_sol << "\n";
  }

  const int max_iters = 25;
  int iter_count = 0;
  symbolic::Polynomial V_sol = V_init / rho_sol;
  while (iter_count < max_iters) {
    // Find the Lagrangian multipliers
    symbolic::Polynomial lambda0_sol;
    Vector2<symbolic::Polynomial> l_sol;
    {
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      prog.AddIndeterminates(z);
      symbolic::Polynomial lambda0;
      std::tie(lambda0, std::ignore) = prog.NewSosPolynomial(
          xz_set, lambda0_degree,
          solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Lambda");
      Vector2<symbolic::Polynomial> l;
      for (int i = 0; i < 2; ++i) {
        std::tie(l(i), std::ignore) = prog.NewSosPolynomial(
            xz_set, l_degrees[i],
            solvers::MathematicalProgram::NonnegativePolynomial::kSos, "l");
      }
      Vector6<symbolic::Polynomial> p;
      for (int i = 0; i < 6; ++i) {
        p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i],
                                      "p" + std::to_string(i));
      }
      AddControlLyapunovConstraint(&prog, x, z_poly, lambda0, V_sol, l, qdot,
                                   deriv_eps, p, state_constraints);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      RemoveTinyCoeff(&prog, 1E-9);
      std::cout << "Smallest coeff in Lagrangian program: "
                << SmallestCoeff(prog) << "\n";
      const auto result = solvers::Solve(prog, std::nullopt, solver_options);
      if (result.is_success()) {
        lambda0_sol = result.GetSolution(lambda0);
        for (int i = 0; i < 2; ++i) {
          l_sol(i) = result.GetSolution(l(i));
        }
      } else {
        drake::log()->error("Failed to find Lagrangian");
      }
      DRAKE_DEMAND(result.is_success());
    }
  }
}

int DoMain() {
  SearchWImplicitTrigDynamics();
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() { return drake::systems::analysis::DoMain(); }
