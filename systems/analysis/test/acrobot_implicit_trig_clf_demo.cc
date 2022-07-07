#include <iomanip>
#include <iostream>
#include <limits>

#include "drake/examples/acrobot/acrobot_geometry.h"
#include "drake/geometry/meshcat_visualizer.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sdpa_free_format.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/test/acrobot.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

void Simulate(const examples::acrobot::AcrobotParams<double>& parameters,
              const Vector6<symbolic::Variable>& x,
              const symbolic::Polynomial& clf, double u_bound, double deriv_eps,
              const Eigen::Vector4d& initial_state, double duration) {
  systems::DiagramBuilder<double> builder;
  auto acrobot = builder.AddSystem<examples::acrobot::AcrobotPlant<double>>();
  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  examples::acrobot::AcrobotGeometry::AddToBuilder(
      &builder, acrobot->get_output_port(0), scene_graph);

  auto meshcat = std::make_shared<geometry::Meshcat>();
  geometry::MeshcatVisualizerParams meshcat_params{};
  meshcat_params.role = geometry::Role::kIllustration;
  auto visualizer = &geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, *scene_graph, meshcat, meshcat_params);
  unused(visualizer);

  const Eigen::Vector2d Au(1, -1);
  const Eigen::Vector2d bu(u_bound, u_bound);
  const Vector1d u_star(0);
  const Vector1d Ru(1);

  Vector6<symbolic::Polynomial> f;
  Vector6<symbolic::Polynomial> G;
  symbolic::Polynomial dynamics_numerator;
  TrigPolyDynamics(parameters, x, &f, &G, &dynamics_numerator);

  const double vdot_cost = 0;
  auto clf_controller = builder.AddSystem<ClfController>(
      x, f, G, dynamics_numerator, clf, deriv_eps, Au, bu, u_star, Ru,
      vdot_cost);
  auto state_logger = LogVectorOutput(acrobot->get_output_port(0), &builder);
  auto clf_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->clf_output_index()),
      &builder);
  auto control_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      &builder);
  unused(control_logger);
  auto trig_state_converter = builder.AddSystem<ToTrigStateConverter<double>>();
  builder.Connect(acrobot->get_output_port(0),
                  trig_state_converter->get_input_port());
  builder.Connect(
      trig_state_converter->get_output_port(),
      clf_controller->get_input_port(clf_controller->x_input_index()));
  builder.Connect(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      acrobot->get_input_port());

  symbolic::Environment env;
  env.insert(x, ToTrigState<double>(initial_state));
  std::cout << std::setprecision(10)
            << "V(initial_state): " << clf.Evaluate(env) << "\n";

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  Simulator<double> simulator(*diagram);
  // ResetIntegratorFromFlags(&simulator, "implicit_euler", 0.0002);
  simulator.get_mutable_context().SetContinuousState(initial_state);
  diagram->Publish(simulator.get_context());
  std::cout << "Refresh meshcat brower and press to continue\n";
  std::cin.get();

  simulator.AdvanceTo(duration);
  std::cout << "finish simulation\n";

  std::cout << fmt::format(
      "final state: {}, final V: {}\n",
      state_logger->FindLog(simulator.get_context())
          .data()
          .rightCols<1>()
          .transpose(),
      clf_logger->FindLog(simulator.get_context()).data().rightCols<1>());
  std::cout << "V: " << std::setprecision(10)
            << clf_logger->FindLog(simulator.get_context()).data() << "\n";
  std::cout << "u: " << control_logger->FindLog(simulator.get_context()).data()
            << "\n";
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

// Add the constraint (1+λ₀(x, z))xᵀx(V(x)−ρ) − l₁(x, z) * (∂V/∂q*q̇+∂V/∂v*z¹+εV)
// + l₂(x, z)(∂V/∂q*q̇+∂V/∂v*z²+εV)−p(x, z)c(x, z) is sos
symbolic::Polynomial AddControlLyapunovConstraint(
    solvers::MathematicalProgram* prog, const Vector6<symbolic::Variable>& x,
    const Vector4<symbolic::Polynomial>& z_poly,
    const symbolic::Polynomial& lambda0, const symbolic::Polynomial& V,
    double rho, const Vector2<symbolic::Polynomial>& l,
    const Vector4<symbolic::Polynomial>& qdot, double deriv_eps,
    const Vector6<symbolic::Polynomial>& p,
    const Vector6<symbolic::Polynomial>& state_constraints) {
  symbolic::Polynomial vdot_sos =
      (1 + lambda0) *
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x)) * (V - rho);
  const RowVector4<symbolic::Polynomial> dVdq = V.Jacobian(x.head<4>());
  vdot_sos -= l.sum() * (dVdq.dot(qdot) + deriv_eps * V);
  const RowVector2<symbolic::Polynomial> dVdv = V.Jacobian(x.tail<2>());
  vdot_sos -= l(0) * dVdv.dot(z_poly.head<2>());
  vdot_sos -= l(1) * dVdv.dot(z_poly.tail<2>());
  vdot_sos -= p.dot(state_constraints);
  prog->AddSosConstraint(vdot_sos);
  return vdot_sos;
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

  const double deriv_eps = 0.1;
  const int lambda0_degree = 0;
  const std::vector<int> l_degrees{{0, 0}};
  const std::vector<int> p_degrees{{2, 2, 3, 3, 3, 3}};
  // double rho_sol;
  //{
  //  // Maximize rho
  //  solvers::MathematicalProgram prog;
  //  prog.AddIndeterminates(x);
  //  prog.AddIndeterminates(z);
  //  const int d_degree = lambda0_degree / 2 + 1;
  //  const symbolic::Variable rho = prog.NewContinuousVariables<1>("rho")(0);
  //  symbolic::Polynomial vdot_sos =
  //      symbolic::Polynomial(
  //          pow(x.cast<symbolic::Expression>().dot(x), d_degree)) *
  //      (V_init - rho);
  //  Vector2<symbolic::Polynomial> l;
  //  const RowVector4<symbolic::Polynomial> dVdq =
  //  V_init.Jacobian(x.head<4>()); for (int i = 0; i < 2; ++i) {
  //    std::tie(l(i), std::ignore) = prog.NewSosPolynomial(
  //        xz_set, l_degrees[i],
  //        solvers::MathematicalProgram::NonnegativePolynomial::kSos, "l");
  //  }
  //  const RowVector2<symbolic::Polynomial> dVdv =
  //  V_init.Jacobian(x.tail<2>()); vdot_sos -= l.sum() * (dVdq.dot(qdot) +
  //  deriv_eps * V_init);

  //  vdot_sos -= l(0) * (dVdv.dot(z_poly.head<2>())) +
  //              l(1) * (dVdv.dot(z_poly.tail<2>()));
  //  Vector6<symbolic::Polynomial> p;
  //  for (int i = 0; i < 6; ++i) {
  //    p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i], "p");
  //  }
  //  vdot_sos -= p.dot(state_constraints);
  //  prog.AddSosConstraint(vdot_sos);
  //  prog.AddLinearCost(Vector1d(-1), 0, Vector1<symbolic::Variable>(rho));
  //  RemoveTinyCoeff(&prog, 1E-8);
  //  solvers::SolverOptions solver_options;
  //  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  //  std::cout << "Smallest coeff: " << SmallestCoeff(prog) << "\n";
  //  std::cout << "Largest coeff: " << LargestCoeff(prog) << "\n";
  //  const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  //  DRAKE_DEMAND(result.is_success());
  //  rho_sol = result.GetSolution(rho);
  //  std::cout << "V_init <= " << rho_sol << "\n";
  //}

  const int max_iters = 1;
  int iter_count = 0;
  symbolic::Polynomial V_sol = V_init;
  const double rho = 0.05;
  while (iter_count < max_iters) {
    symbolic::Polynomial lambda0_sol;
    Vector2<symbolic::Polynomial> l_sol;
    // Find the Lagrangian multipliers
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
      AddControlLyapunovConstraint(&prog, x, z_poly, lambda0, V_sol, rho, l,
                                   qdot, deriv_eps, p, state_constraints);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      RemoveTinyCoeff(&prog, 1E-9);
      std::cout << "Smallest coeff in Lagrangian program: "
                << SmallestCoeff(prog) << "\n";
      std::cout << "Largest coeff in Lagrangian program: " << LargestCoeff(prog)
                << "\n";
      const auto result = solvers::Solve(prog, std::nullopt, solver_options);
      if (result.is_success()) {
        lambda0_sol = result.GetSolution(lambda0);
        const double lsol_tol = 5E-5;
        lambda0_sol = lambda0_sol.RemoveTermsWithSmallCoefficients(lsol_tol);
        for (int i = 0; i < 2; ++i) {
          l_sol(i) = result.GetSolution(l(i));
          l_sol(i) = l_sol(i).RemoveTermsWithSmallCoefficients(lsol_tol);
        }
      } else {
        drake::log()->error("Failed to find Lagrangian");
        return;
      }
    }

    // Now search for Lyapunov.
    {
      solvers::MathematicalProgram prog;
      prog.AddIndeterminates(x);
      prog.AddIndeterminates(z);
      symbolic::Polynomial V;
      V = NewFreePolynomialPassOrigin(&prog, x_set, V_degree, "V",
                                      symbolic::internal::DegreeType::kAny,
                                      symbolic::Variables{});
      // First add the constraint that V −ε₁(xᵀx)ᵈ − p₁(x)c₁(x) is sos.
      const double positivity_eps = 0.01;
      const int d = V_degree / 2;
      symbolic::Polynomial positivity_sos =
          V - positivity_eps * symbolic::Polynomial(pow(
                                   x.cast<symbolic::Expression>().dot(x), d));
      const std::vector<int> positivity_lagrangian_degrees{
          {V_degree - 2, V_degree - 2}};
      Vector2<symbolic::Polynomial> positivity_lagrangian;
      for (int i = 0; i < 2; ++i) {
        positivity_lagrangian(i) =
            prog.NewFreePolynomial(x_set, positivity_lagrangian_degrees[i]);
      }
      positivity_sos -= positivity_lagrangian.dot(StateEqConstraints(x));
      prog.AddSosConstraint(positivity_sos);
      // Now add the constraint on Vdot.
      Vector6<symbolic::Polynomial> p;
      for (int i = 0; i < 6; ++i) {
        p(i) = prog.NewFreePolynomial(xz_set, p_degrees[i],
                                      "p" + std::to_string(i));
      }
      const symbolic::Polynomial vdot_sos = AddControlLyapunovConstraint(
          &prog, x, z_poly, lambda0_sol, V, rho, l_sol, qdot, deriv_eps, p,
          state_constraints);

      // Now minimize V on x_samples.
      Eigen::Matrix<double, 6, 3> x_samples;
      x_samples.col(0) = ToTrigState<double>(Eigen::Vector4d(0, 0, 0, 0));
      x_samples.col(1) =
          ToTrigState<double>(Eigen::Vector4d(M_PI + 0.1, 0, 0, 0));
      x_samples.col(2) =
          ToTrigState<double>(Eigen::Vector4d(M_PI + 0.1, 0.2, 0, 0));
      OptimizePolynomialAtSamples(&prog, V, x, x_samples,
                                  OptimizePolynomialMode::kMinimizeMaximal);
      solvers::SolverOptions solver_options;
      solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
      const double backoff_scale = 0.02;
      const auto result_lyapunov = SearchWithBackoff(
          &prog, solvers::MosekSolver::id(), solver_options, backoff_scale);
      if (result_lyapunov.is_success()) {
        V_sol = result_lyapunov.GetSolution(V);
        drake::log()->info(
            "Optimal cost = {}",
            V_sol.EvaluateIndeterminates(x, x_samples).maxCoeff());

        std::cout << "V_sol: " << V_sol.Expand() << "\n";
        Save(V_sol, "acrobot_implicit_trig_clf.txt");
        internal::PrintPsdConstraintStat(prog, result_lyapunov);
      } else {
        drake::log()->error("Failed to find Lyapunov");
        return;
      }
    }
    iter_count += 1;
  }

  V_sol = Load(x_set, "acrobot_implicit_trig_clf.txt");
  Vector6<symbolic::Polynomial> f_numerator;
  Vector6<symbolic::Polynomial> G_numerator;
  symbolic::Polynomial dynamics_denominator;
  TrigPolyDynamics(parameters, x, &f_numerator, &G_numerator,
                   &dynamics_denominator);
  const VdotCalculator vdot_calculator(x, V_sol, f_numerator, G_numerator,
                                       dynamics_denominator,
                                       Eigen::RowVector2d(-u_max, u_max));
  const double duration = 5;
  Simulate(parameters, x, V_sol, u_max, deriv_eps, Eigen::Vector4d(0, 0, 0, 0),
           duration);
}

int DoMain() {
  SearchWImplicitTrigDynamics();
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
