#include <math.h>

#include <iostream>
#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/test/quadrotor2d.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {

const double kInf = std::numeric_limits<double>::infinity();

[[maybe_unused]] controllers::LinearQuadraticRegulatorResult TrigDynamicsLQR() {
  Eigen::VectorXd lqr_Q_diag(7);
  lqr_Q_diag << 1, 1, 1, 1, 10, 10, 10;
  const Eigen::MatrixXd lqr_Q = lqr_Q_diag.asDiagonal();
  const Eigen::Matrix2d lqr_R = 10 * Eigen::Matrix2d::Identity();
  const auto lqr_result = SynthesizeQuadrotor2dTrigLqr(lqr_Q, lqr_R);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(lqr_result.S);
  std::cout << "minimal eigenvalue of LQR S: " << es.eigenvalues().minCoeff()
            << "\n";
  return lqr_result;
}

[[maybe_unused]] void ValidateLQRasLyapunov() {
  const auto lqr_result = TrigDynamicsLQR();
  Eigen::Matrix<symbolic::Variable, 7, 1> x;
  for (int i = 0; i < 7; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  Quadrotor2dTrigPlant<double> quadrotor;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  symbolic::Polynomial V(x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  V = V.RemoveTermsWithSmallCoefficients(1E-8);
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  VectorX<symbolic::Polynomial> controller(2);
  for (int i = 0; i < controller.rows(); ++i) {
    controller(i) =
        symbolic::Polynomial(-lqr_result.K.row(i).dot(x) + thrust_equilibrium);
  }
  const VectorX<symbolic::Polynomial> xdot = f + G * controller;
  symbolic::Polynomial Vdot = dVdx.dot(xdot);
  Vdot = Vdot.RemoveTermsWithSmallCoefficients(1E-8);
  const symbolic::Variables x_set(x);
  const symbolic::Polynomial state_constraints(x(2) * x(2) + x(3) * x(3) +
                                               2 * x(3));

  const double rho = 0.01;
  {
    // Use Pablo's formulation
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    const int d = 1;
    const symbolic::Polynomial x_square_d(
        pow(x.cast<symbolic::Expression>().dot(x), d));
    const int lambda0_degree = 2;
    symbolic::Polynomial lambda0;
    std::tie(lambda0, std::ignore) =
        prog.NewSosPolynomial(x_set, lambda0_degree);
    symbolic::Polynomial sos_condition = (1 + lambda0) * x_square_d * (V - rho);
    const int l_degree = 4;
    symbolic::Polynomial l;
    std::tie(l, std::ignore) = prog.NewSosPolynomial(x_set, l_degree);
    sos_condition -= l * Vdot;
    const int p_degree = 5;
    const symbolic::Polynomial p = prog.NewFreePolynomial(x_set, p_degree);
    sos_condition -= p * state_constraints;
    prog.AddSosConstraint(sos_condition);
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  }
  {
    // Use V <= rho => Vdot <= 0
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    const int lambda0_degree = 2;
    symbolic::Polynomial lambda0;
    std::tie(lambda0, std::ignore) =
        prog.NewSosPolynomial(x_set, lambda0_degree);
    symbolic::Polynomial sos_condition = (1 + lambda0) * -Vdot;
    const int l_degree = 4;
    symbolic::Polynomial l;
    std::tie(l, std::ignore) = prog.NewSosPolynomial(x_set, l_degree);
    sos_condition -= l * (rho - V);
    const int p_degree = 4;
    const symbolic::Polynomial p = prog.NewFreePolynomial(x_set, p_degree);
    sos_condition -= p * state_constraints;
    prog.AddSosConstraint(sos_condition);
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
    const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  }
}

[[maybe_unused]] symbolic::Polynomial FindTrigRegionalClf(
    int V_degree, const Eigen::Matrix<symbolic::Variable, 7, 1>& x) {
  Quadrotor2dTrigPlant<double> quadrotor2d;
  const auto lqr_result = TrigDynamicsLQR();
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
  const int d = V_degree / 2;
  const double kappa = 0.0001;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      Quadrotor2dStateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{
      {static_cast<int>(std::ceil((V_degree + 1) / 2.) * 2 - 2)}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{V_degree - 2};
  const std::vector<int> derivative_cin_lagrangian_degrees =
      derivative_ceq_lagrangian_degrees;

  auto ret = FindCandidateRegionalLyapunov(
      x, dynamics, std::nullopt /* dynamics_denominator*/, V_degree,
      positivity_eps, d, kappa, state_eq_constraints,
      positivity_ceq_lagrangian_degrees, derivative_ceq_lagrangian_degrees,
      state_ineq_constraints, positivity_cin_lagrangian_degrees,
      derivative_cin_lagrangian_degrees);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*(ret.prog), std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(ret.V);
  return V_sol;
}

[[maybe_unused]] symbolic::Polynomial FindTrigClfInitBySample(
    int V_degree, const Eigen::Matrix<symbolic::Variable, 7, 1>& x) {
  Quadrotor2dTrigPlant<double> quadrotor;
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  // Find the initial V_init by sampling states.
  const Eigen::VectorXd pos_x_samples =
      Eigen::VectorXd::LinSpaced(8, -0.02, 0.02);
  const Eigen::VectorXd pos_y_samples =
      Eigen::VectorXd::LinSpaced(10, -0.02, 0.02);
  const Eigen::VectorXd theta_samples =
      Eigen::VectorXd::LinSpaced(10, -0.02 * M_PI, 0.03 * M_PI);
  const Eigen::VectorXd vel_x_samples =
      Eigen::VectorXd::LinSpaced(10, -0.02, 0.01);
  const Eigen::VectorXd vel_y_samples =
      Eigen::VectorXd::LinSpaced(10, -0.03, 0.02);
  const Eigen::VectorXd thetadot_samples =
      Eigen::VectorXd::LinSpaced(10, -0.03, 0.04);
  const std::vector<Eigen::VectorXd> state_samples(
      {pos_x_samples, pos_y_samples, theta_samples, vel_x_samples,
       vel_y_samples, thetadot_samples});
  const Eigen::MatrixXd state_mesh = Meshgrid(state_samples);
  Eigen::MatrixXd x_val(7, state_mesh.cols());
  Eigen::MatrixXd xdot_val(7, state_mesh.cols());
  // Synthesize an LQR controller to compute the action.
  auto context = quadrotor.CreateDefaultContext();
  context->SetContinuousState(Vector6d::Zero());
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const Eigen::Vector2d u_equilibrium(thrust_equilibrium, thrust_equilibrium);
  quadrotor.get_input_port().FixValue(context.get(), u_equilibrium);
  auto linearized_quadrotor = Linearize(quadrotor, *context);
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10, 10;
  const Eigen::Matrix<double, 6, 6> lqr_Q = lqr_Q_diag.asDiagonal();
  auto lqr_result = controllers::LinearQuadraticRegulator(
      linearized_quadrotor->A(), linearized_quadrotor->B(), lqr_Q,
      10 * Eigen::Matrix2d::Identity());

  for (int i = 0; i < state_mesh.cols(); ++i) {
    x_val.col(i) = ToQuadrotor2dTrigState<double>(state_mesh.col(i));
    xdot_val.col(i) =
        TrigDynamics<double>(quadrotor, x_val.col(i),
                             -lqr_result.K * state_mesh.col(i) + u_equilibrium);
  }

  MatrixX<symbolic::Expression> V_init_gram;
  symbolic::Polynomial V_init;
  const double positivity_eps = 0.1;
  const int d = 1;
  const Vector1<symbolic::Polynomial> state_constraints(
      symbolic::Polynomial(x(2) * x(2) + x(3) * x(3) + 2 * x(3)));
  const std::vector<int> eq_lagrangian_degrees{V_degree - 2};
  VectorX<symbolic::Polynomial> eq_lagrangian;

  auto prog_V_init = FindCandidateLyapunov(
      x, V_degree, positivity_eps, d, state_constraints, eq_lagrangian_degrees,
      x_val, xdot_val, &V_init, &eq_lagrangian);
  for (const auto& [monomial, coeff] : V_init.monomial_to_coefficient_map()) {
    // The polynomial cannot have large coefficients.
    prog_V_init->AddLinearConstraint(coeff, -50, 50);
  }
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result_init =
      solvers::Solve(*prog_V_init, std::nullopt, solver_options);
  DRAKE_DEMAND(result_init.is_success());
  const symbolic::Polynomial V_init_sol = result_init.GetSolution(V_init);
  Save(V_init_sol, "quadrotor2d_trig_clf_init.txt");
  return V_init_sol;
}

void ValidateTrigClfInit(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>& x,
    const symbolic::Polynomial& V_init, const VectorX<symbolic::Polynomial>& f,
    const MatrixX<symbolic::Polynomial>& G,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices) {
  // Now sample many points and evaluate Vdot.
  const RowVectorX<symbolic::Polynomial> dVdx = V_init.Jacobian(x);
  Eigen::MatrixXd state_validate = Eigen::MatrixXd::Random(6, 1000000) * 0.001;
  Eigen::MatrixXd x_validate(7, state_validate.cols());

  const auto lqr_result = TrigDynamicsLQR();
  Quadrotor2dTrigPlant<double> quadrotor;
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  Eigen::MatrixXd xdot_lqr_validate(7, x_validate.cols());
  for (int i = 0; i < state_validate.cols(); ++i) {
    x_validate.col(i) = ToQuadrotor2dTrigState<double>(state_validate.col(i));
    const Eigen::Vector2d u_lqr =
        -lqr_result.K * x_validate.col(i) +
        Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium);
    xdot_lqr_validate.col(i) =
        TrigDynamics<double>(quadrotor, x_validate.col(i), u_lqr);
  }
  Eigen::MatrixXd dVdx_validate(7, x_validate.cols());
  for (int i = 0; i < x.rows(); ++i) {
    dVdx_validate.row(i) =
        dVdx(i).EvaluateIndeterminates(x, x_validate).transpose();
  }
  std::cout << "Vdot with lqr controller by samples: "
            << (dVdx_validate.array() * xdot_lqr_validate.array())
                   .colwise()
                   .sum()
                   .maxCoeff()
            << "\n";
  VdotCalculator vdot_calculator(
      x, V_init, f, G, std::nullopt /* dynamics numerator */, u_vertices);
  const Eigen::VectorXd Vdot_validate = vdot_calculator.CalcMin(x_validate);
  std::cout << "Vdot CLF max by sample: " << Vdot_validate.maxCoeff() << "\n";
  symbolic::Variable max_Vdot;
  auto prog_validate =
      ConstructMaxVdotProgram(x, V_init, f, G, u_vertices, &max_Vdot);
  // Add constraint x(2) * x(2) + (x(3) + 1) * (x(3) + 1) = 1
  prog_validate->AddConstraint(
      std::make_shared<solvers::QuadraticConstraint>(
          2 * Eigen::Matrix2d::Identity(), Eigen::Vector2d(0, 2), 0, 0),
      x.segment<2>(2));
  prog_validate->AddBoundingBoxConstraint(-Eigen::VectorXd::Ones(7) * 0.01,
                                          Eigen::VectorXd::Ones(7) * 0.01, x);
  // Add constraint that x is slightly away from the origin.
  prog_validate->AddBoundingBoxConstraint(0.001, kInf, x(2));
  double max_Vdot_val = -kInf;
  Eigen::VectorXd max_Vdot_x = Eigen::VectorXd::Zero(7);
  for (int i = 0; i < 100; ++i) {
    prog_validate->SetInitialGuess(x, Eigen::VectorXd::Random(7) * 0.01);
    const auto result_validate = solvers::Solve(*prog_validate);
    DRAKE_DEMAND(result_validate.is_success());
    if (result_validate.GetSolution(max_Vdot) > max_Vdot_val) {
      max_Vdot_val = result_validate.GetSolution(max_Vdot);
      max_Vdot_x = result_validate.GetSolution(x);
    }
  }
  std::cout << "max Vdot: " << max_Vdot_val << " at " << max_Vdot_x.transpose()
            << "\n";
}

[[maybe_unused]] void SearchWTrigDynamics() {
  Quadrotor2dTrigPlant<double> quadrotor;
  Eigen::Matrix<symbolic::Variable, 7, 1> x;
  for (int i = 0; i < 7; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);

  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const double thrust_max = 3 * thrust_equilibrium;
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(
      Quadrotor2dStateEqConstraint(x));
  symbolic::Polynomial V_init;
  const int V_degree = 2;
  {
    V_init = FindTrigRegionalClf(V_degree, x);
    // V_init = FindTrigClfInitBySample(V_degree, x);
    // const auto lqr_result = TrigDynamicsLQR();
    // V_init = symbolic::Polynomial(
    //    x.cast<symbolic::Expression>().dot(lqr_result.S * x));
    V_init = V_init.RemoveTermsWithSmallCoefficients(1e-6);
    std::cout << "V_init: " << V_init << "\n";

    // ValidateTrigClfInit(x, V_init, f, G, u_vertices);
  }
  //{ ValidateLQRasLyapunov(); }

  const ControlLyapunov dut(x, f, G, std::nullopt /* dynamics denominator */,
                            u_vertices, state_constraints);
  const int lambda0_degree = 0;
  const std::vector<int> l_degrees{{2, 2, 2, 2}};
  const std::vector<int> p_degrees{{4}};
  const double kappa = 0.2;
  {
    // Maximize rho such that V(x) <= rho defines a valid ROA.
    const int d_degree = lambda0_degree / 2 + 1;
    auto lagrangian_ret = dut.ConstructLagrangianProgram(
        V_init, symbolic::Polynomial(), d_degree, l_degrees, p_degrees, kappa);
    solvers::SolverOptions solver_options;
    // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole,
    // 1);
    const auto result =
        solvers::Solve(*(lagrangian_ret.prog), std::nullopt, solver_options);
    DRAKE_DEMAND(result.is_success());
    const double rho_sol = result.GetSolution(lagrangian_ret.rho);
    std::cout << fmt::format("V_init(x) <= {}\n", rho_sol);
    V_init = V_init / rho_sol;
  }

  {
    ControlLyapunov::SearchOptions search_options;
    search_options.d_converge_tol = 0.;
    search_options.bilinear_iterations = 100;
    search_options.lyap_step_backoff_scale = 0.01;
    search_options.lsol_tiny_coeff_tol = 1E-8;
    search_options.lyap_tiny_coeff_tol = 1E-8;
    search_options.lagrangian_step_solver_options = solvers::SolverOptions();
    search_options.lagrangian_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 0);
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    search_options.lyap_step_solver_options->SetOption(
        solvers::CommonSolverOption::kPrintToConsole, 0);
    Eigen::MatrixXd state_samples(6, 4);
    state_samples.col(0) << 0, 0.5, 0, 0, 0, 0;
    state_samples.col(1) << 1, 0., 0, 0, 0, 0;
    state_samples.col(2) << 1, 0., M_PI / 2, 0, 0, 0;
    state_samples.col(3) << 1, 0., -M_PI / 2, 0, 0, 0;
    Eigen::MatrixXd x_samples(7, state_samples.cols());
    for (int i = 0; i < state_samples.cols(); ++i) {
      x_samples.col(i) = ToQuadrotor2dTrigState<double>(state_samples.col(i));
    }

    const double positivity_eps = 0.0001;
    const int positivity_d = V_degree / 2;
    const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    const bool minimize_max = true;
    SearchResultDetails search_result_details;
    const auto search_result =
        dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                   positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                   kappa, x_samples, std::nullopt /* in_roa_samples */,
                   minimize_max, search_options);
    std::cout
        << "V(x_samples): "
        << search_result.V.EvaluateIndeterminates(x, x_samples).transpose()
        << "\n";
  }
}

int DoMain() {
  SearchWTrigDynamics();
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
