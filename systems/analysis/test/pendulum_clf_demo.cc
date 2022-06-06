#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/test/pendulum.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

class ControlledPendulum final : public LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlledPendulum)

  ControlledPendulum(symbolic::Polynomial clf, Vector2<symbolic::Variable> x,
                     Eigen::Vector2d x_des, double u_bound, double deriv_eps)
      : LeafSystem<double>(),
        clf_{std::move(clf)},
        x_{std::move(x)},
        x_des_{std::move(x_des)},
        u_bound_{u_bound},
        deriv_eps_{deriv_eps} {
    auto state_index = this->DeclareContinuousState(2);
    state_output_index_ =
        this->DeclareStateOutputPort("state", state_index).get_index();
    clf_output_index_ =
        this->DeclareVectorOutputPort("clf", 1, &ControlledPendulum::CalcClf)
            .get_index();
    const RowVector2<symbolic::Polynomial> dVdx = clf_.Jacobian(x_);
    pendulum_.ControlAffineDynamics(x_, x_des_(0), u_bound_, &f_, &G_);
    dVdx_times_f_ = dVdx.dot(f_);
    dVdx_times_G_ = dVdx.dot(G_);
  }

  ~ControlledPendulum() final{};

  const OutputPortIndex& state_output_index() const {
    return state_output_index_;
  }

  const OutputPortIndex& clf_output_index() const { return clf_output_index_; }

 private:
  virtual void DoCalcTimeDerivatives(
      const systems::Context<double>& context,
      systems::ContinuousState<double>* derivatives) const final {
    solvers::MathematicalProgram prog;
    auto u = prog.NewContinuousVariables<1>();
    prog.AddBoundingBoxConstraint(-1, 1, u(0));
    prog.AddQuadraticCost(Eigen::Matrix<double, 1, 1>::Constant(1),
                          Eigen::Matrix<double, 1, 1>::Zero(), u);
    const Vector2<double> x_val =
        context.get_continuous_state_vector().CopyToVector();
    const Vector2<double> x_bar_val = x_val - x_des_;

    symbolic::Environment env;
    env.insert(x_, x_bar_val);
    const double dVdx_times_f_val = dVdx_times_f_.Evaluate(env);
    const double dVdx_times_G_val = dVdx_times_G_.Evaluate(env);
    const double V_val = clf_.Evaluate(env);
    // dVdx * G * u + dVdx * f <= -eps * V
    prog.AddLinearConstraint(
        Eigen::Matrix<double, 1, 1>::Constant(dVdx_times_G_val), -kInf,
        -deriv_eps_ * V_val - dVdx_times_f_val, u);
    const auto result = solvers::Solve(prog);
    DRAKE_DEMAND(result.is_success());
    const double u_val = result.GetSolution(u(0)) * u_bound_;
    auto& derivative_vector = derivatives->get_mutable_vector();
    derivative_vector.SetAtIndex(0, x_val(1));
    derivative_vector.SetAtIndex(
        1, pendulum_.ComputeThetaddot(x_val(0), x_val(1), u_val));
  }

  void CalcClf(const Context<double>& context,
               BasicVector<double>* output) const {
    const Vector2<double> x_val =
        context.get_continuous_state_vector().CopyToVector();
    const Vector2<double> x_bar_val = x_val - x_des_;

    symbolic::Environment env;
    env.insert(x_, x_bar_val);
    Eigen::VectorBlock<VectorX<double>> clf_vec = output->get_mutable_value();
    clf_vec(0) = clf_.Evaluate(env);
  }

  symbolic::Polynomial clf_;
  Vector2<symbolic::Variable> x_;
  Eigen::Vector2d x_des_;
  double u_bound_;
  double deriv_eps_;
  Pendulum pendulum_;
  Vector2<symbolic::Polynomial> f_;
  Vector2<symbolic::Polynomial> G_;
  symbolic::Polynomial dVdx_times_f_;
  symbolic::Polynomial dVdx_times_G_;
  OutputPortIndex state_output_index_;
  OutputPortIndex clf_output_index_;
};

void Simulate(const Vector2<symbolic::Variable>& x,
              const Eigen::Vector2d& x_des, const symbolic::Polynomial& clf,
              double u_bound, double deriv_eps, const Eigen::Vector2d& x0,
              double duration) {
  systems::DiagramBuilder<double> builder;
  auto system =
      builder.AddSystem<ControlledPendulum>(clf, x, x_des, u_bound, deriv_eps);
  auto state_logger = LogVectorOutput(
      system->get_output_port(system->state_output_index()), &builder);
  auto clf_logger = LogVectorOutput(
      system->get_output_port(system->clf_output_index()), &builder);
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  Simulator<double> simulator(*diagram);
  simulator.get_mutable_context().SetContinuousState(x0);

  simulator.AdvanceTo(duration);
  std::cout << "finish simulation\n";

  std::cout << fmt::format(
      "final state: {}, final V: {}\n",
      state_logger->FindLog(simulator.get_context())
          .data()
          .rightCols<1>()
          .transpose(),
      clf_logger->FindLog(simulator.get_context()).data().rightCols<1>());
}

[[maybe_unused]] void SearchWTrigPoly() {
  const double u_bound = 25;
  const Vector3<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"),
                                      symbolic::Variable("x2"));
  examples::pendulum::PendulumPlant<double> pendulum;
  const double theta_des = M_PI;
  Vector3<symbolic::Polynomial> f;
  Vector3<symbolic::Polynomial> G;
  TrigPolyDynamics(pendulum, x, theta_des, &f, &G);
  for (int i = 0; i < 3; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1e-8);
    G(i) = G(i).RemoveTermsWithSmallCoefficients(1e-8);
  }
  const Eigen::RowVector2d u_vertices(u_bound, -u_bound);
  const double sin_theta_des = theta_des == M_PI ? 0 : std::sin(theta_des);
  const double cos_theta_des = theta_des == M_PI ? -1 : std::cos(theta_des);
  Vector1<symbolic::Polynomial> state_constraints(symbolic::Polynomial(
      (x(0) + sin_theta_des) * (x(0) + sin_theta_des) +
      (x(1) + cos_theta_des) * (x(1) + cos_theta_des) - 1));
  state_constraints(0) = state_constraints(0).Expand();
  state_constraints(0) =
      state_constraints(0).RemoveTermsWithSmallCoefficients(1E-8);
  const ControlLyapunov dut(x, f, G, u_vertices, state_constraints);
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees{4, 4};
  const std::vector<int> p_degrees{5};
  const int V_degree = 2;
  const Eigen::Vector3d x_star(0, 0, 0);
  const Eigen::Matrix3d S = Eigen::Matrix3d::Identity();

  ControlLyapunov::SearchOptions search_options;
  search_options.lyap_step_solver = solvers::ScsSolver::id();
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  // search_options.lyap_step_solver_options->SetOption(
  //    solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.backoff_scale = 0.03;
  search_options.lsol_tiny_coeff_tol = 1E-5;
  // There are tiny coefficients coming from numerical roundoff error.
  search_options.lyap_tiny_coeff_tol = 1E-8;
  const double rho_min = 0.01;
  const double rho_max = 15;
  const double rho_bisection_tol = 0.01;
  const ControlLyapunov::RhoBisectionOption rho_bisection_option(
      rho_min, rho_max, rho_bisection_tol);
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  symbolic::Polynomial r;
  VectorX<symbolic::Polynomial> p;
  double rho;
  const double deriv_eps = 0.0;
  // Compute V_init from LQR controller.
  const Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  const Vector1d R(1);
  const auto lqr_result = TrigDynamicsLQR(pendulum, theta_des, Q, R);
  symbolic::Polynomial V_init;
  {
    // Now take many samples around (theta_des, 0).
    const Eigen::VectorXd theta_samples =
        Eigen::VectorXd::LinSpaced(-0.2 + theta_des, 0.2 + theta_des, 10);
    const Eigen::VectorXd thetadot_samples =
        Eigen::VectorXd::LinSpaced(-0.3, 0.3, 10);
    Eigen::Matrix3Xd x_val(3, theta_samples.cols() * thetadot_samples.cols());
    Eigen::Matrix3Xd xdot_val(3, x_val.cols());
    int x_count = 0;
    for (int i = 0; i < theta_samples.rows(); ++i) {
      for (int j = 0; j < thetadot_samples.rows(); ++j) {
        x_val.col(x_count) =
            ToTrigState(theta_samples(i), thetadot_samples(j), theta_des);
        const double u = -lqr_result.K.row(0).dot(x_val.col(x_count)) +
                         EquilibriumTorque(pendulum, theta_des);
        xdot_val.col(x_count) = TrigDynamics<double>(
            pendulum, x_val.col(x_count), theta_des, Vector1d(u));
        x_count++;
      }
    }
    const int V_init_degree = 2;
    MatrixX<symbolic::Expression> V_init_gram;
    auto prog_V_init = FindCandidateLyapunov(x, V_init_degree, x_val, xdot_val,
                                             &V_init, &V_init_gram);
    const auto result_init = solvers::Solve(*prog_V_init);
    DRAKE_DEMAND(result_init.is_success());
    V_init = result_init.GetSolution(V_init);
  }

  V_init = V_init.RemoveTermsWithSmallCoefficients(1e-6);
  // First maximize rho such that V(x)<=rho defines a valid ROA.
  {
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    symbolic::Variable rho_var;
    symbolic::Polynomial vdot_sos;
    VectorX<symbolic::Monomial> vdot_monomials;
    MatrixX<symbolic::Variable> vdot_gram;
    auto prog = dut.ConstructLagrangianProgram(
        V_init, symbolic::Polynomial(), 2, l_degrees, p_degrees, deriv_eps, &l,
        &l_grams, &p, &rho_var, &vdot_sos, &vdot_monomials, &vdot_gram);
    const auto result = solvers::Solve(*prog);
    DRAKE_DEMAND(result.is_success());
    std::cout << fmt::format("V_init(x) <= {}\n", result.GetSolution(rho_var));
  }

  symbolic::Polynomial V_sol;
  dut.Search(V_init, lambda0_degree, l_degrees, V_degree, p_degrees, deriv_eps,
             x_star, S, V_degree - 2, search_options, rho_bisection_option,
             &V_sol, &lambda0, &l, &r, &p, &rho);
  // Check if theta = 0, thetadot = 0 is in the verified ROA.
  std::cout << fmt::format("V at (theta, thetadot)=(0, 0) = {}\n",
                           V_sol.EvaluateIndeterminates(
                               x, ToTrigState<double>(0., 0, theta_des))(0));
}

[[maybe_unused]] void Search() {
  examples::pendulum::PendulumPlant<double> pendulum;
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  const double theta_des = M_PI;
  Vector2<symbolic::Polynomial> f;
  Vector2<symbolic::Polynomial> G;
  ControlAffineDynamics(pendulum, x, theta_des, &f, &G);
  const double u_bound = 25;
  const Eigen::RowVector2d u_vertices(-u_bound, u_bound);
  VectorX<symbolic::Polynomial> state_constraints(0);
  ControlLyapunov dut(x, f, G, u_vertices, state_constraints);

  auto context = pendulum.CreateDefaultContext();
  context->SetContinuousState(Eigen::Vector2d(theta_des, 0));
  pendulum.get_input_port(0).FixValue(context.get(), Vector1d(0));
  const Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
  const Vector1d R(0.01);
  const auto linear_system = Linearize(pendulum, *context);
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      linear_system->A(), linear_system->B(), Q, R);
  symbolic::Polynomial V_init(
      x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  V_init = V_init.RemoveTermsWithSmallCoefficients(1E-8);
  std::cout << "V_init: " << V_init << "\n";
  const double deriv_eps = 0.5;
  {
    const symbolic::Polynomial lambda0{};
    const int d_degree = 2;
    const std::vector<int> l_degrees{2, 2};
    const std::vector<int> p_degrees{};
    VectorX<symbolic::Polynomial> l;
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    VectorX<symbolic::Polynomial> p;
    symbolic::Variable rho;
    symbolic::Polynomial vdot_sos;
    VectorX<symbolic::Monomial> vdot_monomials;
    MatrixX<symbolic::Variable> vdot_gram;
    auto prog = dut.ConstructLagrangianProgram(
        V_init, lambda0, d_degree, l_degrees, p_degrees, deriv_eps, &l,
        &l_grams, &p, &rho, &vdot_sos, &vdot_monomials, &vdot_gram);
    const auto result = solvers::Solve(*prog);
    DRAKE_DEMAND(result.is_success());
    std::cout << V_init << " <= " << result.GetSolution(rho) << "\n";
  }
  {
    const int lambda0_degree = 2;
    const std::vector<int> l_degrees{2, 2};
    const std::vector<int> p_degrees{};
    const int V_degree = 2;
    const Eigen::Vector2d x_star(0, 0);
    const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();

    ControlLyapunov::SearchOptions search_options;
    search_options.backoff_scale = 0.02;
    // There are tiny coefficients coming from numerical roundoff error.
    search_options.lyap_tiny_coeff_tol = 1E-10;
    const double rho_min = 0.01;
    const double rho_max = 15;
    const double rho_bisection_tol = 0.01;
    const ControlLyapunov::RhoBisectionOption rho_bisection_option(
        rho_min, rho_max, rho_bisection_tol);
    symbolic::Polynomial V_sol;
    symbolic::Polynomial lambda0;
    VectorX<symbolic::Polynomial> l;
    symbolic::Polynomial r;
    VectorX<symbolic::Polynomial> p;
    double rho;
    dut.Search(V_init, lambda0_degree, l_degrees, V_degree, p_degrees,
               deriv_eps, x_star, S, V_degree - 2, search_options,
               rho_bisection_option, &V_sol, &lambda0, &l, &r, &p, &rho);
  }
}

[[maybe_unused]] void SearchWBoxBounds(const symbolic::Polynomial& V_init,
                                       const Vector2<symbolic::Variable>& x,
                                       const Vector2<symbolic::Polynomial>& f,
                                       const Vector2<symbolic::Polynomial>& G,
                                       symbolic::Polynomial* V_sol,
                                       double* deriv_eps_sol) {
  const bool symmetric_dynamics = internal::IsDynamicsSymmetric(f, G);
  const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
  std::vector<std::vector<symbolic::Polynomial>> l_given(1);
  l_given[0].resize(num_vdot_sos);
  for (int i = 0; i < num_vdot_sos; ++i) {
    l_given[0][i] = symbolic::Polynomial();
  }
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees(1);
  lagrangian_degrees[0].resize(num_vdot_sos);
  for (int i = 0; i < num_vdot_sos; ++i) {
    lagrangian_degrees[0][i] = {0, 4, 4};
  }
  std::vector<int> b_degrees(1);
  b_degrees[0] = 4;

  const double positivity_eps = 0.;
  ControlLyapunovBoxInputBound searcher(f, G, x, positivity_eps);
  const Eigen::Vector2d x_star(0, 0);
  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  // int s_degree = 0;
  // symbolic::Polynomial t_given{0};
  const int V_degree = 2;
  const double deriv_eps_lower = 0.5;
  const double deriv_eps_upper = deriv_eps_lower;
  ControlLyapunovBoxInputBound::SearchOptions search_options;
  search_options.bilinear_iterations = 15;
  search_options.backoff_scale = 0.02;
  search_options.lyap_tiny_coeff_tol = 1E-10;
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  // search_options.lyap_step_solver_options->SetOption(solvers::CommonSolverOption::kPrintToConsole,
  // 1);
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  // search_options.lagrangian_step_solver_options->SetOption(solvers::CommonSolverOption::kPrintToConsole,
  // 1);
  const double rho_min = 0.01;
  const double rho_max = 15;
  const double rho_bisection_tol = 0.01;
  const int r_degree = V_degree - 2;
  const ControlLyapunovBoxInputBound::RhoBisectionOption rho_bisection_option(
      rho_min, rho_max, rho_bisection_tol);
  auto clf_result =
      searcher.Search(V_init, l_given, lagrangian_degrees, b_degrees, x_star, S,
                      r_degree, V_degree, deriv_eps_lower, deriv_eps_upper,
                      search_options, rho_bisection_option);
  *V_sol = clf_result.V;
  *deriv_eps_sol = clf_result.deriv_eps;
}

[[maybe_unused]] void SearchWTaylorDynamics() {
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));

  const Pendulum pendulum;
  Vector2<symbolic::Polynomial> f;
  Vector2<symbolic::Polynomial> G;
  const double u_bound = 25;
  Eigen::Matrix2d A;
  Eigen::Vector2d B;
  const double theta_des = M_PI;
  const Vector2<double> x_des(theta_des, 0);
  pendulum.DynamicsGradient(theta_des, u_bound, &A, &B);
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      A, B, Eigen::Matrix2d::Identity(), Vector1<double>::Identity());
  pendulum.ControlAffineDynamics(x, M_PI, u_bound, &f, &G);
  for (int i = 0; i < 2; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    G(i, 0) = G(i, 0).RemoveTermsWithSmallCoefficients(1E-6);
  }
  symbolic::Polynomial V_init(x.dot(lqr_result.S * x));

  symbolic::Polynomial V_sol;
  double deriv_eps_sol;
  SearchWBoxBounds(V_init, x, f, G, &V_sol, &deriv_eps_sol);

  std::cout << "clf: " << V_sol << "\n";
  Simulate(x, x_des, V_sol, u_bound, deriv_eps_sol,
           Eigen::Vector2d(M_PI + M_PI * 0.6, 0), 10);
}

int DoMain() {
  // Search();
  // SearchWTaylorDynamics();
  SearchWTrigPoly();
  return 0;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
