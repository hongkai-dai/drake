#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

struct Pendulum {
  double compute_thetaddot(double theta, double theta_dot, double u) const {
    const double theta_ddot =
        (u - mass * gravity * length * std::sin(theta) - damping * theta_dot) /
        (mass * length * length);
    return theta_ddot;
  }

  void control_affine_dynamics(const Vector2<symbolic::Variable>& x,
                               double theta_des, double u_bound,
                               Vector2<symbolic::Polynomial>* f,
                               Vector2<symbolic::Polynomial>* G) const {
    (*G)(0) = symbolic::Polynomial();
    (*G)(1) = symbolic::Polynomial(u_bound / (mass * length * length));
    (*f)(0) = symbolic::Polynomial(x(1));
    (*f)(1) = symbolic::Polynomial(
        (-mass * gravity * length *
             (std::sin(theta_des) + std::cos(theta_des) * x(0) -
              std::sin(theta_des) / 2 * pow(x(0), 2) -
              std::cos(theta_des) / 6 * pow(x(0), 3)) -
         damping * x(1)) /
        (mass * length * length));
  }

  void dynamics_gradient(double theta, double u_bound, Eigen::Matrix2d* A,
                         Eigen::Vector2d* B) const {
    *A << 0, 1,
        -mass * gravity * length * std::cos(theta) / (mass * length * length),
        -damping / (mass * length * length);
    *B << 0, u_bound / (mass * length * length);
  }

  double mass{1};
  double gravity{9.81};
  double length{1};
  double damping{0.1};
};

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
    pendulum_.control_affine_dynamics(x_, x_des_(0), u_bound_, &f_, &G_);
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
        1, pendulum_.compute_thetaddot(x_val(0), x_val(1), u_val));
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

[[maybe_unused]] void Search(const symbolic::Polynomial& V_init,
                             const Vector2<symbolic::Variable>& x,
                             const Vector2<symbolic::Polynomial>& f,
                             const Vector2<symbolic::Polynomial>& G,
                             symbolic::Polynomial* V_sol,
                             double* deriv_eps_sol) {
  const Eigen::RowVector2d u_vertices(1, -1);
  const ControlLyapunov dut(x, f, G, u_vertices);
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees{2, 2};
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
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  symbolic::Polynomial r;
  double rho;
  *deriv_eps_sol = 0.5;
  dut.Search(V_init, lambda0_degree, l_degrees, V_degree, *deriv_eps_sol,
             x_star, S, V_degree - 2, search_options, rho_bisection_option,
             V_sol, &lambda0, &l, &r, &rho);
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

int DoMain() {
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
  pendulum.dynamics_gradient(theta_des, u_bound, &A, &B);
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      A, B, Eigen::Matrix2d::Identity(), Vector1<double>::Identity());
  pendulum.control_affine_dynamics(x, M_PI, u_bound, &f, &G);
  for (int i = 0; i < 2; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    G(i, 0) = G(i, 0).RemoveTermsWithSmallCoefficients(1E-6);
  }
  symbolic::Polynomial V_init(x.dot(lqr_result.S * x));

  symbolic::Polynomial V_sol;
  double deriv_eps_sol;
  Search(V_init, x, f, G, &V_sol, &deriv_eps_sol);
  // SearchWBoxBounds(V_init, x, f, G, &V_sol, &deriv_eps_sol);

  std::cout << "clf: " << V_sol << "\n";
  Simulate(x, x_des, V_sol, u_bound, deriv_eps_sol,
           Eigen::Vector2d(M_PI + M_PI * 0.6, 0), 10);
  return 0;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
