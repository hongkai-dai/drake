#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();
struct Quadrotor {
  // This is the dynamics with un-normalized u.
  template <typename T>
  Vector6<T> CalcDynamics(const Vector6<T>& x, const Vector2<T>& u) const {
    Vector6<T> xdot;
    xdot.template head<3>() = x.template tail<3>();
    using std::cos;
    using std::sin;
    xdot.template tail<3>() << -sin(x(2)) / mass * (u(0) + u(1)),
        cos(x(2)) / mass * (u(0) + u(1)) - gravity,
        length / inertia * (u(0) - u(1));
    return xdot;
  }

  template <typename T>
  Vector2<T> NormalizeU(const Vector2<T>& u) const {
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    return (u - u_mid) / (u_max / 2);
  }

  template <typename T>
  Vector2<T> UnnormalizeU(const Vector2<T>& u_normalize) const {
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    return u_normalize * (u_max / 2) + u_mid;
  }

  // This assumes normalized u, namely -1 <= u <= 1
  template <typename T>
  void ControlAffineDynamics(const Vector6<T>& x, Vector6<T>* f,
                             Eigen::Matrix<T, 6, 2>* G) const {
    f->template head<3>() = x.template tail<3>();
    G->template topRows<3>().setZero();
    using std::cos;
    using std::sin;
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    const T s2 = sin(x(2));
    const T c2 = cos(x(2));
    (*f)(3) = -s2 / mass * (u_mid(0) + u_mid(1));
    (*f)(4) = -gravity + c2 / mass * (u_mid(0) + u_mid(1));
    (*f)(5) = length / inertia * (u_mid(0) - u_mid(1));
    (*G)(3, 0) = -s2 / mass * u_max / 2;
    (*G)(3, 1) = (*G)(3, 0);
    (*G)(4, 0) = c2 / mass * u_max / 2;
    (*G)(4, 1) = (*G)(4, 0);
    (*G)(5, 0) = length / inertia * u_max / 2;
    (*G)(5, 1) = -(*G)(5, 0);
  }

  // This assumes normalized u.
  void PolynomialControlAffineDynamics(
      const Vector6<symbolic::Variable>& x, Vector6<symbolic::Polynomial>* f,
      Eigen::Matrix<symbolic::Polynomial, 6, 2>* G) const {
    for (int i = 0; i < 3; ++i) {
      (*f)(i) = symbolic::Polynomial(symbolic::Monomial(x(i + 3)));
      (*G)(i, 0) = symbolic::Polynomial();
      (*G)(i, 1) = symbolic::Polynomial();
    }
    // Use taylor expansion for sin and cos around theta=0.
    const symbolic::Polynomial s2{{{symbolic::Monomial(x(2)), 1},
                                   {symbolic::Monomial(x(2), 3), -1. / 6}}};
    const symbolic::Polynomial c2{
        {{symbolic::Monomial(), 1}, {symbolic::Monomial(x(2), 2), -1. / 2}}};
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    (*f)(3) = -s2 / mass * (u_mid(0) + u_mid(1));
    (*f)(4) = -gravity + c2 / mass * (u_mid(0) + u_mid(1));
    (*f)(5) = symbolic::Polynomial();
    (*G)(3, 0) = -s2 / mass * u_max / 2;
    (*G)(3, 1) = (*G)(3, 0);
    (*G)(4, 0) = c2 / mass * u_max / 2;
    (*G)(4, 1) = (*G)(4, 0);
    (*G)(5, 0) = symbolic::Polynomial(
        {{symbolic::Monomial(), length / inertia * u_max / 2}});
    (*G)(5, 1) = -(*G)(5, 0);
  }

  double length{0.25};
  double mass{0.486};
  double inertia{0.00383};
  double gravity{9.81};
  double u_max{mass * gravity * 2};
};

class ControlledQuadrotor final : public LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlledQuadrotor)

  ControlledQuadrotor(symbolic::Polynomial clf, Vector6<symbolic::Variable> x,
                      double deriv_eps)
      : LeafSystem<double>(),
        clf_{std::move(clf)},
        x_{std::move(x)},
        deriv_eps_{deriv_eps} {
    auto state_index = this->DeclareContinuousState(6);
    state_output_index_ =
        this->DeclareStateOutputPort("state", state_index).get_index();
    clf_output_index_ =
        this->DeclareVectorOutputPort("clf", 1, &ControlledQuadrotor::CalcClf)
            .get_index();
    u_output_index_ =
        this->DeclareVectorOutputPort("u", 2, &ControlledQuadrotor::CalcU)
            .get_index();
    const RowVector6<symbolic::Polynomial> dVdx = clf_.Jacobian(x_);

    quadrotor_.PolynomialControlAffineDynamics(x, &f_, &G_);
    dVdx_times_f_ = dVdx.dot(f_);
    dVdx_times_G_ = dVdx * G_;
  }

  ~ControlledQuadrotor() final{};

  const OutputPortIndex& state_output_index() const {
    return state_output_index_;
  }

  const OutputPortIndex& clf_output_index() const { return clf_output_index_; }

 private:
  virtual void DoCalcTimeDerivatives(
      const systems::Context<double>& context,
      systems::ContinuousState<double>* derivatives) const final {
    BasicVector<double> u_output(2);
    this->CalcU(context, &u_output);

    auto& derivative_vector = derivatives->get_mutable_vector();

    const Vector6d x_val = context.get_continuous_state_vector().CopyToVector();
    const Eigen::Vector2d u_val = u_output.get_value();
    const Vector6d xdot = quadrotor_.CalcDynamics(x_val, u_val);
    for (int i = 0; i < 6; ++i) {
      derivative_vector.SetAtIndex(i, xdot(i));
    }
  }

  void CalcU(const Context<double>& context,
             BasicVector<double>* output) const {
    solvers::MathematicalProgram prog;
    auto u = prog.NewContinuousVariables<2>();
    prog.AddBoundingBoxConstraint(-1, 1, u);
    prog.AddQuadraticCost(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(),
                          u);
    const Vector6d x_val = context.get_continuous_state_vector().CopyToVector();

    symbolic::Environment env;
    env.insert(x_, x_val);
    const double dVdx_times_f_val = dVdx_times_f_.Evaluate(env);
    Eigen::RowVector2d dVdx_times_G_val;
    for (int i = 0; i < 2; ++i) {
      dVdx_times_G_val(i) = dVdx_times_G_(i).Evaluate(env);
    }
    const double V_val = clf_.Evaluate(env);
    // dVdx * G * u + dVdx * f <= -eps * V
    prog.AddLinearConstraint(dVdx_times_G_val, -kInf,
                             -deriv_eps_ * V_val - dVdx_times_f_val, u);
    const auto result = solvers::Solve(prog);
    DRAKE_DEMAND(result.is_success());
    const Eigen::Vector2d u_val =
        quadrotor_.UnnormalizeU(result.GetSolution(u));
    output->get_mutable_value() = u_val;
  }

  void CalcClf(const Context<double>& context,
               BasicVector<double>* output) const {
    const Vector6<double> x_val =
        context.get_continuous_state_vector().CopyToVector();

    symbolic::Environment env;
    env.insert(x_, x_val);
    Eigen::VectorBlock<VectorX<double>> clf_vec = output->get_mutable_value();
    clf_vec(0) = clf_.Evaluate(env);
  }

  symbolic::Polynomial clf_;
  Vector6<symbolic::Variable> x_;
  double deriv_eps_;
  Quadrotor quadrotor_;
  Vector6<symbolic::Polynomial> f_;
  Eigen::Matrix<symbolic::Polynomial, 6, 2> G_;
  symbolic::Polynomial dVdx_times_f_;
  RowVector2<symbolic::Polynomial> dVdx_times_G_;
  OutputPortIndex state_output_index_;
  OutputPortIndex clf_output_index_;
  OutputPortIndex u_output_index_;
};

void Simulate(const Vector6<symbolic::Variable>& x,
              const symbolic::Polynomial& clf, double deriv_eps,
              const Vector6d& x0, double duration) {
  systems::DiagramBuilder<double> builder;
  auto system = builder.AddSystem<ControlledQuadrotor>(clf, x, deriv_eps);
  auto state_logger = LogVectorOutput(
      system->get_output_port(system->state_output_index()), &builder);
  auto clf_logger = LogVectorOutput(
      system->get_output_port(system->clf_output_index()), &builder);
  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  Simulator<double> simulator(*diagram);
  ResetIntegratorFromFlags(&simulator, "implicit_euler", 0.01);

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

[[maybe_unused]] void search(const Vector6<symbolic::Variable>& x,
                             const symbolic::Polynomial& V_init,
                             const Vector6<symbolic::Polynomial>& f,
                             const Eigen::Matrix<symbolic::Polynomial, 6, 2>& G,
                             symbolic::Polynomial* V_sol,
                             double* deriv_eps_sol) {
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 1, 1, -1, -1,
                1, -1, 1, -1;
  // clang-format on
  ControlLyapunov dut(x, f, G, u_vertices);
  const double deriv_eps = 0.2;
  *deriv_eps_sol = deriv_eps;
  const int lambda0_degree = 2;
  std::vector<int> l_degrees = {2, 2, 2, 2};
  const int V_degree = 2;
  const Vector6d x_star = Vector6d::Zero();
  const Matrix6<double> S = Matrix6<double>::Identity();
  ControlLyapunov::SearchOptions search_options;
  search_options.bilinear_iterations = 10;
  search_options.backoff_scale = 0.02;
  search_options.lyap_tiny_coeff_tol = 2E-7;
  search_options.Vsol_tiny_coeff_tol = 1E-7;
  search_options.lsol_tiny_coeff_tol = 1E-7;
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  // search_options.lyap_step_solver_options->SetOption(solvers::CommonSolverOption::kPrintToConsole,
  // 1);

  ControlLyapunov::RhoBisectionOption rho_bisection_option(0.01, 2, 0.01);
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  symbolic::Polynomial r;
  double rho_sol;

  dut.Search(V_init, lambda0_degree, l_degrees, V_degree, deriv_eps, x_star, S,
             V_degree - 2, search_options, rho_bisection_option, V_sol,
             &lambda0, &l, &r, &rho_sol);
}

[[maybe_unused]] void search_w_box_bounds(
    const Vector6<symbolic::Variable>& x, const symbolic::Polynomial& V_init,
    const Vector6<symbolic::Polynomial>& f,
    const Eigen::Matrix<symbolic::Polynomial, 6, 2>& G,
    symbolic::Polynomial* V_sol, double* deriv_eps_sol) {
  const int V_degree = 2;

  const double positivity_eps{0};
  const ControlLyapunovBoxInputBound dut(f, G, x, positivity_eps);
  ControlLyapunovBoxInputBound::SearchOptions search_options;
  search_options.bilinear_iterations = 10;
  search_options.backoff_scale = 0.02;
  // I run into numerical problem if I only search for l.
  search_options.search_l_and_b = true;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  // search_options.lagrangian_step_solver_options->SetOption(
  //    solvers::CommonSolverOption::kPrintToConsole, 1);
  ControlLyapunovBoxInputBound::RhoBisectionOption rho_bisection_option(0.01, 5,
                                                                        0.01);

  const int nu = 2;
  const int num_vdot_sos = 2;
  std::vector<std::vector<symbolic::Polynomial>> l_given(nu);
  for (int i = 0; i < nu; ++i) {
    l_given[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      l_given[i][j] = symbolic::Polynomial();
    }
  }
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees(nu);
  for (int i = 0; i < nu; ++i) {
    lagrangian_degrees[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      lagrangian_degrees[i][j] = {0, 2, 4};
    }
  }
  std::vector<int> b_degrees(nu);
  for (int i = 0; i < nu; ++i) {
    b_degrees[i] = 4;
  }
  const Vector6d x_star = Vector6d::Zero();
  const Eigen::Matrix<double, 6, 6> S = Eigen::Matrix<double, 6, 6>::Identity();
  const int r_degree = V_degree - 2;
  const double deriv_eps_lower = 0.2;
  const double deriv_eps_upper = kInf;

  const auto search_result =
      dut.Search(V_init, l_given, lagrangian_degrees, b_degrees, x_star, S,
                 r_degree, V_degree, deriv_eps_lower, deriv_eps_upper,
                 search_options, rho_bisection_option);
  *V_sol = search_result.V;
  *deriv_eps_sol = search_result.deriv_eps;
}

int DoMain() {
  Quadrotor quadrotor;

  // Synthesize an LQR controller.
  const Vector6d x_des = Vector6d::Zero();
  const double thrust_equilibrium = quadrotor.mass * quadrotor.gravity / 2;
  const Eigen::Vector2d u_des = quadrotor.NormalizeU(
      Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium));
  const auto x_des_ad = math::InitializeAutoDiff(x_des);
  Vector6<AutoDiffXd> f_des;
  Eigen::Matrix<AutoDiffXd, 6, 2> G_des;
  quadrotor.ControlAffineDynamics<AutoDiffXd>(x_des_ad, &f_des, &G_des);
  const auto xdot_des = f_des + G_des * u_des;
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10, 10;
  const Eigen::Matrix<double, 6, 6> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      math::ExtractGradient(xdot_des), math::ExtractValue(G_des), lqr_Q,
      Eigen::Matrix2d::Identity() * 10);

  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable(fmt::format("x{}", i));
  }

  Vector6<symbolic::Polynomial> f;
  Eigen::Matrix<symbolic::Polynomial, 6, 2> G;
  quadrotor.PolynomialControlAffineDynamics(x, &f, &G);
  for (int i = 0; i < 6; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    for (int j = 0; j < 2; ++j) {
      G(i, j) = G(i, j).RemoveTermsWithSmallCoefficients(1E-6);
    }
  }

  symbolic::Polynomial V_init(
      1 * x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  V_init = V_init.RemoveTermsWithSmallCoefficients(1E-8);
  std::cout << "V_init: " << V_init << "\n";

  Eigen::MatrixXd x_samples = Eigen::MatrixXd::Random(6, 100000) * 0.15;
  const VdotCalculator vdot_calculator(x, V_init, f, G);
  const Eigen::VectorXd vdot_samples = vdot_calculator.CalcMin(x_samples);
  const Eigen::VectorXd v_samples = V_init.EvaluateIndeterminates(x, x_samples);
  for (int i = 0; i < v_samples.rows(); ++i) {
    if (v_samples(i) < 1 && vdot_samples(i) > 0) {
      std::cout << fmt::format("v = {}, vdot = {}\n", v_samples(i),
                               vdot_samples(i));
    }
  }

  symbolic::Polynomial V_sol;
  double deriv_eps_sol;
  search(x, V_init, f, G, &V_sol, &deriv_eps_sol);
  // search_w_box_bounds(x, V_init, f, G, &V_sol, &deriv_eps_sol);

  Vector6d x0 = Vector6d::Zero();
  x0(0) += 0.5;
  x0(2) += 0.2 * M_PI;
  Simulate(x, V_sol, deriv_eps_sol, x0, 10);
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
