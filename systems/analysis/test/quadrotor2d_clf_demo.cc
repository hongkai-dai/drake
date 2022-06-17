#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
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

[[maybe_unused]] void Simulate(
    const Vector6<symbolic::Variable>& x,
    const Vector6<symbolic::Polynomial>& f,
    const Eigen::Matrix<symbolic::Polynomial, 6, 2>& G, double thrust_max,
    const symbolic::Polynomial& clf, double deriv_eps, const Vector6d& x0,
    double duration) {
  systems::DiagramBuilder<double> builder;
  auto quadrotor = builder.AddSystem<QuadrotorPlant<double>>();
  Eigen::Matrix<double, 4, 2> Au;
  Au.topRows<2>() = Eigen::Matrix2d::Identity();
  Au.bottomRows<2>() = -Eigen::Matrix2d::Identity();
  const Eigen::Vector4d bu(thrust_max, thrust_max, 0, 0);
  const Eigen::Vector2d u_star(0, 0);
  const Eigen::Matrix2d Ru = Eigen::Matrix2d::Identity();
  auto clf_controller = builder.AddSystem<ClfController>(
      x, f, G, clf, deriv_eps, Au, bu, u_star, Ru);

  auto state_logger =
      LogVectorOutput(quadrotor->get_state_output_port(), &builder);
  auto clf_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->clf_output_index()),
      &builder);
  builder.Connect(quadrotor->get_state_output_port(),
                  clf_controller->get_input_port());
  builder.Connect(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      quadrotor->get_input_port());
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

[[maybe_unused]] void SearchWTaylorDynamics() {
  QuadrotorPlant<double> quadrotor;

  // Synthesize an LQR controller.
  auto context = quadrotor.CreateDefaultContext();
  context->SetContinuousState(Vector6d::Zero());
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const Eigen::Vector2d u_des(thrust_equilibrium, thrust_equilibrium);
  quadrotor.get_input_port().FixValue(context.get(), u_des);
  auto linearized_quadrotor = Linearize(quadrotor, *context);
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10, 10;
  const Eigen::Matrix<double, 6, 6> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      linearized_quadrotor->A(), linearized_quadrotor->B(), lqr_Q,
      Eigen::Matrix2d::Identity() * 10);

  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable(fmt::format("x{}", i));
  }

  Vector6<symbolic::Polynomial> f;
  Eigen::Matrix<symbolic::Polynomial, 6, 2> G;

  PolynomialControlAffineDynamics(quadrotor, x, &f, &G);

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

  symbolic::Polynomial V_sol;
  const double deriv_eps = 0.2;
  const double thrust_max = 3 * thrust_equilibrium;
  {
    Eigen::Matrix<double, 2, 4> u_vertices;
    // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
    // clang-format on
    VectorX<symbolic::Polynomial> state_constraints(0);
    ControlLyapunov dut(x, f, G, u_vertices, state_constraints);
    {
      const symbolic::Polynomial lambda0{};
      const int d_degree = 2;
      const std::vector<int> l_degrees = {2, 2, 2, 2};
      const std::vector<int> p_degrees = {};
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
      const std::vector<int> l_degrees = {2, 2, 2, 2};
      const std::vector<int> p_degrees = {};
      const std::vector<int> ellipsoid_c_lagrangian_degrees = {};
      const int V_degree = 2;
      const double positivity_eps = 0.0001;
      const int positivity_d = V_degree / 2;
      const std::vector<int> positivity_eq_lagrangian_degrees{};
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
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
      VectorX<symbolic::Polynomial> p_sol;
      VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;

      dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
                 positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
                 ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S,
                 V_degree - 2, search_options, rho_bisection_option, &V_sol,
                 &lambda0, &l, &r, &p_sol, &positivity_eq_lagrangian_sol,
                 &rho_sol, &ellipsoid_c_lagrangian_sol);
    }
  }

  Vector6d x0 = Vector6d::Zero();
  x0(0) += 0.5;
  x0(2) += 0.2 * M_PI;
  Simulate(x, f, G, thrust_max, V_sol, deriv_eps, x0, 10);
}

int DoMain() {
  SearchWTaylorDynamics();
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
