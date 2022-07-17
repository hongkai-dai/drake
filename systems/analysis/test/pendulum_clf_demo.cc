#include <math.h>

#include <iostream>
#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/examples/pendulum/pendulum_geometry.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/geometry/meshcat_visualizer.h"
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
#include "drake/systems/primitives/adder.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/linear_system.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

void Simulate(const Vector2<symbolic::Variable>& x, double theta_des,
              const symbolic::Polynomial& clf, double u_bound, double deriv_eps,
              const Eigen::Vector2d& x0, double duration) {
  systems::DiagramBuilder<double> builder;
  auto pendulum =
      builder.AddSystem<examples::pendulum::PendulumPlant<double>>();
  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  examples::pendulum::PendulumGeometry::AddToBuilder(
      &builder, pendulum->get_state_output_port(), scene_graph);
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

  Vector2<symbolic::Polynomial> f;
  Vector2<symbolic::Polynomial> G;
  ControlAffineDynamics(*pendulum, x, theta_des, &f, &G);

  const double vdot_cost = 0;
  auto clf_controller = builder.AddSystem<ClfController>(
      x, f, G, std::nullopt /* dynamics numerator */, clf, deriv_eps, Au, bu,
      u_star, Ru, vdot_cost);
  auto state_logger =
      LogVectorOutput(pendulum->get_state_output_port(), &builder);
  auto clf_logger = LogVectorOutput(
      clf_controller->get_output_port(clf_controller->clf_output_index()),
      &builder);
  // Shift the state by (-theta_des, 0) as the input to the clf controller.
  auto state_shifter = builder.AddSystem<ConstantVectorSource<double>>(
      Eigen::Vector2d(-theta_des, 0));
  const int nx = pendulum->num_continuous_states();
  auto state_adder = builder.AddSystem<Adder<double>>(2, nx);
  builder.Connect(pendulum->get_state_output_port(),
                  state_adder->get_input_port(0));
  builder.Connect(state_shifter->get_output_port(),
                  state_adder->get_input_port(1));
  builder.Connect(
      state_adder->get_output_port(),
      clf_controller->get_input_port(clf_controller->x_input_index()));
  builder.Connect(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      pendulum->get_input_port());
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

void SimulateTrigClf(const Vector3<symbolic::Variable>& x, double theta_des,
                     const symbolic::Polynomial& clf, double u_bound,
                     double deriv_eps, double theta0, double thetadot0,
                     double duration) {
  systems::DiagramBuilder<double> builder;
  auto pendulum =
      builder.AddSystem<examples::pendulum::PendulumPlant<double>>();
  auto scene_graph = builder.AddSystem<geometry::SceneGraph<double>>();
  examples::pendulum::PendulumGeometry::AddToBuilder(
      &builder, pendulum->get_state_output_port(), scene_graph);
  auto meshcat = std::make_shared<geometry::Meshcat>();
  geometry::MeshcatVisualizerParams meshcat_params{};
  meshcat_params.role = geometry::Role::kIllustration;
  auto visualizer = &geometry::MeshcatVisualizer<double>::AddToBuilder(
      &builder, *scene_graph, meshcat, meshcat_params);
  unused(visualizer);

  Vector3<symbolic::Polynomial> f;
  Vector3<symbolic::Polynomial> G;
  TrigPolyDynamics(*pendulum, x, theta_des, &f, &G);
  Vector1d u_star(0);
  const double vdot_cost = 0;
  auto clf_controller = builder.AddSystem<ClfController>(
      x, f, G, std::nullopt /* dynamics numerator */, clf, deriv_eps,
      Eigen::Vector2d(1, -1), Eigen::Vector2d(u_bound, u_bound), u_star,
      Vector1d::Ones(), vdot_cost);

  auto state_converter = builder.AddSystem<TrigStateConverter>(theta_des);

  builder.Connect(pendulum->get_state_output_port(),
                  state_converter->get_input_port());
  builder.Connect(
      state_converter->get_output_port(),
      clf_controller->get_input_port(clf_controller->x_input_index()));
  builder.Connect(
      clf_controller->get_output_port(clf_controller->control_output_index()),
      pendulum->get_input_port());

  auto diagram = builder.Build();
  auto context = diagram->CreateDefaultContext();

  Simulator<double> simulator(*diagram);
  simulator.get_mutable_context().SetContinuousState(
      Eigen::Vector2d(theta0, thetadot0));
  diagram->Publish(simulator.get_context());
  std::cout << "Refresh meshcat brower and press to continue\n";
  std::cin.get();

  simulator.AdvanceTo(duration);
  std::cout << "Final state: "
            << simulator.get_context()
                   .get_continuous_state()
                   .get_vector()
                   .CopyToVector()
                   .transpose()
            << "\n";
}

[[maybe_unused]] void SearchWTrigPoly(bool max_ellipsoid) {
  const double u_bound = 30;
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
  symbolic::Polynomial V_init;
  {
    // Now take many samples around (theta_des, 0).
    const Eigen::VectorXd theta_samples =
        Eigen::VectorXd::LinSpaced(10, -0.2 + theta_des, 0.2 + theta_des);
    const Eigen::VectorXd thetadot_samples =
        Eigen::VectorXd::LinSpaced(10, -0.3, 0.3);
    Eigen::Matrix3Xd x_val(3, theta_samples.rows() * thetadot_samples.rows());
    Eigen::Matrix3Xd xdot_val(3, x_val.cols());
    int x_count = 0;
    // Compute control action from LQR controller.
    const Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
    const Vector1d R(1);
    const auto lqr_result = TrigDynamicsLQR(pendulum, theta_des, Q, R);
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
    {
      const double positivity_eps = 0;
      const int d = 0;
      const VectorX<symbolic::Polynomial> state_constraints_init(0);
      const std::vector<int> c_lagrangian_degrees{};
      VectorX<symbolic::Polynomial> c_lagrangian;
      auto prog_V_init = FindCandidateLyapunov(
          x, V_init_degree, positivity_eps, d, state_constraints_init,
          c_lagrangian_degrees, x_val, xdot_val, &V_init, &c_lagrangian);
      const auto result_init = solvers::Solve(*prog_V_init);
      DRAKE_DEMAND(result_init.is_success());
      V_init = result_init.GetSolution(V_init);
    }
  }
  const ControlLyapunov dut(x, f, G, std::nullopt /*dynamics denominator */,
                            u_vertices, state_constraints);
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees{4, 4};
  const std::vector<int> p_degrees{8};
  const int V_degree = 4;
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p;
  double rho;
  const double deriv_eps = 0.25;

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
    V_init = V_init / result.GetSolution(rho_var);
  }

  const double positivity_eps = 0.0001;
  const int positivity_d = 1;
  const std::vector<int> positivity_eq_lagrangian_degrees{{V_degree - 2}};
  symbolic::Polynomial V_sol;
  VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
  if (max_ellipsoid) {
    const Eigen::Vector3d x_star(0, -0.0, 0);
    const Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    const std::vector<int> ellipsoid_c_lagrangian_degrees{0};
    ControlLyapunov::SearchOptions search_options;
    search_options.bilinear_iterations = 50;
    // search_options.lyap_step_solver = solvers::CsdpSolver::id();
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    // search_options.lyap_step_solver_options->SetOption(
    //    solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.backoff_scale = 0.0;
    search_options.lsol_tiny_coeff_tol = 1E-5;
    // There are tiny coefficients coming from numerical roundoff error.
    search_options.lyap_tiny_coeff_tol = 1E-7;
    const double rho_min = 0.01;
    const double rho_max = 15;
    const double rho_bisection_tol = 0.01;
    const ControlLyapunov::RhoBisectionOption rho_bisection_option(
        rho_min, rho_max, rho_bisection_tol);
    symbolic::Polynomial r;
    VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
    dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
               positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
               ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S,
               V_degree - 2, search_options, rho_bisection_option, &V_sol,
               &lambda0, &l, &r, &p, &positivity_eq_lagrangian_sol, &rho,
               &ellipsoid_c_lagrangian_sol);
  } else {
    ControlLyapunov::SearchOptions search_options;
    search_options.rho_converge_tol = 0.;
    search_options.bilinear_iterations = 50;
    // search_options.lyap_step_solver = solvers::CsdpSolver::id();
    search_options.lyap_step_solver_options = solvers::SolverOptions();
    // search_options.lyap_step_solver_options->SetOption(
    //    solvers::CommonSolverOption::kPrintToConsole, 1);
    search_options.backoff_scale = 0.0;
    search_options.lsol_tiny_coeff_tol = 1E-5;
    // There are tiny coefficients coming from numerical roundoff error.
    search_options.lyap_tiny_coeff_tol = 1E-7;
    Eigen::MatrixXd x_samples(3, 1);
    x_samples.col(0) = ToTrigState<double>(0., 0, theta_des);

    symbolic::Polynomial lambda0_sol;
    VectorX<symbolic::Polynomial> l_sol;
    VectorX<symbolic::Polynomial> p_sol;
    SearchResultDetails search_result_details;
    dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
               positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
               deriv_eps, x_samples, std::nullopt /* in_roa_samples */, true,
               search_options, &V_sol, &positivity_eq_lagrangian_sol,
               &lambda0_sol, &l_sol, &p_sol, &search_result_details);
  }
  const double theta0 = 0.0;
  const double thetadot0 = 0.0;
  std::cout << fmt::format(
      "V at (theta, thetadot)=({}, {}) = {}\n", theta0, thetadot0,
      V_sol.EvaluateIndeterminates(
          x, ToTrigState<double>(theta0, thetadot0, theta_des))(0));

  SimulateTrigClf(x, theta_des, V_sol, u_bound, deriv_eps, theta0, thetadot0,
                  30);
}

[[maybe_unused]] void SearchWTaylorDynamics() {
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
  ControlLyapunov dut(x, f, G, std::nullopt /* dynamics denominator */,
                      u_vertices, state_constraints);

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
    const double positivity_eps = 0.0001;
    const int positivity_d = 1;
    const std::vector<int> positivity_eq_lagrangian_degrees{};
    const std::vector<int> p_degrees{};
    const std::vector<int> ellipsoid_c_lagrangian_degrees{};
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
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    double rho;
    VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
    dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
               positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
               ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S,
               V_degree - 2, search_options, rho_bisection_option, &V_sol,
               &lambda0, &l, &r, &p, &positivity_eq_lagrangian, &rho,
               &ellipsoid_c_lagrangian_sol);
    Simulate(x, theta_des, V_sol, u_bound, deriv_eps,
             Eigen::Vector2d(M_PI + 0.6 * M_PI, 0), 10);
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

int DoMain() {
  // SearchWTaylorDynamics();
  SearchWTrigPoly(false);
  return 0;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
