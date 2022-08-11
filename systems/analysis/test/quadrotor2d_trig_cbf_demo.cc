#include <math.h>

#include <iostream>
#include <limits>

#include "drake/geometry/scene_graph.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/test/quadrotor2d.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

enum Scenario {
  kCeilingGround,
  kBox,
};

Eigen::Matrix<double, 4, 2> GetCbfAu() {
  Eigen::Matrix<double, 4, 2> Au;
  // clang-format off
  Au << 1, 0,
        0, 1,
        -1, 0,
        0, -1;
  // clang-format on
  return Au;
}

class QuadrotorCbfController : public CbfController {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuadrotorCbfController)

  QuadrotorCbfController(Eigen::Matrix<symbolic::Variable, 7, 1> x,
                         Eigen::Matrix<symbolic::Polynomial, 7, 1> f,
                         Eigen::Matrix<symbolic::Polynomial, 7, 2> G,
                         symbolic::Polynomial cbf, double deriv_eps,
                         double thrust_max)
      : CbfController(x, f, G, std::nullopt /* dynamics_denominator */, cbf,
                      deriv_eps),
        thrust_max_{thrust_max} {}

  virtual ~QuadrotorCbfController() {}

 private:
  void DoCalcControl(const Context<double>& context,
                     BasicVector<double>* output) const {
    const Eigen::VectorXd x_val = this->x_input_port().Eval(context);
    symbolic::Environment env;
    env.insert(this->x(), x_val);

    solvers::MathematicalProgram prog;
    const int nu = 2;
    auto u = prog.NewContinuousVariables<2>("u");
    prog.AddBoundingBoxConstraint(0, thrust_max_, u);
    prog.AddQuadraticCost(Eigen::Matrix2d::Identity(), Eigen::Vector2d::Zero(),
                          0, u);
    const double dhdx_times_f_val = dhdx_times_f().Evaluate(env);
    Eigen::RowVector2d dhdx_times_G_val;
    for (int i = 0; i < nu; ++i) {
      dhdx_times_G_val(i) = this->dhdx_times_G()(i).Evaluate(env);
    }
    const double h_val = this->cbf().Evaluate(env);
    // dhdx * G * u + dhdx * f >= -eps * h
    prog.AddLinearConstraint(dhdx_times_G_val,
                             -deriv_eps() * h_val - dhdx_times_f_val, kInf, u);
    const auto result = solvers::Solve(prog);
    if (!result.is_success()) {
      drake::log()->info("-dhdx*f - eps*h={}, dhdx*G={}",
                         -dhdx_times_f_val - deriv_eps() * h_val,
                         dhdx_times_G_val);
      abort();
    }
    const Eigen::Vector2d u_val = result.GetSolution(u);
    output->get_mutable_value() = u_val;
  }

 private:
  double thrust_max_;
};

void Simulate(const Eigen::Matrix<symbolic::Variable, 7, 1>& x,
              const symbolic::Polynomial& cbf, double thrust_max,
              double deriv_eps, const Vector6d& initial_state,
              double duration) {
  systems::DiagramBuilder<double> builder;

  auto quadrotor = builder.AddSystem<Quadrotor2dTrigPlant<double>>();

  auto state_converter =
      builder.AddSystem<Quadrotor2dTrigStateConverter<double>>();

  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(*quadrotor, x, &f, &G);

  auto cbf_controller = builder.AddSystem<QuadrotorCbfController>(
      x, f, G, cbf, deriv_eps, thrust_max);

  auto state_logger =
      LogVectorOutput(quadrotor->get_state_output_port(), &builder);

  auto cbf_logger =
      LogVectorOutput(cbf_controller->cbf_output_port(), &builder);

  auto control_logger =
      LogVectorOutput(cbf_controller->control_output_port(), &builder);

  builder.Connect(quadrotor->get_state_output_port(),
                  state_converter->get_input_port());
  builder.Connect(state_converter->get_output_port(),
                  cbf_controller->x_input_port());
  builder.Connect(cbf_controller->control_output_port(),
                  quadrotor->get_actuation_input_port());

  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);

  ResetIntegratorFromFlags(&simulator, "implicit_euler", 0.01);

  simulator.get_mutable_context().SetContinuousState(initial_state);
  simulator.AdvanceTo(duration);

  const Eigen::MatrixXd state_data =
      state_logger->FindLog(simulator.get_context()).data();
  std::cout << "z_min: " << state_data.row(1).minCoeff() << "\n";
  std::cout << "z_max: " << state_data.row(1).maxCoeff() << "\n";

  const Eigen::RowVectorXd cbf_data =
      cbf_logger->FindLog(simulator.get_context()).data().row(0);
  std::cout << "cbf min: " << cbf_data.minCoeff() << "\n";

  unused(control_logger);
  unused(cbf_logger);
}

symbolic::Polynomial FindCbfInit(
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, int h_degree) {
  // I first synthesize an LQR controller, and find a Lyapunov function V for
  // this LQR controller, then -V + eps should satisfy the CBF condition within
  // a small neighbourhood.
  Eigen::VectorXd lqr_Q_diag(7);
  lqr_Q_diag << 1, 1, 1, 1, 10, 10, 10;
  const Eigen::MatrixXd lqr_Q = lqr_Q_diag.asDiagonal();
  const Eigen::Matrix2d lqr_R = 10 * Eigen::Matrix2d::Identity();
  const auto lqr_result = SynthesizeQuadrotor2dTrigLqr(lqr_Q, lqr_R);
  Quadrotor2dTrigPlant<double> quadrotor2d;
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
  const int d = h_degree / 2;
  const double deriv_eps = 0.0001;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      Quadrotor2dStateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{h_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{
      {static_cast<int>(std::ceil((h_degree + 1) / 2.) * 2 - 2)}};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{h_degree - 2};
  const std::vector<int> derivative_cin_lagrangian_degrees =
      derivative_ceq_lagrangian_degrees;
  symbolic::Polynomial V;
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian;
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian;
  VectorX<symbolic::Polynomial> derivative_cin_lagrangian;
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian;
  symbolic::Polynomial positivity_sos_condition;
  symbolic::Polynomial derivative_sos_condition;
  auto prog = FindCandidateRegionalLyapunov(
      x, dynamics, std::nullopt /* dynamics_denominator*/, h_degree,
      positivity_eps, d, deriv_eps, state_eq_constraints,
      positivity_ceq_lagrangian_degrees, derivative_ceq_lagrangian_degrees,
      state_ineq_constraints, positivity_cin_lagrangian_degrees,
      derivative_cin_lagrangian_degrees, &V, &positivity_cin_lagrangian,
      &positivity_ceq_lagrangian, &derivative_cin_lagrangian,
      &derivative_ceq_lagrangian, &positivity_sos_condition,
      &derivative_sos_condition);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  DRAKE_DEMAND(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(V);
  return -V_sol + 0.1;
}

[[maybe_unused]] symbolic::Polynomial SearchWithSlackA(
    Scenario scenario, const Quadrotor2dTrigPlant<double>& quadrotor,
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, double thrust_max,
    double deriv_eps,
    const std::vector<VectorX<symbolic::Polynomial>>& unsafe_regions,
    const Eigen::MatrixXd& x_safe,
    const std::optional<std::string>& load_cbf_file) {
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(
      Quadrotor2dStateEqConstraint(x));

  const std::optional<symbolic::Polynomial> dynamics_denominator = std::nullopt;

  const double beta_minus = -0.;
  std::optional<double> beta_plus = std::nullopt;
  if (scenario == Scenario::kBox) {
    beta_plus = 0.1;
  }
  const ControlBarrier dut(f, G, dynamics_denominator, x, beta_minus, beta_plus,
                           unsafe_regions, u_vertices, state_constraints);

  // Set h_init to some values, it should satisfy h_init(x_safe) >= 0.
  const int h_degree = 2;
  const symbolic::Variables x_set(x);
  // symbolic::Polynomial h_init = FindCbfInit(x, h_degree);
  symbolic::Polynomial h_init;
  if (load_cbf_file.has_value()) {
    h_init = Load(x_set, load_cbf_file.value());
  } else {
    switch (scenario) {
      case Scenario::kCeilingGround: {
        h_init =
            symbolic::Polynomial(1 - x.cast<symbolic::Expression>().dot(x));
        break;
      }
      case Scenario::kBox: {
        h_init = symbolic::Polynomial(
            pow(x(0) - 0.8, 2) +
            x.tail<5>().cast<symbolic::Expression>().dot(x.tail<5>()) - 1);
        break;
      }
    }
  }
  const Eigen::VectorXd h_init_x_safe =
      h_init.EvaluateIndeterminates(x, x_safe);
  std::cout << "h_init(x_safe): " << h_init_x_safe.transpose() << "\n";
  if ((h_init_x_safe.array() < 0).any()) {
    h_init -= h_init_x_safe.minCoeff();
    h_init += 0.1;
  }

  const int lambda0_degree = 4;
  const std::optional<int> lambda1_degree = std::nullopt;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_eq_lagrangian_degrees = {h_degree +
                                                       lambda0_degree - 2};
  const int hdot_a_degree = h_degree + lambda0_degree;

  const std::vector<int> t_degrees = {0, 0};
  std::vector<std::vector<int>> s_degrees;
  std::vector<std::vector<int>> unsafe_eq_lagrangian_degrees;
  std::vector<std::optional<int>> unsafe_a_degrees;
  Eigen::VectorXd h_x_safe_min = Eigen::VectorXd::Constant(x_safe.cols(), 0.01);
  switch (scenario) {
    case Scenario::kCeilingGround: {
      s_degrees = {{h_degree - 2}, {h_degree - 2}};
      unsafe_eq_lagrangian_degrees = {{h_degree - 2}, {h_degree - 2}};
      unsafe_a_degrees = {h_degree, h_degree};
      break;
    }
    case Scenario::kBox: {
      s_degrees = {{h_degree - 2, h_degree - 2, h_degree - 2, h_degree - 2}};
      unsafe_eq_lagrangian_degrees = {{h_degree - 2}};
      unsafe_a_degrees = {h_degree};
      break;
    }
  };

  double hdot_a_zero_tol = 3E-9;
  double unsafe_a_zero_tol = 1E-9;
  ControlBarrier::SearchWithSlackAOptions search_options(
      hdot_a_zero_tol, unsafe_a_zero_tol, true, 1, std::vector<double>{1., 1.});
  search_options.bilinear_iterations = 100;
  const auto search_result = dut.SearchWithSlackA(
      h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      hdot_eq_lagrangian_degrees, hdot_a_degree, t_degrees, s_degrees,
      unsafe_eq_lagrangian_degrees, unsafe_a_degrees, x_safe, h_x_safe_min,
      search_options);
  std::cout << search_result.h << "\n";

  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto search_lagrangian_ret = dut.SearchLagrangian(
      search_result.h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      hdot_eq_lagrangian_degrees, std::nullopt /* hdot_a_degree */, t_degrees,
      s_degrees, unsafe_eq_lagrangian_degrees,
      std::vector<std::optional<int>>(unsafe_regions.size(),
                                      std::nullopt) /* unsafe_a_degrees */,
      search_options, std::nullopt /* backoff_scale */);
  drake::log()->info("h_sol is valid? {}", search_lagrangian_ret.success);
  return search_result.h;
}

[[maybe_unused]] symbolic::Polynomial Search(
    const Quadrotor2dTrigPlant<double>& quadrotor,
    const Eigen::Matrix<symbolic::Variable, 7, 1>& x, double thrust_max,
    double deriv_eps,
    const std::vector<VectorX<symbolic::Polynomial>>& unsafe_regions) {
  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  TrigPolyDynamics(quadrotor, x, &f, &G);
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const Vector1<symbolic::Polynomial> state_constraints(
      Quadrotor2dStateEqConstraint(x));

  const std::optional<symbolic::Polynomial> dynamics_denominator = std::nullopt;

  const double beta_minus = -1;
  const std::optional<double> beta_plus = std::nullopt;
  const ControlBarrier dut(f, G, dynamics_denominator, x, beta_minus, beta_plus,
                           unsafe_regions, u_vertices, state_constraints);

  const int h_degree = 2;
  symbolic::Polynomial h_init = FindCbfInit(x, h_degree);
  const int lambda0_degree = 4;
  const std::optional<int> lambda1_degree = std::nullopt;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_eq_lagrangian_degrees{h_degree - 1};
  const std::vector<int> t_degree = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{h_degree - 2},
                                                   {h_degree - 2}};
  const std::vector<std::vector<int>>
      unsafe_state_constraints_lagrangian_degrees{{h_degree - 1},
                                                  {h_degree - 1}};
  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  std::vector<std::variant<ControlBarrier::EllipsoidBisectionOption,
                           ControlBarrier::EllipsoidMaximizeOption>>
      ellipsoid_options;
  ellipsoids.emplace_back(Eigen::Matrix<double, 7, 1>::Zero(),
                          Eigen::Matrix<double, 7, 7>::Identity(), 0,
                          h_degree - 2, std::vector<int>{{h_degree - 1}});
  ellipsoid_options.push_back(
      ControlBarrier::EllipsoidBisectionOption(0, 2, 0.01));

  const Eigen::Matrix<double, 7, 1> x_anchor =
      Eigen::Matrix<double, 7, 1>::Zero();
  const double h_x_anchor_max = h_init.EvaluateIndeterminates(x, x_anchor)(0);
  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.barrier_step_solver_options = solvers::SolverOptions();
  search_options.barrier_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto search_ret = dut.Search(
      h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      hdot_eq_lagrangian_degrees, t_degree, s_degrees,
      unsafe_state_constraints_lagrangian_degrees, x_anchor, h_x_anchor_max,
      search_options, &ellipsoids, &ellipsoid_options);
  std::cout << "h_sol: " << search_ret.h << "\n";

  Eigen::Matrix<double, 6, Eigen::Dynamic> state_samples(6, 5);
  state_samples.col(0) << 0.2, 0.2, 0, 0, 0, 0;
  state_samples.col(1) << 0.2, -0.3, 0, 0, 0, 0;
  state_samples.col(2) << 0, 0.4, 0.1 * M_PI, 0, 0, 0;
  state_samples.col(3) << 0, -0.3, 0.2 * M_PI, 0, 0, 0;
  state_samples.col(4) << 0, 0.2, 0.3 * M_PI, 0.1, 0.2, 0;
  Eigen::Matrix<double, 7, Eigen::Dynamic> x_samples(7, state_samples.cols());
  for (int i = 0; i < state_samples.cols(); ++i) {
    x_samples.col(i) = ToQuadrotor2dTrigState<double>(state_samples.col(i));
  }
  std::cout << "h at samples: "
            << search_ret.h.EvaluateIndeterminates(x, x_samples).transpose()
            << "\n";

  return search_ret.h;
}

int DoMain() {
  const Quadrotor2dTrigPlant<double> plant{};
  Eigen::Matrix<symbolic::Variable, 7, 1> x;
  for (int i = 0; i < 7; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const double thrust_equilibrium = EquilibriumThrust(plant);
  const double thrust_max = 3 * thrust_equilibrium;
  const double deriv_eps = 0.5;
  // The unsafe region is the ground and the ceiling.
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions;
  Eigen::MatrixXd safe_states;
  const Scenario scenario{Scenario::kCeilingGround};
  switch (scenario) {
    case Scenario::kCeilingGround: {
      unsafe_regions.resize(2);
      unsafe_regions[0].resize(1);
      unsafe_regions[0](0) = symbolic::Polynomial(x(1) + 0.3);
      unsafe_regions[1].resize(1);
      unsafe_regions[1](0) = symbolic::Polynomial(0.5 - x(1));
      safe_states.resize(6, 1);
      safe_states.col(0) << 0, 0, 0, 0, 0, 0;
      // safe_states.col(1) << 0, 0.2, 0, 0, 0, 0;
      // safe_states.col(2) << 0, -0.2, 0, 0, 0, 0;
      // safe_states.col(3) << 0, 0.45, 0.1, 0, 0, 0;
      break;
    }
    case Scenario::kBox: {
      // A box 0.6 <= px <= 1
      //       -0.2 <= pz <= 0.2
      unsafe_regions.resize(1);
      unsafe_regions[0].resize(4);
      unsafe_regions[0](0) = symbolic::Polynomial(x(0) - 1);
      unsafe_regions[0](1) = symbolic::Polynomial(0.6 - x(0));
      unsafe_regions[0](2) = symbolic::Polynomial(x(1) - 0.1);
      unsafe_regions[0](3) = symbolic::Polynomial(-0.2 - x(1));
      safe_states.resize(6, 2);
      safe_states.col(0) << 0, 0, 0, 0, 0, 0;
      safe_states.col(1) << 0.5, 0, 0, 0, 0, 0;
      break;
    }
  }

  Eigen::MatrixXd x_safe(7, safe_states.cols());
  for (int i = 0; i < safe_states.cols(); ++i) {
    x_safe.col(i) = ToQuadrotor2dTrigState<double>(safe_states.col(i));
  }

  std::optional<std::string> load_cbf_file = std::nullopt;
  // load_cbf_file =
  //    "/home/hongkaidai/sos_clf_cbf_data/quadrotor2d_cbf/"
  //    "quadrotor2d_trig_cbf_box2.txt";
  const symbolic::Polynomial h_sol =
      SearchWithSlackA(scenario, plant, x, thrust_max, deriv_eps,
                       unsafe_regions, x_safe, load_cbf_file);
  Save(h_sol,
       "/home/hongkaidai/sos_clf_cbf_data/quadrotor2d_cbf/"
       "quadrotor2d_trig_cbf_box3.txt");
  // Save(h_sol, "sos_data/quadrotor2d_trig_cbf5.txt");

  // Simulate(x, h_sol, thrust_max, deriv_eps, Vector6d::Zero(), 10);

  // const symbolic::Polynomial h_sol =
  //    Search(plant, x, thrust_max, deriv_eps, unsafe_regions);
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
