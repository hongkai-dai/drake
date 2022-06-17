#include "drake/systems/analysis/test/pendulum.h"

#include <vector>

#include <eigen3/Eigen/src/Core/Matrix.h>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(ControlAffineDynamics, Test) {
  const examples::pendulum::PendulumPlant<double> pendulum;
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  const double theta_des = M_PI;
  Vector2<symbolic::Polynomial> f;
  Vector2<symbolic::Polynomial> G;
  ControlAffineDynamics(pendulum, x, theta_des, &f, &G);

  std::vector<Eigen::Vector3d> xu_samples;
  xu_samples.emplace_back(0.2, 0.5, -1);
  xu_samples.emplace_back(-0.2, 2, 1.2);
  xu_samples.emplace_back(0.1, 0.2, 1.2);
  auto context = pendulum.CreateDefaultContext();
  for (const auto& xu : xu_samples) {
    symbolic::Environment env;
    env.insert(x, xu.head<2>());
    Eigen::Vector2d f_val;
    Eigen::Vector2d G_val;
    for (int i = 0; i < 2; ++i) {
      f_val(i) = f(i).Evaluate(env);
      G_val(i) = G(i).Evaluate(env);
    }
    const Eigen::Vector2d xdot = f_val + G_val * xu(2);

    context->SetContinuousState(Eigen::Vector2d(theta_des + xu(0), xu(1)));
    pendulum.get_input_port(0).FixValue(context.get(), Vector1d(xu(2)));
    const ContinuousState<double>& derivatives =
        pendulum.EvalTimeDerivatives(*context);
    EXPECT_TRUE(
        CompareMatrices(xdot, derivatives.get_vector().CopyToVector(), 1E-3));
  }
}

GTEST_TEST(TrigDynamics, Test) {
  const examples::pendulum::PendulumPlant<double> pendulum;
  const double theta_des = M_PI;

  std::vector<Eigen::Vector3d> theta_thetadot_u_samples;
  theta_thetadot_u_samples.emplace_back(0.2, 0.5, 2);
  theta_thetadot_u_samples.emplace_back(1.2, -0.5, 5);
  theta_thetadot_u_samples.emplace_back(2.1, -3.5, -15);
  theta_thetadot_u_samples.emplace_back(-1.1, -3.5, -10);
  theta_thetadot_u_samples.emplace_back(5, -3.5, -15);

  auto context = pendulum.CreateDefaultContext();
  for (const auto& theta_thetadot_u : theta_thetadot_u_samples) {
    const double theta = theta_thetadot_u(0);
    const Eigen::Vector3d x(std::sin(theta) - std::sin(theta_des),
                            std::cos(theta) - std::cos(theta_des),
                            theta_thetadot_u(1));
    const Vector1d u = theta_thetadot_u.tail<1>();
    const Eigen::Vector3d xdot =
        TrigDynamics<double>(pendulum, x, theta_des, u);
    context->SetContinuousState(theta_thetadot_u.head<2>());
    pendulum.get_input_port(0).FixValue(context.get(), u);
    const ContinuousState<double>& derivatives =
        pendulum.EvalTimeDerivatives(*context);
    const double thetadot = theta_thetadot_u(1);
    const Eigen::Vector3d xdot_expected(std::cos(theta) * thetadot,
                                        -std::sin(theta) * thetadot,
                                        derivatives.CopyToVector()(1));
    EXPECT_TRUE(CompareMatrices(xdot, xdot_expected, 1E-10));
  }
}

GTEST_TEST(TrigPolyDynamics, Test) {
  const examples::pendulum::PendulumPlant<double> pendulum;
  const double theta_des = M_PI;

  std::vector<Eigen::Vector3d> theta_thetadot_u_samples;
  theta_thetadot_u_samples.emplace_back(0.2, 0.5, 2);
  theta_thetadot_u_samples.emplace_back(1.2, -0.5, 5);
  theta_thetadot_u_samples.emplace_back(2.1, -3.5, -15);
  theta_thetadot_u_samples.emplace_back(5, -3.5, -15);

  const Vector3<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"),
                                      symbolic::Variable("x2"));
  Vector3<symbolic::Polynomial> f;
  Vector3<symbolic::Polynomial> G;
  TrigPolyDynamics(pendulum, x, theta_des, &f, &G);

  for (const auto& theta_thetadot_u : theta_thetadot_u_samples) {
    const double theta = theta_thetadot_u(0);
    symbolic::Environment env;
    const Eigen::Vector3d x_val(std::sin(theta) - std::sin(theta_des),
                                std::cos(theta) - std::cos(theta_des),
                                theta_thetadot_u(1));
    env.insert(x, x_val);
    Eigen::Vector3d f_val;
    Eigen::Vector3d G_val;
    for (int i = 0; i < 3; ++i) {
      f_val(i) = f(i).Evaluate(env);
      G_val(i) = G(i).Evaluate(env);
    }
    const Eigen::Vector3d xdot_val = f_val + G_val * theta_thetadot_u(2);
    const Vector1d u = theta_thetadot_u.tail<1>();
    const Eigen::Vector3d xdot_val_expected =
        TrigDynamics<double>(pendulum, x_val, theta_des, u);
    EXPECT_TRUE(CompareMatrices(xdot_val, xdot_val_expected, 1E-10));
  }
}

class TrigDynamicsLQRController : public LeafSystem<double> {
 public:
  TrigDynamicsLQRController(
      const examples::pendulum::PendulumPlant<double>& pendulum,
      double theta_des, const Eigen::Ref<const Eigen::Matrix3d>& Q,
      const Eigen::Ref<const Vector1d>& R)
      : LeafSystem<double>(),
        lqr_result_{TrigDynamicsLQR(pendulum, theta_des, Q, R)},
        theta_des_{theta_des} {
    DeclareVectorInputPort("theta_thetadot", 2);
    DeclareVectorOutputPort("tau", 1, &TrigDynamicsLQRController::CalcTau);
    auto context = pendulum.CreateDefaultContext();
    const auto& p = pendulum.get_parameters(*context);
    u_des_ = p.mass() * p.gravity() * p.length() * std::sin(theta_des_);
  }

  void CalcTau(const Context<double>& context, BasicVector<double>* tau) const {
    const Eigen::Vector2d theta_thetadot = this->get_input_port().Eval(context);
    const double theta = theta_thetadot(0);
    const Eigen::Vector3d x(std::sin(theta) - std::sin(theta_des_),
                            std::cos(theta) - std::cos(theta_des_),
                            theta_thetadot(1));
    tau->SetFromVector(-lqr_result_.K * x + Vector1d(u_des_));
  }

  const controllers::LinearQuadraticRegulatorResult& lqr_result() const {
    return lqr_result_;
  }

 private:
  controllers::LinearQuadraticRegulatorResult lqr_result_;
  double theta_des_;
  double u_des_;
};

GTEST_TEST(TrigDynamicsLQR, Test) {
  // Construct a pendulum stabilized by the trig_lqr controller. Simulate the
  // system.
  const Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  const Vector1d R(1);
  const double theta_des = M_PI;

  DiagramBuilder<double> builder;
  auto pendulum =
      builder.AddSystem<examples::pendulum::PendulumPlant<double>>();
  auto controller =
      builder.AddSystem<TrigDynamicsLQRController>(*pendulum, theta_des, Q, R);
  builder.Connect(pendulum->get_state_output_port(),
                  controller->get_input_port());
  builder.Connect(controller->get_output_port(), pendulum->get_input_port());

  auto diagram = builder.Build();

  Simulator<double> simulator(*diagram);
  auto& context = simulator.get_mutable_context();
  auto& pendulum_context =
      diagram->GetMutableSubsystemContext(*pendulum, &context);
  pendulum_context.get_mutable_continuous_state_vector().SetFromVector(
      Eigen::Vector2d(theta_des + 0.5, 0.3));
  simulator.AdvanceTo(5);
  EXPECT_TRUE(CompareMatrices(
      pendulum_context.get_continuous_state_vector().CopyToVector(),
      Eigen::Vector2d(theta_des, 0), 1E-5));
}

GTEST_TEST(PendulumROA, TestTrigLQR) {
  // For pendulum with trigonometric dynamics, construct its LQR controller and
  // verify the ROA.
  const Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  const Vector1d R(1);
  const double theta_des = M_PI;
  examples::pendulum::PendulumPlant<double> pendulum;

  solvers::MathematicalProgram prog;
  auto x = prog.NewIndeterminates<3>("x");
  // Find a candidate V_init by using the LQR controller gain and form a
  // closed-loop system on the trigonometric dynamics.
  const auto lqr_result = TrigDynamicsLQR(pendulum, theta_des, Q, R);
  symbolic::Polynomial V_init;
  {
    auto context = pendulum.CreateDefaultContext();
    context->SetContinuousState(Eigen::Vector2d(theta_des, 0));
    pendulum.get_input_port().FixValue(
        context.get(), Vector1d(EquilibriumTorque(pendulum, theta_des)));
    auto linearized_pendulum = Linearize(pendulum, *context);
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
    const double positivity_eps = 0;
    const int d = 0;
    const VectorX<symbolic::Polynomial> state_constraints_init(0);
    const std::vector<int> c_lagrangian_degrees{};
    VectorX<symbolic::Polynomial> c_lagrangian;
    auto prog_V_init = FindCandidateLyapunov(
        x, V_init_degree, positivity_eps, d, state_constraints_init,
        c_lagrangian_degrees, x_val, xdot_val, &V_init, &c_lagrangian);
    const auto result_init = solvers::Solve(*prog_V_init);
    ASSERT_TRUE(result_init.is_success());
    V_init = result_init.GetSolution(V_init);
  }
  Vector3<symbolic::Polynomial> f;
  Vector3<symbolic::Polynomial> G;
  TrigPolyDynamics(pendulum, x, theta_des, &f, &G);
  const double u_des = EquilibriumTorque(pendulum, theta_des);
  const symbolic::Polynomial pi(
      -lqr_result.K.row(0).dot(x.cast<symbolic::Expression>()) + u_des);
  Vector3<symbolic::Polynomial> xdot = f + G * pi;
  for (int i = 0; i < 3; ++i) {
    xdot(i) = xdot(i).RemoveTermsWithSmallCoefficients(1E-6);
  }

  // Construct a program
  // max ρ
  // s.t (xᵀx)ᵈ(V(x)−ρ) − l(x)V̇(x) − p(x)c(x) is sos
  //     l(x) is sos
  const int d = 2;
  symbolic::Polynomial V = V_init;
  V = V.RemoveTermsWithSmallCoefficients(1E-7);
  const symbolic::Polynomial Vdot = V.Jacobian(x).dot(xdot);
  std::cout << "V: " << V << "\n";
  std::cout << "Vdot: " << Vdot << "\n";

  {
    // Print V and Vdot at sampled value. If I use V as x.dot(S * x) where S is
    // from the linearized trigonometric dynamics, then the corresponding Vdot
    // (using trigonometric nonlinear dynamics) is often positive.
    symbolic::Environment env;
    double theta = 0.1;
    env.insert(x, ToTrigState(theta, 0.2, theta_des));
    std::cout << "V_val: " << V.Evaluate(env) << "\n";
    std::cout << "Vdot_val: " << Vdot.Evaluate(env) << "\n";
  }
  const int l_degree = 2;
  symbolic::Polynomial l;
  const symbolic::Variables x_set(x);
  std::tie(l, std::ignore) = prog.NewSosPolynomial(x_set, l_degree);
  const double sin_theta_des = std::sin(theta_des);
  const double cos_theta_des = std::cos(theta_des);
  const symbolic::Polynomial c((x(0) + sin_theta_des) * (x(0) + sin_theta_des) +
                               (x(1) + cos_theta_des) * (x(1) + cos_theta_des) -
                               1);
  const int p_degree = 6;
  const symbolic::Polynomial p = prog.NewFreePolynomial(x_set, p_degree);
  using std::pow;
  const symbolic::Polynomial x_squared_d(
      pow(x.cast<symbolic::Expression>().dot(x), d));
  const symbolic::Variable rho = prog.NewContinuousVariables<1>("rho")(0);
  prog.AddSosConstraint(x_squared_d * (V - symbolic::Polynomial(rho, x_set)) -
                        l * Vdot - p * c);
  prog.AddLinearCost(-rho);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  // For some reason Mosek doesn't work well, it runs into Unknown error.
  solvers::CsdpSolver csdp_solver;
  const auto result = csdp_solver.Solve(prog, std::nullopt, solver_options);
  std::cout << result.get_solution_result() << "\n";
  ASSERT_TRUE(result.is_success());
  const double rho_sol = result.GetSolution(rho);
  std::cout << "rho_sol: " << rho_sol << "\n";
  EXPECT_GT(rho_sol, 0.001);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
