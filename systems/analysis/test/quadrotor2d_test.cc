#include "drake/systems/analysis/test/quadrotor2d.h"

#include <gtest/gtest.h>

#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(Quadrotor, TestPolyDynamics) {
  const Quadrotor2dTrigPlant<double> dut;
  Vector6d x;
  x << 0.2, 0.3, 0.5, 1.2, 2.1, -2.5;
  const Eigen::Vector2d u(0.5, 0.9);
  auto context = dut.CreateDefaultContext();
  context->SetContinuousState(x);
  dut.get_input_port().FixValue(context.get(), u);
  const ContinuousState<double>& derivatives =
      dut.EvalTimeDerivatives(*context);
  const auto derivatives_val = derivatives.CopyToVector();

  const auto x_trig = ToQuadrotor2dTrigState<double>(x);
  EXPECT_TRUE(CompareMatrices(x_trig.tail<2>(), x.tail<2>()));
  EXPECT_TRUE(CompareMatrices(x_trig.tail<3>(), x.tail<3>()));
  EXPECT_EQ(x_trig(2), std::sin(x(2)));
  EXPECT_EQ(x_trig(3), std::cos(x(2)) - 1);

  const auto xdot_trig = TrigDynamics<double>(dut, x_trig, u);
  EXPECT_TRUE(CompareMatrices(xdot_trig.head<2>(), derivatives_val.head<2>()));
  EXPECT_TRUE(CompareMatrices(xdot_trig.tail<3>(), derivatives_val.tail<3>()));
  EXPECT_EQ(xdot_trig(2), std::cos(x(2)) * x(5));
  EXPECT_EQ(xdot_trig(3), -std::sin(x(2)) * x(5));

  Eigen::Matrix<symbolic::Polynomial, 7, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 7, 2> G;
  Eigen::Matrix<symbolic::Variable, 7, 1> x_trig_sym;
  for (int i = 0; i < 7; ++i) {
    x_trig_sym(i) = symbolic::Variable("x" + std::to_string(i));
  }
  TrigPolyDynamics(dut, x_trig_sym, &f, &G);
  symbolic::Environment env;
  env.insert(x_trig_sym, x_trig);
  for (int i = 0; i < 7; ++i) {
    EXPECT_NEAR(xdot_trig(i),
                f(i).Evaluate(env) + G(i, 0).Evaluate(env) * u(0) +
                    G(i, 1).Evaluate(env) * u(1),
                1E-7);
  }
}

GTEST_TEST(TwinQuadrotor, Test) {
  Quadrotor2dTrigPlant<double> quadrotor;
  Vector6d state1;
  Vector6d state2;
  state1 << 0.5, 2.1, 1.3, -0.5, 2.1, 2.5;
  state2 << -0.2, 1.1, 0.5, 1.5, -0.3, 2;
  const auto x1_val = ToQuadrotor2dTrigState<double>(state1);
  const auto x2_val = ToQuadrotor2dTrigState<double>(state2);
  Eigen::Matrix<symbolic::Variable, 12, 1> x;
  for (int i = 0; i < 12; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 12, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 12, 4> G;
  TrigPolyDynamicsTwinQuadrotor(quadrotor, x, &f, &G);
  const Eigen::Vector2d u1(1, 3);
  const Eigen::Vector2d u2(2, 5);
  Eigen::Matrix<double, 12, 1> x_val;
  x_val << x1_val.tail<5>(), x2_val.head<2>() - x1_val.head<2>(),
      x2_val.tail<5>();
  symbolic::Environment env;
  env.insert(x, x_val);
  Eigen::Matrix<double, 12, 1> f_val;
  Eigen::Matrix<double, 12, 4> G_val;
  for (int i = 0; i < 12; ++i) {
    f_val(i) = f(i).Evaluate(env);
    for (int j = 0; j < 4; ++j) {
      G_val(i, j) = G(i, j).Evaluate(env);
    }
  }
  Eigen::VectorXd xdot_val =
      f_val + G_val.leftCols<2>() * u1 + G_val.rightCols<2>() * u2;
  const auto xdot1_val = TrigDynamics<double>(quadrotor, x1_val, u1);
  const auto xdot2_val = TrigDynamics<double>(quadrotor, x2_val, u2);
  Eigen::Matrix<double, 12, 1> xdot_val_expected;
  xdot_val_expected << xdot1_val.tail<5>(),
      xdot2_val.head<2>() - xdot1_val.head<2>(), xdot2_val.tail<5>();
  EXPECT_TRUE(CompareMatrices(xdot_val, xdot_val_expected, 1E-12));

  const auto eq_constraints = TwinQuadrotor2dStateEqConstraint(x);
  for (int i = 0; i < eq_constraints.rows(); ++i) {
    EXPECT_NEAR(eq_constraints(i).Evaluate(env), 0, 1E-12);
  }
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
