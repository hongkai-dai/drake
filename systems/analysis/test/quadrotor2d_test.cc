#include "drake/systems/analysis/test/quadrotor2d.h"

#include <gtest/gtest.h>

#include "drake/common/symbolic.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(Quadrotor, TestPolyDynamics) {
  const QuadrotorPlant<double> dut;
  Vector6d x;
  x << 0.2, 0.3, 0.5, 1.2, 2.1, -2.5;
  const Eigen::Vector2d u(0.5, 0.9);
  auto context = dut.CreateDefaultContext();
  context->SetContinuousState(x);
  dut.get_input_port().FixValue(context.get(), u);
  const ContinuousState<double>& derivatives =
      dut.EvalTimeDerivatives(*context);
  const auto derivatives_val = derivatives.CopyToVector();

  const auto x_trig = ToTrigState<double>(x);
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
}  // namespace analysis
}  // namespace systems
}  // namespace drake
