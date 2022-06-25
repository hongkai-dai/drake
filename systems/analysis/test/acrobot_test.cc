#include "drake/systems/analysis/test/acrobot.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(Acrobot, DynamicsTest) {
  EXPECT_TRUE(
      CompareMatrices(ToTrigState<double>(Eigen::Vector4d(M_PI, 0, 0, 0)),
                      Vector6d::Zero(), 1E-12));
  examples::acrobot::AcrobotPlant<double> plant;
  auto context = plant.CreateDefaultContext();
  const auto& p = plant.get_parameters(*context);
  const Eigen::Vector4d x_orig(0.2, 1.4, -.5, 0.9);
  const double u = 3.1;
  context->SetContinuousState(x_orig);
  plant.get_input_port().FixValue(context.get(), Vector1d(u));
  const Vector6d x_trig = ToTrigState<double>(x_orig);
  const Eigen::Matrix2d M = MassMatrix<double>(p, x_trig);
  const Eigen::Matrix2d M_expected = plant.MassMatrix(*context);
  EXPECT_TRUE(CompareMatrices(M, M_expected, 1E-12));

  const Eigen::Vector2d bias = DynamicsBiasTerm<double>(p, x_trig);
  const Eigen::Vector2d bias_expected = plant.DynamicsBiasTerm(*context);
  EXPECT_TRUE(CompareMatrices(bias, bias_expected, 1E-12));

  Vector6d n;
  double d;
  TrigDynamics<double>(p, x_trig, u, &n, &d);
  const Vector6d x_trig_dot = n / d;
  const Eigen::Vector4d xdot_orig =
      plant.EvalTimeDerivatives(*context).CopyToVector();
  Vector6d x_trig_dot_expected;
  x_trig_dot_expected(0) = -std::cos(x_orig(0)) * x_orig(2);
  x_trig_dot_expected(1) = std::sin(x_orig(0)) * x_orig(2);
  x_trig_dot_expected(2) = std::cos(x_orig(1)) * x_orig(3);
  x_trig_dot_expected(3) = -std::sin(x_orig(1)) * x_orig(3);
  x_trig_dot_expected(4) = xdot_orig(2);
  x_trig_dot_expected(5) = xdot_orig(3);
  EXPECT_TRUE(CompareMatrices(x_trig_dot, x_trig_dot_expected, 1E-12));

  Vector6<symbolic::Polynomial> f;
  Vector6<symbolic::Polynomial> G;
  symbolic::Polynomial d_poly;
  Vector6<symbolic::Variable> x_var;
  for (int i = 0; i < 6; ++i) {
    x_var(i) = symbolic::Variable("x" + std::to_string(i));
  }
  TrigPolyDynamics(p, x_var, &f, &G, &d_poly);

  symbolic::Environment env;
  env.insert(x_var, x_trig);
  EXPECT_NEAR(d_poly.Evaluate(env), d, 1E-12);
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(f(i).Evaluate(env) / d + G(i).Evaluate(env) / d * u,
                x_trig_dot(i), 1E-12);
  }
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
