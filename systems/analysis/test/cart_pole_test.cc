#include "drake/systems/analysis/test/cart_pole.h"

#include <eigen3/Eigen/src/Core/Matrix.h>
#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"

namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(CartPole, Test) {
  multibody::MultibodyPlant<double> plant(0.);
  multibody::Parser(&plant).AddModelFromFile(
      FindResourceOrThrow("drake/examples/multibody/cart_pole/cart_pole.sdf"));
  plant.Finalize();
  auto context = plant.CreateDefaultContext();
  const Eigen::Vector4d x_val(0.5, 2.1, 0.4, -1.5);
  double u_val = 2;
  plant.get_actuation_input_port().FixValue(context.get(), Vector1d(u_val));
  context->SetContinuousState(x_val);

  // Test mass matrix.
  Eigen::Matrix2d M_expected;
  plant.CalcMassMatrix(*context, &M_expected);

  const Eigen::Matrix<double, 5, 1> x_trig = ToTrigState<double>(x_val);
  const CartPoleParams params{};
  const Eigen::Matrix2d M = MassMatrix<double>(params, x_trig);
  EXPECT_TRUE(CompareMatrices(M, M_expected, 1E-12));

  // Test bias
  Eigen::Vector2d bias_expected;
  plant.CalcBiasTerm(*context, &bias_expected);
  const Eigen::Vector2d bias = CalcBiasTerm<double>(params, x_trig);
  EXPECT_TRUE(CompareMatrices(bias, bias_expected, 1E-12));

  // Test gravity vector
  const Eigen::Vector2d tau_g_expected =
      plant.CalcGravityGeneralizedForces(*context);
  const Eigen::Vector2d tau_g = CalcGravityVector<double>(params, x_trig);
  EXPECT_TRUE(CompareMatrices(tau_g, tau_g_expected, 1E-12));

  // Test TrigDynamics
  const Eigen::Vector4d xdot_orig =
      plant.EvalTimeDerivatives(*context).CopyToVector();
  Eigen::Matrix<double, 5, 1> x_trig_dot_expected;
  x_trig_dot_expected << xdot_orig(0), std::cos(x_val(1)) * x_val(3),
      -std::sin(x_val(1)) * x_val(3), xdot_orig(2), xdot_orig(3);
  Eigen::Matrix<double, 5, 1> n;
  double d;
  TrigDynamics<double>(params, x_trig, u_val, &n, &d);
  EXPECT_TRUE(CompareMatrices(n / d, x_trig_dot_expected, 1E-12));

  // Test TrigPolyDynamics
  Eigen::Matrix<symbolic::Variable, 5, 1> x;
  for (int i = 0; i < 5; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 5, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 5, 1> G;
  symbolic::Polynomial d_poly;
  TrigPolyDynamics(params, x, &f, &G, &d_poly);
  symbolic::Environment env;
  env.insert(x, x_trig);
  EXPECT_NEAR(d_poly.Evaluate(env), d, 1E-12);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR((f(i) + G(i) * u_val).Evaluate(env), n(i), 1E-12);
  }

  const symbolic::Polynomial state_constraint = StateEqConstraint(x);
  EXPECT_EQ(state_constraint.Evaluate(env), 0);

  // Test CalcQdot
  EXPECT_TRUE(CompareMatrices(CalcQdot<double>(x_trig),
                              x_trig_dot_expected.head<3>(), 1E-12));
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
