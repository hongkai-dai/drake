#include "drake/systems/analysis/test/quadrotor.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/examples/quadrotor/quadrotor_plant.h"
#include "drake/math/cross_product.h"

namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(QuadrotorTrigPlant, TestDynamics) {
  QuadrotorTrigPlant<double> dut;
  examples::quadrotor::QuadrotorPlant<double> quadrotor_rpy;

  Eigen::Matrix<double, 12, 1> x_original;
  x_original << 0.5, 2.1, 0.4, 0.6, -0.4, 1.3, 0.5, 1.2, -0.1, 0.4, 0.5, -0.2;
  const Eigen::Matrix<double, 13, 1> x_trig = ToTrigState<double>(x_original);
  const Eigen::Vector4d u(0.4, 1.5, 0.9, 1.8);

  auto context = dut.CreateDefaultContext();
  context->SetContinuousState(x_trig);
  dut.get_input_port().FixValue(context.get(), u);
  const auto& derivatives_trig = dut.EvalTimeDerivatives(*context);
  auto context_rpy = quadrotor_rpy.CreateDefaultContext();
  context_rpy->SetContinuousState(x_original);
  quadrotor_rpy.get_input_port().FixValue(context_rpy.get(), u);
  const auto& derivatives_rpy = quadrotor_rpy.EvalTimeDerivatives(*context_rpy);
  const Eigen::Matrix<double, 13, 1> xdot_trig =
      derivatives_trig.CopyToVector();
  const Eigen::Matrix<double, 12, 1> xdot_rpy = derivatives_rpy.CopyToVector();
  unused(xdot_rpy);
  const math::RollPitchYaw<double> rpy(x_original.segment<3>(3));
  const Eigen::Vector3d rpyDt = x_original.tail<3>();
  const Eigen::Vector3d omega_WB_B =
      rpy.CalcAngularVelocityInChildFromRpyDt(rpyDt);
  const Eigen::Quaterniond quat = rpy.ToQuaternion();
  Eigen::Matrix4d Omega = Eigen::Matrix4d::Zero();
  Omega.block<1, 3>(0, 1) = -omega_WB_B;
  Omega.block<3, 1>(1, 0) = omega_WB_B;
  Omega.block<3, 3>(1, 1) = -math::VectorToSkewSymmetric(omega_WB_B);
  const Eigen::Vector4d quatDt_expected =
      0.5 * Omega * Eigen::Vector4d(quat.w(), quat.x(), quat.y(), quat.z());
  // quatDt
  EXPECT_TRUE(CompareMatrices(xdot_trig.head<4>(), quatDt_expected, 1e-10));
  // v_WB
  EXPECT_TRUE(
      CompareMatrices(xdot_trig.segment<3>(4), x_original.segment<3>(6)));
  // a_WB
  EXPECT_TRUE(
      CompareMatrices(xdot_trig.segment<3>(7), xdot_rpy.segment<3>(6), 1E-10));
  // alpha_WB_B
  EXPECT_TRUE(CompareMatrices(
      rpy.CalcRpyDDtFromAngularAccelInChild(rpyDt, xdot_trig.tail<3>()),
      xdot_rpy.tail<3>(), 1E-10));
}

GTEST_TEST(QuadrotorTrigPlant, TrigPolyDynamics) {
  QuadrotorTrigPlant<double> dut;
  Eigen::Matrix<symbolic::Variable, 13, 1> x;
  for (int i = 0; i < 13; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  Eigen::Matrix<symbolic::Polynomial, 13, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 13, 4> G;
  TrigPolyDynamics(dut, x, &f, &G);

  Eigen::Matrix<double, 12, 1> x_orig;
  x_orig << 0.5, 0.3, 1.2, -0.4, 0.6, -0.8, 1.2, -0.3, 1.5, -0.4, 1.2, 0.7;
  const auto x_trig = ToTrigState<double>(x_orig);
  const Eigen::Vector4d u(0.5, 0.9, 1.2, 0.3);
  auto context = dut.CreateDefaultContext();
  context->SetContinuousState(x_trig);
  dut.get_input_port().FixValue(context.get(), u);
  const auto& derivatives = dut.EvalTimeDerivatives(*context);
  const auto xdot = derivatives.CopyToVector();
  symbolic::Environment env;
  env.insert(x, x_trig);
  Eigen::Matrix<double, 13, 1> f_val;
  Eigen::Matrix<double, 13, 4> G_val;
  for (int i = 0; i < 13; ++i) {
    f_val(i) = f(i).Evaluate(env);
    for (int j = 0; j < 4; ++j) {
      G_val(i, j) = G(i, j).Evaluate(env);
    }
  }
  EXPECT_TRUE(CompareMatrices(xdot, f_val + G_val * u, 1E-9));
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
