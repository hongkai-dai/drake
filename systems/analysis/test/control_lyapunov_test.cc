#include "drake/systems/analysis/control_lyapunov.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"

namespace drake {
namespace systems {
namespace analysis {

class ControlLyapunovTest : public ::testing::Test {
 protected:
  const symbolic::Variable x0_{"x0"};
  const symbolic::Variable x1_{"x1"};
  const Vector2<symbolic::Variable> x_{x0_, x1_};
  const symbolic::Variables x_vars_{{x0_, x1_}};
  const symbolic::Variable a_{"a"};
  const symbolic::Variable b_{"b"};
};

TEST_F(ControlLyapunovTest, VdotCalculator) {
  const Vector2<symbolic::Polynomial> f_{
      2 * x0_, symbolic::Polynomial{a_ * x1_ * x0_, x_vars_}};
  Matrix2<symbolic::Polynomial> G_;
  G_ << symbolic::Polynomial{3.}, symbolic::Polynomial{a_, x_vars_},
      symbolic::Polynomial{x0_}, symbolic::Polynomial{b_ * x1_, x_vars_};
  const symbolic::Polynomial V1(x0_ * x0_ + 2 * x1_ * x1_);
  const VdotCalculator dut1(x_, V1, f_, G_);

  const Eigen::Vector2d u_val(2., 3.);
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, dut1.Calc(u_val),
               (Eigen::Matrix<symbolic::Polynomial, 1, 2>(2 * x0_, 4 * x1_) *
                (f_ + G_ * u_val))(0));

  const symbolic::Polynomial V2(2 * a_ * x0_ * x1_ * x1_, x_vars_);
  const VdotCalculator dut2(x_, V2, f_, G_);
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, dut2.Calc(u_val),
               (Eigen::Matrix<symbolic::Polynomial, 1, 2>(
                    symbolic::Polynomial(2 * a_ * x1_ * x1_, x_vars_),
                    symbolic::Polynomial(4 * a_ * x0_ * x1_, x_vars_)) *
                (f_ + G_ * u_val))(0));
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
