#include "drake/systems/analysis/clf_cbf_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"

namespace drake {
namespace systems {
namespace analysis {
GTEST_TEST(EvaluatePolynomial, Test1) {
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  symbolic::Variable a("a");
  symbolic::Variable b("b");
  symbolic::Variable c("c");
  const symbolic::Polynomial p{
      {{symbolic::Monomial(), a},
       {symbolic::Monomial(x(0), 2), b},
       {symbolic::Monomial({{x(0), 1}, {x(1), 2}}), c}}};

  Eigen::Matrix<double, 2, 3> x_vals;
  x_vals << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd coeff_mat;
  VectorX<symbolic::Variable> v;
  EvaluatePolynomial(p, x, x_vals, &coeff_mat, &v);
  EXPECT_EQ(coeff_mat.rows(), x_vals.cols());
  EXPECT_EQ(coeff_mat.cols(), p.monomial_to_coefficient_map().size());
  for (int i = 0; i < x_vals.cols(); ++i) {
    symbolic::Environment env;
    env.insert(x, x_vals.col(i));
    EXPECT_PRED2(symbolic::test::ExprEqual,
                 p.EvaluatePartial(env).ToExpression(),
                 coeff_mat.row(i).dot(v).Expand());
  }
}

GTEST_TEST(EvaluatePolynomial, Test2) {
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  symbolic::Variable a("a");
  symbolic::Variable b("b");
  symbolic::Variable c("c");
  const symbolic::Polynomial p{
      {{symbolic::Monomial(), a + 1},
       {symbolic::Monomial(x(0), 2), b + c},
       {symbolic::Monomial({{x(0), 1}, {x(1), 2}}), 2 * b - a + 1}}};

  Eigen::Matrix<double, 2, 5> x_vals;
  // clang-format off
  x_vals << 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10;
  // clang-format on
  VectorX<symbolic::Expression> p_vals;
  EvaluatePolynomial(p, x, x_vals, &p_vals);
  EXPECT_EQ(p_vals.rows(), x_vals.cols());
  for (int i = 0; i < p_vals.rows(); ++i) {
    symbolic::Environment env;
    env.insert(x, x_vals.col(i));
    EXPECT_PRED2(symbolic::test::ExprEqual, p_vals(i).Expand(),
                 p.EvaluatePartial(env).ToExpression().Expand());
  }
}

GTEST_TEST(SplitCandidateStates, Test) {
  symbolic::Variable x0("x0");
  symbolic::Variable x1("x1");
  Vector2<symbolic::Variable> x(x0, x1);

  const symbolic::Polynomial p(1 + 2 * x0 * x1 + 2 * x1 * x1 - 3 * x0 * x0);
  Eigen::Matrix<double, 2, 7> x_vals;
  // clang-format off
  x_vals << 1, 0, -1, 2, 1, 2, -1,
            0, 1,  2, 1, 2, 2, -1;
  // clang-format on
  Eigen::MatrixXd positive_states;
  Eigen::MatrixXd negative_states;
  SplitCandidateStates(p, x, x_vals, &positive_states, &negative_states);
  EXPECT_EQ(positive_states.cols() + negative_states.cols(), x_vals.cols());
  EXPECT_TRUE(
      (p.EvaluateIndeterminates(x, positive_states).array() >= 0).all());
  EXPECT_TRUE((p.EvaluateIndeterminates(x, negative_states).array() < 0).all());
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
