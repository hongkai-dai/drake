#include "drake/systems/analysis/control_barrier.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sdpa_free_format.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

class SimpleLinearSystemTest : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimpleLinearSystemTest)

  SimpleLinearSystemTest() {
    A_ << 1, 2, -1, 3;
    B_ << 1, 0.5, 0.5, 1;
    x_ << symbolic::Variable("x0"), symbolic::Variable("x1");
    x_set_ = symbolic::Variables(x_);

    // clang-format off
    u_vertices_ << 1, 1, -1, -1,
                   1, -1, 1, -1;
    // clang-format on
    symbolic::Variables x_set(x_);
    f_[0] = symbolic::Polynomial(A_.row(0).dot(x_), x_set);
    f_[1] = symbolic::Polynomial(A_.row(1).dot(x_), x_set);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        G_(i, j) = symbolic::Polynomial(B_(i, j));
      }
    }
  }

 protected:
  Eigen::Matrix2d A_;
  Eigen::Matrix2d B_;
  Vector2<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  Eigen::Matrix<double, 2, 4> u_vertices_;
  Vector2<symbolic::Polynomial> f_;
  Matrix2<symbolic::Polynomial> G_;
};

TEST_F(SimpleLinearSystemTest, SearchControlBarrier) {
  // Test SearchControlBarrier
  Eigen::Matrix<double, 2, 5> candidate_safe_states;
  // clang-format off
  candidate_safe_states << 0.1, 0.1, -0.1, -0.1, 0,
                           0.1, -0.1, 0.1, -0.1, 0;
  // clang-format on
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions;
  // The unsafe region is -2 <= x(0) <= -1
  unsafe_regions.push_back(Vector2<symbolic::Polynomial>(
      symbolic::Polynomial(x_(0) + 1), symbolic::Polynomial(-x_(0) - 2)));
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 1, 1, -1, -1,
                1, -1, 1, -1;
  // clang-format on
  u_vertices *= 10;
  const SearchControlBarrier dut(f_, G_, x_, candidate_safe_states,
                                 unsafe_regions, u_vertices);

  const symbolic::Polynomial h_init(1 - x_(0) * x_(0) - x_(1) * x_(1));
  const double deriv_eps = 0.1;
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  symbolic::Polynomial lambda0;
  MatrixX<symbolic::Variable> lambda0_gram;
  VectorX<symbolic::Polynomial> l;
  std::vector<MatrixX<symbolic::Variable>> l_grams;
  symbolic::Polynomial hdot_sos;
  VectorX<symbolic::Monomial> hdot_monomials;
  MatrixX<symbolic::Variable> hdot_gram;
  auto prog_lagrangian = dut.ConstructLagrangianProgram(
      h_init, deriv_eps, lambda0_degree, l_degrees, &lambda0, &lambda0_gram, &l,
      &l_grams, &hdot_sos, &hdot_monomials, &hdot_gram);
  auto result_lagrangian = solvers::Solve(*prog_lagrangian);
  ASSERT_TRUE(result_lagrangian.is_success());
  const Eigen::MatrixXd lambda0_gram_sol =
      result_lagrangian.GetSolution(lambda0_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(lambda0_gram_sol));
  const symbolic::Polynomial lambda0_sol =
      result_lagrangian.GetSolution(lambda0);
  symbolic::Polynomial hdot_sos_expected = (1 + lambda0_sol) * (-1 - h_init);
  std::vector<symbolic::Polynomial> l_sol(u_vertices.cols());
  RowVectorX<symbolic::Polynomial> dhdx = h_init.Jacobian(x_);
  for (int i = 0; i < u_vertices.cols(); ++i) {
    EXPECT_TRUE(
        math::IsPositiveDefinite(result_lagrangian.GetSolution(l_grams[i])));
    l_sol[i] = result_lagrangian.GetSolution(l[i]);
    hdot_sos_expected -= l_sol[i] * (-deriv_eps * h_init - dhdx.dot(f_) -
                                     dhdx.dot(G_ * u_vertices.col(i)));
  }
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               result_lagrangian.GetSolution(hdot_sos).Expand(),
               hdot_sos_expected.Expand(), 1E-5);
  const Eigen::MatrixXd hdot_gram_sol =
      result_lagrangian.GetSolution(hdot_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(hdot_gram_sol));
  EXPECT_PRED3(symbolic::test::PolynomialEqual, hdot_sos_expected.Expand(),
               hdot_monomials.dot(hdot_gram_sol * hdot_monomials), 1E-5);
}

TEST_F(SimpleLinearSystemTest, ConstructLagrangianAndBProgram) {
  Eigen::Matrix<double, 2, 5> candidate_safe_states;
  // clang-format off
  candidate_safe_states << 0.1, 0.1, -0.1, -0.1, 0,
                           0.1, -0.1, 0.1, -0.1, 0;
  // clang-format on
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions;
  unsafe_regions.push_back(
      Vector1<symbolic::Polynomial>(symbolic::Polynomial(x_(0) + 1)));
  const ControlBarrierBoxInputBound dut(f_, G_, x_, candidate_safe_states,
                                        unsafe_regions);

  const symbolic::Polynomial h_init(x_(0) + 0.5);
  const int nu = 2;
  std::vector<std::vector<symbolic::Polynomial>> l_given(nu);
  const int num_hdot_sos = 2;
  for (int i = 0; i < nu; ++i) {
    l_given[i].resize(num_hdot_sos);
    for (int j = 0; j < num_hdot_sos; ++j) {
      l_given[i][j] = symbolic::Polynomial();
    }
  }
  std::vector<std::vector<std::array<int, 2>>> lagrangian_degrees(nu);
  for (int i = 0; i < nu; ++i) {
    lagrangian_degrees[i].resize(num_hdot_sos);
    for (int j = 0; j < num_hdot_sos; ++j) {
      lagrangian_degrees[i][j][0] = 0;
      lagrangian_degrees[i][j][1] = 2;
    }
  }
  std::vector<int> b_degrees(2, 2);

  std::vector<std::vector<std::array<symbolic::Polynomial, 2>>> lagrangians;
  std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 2>>>
      lagrangian_grams;
  VectorX<symbolic::Polynomial> b;
  symbolic::Variable deriv_eps;
  ControlBarrierBoxInputBound::HdotSosConstraintReturn hdot_sos_constraint(nu);

  auto prog = dut.ConstructLagrangianAndBProgram(
      h_init, l_given, lagrangian_degrees, b_degrees, &lagrangians,
      &lagrangian_grams, &b, &deriv_eps, &hdot_sos_constraint);
  const auto result = solvers::Solve(*prog);
  ASSERT_TRUE(result.is_success());
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main(int argc, char** argv) {
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
