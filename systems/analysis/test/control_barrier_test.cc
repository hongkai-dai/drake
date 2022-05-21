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
#include "drake/systems/analysis/clf_cbf_utils.h"

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

TEST_F(SimpleLinearSystemTest, ControlBarrier) {
  // Test ControlBarrier
  Eigen::Matrix<double, 2, 7> candidate_safe_states;
  // clang-format off
  candidate_safe_states << 0.1, 0.1, -0.1, -0.1, 0, 1, 0.5,
                           0.1, -0.1, 0.1, -0.1, 0, 0.5, -1;
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
  const ControlBarrier dut(f_, G_, x_, candidate_safe_states, unsafe_regions,
                           u_vertices);

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
  symbolic::Polynomial lambda0_sol = result_lagrangian.GetSolution(lambda0);
  symbolic::Polynomial hdot_sos_expected = (1 + lambda0_sol) * (-1 - h_init);
  VectorX<symbolic::Polynomial> l_sol(u_vertices.cols());
  RowVectorX<symbolic::Polynomial> dhdx = h_init.Jacobian(x_);
  for (int i = 0; i < u_vertices.cols(); ++i) {
    EXPECT_TRUE(
        math::IsPositiveDefinite(result_lagrangian.GetSolution(l_grams[i])));
    l_sol(i) = result_lagrangian.GetSolution(l[i]);
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

  const int t_degree = 0;
  const std::vector<int> s_degrees = {0, 0};
  symbolic::Polynomial t;
  MatrixX<symbolic::Variable> t_gram;
  std::vector<VectorX<symbolic::Polynomial>> s(1);
  std::vector<std::vector<MatrixX<symbolic::Variable>>> s_grams(1);
  symbolic::Polynomial unsafe_sos_poly;
  MatrixX<symbolic::Variable> unsafe_sos_poly_gram;
  auto prog_unsafe = dut.ConstructUnsafeRegionProgram(
      h_init, 0, t_degree, s_degrees, &t, &t_gram, &(s[0]), &(s_grams[0]),
      &unsafe_sos_poly, &unsafe_sos_poly_gram);
  const auto result_unsafe = solvers::Solve(*prog_unsafe);
  ASSERT_TRUE(result_unsafe.is_success());
  const symbolic::Polynomial t_sol = result_unsafe.GetSolution(t);
  EXPECT_TRUE(math::IsPositiveDefinite(result_unsafe.GetSolution(t_gram)));
  VectorX<symbolic::Polynomial> s_sol(s[0].rows());
  EXPECT_EQ(s[0].rows(), unsafe_regions[0].rows());
  for (int i = 0; i < s[0].rows(); ++i) {
    s_sol(i) = result_unsafe.GetSolution(s[0](i));
    EXPECT_TRUE(
        math::IsPositiveDefinite(result_unsafe.GetSolution(s_grams[0][i])));
  }
  symbolic::Polynomial unsafe_sos_poly_expected =
      (1 + t_sol) * -h_init + s_sol.dot(unsafe_regions[0]);
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               unsafe_sos_poly_expected.Expand(),
               result_unsafe.GetSolution(unsafe_sos_poly).Expand(), 1E-5);
  EXPECT_TRUE(math::IsPositiveDefinite(
      result_unsafe.GetSolution(unsafe_sos_poly_gram)));

  // Now search for barrier given Lagrangians.
  Eigen::MatrixXd verified_safe_states;
  Eigen::MatrixXd unverified_candidate_states;
  SplitCandidateStates(h_init, x_, candidate_safe_states, &verified_safe_states,
                       &unverified_candidate_states);

  const int h_degree = 2;
  symbolic::Polynomial h;
  std::vector<symbolic::Polynomial> unsafe_sos_polys;
  std::vector<MatrixX<symbolic::Variable>> unsafe_sos_poly_grams;
  lambda0_sol = lambda0_sol.RemoveTermsWithSmallCoefficients(1e-10);
  const double eps = 1E-3;
  auto prog_barrier = dut.ConstructBarrierProgram(
      lambda0_sol, l_sol, {t_sol}, h_degree, deriv_eps, {s_degrees},
      verified_safe_states, unverified_candidate_states, eps, &h, &hdot_sos,
      &hdot_gram, &s, &s_grams, &unsafe_sos_polys, &unsafe_sos_poly_grams);
  RemoveTinyCoeff(prog_barrier.get(), 1E-10);
  const auto result_barrier = solvers::Solve(*prog_barrier);
  ASSERT_TRUE(result_barrier.is_success());
  const auto h_sol = result_barrier.GetSolution(h);
  // Check h_sol on verified_safe_states;
  EXPECT_TRUE(
      (h_sol.EvaluateIndeterminates(x_, verified_safe_states).array() >= 0)
          .all());
  // Check cost.
  const auto h_unverified_vals =
      h_sol.EvaluateIndeterminates(x_, unverified_candidate_states);
  EXPECT_NEAR(
      -result_barrier.get_optimal_cost(),
      (h_unverified_vals.array() >= eps)
          .select(Eigen::VectorXd::Constant(h_unverified_vals.rows(), eps),
                  h_unverified_vals)
          .sum(),
      1E-5);
  // Check sos for unsafe regions.
  EXPECT_EQ(s.size(), unsafe_regions.size());
  for (int i = 0; i < static_cast<int>(s.size()); ++i) {
    EXPECT_EQ(s[i].rows(), unsafe_regions[i].rows());
    s_sol.resize(s[i].rows());
    for (int j = 0; j < s[i].rows(); ++j) {
      EXPECT_TRUE(
          math::IsPositiveDefinite(result_barrier.GetSolution(s_grams[i][j])));
      s_sol[j] = result_barrier.GetSolution(s[i](j));
    }
    unsafe_sos_poly_expected =
        (1 + t_sol) * -h_sol + s_sol.dot(unsafe_regions[i]);
    EXPECT_PRED3(
        symbolic::test::PolynomialEqual, unsafe_sos_poly_expected.Expand(),
        result_barrier.GetSolution(unsafe_sos_polys[i]).Expand(), 1E-5);
    EXPECT_TRUE(math::IsPositiveDefinite(
        result_barrier.GetSolution(unsafe_sos_poly_grams[i])));
  }
}

TEST_F(SimpleLinearSystemTest, ControlBarrierSearch) {
  // Test ControlBarrier::Search function.
  Eigen::Matrix<double, 2, 12> candidate_safe_states;
  // clang-format off
  candidate_safe_states << 0.1, 0.1, -0.1, -0.1, 0, 1, 0.5, 0.2, 1., 0.2, 0.4, 0.8,
                           0.1, -0.1, 0.1, -0.1, 0, 0.5, -1, 0.5, -0.1, 1, 1, -1;
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
  const ControlBarrier dut(f_, G_, x_, candidate_safe_states, unsafe_regions,
                           u_vertices);

  const symbolic::Polynomial h_init(1 - x_(0) * x_(0) - x_(1) * x_(1));
  const int h_degree = 2;
  const double deriv_eps = 0.1;
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> t_degree = {0};
  const std::vector<std::vector<int>> s_degrees = {{0, 0}};

  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lsol_tiny_coeff_tol = 1E-8;

  symbolic::Polynomial h_sol;
  symbolic::Polynomial lambda0_sol;
  VectorX<symbolic::Polynomial> l_sol;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  Eigen::MatrixXd verified_safe_states;
  Eigen::MatrixXd unverified_candidate_states;

  dut.Search(h_init, h_degree, deriv_eps, lambda0_degree, l_degrees, t_degree,
             s_degrees, search_options, &h_sol, &lambda0_sol, &l_sol, &t_sol,
             &s_sol, &verified_safe_states, &unverified_candidate_states);
  std::cout << "verified safe states:\n" << verified_safe_states << "\n";
  std::cout << "unverified candidate states:\n"
            << unverified_candidate_states << "\n";
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
