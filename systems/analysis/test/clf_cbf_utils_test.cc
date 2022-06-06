#include "drake/systems/analysis/clf_cbf_utils.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace systems {
namespace analysis {

// Check if the ellipsoid {x | (x-x*)ᵀS(x-x*) <= ρ} in the
// sub-level set {x | f(x) <= 0}
void CheckEllipsoidInSublevelSet(
    const Eigen::Ref<const VectorX<symbolic::Variable>> x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, double rho,
    const symbolic::Polynomial& f) {
  // Check if any point within the ellipsoid also satisfies V(x)<=1.
  // A point on the boundary of ellipsoid (x−x*)ᵀS(x−x*)=ρ
  // can be writeen as x=√ρ*L⁻ᵀ*u+x*
  // where L is the Cholesky decomposition of S, u is a vector with norm < 1.
  Eigen::LLT<Eigen::Matrix2d> llt_solver;
  llt_solver.compute(S);
  const int x_dim = x.rows();
  srand(0);
  Eigen::MatrixXd u_samples = Eigen::MatrixXd::Random(x_dim, 1000);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0., 1.);
  for (int i = 0; i < u_samples.cols(); ++i) {
    u_samples.col(i) /= u_samples.col(i).norm();
    u_samples.col(i) *= distribution(generator);
  }

  Eigen::ColPivHouseholderQR<Eigen::Matrix2d> qr_solver;
  qr_solver.compute(llt_solver.matrixL().transpose());
  for (int i = 0; i < u_samples.cols(); ++i) {
    const Eigen::VectorXd x_val =
        std::sqrt(rho) * qr_solver.solve(u_samples.col(i)) + x_star;
    symbolic::Environment env;
    env.insert(x, x_val);
    EXPECT_LE(f.Evaluate(env), 1E-5);
  }
}

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

GTEST_TEST(MaximizeInnerEllipsoidRho, Test1) {
  // Test a 2D case with known solution.
  // Find the largest x²+4y² <= ρ within the circle 2x²+2y² <= 1
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  const Eigen::Vector2d x_star(0, 0);
  Eigen::Matrix2d S;
  // clang-format off
  S << 1, 0,
       0, 4;
  // clang-format on
  const symbolic::Polynomial V(2 * x(0) * x(0) + 2 * x(1) * x(1));
  const symbolic::Polynomial t(x(0) * x(0) + x(1) * x(1));
  const int s_degree(2);
  const double backoff_scale = 0.;
  double rho_sol;
  symbolic::Polynomial s_sol;
  MaximizeInnerEllipsoidRho(x, x_star, S, V - 1, t, s_degree,
                            solvers::MosekSolver::id(), std::nullopt,
                            backoff_scale, &rho_sol, &s_sol);
  const double tol = 1E-5;
  EXPECT_NEAR(rho_sol, 0.5, tol);

  CheckEllipsoidInSublevelSet(x, x_star, S, rho_sol, V - 1);
}

GTEST_TEST(MaximizeInnerEllipsoidRho, Test2) {
  // Test a case that I cannot compute the solution analytically.
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  const Eigen::Vector2d x_star(1, 2);
  Eigen::Matrix2d S;
  // clang-format off
  S << 1, 2,
       2, 9;
  // clang-format on
  using std::pow;
  const symbolic::Polynomial V(pow(x(0), 4) + pow(x(1), 4) - 2 * x(0) * x(0) -
                               4 * x(1) * x(1) - 20 * x(0) * x(1));
  ASSERT_LE(
      V.Evaluate(symbolic::Environment({{x(0), x_star(0)}, {x(1), x_star(1)}})),
      1);
  {
    // Test the program
    // max ρ
    // s.t (1+t(x))((x-x*)ᵀS(x-x*)-ρ) - s(x)(V(x)-1) is sos
    //     s(x) is sos
    const symbolic::Polynomial t(0);
    const int s_degree = 2;
    const double backoff_scale = 0.05;
    double rho_sol;
    symbolic::Polynomial s_sol;
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
    // I am really surprised that this SOS finds a solution with rho > 0. AFAIK,
    // t(x) is constant, hence (1+t(x))((x-x*)ᵀS(x-x*)-ρ) is a degree 2
    // polynomial, while -s(x)*(V(x)-1) has much higher degree (>6) with
    // negative leading terms. The resulting polynomial cannot be sos.
    MaximizeInnerEllipsoidRho(x, x_star, S, V - 1, t, s_degree,
                              solvers::MosekSolver::id(), solver_options,
                              backoff_scale, &rho_sol, &s_sol);
    CheckEllipsoidInSublevelSet(x, x_star, S, rho_sol, V - 1);
  }

  {
    // Test the bisection approach
    const int r_degree = 2;
    const double rho_max = 1;
    const double rho_min = 0.2;
    const double rho_tol = 0.1;
    double rho_sol;
    symbolic::Polynomial r_sol;
    MaximizeInnerEllipsoidRho(x, x_star, S, V - 1, r_degree, rho_max, rho_min,
                              solvers::MosekSolver::id(), std::nullopt, rho_tol,
                              &rho_sol, &r_sol);
    CheckEllipsoidInSublevelSet(x, x_star, S, rho_sol, V - 1);
    // Check if r_sol is sos.
    Eigen::Matrix2Xd x_check = Eigen::Matrix2Xd::Random(2, 100);
    const symbolic::Polynomial sos_cond =
        1 - V + r_sol * internal::EllipsoidPolynomial(x, x_star, S, rho_sol);
    for (int i = 0; i < x_check.cols(); ++i) {
      symbolic::Environment env;
      env.insert(x, x_check.col(i));
      EXPECT_GE(r_sol.Evaluate(env), 0.);
      EXPECT_GE(sos_cond.Evaluate(env), 0.);
    }
  }
}

GTEST_TEST(FindCandidateLyapunov, Test) {
  // Find the candidate Lyapunov for a stable linear system xdot = A*x
  Eigen::Matrix2d A;
  A << -1, 2, 0, -3;
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  Eigen::Matrix<double, 2, 5> x_val;
  // clang-format off
  x_val << 0.2, 1, -2, 0.5, 1,
           0.3, -0.5, 1.2, -0.4, 2;
  // clang-format on
  const Eigen::Matrix<double, 2, 5> xdot_val = A * x_val;

  symbolic::Polynomial V;
  MatrixX<symbolic::Expression> V_gram;
  const int V_degree = 2;
  auto prog = FindCandidateLyapunov(x, V_degree, x_val, xdot_val, &V, &V_gram);
  const auto result = solvers::Solve(*prog);
  ASSERT_TRUE(result.is_success());
  EXPECT_EQ(V_gram.rows(), 2);
  const auto V_sol = result.GetSolution(V);
  Eigen::Matrix2d V_gram_sol;
  for (int i = 0; i < V_gram.rows(); ++i) {
    for (int j = 0; j < V_gram.cols(); ++j) {
      const symbolic::Expression V_gram_ij = result.GetSolution(V_gram(i, j));
      V_gram_sol(i, j) = symbolic::get_constant_value(V_gram_ij);
    }
  }
  EXPECT_TRUE(math::IsPositiveDefinite(V_gram_sol));
  EXPECT_TRUE((V_sol.EvaluateIndeterminates(x, x_val).array() >= 0).all());
  const RowVector2<symbolic::Polynomial> dVdx = V_sol.Jacobian(x);
  double cost_expected = 0;
  for (int i = 0; i < x_val.cols(); ++i) {
    symbolic::Environment env;
    env.insert(x, x_val.col(i));
    Eigen::RowVector2d dVdx_val;
    for (int j = 0; j < x.rows(); ++j) {
      dVdx_val(j) = dVdx(j).EvaluateIndeterminates(x, x_val.col(i))(0);
    }
    EXPECT_LE(dVdx_val.dot(xdot_val.col(i)), 0);
    cost_expected += dVdx_val.dot(xdot_val.col(i));
  }
  EXPECT_NEAR(result.get_optimal_cost(), cost_expected, 1E-5);
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
int main(int argc, char** argv) {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
