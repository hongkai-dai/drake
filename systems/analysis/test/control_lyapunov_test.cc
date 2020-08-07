#include "drake/systems/analysis/control_lyapunov.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

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

class SimpleLinearSystemTest : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SimpleLinearSystemTest)

  SimpleLinearSystemTest() {
    A_ << 1, 2, -1, 3;
    B_ << 1, 0.5, 0.5, 1;
    x_ << symbolic::Variable("x0"), symbolic::Variable("x1");
  }

 protected:
  Eigen::Matrix2d A_;
  Eigen::Matrix2d B_;
  Vector2<symbolic::Variable> x_;
};

void CheckMaximizeEpsGivenVBoxInputBoundResult(
    const MaximizeEpsGivenVBoxInputBound& dut,
    const solvers::MathematicalProgramResult& result,
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& f,
    const MatrixX<symbolic::Polynomial>& G,
    const VectorX<symbolic::Variable>& x, double tol) {
  ASSERT_TRUE(result.is_success());
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  const double eps_val = result.GetSolution(dut.eps());
  const int nu = G.cols();
  VectorX<symbolic::Polynomial> b_result(nu);
  for (int i = 0; i < nu; ++i) {
    b_result(i) = result.GetSolution(dut.b()(i));
  }

  EXPECT_TRUE(symbolic::test::PolynomialEqual((dVdx * f)(0) + eps_val * V,
                                              b_result.sum(), tol));
  std::vector<std::array<symbolic::Polynomial, 6>> lagrangians_result(nu);
  for (int i = 0; i < nu; ++i) {
    for (int j = 0; j < 6; ++j) {
      lagrangians_result[i][j] = result.GetSolution(dut.lagrangians()[i][j]);
    }
    const symbolic::Polynomial dVdx_times_Gi = (dVdx * G.col(i))(0);
    const symbolic::Polynomial p1 =
        (lagrangians_result[i][0] + 1) * (dVdx_times_Gi - b_result(i)) -
        lagrangians_result[i][2] * dVdx_times_Gi -
        lagrangians_result[i][4] * (1 - V);
    const VectorX<symbolic::Monomial>& monomials1 =
        dut.constraint_grams()[i][0].second;
    const Eigen::MatrixXd grams1 =
        result.GetSolution(dut.constraint_grams()[i][0].first);
    EXPECT_TRUE(symbolic::test::PolynomialEqual(
        p1, monomials1.dot(grams1 * monomials1), tol));
    const symbolic::Polynomial p2 =
        (lagrangians_result[i][1] + 1) * (-dVdx_times_Gi - b_result(i)) +
        lagrangians_result[i][3] * dVdx_times_Gi -
        lagrangians_result[i][5] * (1 - V);
    const VectorX<symbolic::Monomial>& monomials2 =
        dut.constraint_grams()[i][1].second;
    const Eigen::MatrixXd grams2 =
        result.GetSolution(dut.constraint_grams()[i][1].first);
    EXPECT_TRUE(symbolic::test::PolynomialEqual(
        p2, monomials2.dot(grams2 * monomials2), tol));

    // Now check if the gram matrices are psd.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(grams1);
    EXPECT_TRUE((es.eigenvalues().array() >= -tol).all());
    es.compute(grams2);
    EXPECT_TRUE((es.eigenvalues().array() >= -tol).all());
  }
}

// Sample many points inside the level set V(x) <= 1, and verify that min_u
// Vdot(x, u) (-1 <= u <= 1) is less than -eps * V.
void ValidateRegionOfAttractionBySample(const VectorX<symbolic::Polynomial>& f,
                                        const MatrixX<symbolic::Polynomial>& G,
                                        const symbolic::Polynomial& V,
                                        const VectorX<symbolic::Variable>& x,
                                        const Eigen::MatrixXd& u_vertices,
                                        double eps, int num_samples,
                                        double abs_tol, double rel_tol) {
  int sample_count = 0;
  const int nx = f.rows();
  const int nu = G.cols();
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  const symbolic::Polynomial dVdx_times_f = (dVdx * f)(0);
  const RowVectorX<symbolic::Polynomial> dVdx_times_G = dVdx * G;
  const int num_u_vertices = u_vertices.cols();
  while (sample_count < num_samples) {
    const Eigen::VectorXd x_val = Eigen::VectorXd::Random(nx);
    symbolic::Environment env;
    env.insert(x, x_val);
    const double V_val = V.Evaluate(env);
    EXPECT_GE(V_val, -abs_tol);
    if (V_val <= 1) {
      const double dVdx_times_f_val = dVdx_times_f.Evaluate(env);
      Eigen::RowVectorXd dVdx_times_G_val(nu);
      for (int i = 0; i < nu; ++i) {
        dVdx_times_G_val(i) = dVdx_times_G(i).Evaluate(env);
      }
      const double Vdot =
          (dVdx_times_f_val * Eigen::RowVectorXd::Ones(num_u_vertices) +
           dVdx_times_G_val * u_vertices)
              .array()
              .minCoeff();
      EXPECT_TRUE(-Vdot / V_val > eps - rel_tol ||
                  -Vdot > V_val * eps - abs_tol);

      sample_count++;
    }
  }
}

TEST_F(SimpleLinearSystemTest, MaximizeEpsGivenVBoxInputBound) {
  // We compute the LQR cost-to-go as the candidate Lyapunov function.
  const controllers::LinearQuadraticRegulatorResult lqr_result =
      controllers::LinearQuadraticRegulator(A_, B_, Eigen::Matrix2d::Identity(),
                                            Eigen::Matrix2d::Identity());

  const symbolic::Variables x_set{x_};
  // We multiply the LQR cost-to-go by a factor (10 here), so that we start with
  // a very small neighbourhood around the origin as the initial guess of ROA
  // V(x) <= 1
  const symbolic::Polynomial V(x_.dot(100 * lqr_result.S * x_), x_set);

  Vector2<symbolic::Polynomial> f;
  f[0] = symbolic::Polynomial(A_.row(0).dot(x_), x_set);
  f[1] = symbolic::Polynomial(A_.row(1).dot(x_), x_set);
  Matrix2<symbolic::Polynomial> G;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      G(i, j) = symbolic::Polynomial(B_(i, j));
    }
  }
  // Set l_[i][0] and l_[i][1] to 1 + x.dot(x)
  const int nu = 2;
  std::vector<std::array<symbolic::Polynomial, 2>> l_given{nu};
  for (int i = 0; i < nu; ++i) {
    l_given[i][0] =
        symbolic::Polynomial(1 + x_.cast<symbolic::Expression>().dot(x_));
    l_given[i][1] =
        symbolic::Polynomial(1 + x_.cast<symbolic::Expression>().dot(x_));
  }
  std::vector<std::array<int, 6>> lagrangian_degrees{nu};
  for (int i = 0; i < nu; ++i) {
    lagrangian_degrees[i][0] = 2;
    lagrangian_degrees[i][1] = 2;
    for (int j = 2; j < 6; ++j) {
      lagrangian_degrees[i][j] = 2;
    }
  }

  std::vector<int> b_degrees(nu, 2);

  MaximizeEpsGivenVBoxInputBound dut(V, f, G, l_given, lagrangian_degrees,
                                     b_degrees, x_);

  solvers::MosekSolver mosek_solver;
  solvers::CsdpSolver csdp_solver;
  mosek_solver.set_stream_logging(true, "");
  const auto result = csdp_solver.Solve(dut.prog());
  EXPECT_TRUE(result.is_success());
  CheckMaximizeEpsGivenVBoxInputBoundResult(dut, result, V, f, G, x_, 1.3E-5);

  const double eps = 3;
  Eigen::Matrix<double, 2, 4> u_vertices;
  u_vertices << 1, 1, -1, -1, 1, -1, 1, -1;
  ValidateRegionOfAttractionBySample(f, G, V, x_, u_vertices, eps, 100, 1E-5,
                                     1E-3);
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
