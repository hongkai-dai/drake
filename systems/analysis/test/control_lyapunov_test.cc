#include "drake/systems/analysis/control_lyapunov.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/symbolic_monomial_util.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

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
    x_set_ = symbolic::Variables(x_);
  }

  void InitializeWithLQR(
      symbolic::Polynomial* V, Vector2<symbolic::Polynomial>* f,
      Matrix2<symbolic::Polynomial>* G,
      std::vector<std::array<symbolic::Polynomial, 2>>* l_given,
      std::vector<std::array<int, 6>>* lagrangian_degrees) {
    // We first compute LQR cost-to-go as the candidate Lyapunov function.
    const controllers::LinearQuadraticRegulatorResult lqr_result =
        controllers::LinearQuadraticRegulator(
            A_, B_, Eigen::Matrix2d::Identity(), Eigen::Matrix2d::Identity());

    const symbolic::Variables x_set{x_};
    // We multiply the LQR cost-to-go by a factor (100 here), so that we start
    // with a very small neighbourhood around the origin as the initial guess of
    // ROA V(x) <= 1
    *V = symbolic::Polynomial(x_.dot(100 * lqr_result.S * x_), x_set);

    (*f)[0] = symbolic::Polynomial(A_.row(0).dot(x_), x_set);
    (*f)[1] = symbolic::Polynomial(A_.row(1).dot(x_), x_set);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        (*G)(i, j) = symbolic::Polynomial(B_(i, j));
      }
    }
    // Set l_[i][0] and l_[i][1] to 1 + x.dot(x)
    const int nu = 2;
    l_given->resize(nu);
    for (int i = 0; i < nu; ++i) {
      (*l_given)[i][0] =
          symbolic::Polynomial(1 + x_.cast<symbolic::Expression>().dot(x_));
      (*l_given)[i][1] =
          symbolic::Polynomial(1 + x_.cast<symbolic::Expression>().dot(x_));
    }
    lagrangian_degrees->resize(nu);
    for (int i = 0; i < nu; ++i) {
      (*lagrangian_degrees)[i][0] = 2;
      (*lagrangian_degrees)[i][1] = 2;
      for (int j = 2; j < 6; ++j) {
        (*lagrangian_degrees)[i][j] = 2;
      }
    }
  }

 protected:
  Eigen::Matrix2d A_;
  Eigen::Matrix2d B_;
  Vector2<symbolic::Variable> x_;
  symbolic::Variables x_set_;
};

void CheckSearchLagrangianAndBResult(
    const SearchLagrangianAndBGivenVBoxInputBound& dut,
    const solvers::MathematicalProgramResult& result,
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& f,
    const MatrixX<symbolic::Polynomial>& G,
    const VectorX<symbolic::Variable>& x, double tol) {
  ASSERT_TRUE(result.is_success());
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  const double deriv_eps_val = result.GetSolution(dut.deriv_eps());
  const int nu = G.cols();
  VectorX<symbolic::Polynomial> b_result(nu);
  for (int i = 0; i < nu; ++i) {
    b_result(i) = result.GetSolution(dut.b()(i));
  }

  EXPECT_TRUE(symbolic::test::PolynomialEqual((dVdx * f)(0) + deriv_eps_val * V,
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
    const std::array<symbolic::Polynomial, 2> p_expected =
        dut.vdot_sos_constraint().ComputeSosConstraint(i, result);
    EXPECT_TRUE(symbolic::test::PolynomialEqual(p1, p_expected[0], tol));
    const symbolic::Polynomial p2 =
        (lagrangians_result[i][1] + 1) * (-dVdx_times_Gi - b_result(i)) +
        lagrangians_result[i][3] * dVdx_times_Gi -
        lagrangians_result[i][5] * (1 - V);
    EXPECT_TRUE(symbolic::test::PolynomialEqual(p2, p_expected[1], tol));

    // Now check if the gram matrices are psd.
    const Eigen::MatrixXd grams1 =
        result.GetSolution(dut.vdot_sos_constraint().grams[i][0]);
    const Eigen::MatrixXd grams2 =
        result.GetSolution(dut.vdot_sos_constraint().grams[i][1]);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    es.compute(grams1);
    EXPECT_TRUE((es.eigenvalues().array() >= -tol).all());
    es.compute(grams2);
    EXPECT_TRUE((es.eigenvalues().array() >= -tol).all());
  }
}

void CheckSearchLagrangianResult(const SearchLagrangianGivenVBoxInputBound& dut,
                                 double tol, double psd_tol) {
  // Check if the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  // are satisfied.
  const RowVectorX<symbolic::Polynomial> dVdx = dut.V().Jacobian(dut.x());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_solver;
  for (int i = 0; i < dut.nu(); ++i) {
    std::array<symbolic::Polynomial, 6> l_sol;
    const symbolic::Polynomial dVdx_times_Gi = dVdx.dot(dut.G().col(i));
    for (int j = 0; j < 2; ++j) {
      // Surprisingly Mosek really doesn't like solving Lagrangian problem. I
      // often observe numerical errors.
      solvers::CsdpSolver csdp_solver;
      solvers::SolverOptions solver_options;
      const auto result_ij =
          csdp_solver.Solve(dut.prog(i, j), std::nullopt, solver_options);
      ASSERT_TRUE(result_ij.is_success());
      for (int k = 0; k < 3; ++k) {
        l_sol[j + 2 * k] =
            result_ij.GetSolution(dut.lagrangians()[i][j + 2 * k]);
      }
      const symbolic::Polynomial p =
          j == 0 ? (l_sol[0] + 1) * (dVdx_times_Gi - dut.b()(i)) -
                       l_sol[2] * dVdx_times_Gi - l_sol[4] * (1 - dut.V())
                 : (l_sol[1] + 1) * (-dVdx_times_Gi - dut.b()(i)) +
                       l_sol[3] * dVdx_times_Gi - l_sol[5] * (1 - dut.V());
      const symbolic::Polynomial p_expected =
          dut.vdot_sos_constraint().ComputeSosConstraint(i, j, result_ij);
      EXPECT_PRED3(symbolic::test::PolynomialEqual, p, p_expected, tol);
      // Check if the Gram matrices are PSD.
      const Eigen::MatrixXd gram_sol =
          result_ij.GetSolution(dut.vdot_sos_constraint().grams[i][j]);
      es_solver.compute(gram_sol);
      EXPECT_TRUE((es_solver.eigenvalues().array() > -psd_tol).all());
    }
  }
}

// Sample many points inside the level set V(x) <= 1, and verify that min_u
// Vdot(x, u) (-1 <= u <= 1) is less than -deriv_eps * V.
void ValidateRegionOfAttractionBySample(const VectorX<symbolic::Polynomial>& f,
                                        const MatrixX<symbolic::Polynomial>& G,
                                        const symbolic::Polynomial& V,
                                        const VectorX<symbolic::Variable>& x,
                                        const Eigen::MatrixXd& u_vertices,
                                        double deriv_eps, int num_samples,
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
      EXPECT_TRUE(-Vdot / V_val > deriv_eps - rel_tol ||
                  -Vdot > V_val * deriv_eps - abs_tol);

      sample_count++;
    }
  }
}

TEST_F(SimpleLinearSystemTest, SearchLagrangianAndBGivenVBoxInputBound) {
  // We first compute LQR cost-to-go as the candidate Lyapunov function, and
  // show that we can search the Lagrangians. Then we fix the Lagrangian, and
  // show that we can search the Lyapunov. We compute the LQR cost-to-go as the
  // candidate Lyapunov function.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::array<symbolic::Polynomial, 2>> l_given;
  std::vector<std::array<int, 6>> lagrangian_degrees;
  InitializeWithLQR(&V, &f, &G, &l_given, &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);

  SearchLagrangianAndBGivenVBoxInputBound dut_search_l_b(
      V, f, G, l_given, lagrangian_degrees, b_degrees, x_);
  dut_search_l_b.get_mutable_prog()->AddBoundingBoxConstraint(
      3, kInf, dut_search_l_b.deriv_eps());

  solvers::MosekSolver mosek_solver;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result =
      mosek_solver.Solve(dut_search_l_b.prog(), std::nullopt, solver_options);
  EXPECT_TRUE(result.is_success());
  CheckSearchLagrangianAndBResult(dut_search_l_b, result, V, f, G, x_, 5.3E-5);

  const double deriv_eps_sol = result.GetSolution(dut_search_l_b.deriv_eps());
  Eigen::Matrix<double, 2, 4> u_vertices;
  u_vertices << 1, 1, -1, -1, 1, -1, 1, -1;
  ValidateRegionOfAttractionBySample(f, G, V, x_, u_vertices, deriv_eps_sol,
                                     100, 1E-5, 1E-3);

  std::vector<std::array<symbolic::Polynomial, 6>> l_result(nu);
  for (int i = 0; i < nu; ++i) {
    for (int j = 0; j < 6; ++j) {
      l_result[i][j] = result.GetSolution(dut_search_l_b.lagrangians()[i][j]);
    }
  }
  // Now check if the Lagrangians are all sos.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_solver;
  const double psd_tol = 1E-6;
  for (int i = 0; i < nu; ++i) {
    for (int j = 2; j < 6; ++j) {
      const auto lagrangian_gram_sol =
          result.GetSolution(dut_search_l_b.lagrangian_grams()[i][j]);
      es_solver.compute(lagrangian_gram_sol);
      EXPECT_TRUE((es_solver.eigenvalues().array() >= -psd_tol).all());
      const VectorX<symbolic::Monomial> lagrangian_monomial_basis =
          symbolic::MonomialBasis(x_set_, lagrangian_degrees[i][j] / 2);
      EXPECT_PRED3(symbolic::test::PolynomialEqual, l_result[i][j],
                   lagrangian_monomial_basis.dot(lagrangian_gram_sol *
                                                 lagrangian_monomial_basis),
                   1e-5);
    }
  }
  // Given the lagrangians, test search Lyapunov.
  Vector2<symbolic::Monomial> V_monomial(symbolic::Monomial(x_(0)),
                                         symbolic::Monomial(x_(1)));
  const Eigen::Vector2d x_equilibrium(0, 0);
  const double positivity_eps = 1E-3;
  SearchLyapunovGivenLagrangianBoxInputBound dut_search_V(
      f, G, V_monomial, positivity_eps, deriv_eps_sol, x_equilibrium, l_result,
      b_degrees, x_);
  const auto result_search_V =
      mosek_solver.Solve(dut_search_V.prog(), std::nullopt, solver_options);
  ASSERT_TRUE(result_search_V.is_success());
  const symbolic::Polynomial V_sol =
      result_search_V.GetSolution(dut_search_V.V());
  ValidateRegionOfAttractionBySample(f, G, V_sol, x_, u_vertices, deriv_eps_sol,
                                     100, 1E-5, 1E-3);
  // Check if the V(x_equilibrium) = 0.
  symbolic::Environment env_x_equilibrium;
  env_x_equilibrium.insert(x_(0), x_equilibrium(0));
  env_x_equilibrium.insert(x_(1), x_equilibrium(1));
  EXPECT_NEAR(V_sol.Evaluate(env_x_equilibrium), 0., 1E-5);
  // Make sure V(x) - ε₁(x-x_des)ᵀ(x-x_des) is SOS.
  const Eigen::MatrixXd positivity_constraint_gram_sol =
      result_search_V.GetSolution(dut_search_V.positivity_constraint_gram());
  EXPECT_PRED3(
      symbolic::test::PolynomialEqual,
      V_sol - positivity_eps *
                  symbolic::Polynomial(
                      (x_ - x_equilibrium).dot(x_ - x_equilibrium), x_set_),
      dut_search_V.positivity_constraint_monomial().dot(
          positivity_constraint_gram_sol *
          dut_search_V.positivity_constraint_monomial()),
      1E-6);
  es_solver.compute(positivity_constraint_gram_sol);
  EXPECT_TRUE((es_solver.eigenvalues().array() >= -psd_tol).all());
}

TEST_F(SimpleLinearSystemTest, MaximizeEllipsoid) {
  // We first compute LQR cost-to-go as the candidate Lyapunov function, and
  // show that we can search the Lagrangians and maximize the ellipsoid
  // contained in the ROA. We then fix the Lagrangians and search for V, and
  // show that we can increase the ellipsoid size.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::array<symbolic::Polynomial, 2>> l_given;
  std::vector<std::array<int, 6>> lagrangian_degrees;
  InitializeWithLQR(&V, &f, &G, &l_given, &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);
  SearchLagrangianAndBGivenVBoxInputBound dut(
      V, f, G, l_given, lagrangian_degrees, b_degrees, x_);
  const Eigen::Vector2d x_star(0.001, 0.0002);
  // First make sure that x_star satisfies V(x*)<=1
  const symbolic::Environment env_xstar(
      {{x_(0), x_star(0)}, {x_(1), x_star(1)}});
  ASSERT_LE(V.Evaluate(env_xstar), 1);
  const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  const int s_degree = 2;
  const symbolic::Variables x_set{x_};
  const symbolic::Polynomial t(x_.cast<symbolic::Expression>().dot(x_), x_set);
  // Set the rate-of-convergence epsilon to >= 0.1
  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.1, kInf, dut.deriv_eps());
  solvers::MosekSolver mosek_solver;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result =
      mosek_solver.Solve(dut.prog(), std::nullopt, solver_options);
  EXPECT_TRUE(result.is_success());
  CheckSearchLagrangianAndBResult(dut, result, V, f, G, x_, 1.7E-5);
  // Retrieve solution of Lagrangian multipliers.
  std::vector<std::array<symbolic::Polynomial, 6>> l_sol(nu);
  for (int i = 0; i < nu; ++i) {
    l_sol[i][0] = l_given[i][0];
    l_sol[i][1] = l_given[i][1];
    for (int j = 2; j < 6; ++j) {
      l_sol[i][j] = result.GetSolution(dut.lagrangians()[i][j]);
    }
  }
  const double deriv_eps_sol = result.GetSolution(dut.deriv_eps());

  double rho_sol;
  symbolic::Polynomial s_sol;
  MaximizeInnerEllipsoidRho(x_, x_star, S, V, t, s_degree,
                            solvers::MosekSolver::id(), std::nullopt, 1.,
                            &rho_sol, &s_sol);

  // Now fix the Lagrangian multiplier and search for V
  const Eigen::Vector2d x_des(0, 0);
  Vector2<symbolic::Monomial> V_monomial(symbolic::Monomial(x_(0)),
                                         symbolic::Monomial(x_(1)));
  const double positivity_eps{1E-2};
  SearchLyapunovGivenLagrangianBoxInputBound dut_search_V(
      f, G, V_monomial, positivity_eps, deriv_eps_sol, x_des, l_sol, b_degrees,
      x_);
  const auto ellipsoid_ret_V =
      dut_search_V.AddEllipsoidInRoaConstraint(x_star, S, t, s_sol);
  dut_search_V.get_mutable_prog()->AddLinearCost(-ellipsoid_ret_V.rho);
  const auto result_search_V = mosek_solver.Solve(dut_search_V.prog());
  ASSERT_TRUE(result_search_V.is_success());
  const double rho_search_V_sol =
      result_search_V.GetSolution(ellipsoid_ret_V.rho);
  EXPECT_GT(rho_search_V_sol, rho_sol);
}

TEST_F(SimpleLinearSystemTest, SearchLagrangianGivenVBoxInputBound) {
  // We first compute LQR cost-to-go as the candidate Lyapunov function. We
  // first fix V and search for Lagangians and b. And then fix V and b to search
  // for Lagrangians only.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::array<symbolic::Polynomial, 2>> l_given;
  std::vector<std::array<int, 6>> lagrangian_degrees;
  InitializeWithLQR(&V, &f, &G, &l_given, &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);
  SearchLagrangianAndBGivenVBoxInputBound dut(
      V, f, G, l_given, lagrangian_degrees, b_degrees, x_);
  const Eigen::Vector2d x_star(0.001, 0.0002);
  // First make sure that x_star satisfies V(x*)<=1
  const symbolic::Environment env_xstar(
      {{x_(0), x_star(0)}, {x_(1), x_star(1)}});
  ASSERT_LE(V.Evaluate(env_xstar), 1);
  const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  const int s_degree = 2;
  const symbolic::Variables x_set{x_};
  const symbolic::Polynomial t(x_.cast<symbolic::Expression>().dot(x_), x_set);

  // Set the rate-of-convergence epsilon to >= 0.1
  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.1, kInf, dut.deriv_eps());
  solvers::MosekSolver mosek_solver;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result =
      mosek_solver.Solve(dut.prog(), std::nullopt, solver_options);
  ASSERT_TRUE(result.is_success());
  VectorX<symbolic::Polynomial> b_sol(nu);
  for (int i = 0; i < nu; ++i) {
    b_sol(i) = result.GetSolution(dut.b()(i));
  }
  double rho_sol;
  symbolic::Polynomial s_sol;
  MaximizeInnerEllipsoidRho(x_, x_star, S, V, t, s_degree,
                            solvers::MosekSolver::id(), std::nullopt, 1.,
                            &rho_sol, &s_sol);

  // Now fix V and b, search for Lagrangians.
  SearchLagrangianGivenVBoxInputBound dut_search_l(V, f, G, b_sol, x_,
                                                   lagrangian_degrees);
  CheckSearchLagrangianResult(dut_search_l, 3E-5, 1E-6);
}

TEST_F(SimpleLinearSystemTest, ControlLyapunovBoxInputBound) {
  // We first compute LQR cost-to-go as the candidate Lyapunov function. We
  // first fix V and search for Lagangians and b. And then fix V and b to search
  // for Lagrangians only.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::array<symbolic::Polynomial, 2>> l_given;
  std::vector<std::array<int, 6>> lagrangian_degrees;
  InitializeWithLQR(&V, &f, &G, &l_given, &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);

  const Eigen::Vector2d x_des(0., 0.);
  const double positivity_eps{0.};
  ControlLyapunovBoxInputBound dut(f, G, x_des, x_, positivity_eps);

  ControlLyapunovBoxInputBound::SearchOptions search_options;

  const Eigen::Vector2d x_star(0.001, 0.002);
  const Eigen::Matrix2d S = Eigen::Vector2d(1, 2).asDiagonal();
  const int s_degree{2};
  const symbolic::Polynomial t_given(x_.cast<symbolic::Expression>().dot(x_),
                                     x_set_);
  const Vector2<symbolic::Monomial> V_monomial(symbolic::Monomial(x_(0)),
                                               symbolic::Monomial(x_(1)));
  const double deriv_eps_lower{0.01};
  const double deriv_eps_upper{kInf};
  // Search without backoff.
  search_options.backoff_scale = 0.;
  search_options.bilinear_iterations = 5;
  const auto search_result = dut.Search(
      V, l_given, lagrangian_degrees, b_degrees, x_star, S, s_degree, t_given,
      V_monomial, deriv_eps_lower, deriv_eps_upper, search_options);
  Eigen::Matrix<double, 2, 4> u_vertices;
  u_vertices << 1, 1, -1, -1, 1, -1, 1, -1;
  EXPECT_GE(search_result.deriv_eps, deriv_eps_lower);
  EXPECT_LE(search_result.deriv_eps, deriv_eps_upper);
  ValidateRegionOfAttractionBySample(f, G, search_result.V, x_, u_vertices,
                                     search_result.deriv_eps, 1000, 1E-5, 1E-3);

  // Search with backoff.
  search_options.backoff_scale = 0.05;
  search_options.lyap_step_solver = solvers::MosekSolver::id();
  search_options.bilinear_iterations = 5;
  const auto search_result_backoff = dut.Search(
      V, l_given, lagrangian_degrees, b_degrees, x_star, S, s_degree, t_given,
      V_monomial, deriv_eps_lower, deriv_eps_upper, search_options);
  ValidateRegionOfAttractionBySample(
      f, G, search_result_backoff.V, x_, u_vertices,
      search_result_backoff.deriv_eps, 1000, 1E-5, 1E-3);
}

void CheckEllipsoidInRoa(const Eigen::Ref<const VectorX<symbolic::Variable>> x,
                         const Eigen::Ref<const Eigen::VectorXd>& x_star,
                         const Eigen::Ref<const Eigen::MatrixXd>& S, double rho,
                         const symbolic::Polynomial& V) {
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
    EXPECT_LE(V.Evaluate(env), 1 + 1E-5);
  }
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
  MaximizeInnerEllipsoidRho(x, x_star, S, V, t, s_degree,
                            solvers::MosekSolver::id(), std::nullopt,
                            backoff_scale, &rho_sol, &s_sol);
  const double tol = 1E-5;
  EXPECT_NEAR(rho_sol, 0.5, tol);

  CheckEllipsoidInRoa(x, x_star, S, rho_sol, V);
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
  const symbolic::Polynomial t(0);
  const int s_degree = 2;
  const double backoff_scale = 0.05;
  double rho_sol;
  symbolic::Polynomial s_sol;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  // I am really surprised that this SOS finds a solution with rho > 0. AFAIK,
  // t(x) is constant, hence (1+t(x))((x-x*)ᵀS(x-x*)-ρ) is a degree 2
  // polynomial, while -s(x)*(V(x)-1) has much higher degree (>6) with negative
  // leading terms. The resulting polynomial cannot be sos.
  MaximizeInnerEllipsoidRho(x, x_star, S, V, t, s_degree,
                            solvers::MosekSolver::id(), solver_options,
                            backoff_scale, &rho_sol, &s_sol);
  CheckEllipsoidInRoa(x, x_star, S, rho_sol, V);
}

GTEST_TEST(EllipsoidPolynomial, Test) {
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));
  const Eigen::Vector2d x_star(2, 3);
  Eigen::Matrix2d S;
  S << 1, 1, 3, 9;
  const double rho = 2;
  const symbolic::Polynomial poly =
      internal::EllipsoidPolynomial(x, x_star, S, rho);
  const symbolic::Polynomial poly_expected(
      (x - x_star).dot(S * (x - x_star)) - rho, symbolic::Variables(x));
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, poly, poly_expected);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
