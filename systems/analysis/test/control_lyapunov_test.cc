#include "drake/systems/analysis/control_lyapunov.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/symbolic_monomial_util.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/scs_solver.h"
#include "drake/solvers/sdpa_free_format.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
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
  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 1, 1, -1, -1,
                1, -1, 1, -1;
  // clang-format on
  const VdotCalculator dut1(x_, V1, f_, G_, u_vertices);

  const Eigen::Vector2d u_val(2., 3.);
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, dut1.Calc(u_val),
               (Eigen::Matrix<symbolic::Polynomial, 1, 2>(2 * x0_, 4 * x1_) *
                (f_ + G_ * u_val))(0));

  const symbolic::Polynomial V2(2 * a_ * x0_ * x1_ * x1_, x_vars_);
  const VdotCalculator dut2(x_, V2, f_, G_, u_vertices);
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, dut2.Calc(u_val),
               (Eigen::Matrix<symbolic::Polynomial, 1, 2>(
                    symbolic::Polynomial(2 * a_ * x1_ * x1_, x_vars_),
                    symbolic::Polynomial(4 * a_ * x0_ * x1_, x_vars_)) *
                (f_ + G_ * u_val))(0));
}

TEST_F(ControlLyapunovTest, VdotCalcMin) {
  const Vector2<symbolic::Polynomial> f{
      2 * x0_, symbolic::Polynomial{3 * x1_ * x0_, x_vars_}};
  Matrix2<symbolic::Polynomial> G;
  G << symbolic::Polynomial{3.}, symbolic::Polynomial{1, x_vars_},
      symbolic::Polynomial{x0_}, symbolic::Polynomial{2 * x1_ * x0_, x_vars_};
  const symbolic::Polynomial V(x0_ * x0_ + 2 * x1_ * x1_ + x0_ * x1_);

  Eigen::Matrix<double, 2, 5> u_vertices;
  // clang-format off
  u_vertices << 1, 1, -1, 0, 0.5,
                1, -1, 1, -1, -2;
  // clang-format on
  const VdotCalculator dut(x_, V, f, G, u_vertices);
  Eigen::Matrix<double, 2, 3> x_vals;
  // clang-format off
  x_vals << 0.4, 1, -0.5,
            0.2, 2, -1;
  // clang-format on
  const Eigen::VectorXd Vdot_vals = dut.CalcMin(x_vals);
  EXPECT_EQ(Vdot_vals.rows(), x_vals.cols());
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x_);
  const symbolic::Polynomial dVdx_times_f = dVdx.dot(f);
  const RowVectorX<symbolic::Polynomial> dVdx_times_G = dVdx * G;
  for (int i = 0; i < x_vals.cols(); ++i) {
    symbolic::Environment env;
    env.insert(x_, x_vals.col(i));
    const double dVdx_times_f_val = dVdx_times_f.Evaluate(env);
    Eigen::RowVectorXd dVdx_times_G_val(dVdx_times_G.cols());
    for (int j = 0; j < dVdx_times_G_val.cols(); ++j) {
      dVdx_times_G_val(j) = dVdx_times_G(j).Evaluate(env);
    }
    EXPECT_NEAR(Vdot_vals(i),
                (dVdx_times_G_val * u_vertices).minCoeff() + dVdx_times_f_val,
                1E-7);
  }
}

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
  }

  void InitializeWithLQR(bool symmetric_dynamics, symbolic::Polynomial* V,
                         Vector2<symbolic::Polynomial>* f,
                         Matrix2<symbolic::Polynomial>* G) {
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
  }

  // This is used for ControlLyapunovBoxInputBound
  void InitializeWithLQR(
      bool symmetric_dynamics, symbolic::Polynomial* V,
      Vector2<symbolic::Polynomial>* f, Matrix2<symbolic::Polynomial>* G,
      std::vector<std::vector<symbolic::Polynomial>>* l_given,
      std::vector<std::vector<std::array<int, 3>>>* lagrangian_degrees) {
    InitializeWithLQR(symmetric_dynamics, V, f, G);
    // Set l_[i][0] and l_[i][1] to 1 + x.dot(x)
    const int nu = 2;
    l_given->resize(nu);
    const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
    for (int i = 0; i < nu; ++i) {
      (*l_given)[i].resize(num_vdot_sos);
      for (int j = 0; j < num_vdot_sos; ++j) {
        (*l_given)[i][j] =
            symbolic::Polynomial(1 + x_.cast<symbolic::Expression>().dot(x_));
      }
    }
    lagrangian_degrees->resize(nu);
    for (int i = 0; i < nu; ++i) {
      (*lagrangian_degrees)[i].resize(num_vdot_sos);
      for (int j = 0; j < num_vdot_sos; ++j) {
        (*lagrangian_degrees)[i][j] = {2, 2, 2};
      }
    }
  }

 protected:
  Eigen::Matrix2d A_;
  Eigen::Matrix2d B_;
  Vector2<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  Eigen::Matrix<double, 2, 4> u_vertices_;
};

void CheckSearchLagrangianAndBResult(
    const solvers::MathematicalProgramResult& result,
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& f,
    const MatrixX<symbolic::Polynomial>& G,
    const VectorX<symbolic::Variable>& x,
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>&
        lagrangian_grams,
    const symbolic::Variable& deriv_eps, const VectorX<symbolic::Polynomial>& b,
    const VdotSosConstraintReturn& vdot_sos_constraint, double tol) {
  ASSERT_TRUE(result.is_success());
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  const double deriv_eps_val = result.GetSolution(deriv_eps);
  const int nu = G.cols();
  VectorX<symbolic::Polynomial> b_result(nu);
  for (int i = 0; i < nu; ++i) {
    b_result(i) = result.GetSolution(b(i));
  }

  EXPECT_TRUE(symbolic::test::PolynomialEqual((dVdx * f)(0) + deriv_eps_val * V,
                                              b_result.sum(), tol));
  std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>
      lagrangians_result(nu);
  const int num_vdot_sos = vdot_sos_constraint.symmetric_dynamics ? 1 : 2;
  for (int i = 0; i < nu; ++i) {
    lagrangians_result[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      for (int k = 0; k < 3; ++k) {
        lagrangians_result[i][j][k] = result.GetSolution(l[i][j][k]);
      }
    }
    const symbolic::Polynomial dVdx_times_Gi = (dVdx * G.col(i))(0);
    const symbolic::Polynomial p1 =
        (lagrangians_result[i][0][0] + 1) * (dVdx_times_Gi - b_result(i)) -
        lagrangians_result[i][0][1] * dVdx_times_Gi -
        lagrangians_result[i][0][2] * (1 - V);
    const std::vector<symbolic::Polynomial> p_expected =
        vdot_sos_constraint.ComputeSosConstraint(i, result);
    EXPECT_TRUE(symbolic::test::PolynomialEqual(p1, p_expected[0], tol));
    if (num_vdot_sos == 2) {
      const symbolic::Polynomial p2 =
          (lagrangians_result[i][1][0] + 1) * (-dVdx_times_Gi - b_result(i)) +
          lagrangians_result[i][1][1] * dVdx_times_Gi -
          lagrangians_result[i][1][2] * (1 - V);
      EXPECT_TRUE(symbolic::test::PolynomialEqual(p2, p_expected[1], tol));
    }

    // Now check if the gram matrices are psd.
    for (int j = 0; j < num_vdot_sos; ++j) {
      const Eigen::MatrixXd gram =
          result.GetSolution(vdot_sos_constraint.grams[i][j]);
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
      es.compute(gram);
      EXPECT_TRUE((es.eigenvalues().array() >= -tol).all());
    }
  }
}

void CheckSearchLagrangianResult(const SearchLagrangianGivenVBoxInputBound& dut,
                                 double tol, double psd_tol) {
  // Check if the constraint
  // (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >=0
  // (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)>= 0
  // are satisfied.
  const RowVectorX<symbolic::Polynomial> dVdx = dut.V().Jacobian(dut.x());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_solver;
  for (int i = 0; i < dut.nu(); ++i) {
    const int num_vdot_sos =
        dut.vdot_sos_constraint().symmetric_dynamics ? 1 : 2;
    std::vector<std::array<symbolic::Polynomial, 3>> l_sol(num_vdot_sos);
    const symbolic::Polynomial dVdx_times_Gi = dVdx.dot(dut.G().col(i));
    for (int j = 0; j < num_vdot_sos; ++j) {
      solvers::MosekSolver mosek_solver;
      solvers::SolverOptions solver_options;
      const auto result_ij =
          mosek_solver.Solve(dut.prog(i, j), std::nullopt, solver_options);
      ASSERT_TRUE(result_ij.is_success());
      const symbolic::Polynomial p_expected =
          dut.vdot_sos_constraint().ComputeSosConstraint(i, j, result_ij);
      for (int k = 0; k < 3; ++k) {
        l_sol[j][k] = result_ij.GetSolution(dut.lagrangians()[i][j][k]);
      }
      const symbolic::Polynomial p =
          j == 0
              ? (l_sol[0][0] + 1) * (dVdx_times_Gi - dut.b()(i)) -
                    l_sol[0][1] * dVdx_times_Gi - l_sol[0][2] * (1 - dut.V())
              : (l_sol[1][0] + 1) * (-dVdx_times_Gi - dut.b()(i)) +
                    l_sol[1][1] * dVdx_times_Gi - l_sol[1][2] * (1 - dut.V());
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
  for (bool symmetric_dynamics : {false, true}) {
    // We first compute LQR cost-to-go as the candidate Lyapunov function, and
    // show that we can search the Lagrangians. Then we fix the Lagrangian, and
    // show that we can search the Lyapunov. We compute the LQR cost-to-go as
    // the candidate Lyapunov function.
    symbolic::Polynomial V;
    Vector2<symbolic::Polynomial> f;
    Matrix2<symbolic::Polynomial> G;
    std::vector<std::vector<symbolic::Polynomial>> l_given;
    std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees;
    InitializeWithLQR(symmetric_dynamics, &V, &f, &G, &l_given,
                      &lagrangian_degrees);
    const int nu{2};
    std::vector<int> b_degrees(nu, 2);

    const double positivity_eps = 1E-3;
    ControlLyapunovBoxInputBound dut(f, G, x_, positivity_eps);
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l;
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>
        lagrangian_grams;
    symbolic::Variable deriv_eps;
    VectorX<symbolic::Polynomial> b;
    VdotSosConstraintReturn vdot_sos_constraint(G.cols(), symmetric_dynamics);
    auto prog_l_and_b = dut.ConstructLagrangianAndBProgram(
        V, l_given, lagrangian_degrees, b_degrees, symmetric_dynamics, &l,
        &lagrangian_grams, &deriv_eps, &b, &vdot_sos_constraint);

    prog_l_and_b->AddBoundingBoxConstraint(3, kInf, deriv_eps);

    solvers::MosekSolver mosek_solver;
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
    const auto result =
        mosek_solver.Solve(*prog_l_and_b, std::nullopt, solver_options);
    EXPECT_TRUE(result.is_success());
    CheckSearchLagrangianAndBResult(result, V, f, G, x_, l, lagrangian_grams,
                                    deriv_eps, b, vdot_sos_constraint, 5.3E-5);

    const double deriv_eps_sol = result.GetSolution(deriv_eps);
    Eigen::Matrix<double, 2, 4> u_vertices;
    u_vertices << 1, 1, -1, -1, 1, -1, 1, -1;
    ValidateRegionOfAttractionBySample(f, G, V, x_, u_vertices, deriv_eps_sol,
                                       100, 1E-5, 1E-3);

    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l_result(nu);
    const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
    for (int i = 0; i < nu; ++i) {
      l_result[i].resize(num_vdot_sos);
      for (int j = 0; j < num_vdot_sos; ++j) {
        for (int k = 0; k < 3; ++k) {
          l_result[i][j][k] = result.GetSolution(l[i][j][k]);
        }
      }
    }
    // Now check if the Lagrangians are all sos.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_solver;
    const double psd_tol = 1E-6;
    for (int i = 0; i < nu; ++i) {
      for (int j = 0; j < num_vdot_sos; ++j) {
        for (int k = 1; k < 3; ++k) {
          const auto lagrangian_gram_sol =
              result.GetSolution(lagrangian_grams[i][j][k]);
          es_solver.compute(lagrangian_gram_sol);
          EXPECT_TRUE((es_solver.eigenvalues().array() >= -psd_tol).all());
          VectorX<symbolic::Monomial> lagrangian_monomial_basis;
          if (k == 2) {
            lagrangian_monomial_basis = ComputeMonomialBasisNoConstant(
                x_set_, lagrangian_degrees[i][j][k] / 2,
                symbolic::internal::DegreeType::kAny);
          } else {
            lagrangian_monomial_basis = symbolic::MonomialBasis(
                x_set_, lagrangian_degrees[i][j][k] / 2);
          }
          EXPECT_PRED3(symbolic::test::PolynomialEqual, l_result[i][j][k],
                       lagrangian_monomial_basis.dot(lagrangian_gram_sol *
                                                     lagrangian_monomial_basis),
                       1e-5);
        }
      }
    }
    // Given the lagrangians, test search Lyapunov.
    const int V_degree = 2;

    symbolic::Polynomial V_new;
    MatrixX<symbolic::Expression> positivity_constraint_gram;
    VectorX<symbolic::Monomial> positivity_constraint_monomial;
    VectorX<symbolic::Polynomial> b_new;
    auto prog_V = dut.ConstructLyapunovProgram(
        l_result, symmetric_dynamics, deriv_eps_sol, V_degree, b_degrees,
        &V_new, &positivity_constraint_gram, &positivity_constraint_monomial,
        &b_new, &vdot_sos_constraint);

    const auto result_search_V =
        mosek_solver.Solve(*prog_V, std::nullopt, solver_options);
    ASSERT_TRUE(result_search_V.is_success());
    const symbolic::Polynomial V_sol = result_search_V.GetSolution(V_new);
    ValidateRegionOfAttractionBySample(f, G, V_sol, x_, u_vertices,
                                       deriv_eps_sol, 100, 1E-5, 1E-3);
    // Check if the V(0) = 0 and V(x) doesn't have a constant term.
    for (const auto& [V_sol_monomial, V_sol_coeff] :
         V_sol.monomial_to_coefficient_map()) {
      EXPECT_GT(V_sol_monomial.total_degree(), 1);
    }
    // Make sure V(x) - ε₁xᵀx is SOS.
    Eigen::MatrixXd positivity_constraint_gram_sol(
        positivity_constraint_gram.rows(), positivity_constraint_gram.cols());
    for (int i = 0; i < positivity_constraint_gram_sol.rows(); ++i) {
      for (int j = 0; j < positivity_constraint_gram_sol.cols(); ++j) {
        positivity_constraint_gram_sol(i, j) = symbolic::get_constant_value(
            result_search_V.GetSolution(positivity_constraint_gram(i, j)));
      }
    }
    EXPECT_PRED3(
        symbolic::test::PolynomialEqual,
        V_sol - positivity_eps *
                    symbolic::Polynomial(
                        x_.cast<symbolic::Expression>().dot(x_), x_set_),
        positivity_constraint_monomial.dot(positivity_constraint_gram_sol *
                                           positivity_constraint_monomial),
        1E-6);
    es_solver.compute(positivity_constraint_gram_sol);
    EXPECT_TRUE((es_solver.eigenvalues().array() >= -psd_tol).all());
  }
}

TEST_F(SimpleLinearSystemTest, MaximizeEllipsoid) {
  for (bool symmetric_dynamics : {false, true}) {
    // We first compute LQR cost-to-go as the candidate Lyapunov function, and
    // show that we can search the Lagrangians and maximize the ellipsoid
    // contained in the ROA. We then fix the Lagrangians and search for V, and
    // show that we can increase the ellipsoid size.
    symbolic::Polynomial V;
    Vector2<symbolic::Polynomial> f;
    Matrix2<symbolic::Polynomial> G;
    std::vector<std::vector<symbolic::Polynomial>> l_given;
    std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees;
    InitializeWithLQR(symmetric_dynamics, &V, &f, &G, &l_given,
                      &lagrangian_degrees);
    const int nu{2};
    std::vector<int> b_degrees(nu, 2);

    const double positivity_eps{1E-2};
    ControlLyapunovBoxInputBound dut(f, G, x_, positivity_eps);
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l;
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>
        lagrangian_grams;
    symbolic::Variable deriv_eps;
    VectorX<symbolic::Polynomial> b;
    VdotSosConstraintReturn vdot_sos_constraint(G.cols(), symmetric_dynamics);
    auto prog_l_and_b = dut.ConstructLagrangianAndBProgram(
        V, l_given, lagrangian_degrees, b_degrees, symmetric_dynamics, &l,
        &lagrangian_grams, &deriv_eps, &b, &vdot_sos_constraint);
    const Eigen::Vector2d x_star(0.001, 0.0002);
    // First make sure that x_star satisfies V(x*)<=1
    const symbolic::Environment env_xstar(
        {{x_(0), x_star(0)}, {x_(1), x_star(1)}});
    ASSERT_LE(V.Evaluate(env_xstar), 1);
    const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
    const int s_degree = 2;
    const symbolic::Variables x_set{x_};
    const symbolic::Polynomial t(x_.cast<symbolic::Expression>().dot(x_),
                                 x_set);
    // Set the rate-of-convergence epsilon to >= 0.1
    prog_l_and_b->AddBoundingBoxConstraint(0.1, kInf, deriv_eps);
    solvers::MosekSolver mosek_solver;
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
    const auto result =
        mosek_solver.Solve(*prog_l_and_b, std::nullopt, solver_options);
    EXPECT_TRUE(result.is_success());
    CheckSearchLagrangianAndBResult(result, V, f, G, x_, l, lagrangian_grams,
                                    deriv_eps, b, vdot_sos_constraint, 1.7E-5);
    // Retrieve solution of Lagrangian multipliers.
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l_sol(nu);
    const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
    for (int i = 0; i < nu; ++i) {
      l_sol[i].resize(num_vdot_sos);
      for (int j = 0; j < num_vdot_sos; ++j) {
        l_sol[i][j][0] = l_given[i][j];
        for (int k = 1; k < 3; ++k) {
          l_sol[i][j][k] = result.GetSolution(l[i][j][k]);
        }
      }
    }
    const double deriv_eps_sol = result.GetSolution(deriv_eps);

    double rho_sol;
    symbolic::Polynomial s_sol;
    MaximizeInnerEllipsoidRho(x_, x_star, S, V - 1, t, s_degree,
                              solvers::MosekSolver::id(), std::nullopt, 1.,
                              &rho_sol, &s_sol);

    // Now fix the Lagrangian multiplier and search for V
    const int V_degree = 2;

    symbolic::Polynomial V_new;
    MatrixX<symbolic::Expression> positivity_constraint_gram;
    VectorX<symbolic::Monomial> positivity_constraint_monomial;
    VectorX<symbolic::Polynomial> b_new;
    auto prog_V = dut.ConstructLyapunovProgram(
        l_sol, symmetric_dynamics, deriv_eps_sol, V_degree, b_degrees, &V_new,
        &positivity_constraint_gram, &positivity_constraint_monomial, &b_new,
        &vdot_sos_constraint);
    const auto result_search_V = mosek_solver.Solve(*prog_V);
    ASSERT_TRUE(result_search_V.is_success());
    // Check if V passes the origin.
    const symbolic::Polynomial V_sol = result_search_V.GetSolution(V_new);
    for (const auto& [V_sol_monomial, V_sol_ceoff] :
         V_sol.monomial_to_coefficient_map()) {
      EXPECT_GT(V_sol_monomial.total_degree(), 1);
    }
  }
}

TEST_F(SimpleLinearSystemTest, SearchLagrangianGivenVBoxInputBound) {
  for (bool symmetric_dynamics : {false, true}) {
    // We first compute LQR cost-to-go as the candidate Lyapunov function. We
    // first fix V and search for Lagangians and b. And then fix V and b to
    // search for Lagrangians only.
    symbolic::Polynomial V;
    Vector2<symbolic::Polynomial> f;
    Matrix2<symbolic::Polynomial> G;
    std::vector<std::vector<symbolic::Polynomial>> l_given;
    std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees;
    InitializeWithLQR(symmetric_dynamics, &V, &f, &G, &l_given,
                      &lagrangian_degrees);
    const int nu{2};
    std::vector<int> b_degrees(nu, 2);

    const double positivity_eps = 0;
    ControlLyapunovBoxInputBound dut(f, G, x_, positivity_eps);
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l;
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>
        lagrangian_grams;
    symbolic::Variable deriv_eps;
    VectorX<symbolic::Polynomial> b;
    VdotSosConstraintReturn vdot_sos_constraint(G.cols(), symmetric_dynamics);
    auto prog_l_and_b = dut.ConstructLagrangianAndBProgram(
        V, l_given, lagrangian_degrees, b_degrees, symmetric_dynamics, &l,
        &lagrangian_grams, &deriv_eps, &b, &vdot_sos_constraint);
    const Eigen::Vector2d x_star(0.001, 0.0002);
    // First make sure that x_star satisfies V(x*)<=1
    const symbolic::Environment env_xstar(
        {{x_(0), x_star(0)}, {x_(1), x_star(1)}});
    ASSERT_LE(V.Evaluate(env_xstar), 1);
    const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
    const int s_degree = 2;
    const symbolic::Variables x_set{x_};
    const symbolic::Polynomial t(x_.cast<symbolic::Expression>().dot(x_),
                                 x_set);

    // Set the rate-of-convergence epsilon to >= 0.1
    prog_l_and_b->AddBoundingBoxConstraint(0.1, kInf, deriv_eps);
    solvers::MosekSolver mosek_solver;
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
    const auto result =
        mosek_solver.Solve(*prog_l_and_b, std::nullopt, solver_options);
    ASSERT_TRUE(result.is_success());
    VectorX<symbolic::Polynomial> b_sol(nu);
    for (int i = 0; i < nu; ++i) {
      b_sol(i) = result.GetSolution(b(i));
    }
    double rho_sol;
    symbolic::Polynomial s_sol;
    MaximizeInnerEllipsoidRho(x_, x_star, S, V - 1, t, s_degree,
                              solvers::MosekSolver::id(), std::nullopt, 1.,
                              &rho_sol, &s_sol);

    // Now fix V and b, search for Lagrangians.
    SearchLagrangianGivenVBoxInputBound dut_search_l(
        V, f, G, symmetric_dynamics, b_sol, x_, lagrangian_degrees);
    CheckSearchLagrangianResult(dut_search_l, 5E-5, 1E-6);
  }
}

TEST_F(SimpleLinearSystemTest, ControlLyapunovBoxInputBound) {
  // We first compute LQR cost-to-go as the candidate Lyapunov function. We
  // first fix V and search for Lagangians and b. And then fix V and b to search
  // for Lagrangians only.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::vector<symbolic::Polynomial>> l_given;
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees;
  const bool symmetric_dynamics = true;
  InitializeWithLQR(symmetric_dynamics, &V, &f, &G, &l_given,
                    &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);

  const double positivity_eps{0.};
  ControlLyapunovBoxInputBound dut(f, G, x_, positivity_eps);

  ControlLyapunovBoxInputBound::SearchOptions search_options;

  const Eigen::Vector2d x_star(0.001, 0.002);
  const Eigen::Matrix2d S = Eigen::Vector2d(1, 2).asDiagonal();
  const int s_degree{2};
  const symbolic::Polynomial t_given(x_.cast<symbolic::Expression>().dot(x_),
                                     x_set_);
  const int V_degree = 2;
  const double deriv_eps_lower{0.01};
  const double deriv_eps_upper{kInf};
  // Search without backoff.
  search_options.backoff_scale = 0.;
  search_options.bilinear_iterations = 5;
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  search_options.lyap_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 0);
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 0);
  const auto search_result = dut.Search(
      V, l_given, lagrangian_degrees, b_degrees, x_star, S, s_degree, t_given,
      V_degree, deriv_eps_lower, deriv_eps_upper, search_options);
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
      V_degree, deriv_eps_lower, deriv_eps_upper, search_options);
  ValidateRegionOfAttractionBySample(
      f, G, search_result_backoff.V, x_, u_vertices,
      search_result_backoff.deriv_eps, 1000, 1E-5, 1E-3);

  // Search with algorithm 1
  const double rho_min = 0.001;
  const double rho_max = 5;
  const double rho_bisection_tol = 0.01;
  const int r_degree = V_degree - 2;
  const ControlLyapunovBoxInputBound::RhoBisectionOption rho_bisection_option(
      rho_min, rho_max, rho_bisection_tol);
  const auto search_result_algo1 = dut.Search(
      V, l_given, lagrangian_degrees, b_degrees, x_star, S, r_degree, V_degree,
      deriv_eps_lower, deriv_eps_upper, search_options, rho_bisection_option);
  ValidateRegionOfAttractionBySample(
      f, G, search_result_algo1.V, x_, u_vertices,
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

TEST_F(SimpleLinearSystemTest, ControlLyapunovBoxInputBound_SearchLyapunov) {
  // Test ControlLyapunovBoxInutBound::SearchLyapunov whose objective is to
  // minimize the maximal value of V(x) on the inner ellipsoid.
  symbolic::Polynomial V;
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  std::vector<std::vector<symbolic::Polynomial>> l_given;
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees;
  const bool symmetric_dynamics = true;
  InitializeWithLQR(symmetric_dynamics, &V, &f, &G, &l_given,
                    &lagrangian_degrees);
  const int nu{2};
  std::vector<int> b_degrees(nu, 2);

  const double positivity_eps{0.};
  ControlLyapunovBoxInputBound dut(f, G, x_, positivity_eps);

  const double deriv_eps_lower = 0.01;
  const double deriv_eps_upper = kInf;
  // First find b and Lagrangian multiplier.
  double deriv_eps;
  VectorX<symbolic::Polynomial> b;
  std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l;
  dut.SearchLagrangianAndB(V, l_given, lagrangian_degrees, b_degrees,
                           deriv_eps_lower, deriv_eps_upper,
                           solvers::MosekSolver::id(), std::nullopt, 0.,
                           &deriv_eps, &b, &l);
  // Find the maximal inscribed ellipsoid.
  const Eigen::Vector2d x_star(0.001, 0.002);
  const Eigen::Matrix2d S = Eigen::Vector2d(1, 2).asDiagonal();
  int r_degree = 0;
  double rho_max = 0.04;
  double rho_min = 0.001;
  double rho_tol = 0.001;
  double rho_sol;
  symbolic::Polynomial r_sol;
  MaximizeInnerEllipsoidRho(x_, x_star, S, V - 1, std::nullopt, r_degree,
                            std::nullopt, rho_max, rho_min,
                            solvers::MosekSolver::id(), std::nullopt, rho_tol,
                            &rho_sol, &r_sol, nullptr);

  // Now find the Lyapunov function.
  const double backoff_scale = 0.01;
  symbolic::Polynomial V_sol;
  VectorX<symbolic::Polynomial> b_sol;
  double d_sol;
  dut.SearchLyapunov(l, b_degrees, V.TotalDegree(), deriv_eps, x_star, S,
                     rho_sol, r_degree, solvers::MosekSolver::id(),
                     std::nullopt, backoff_scale, 0., 0., &V_sol, &b_sol,
                     &r_sol, &d_sol);
  // First validate that V is a valid CLF.
  Eigen::Matrix<double, 2, 4> u_vertices;
  u_vertices << 1, 1, -1, -1, 1, -1, 1, -1;
  ValidateRegionOfAttractionBySample(f, G, V_sol, x_, u_vertices, deriv_eps,
                                     100, 1E-5, 1E-3);
  ASSERT_GT(d_sol, 0);
  EXPECT_LE(d_sol, 1);
  // Now check if the ellipsoid is in the sub-level set {x | V(x) <= d}.
  CheckEllipsoidInRoa(x_, x_star, S, rho_sol, 1. / d_sol * V_sol);

  // Now find the largest ellipsoid in {x | V(x)<=1} again. This ellipsoid
  // should be larger than the one before searching for V.
  double rho_sol_new;
  symbolic::Polynomial r_sol_new;
  MaximizeInnerEllipsoidRho(x_, x_star, S, V_sol - 1, std::nullopt, r_degree,
                            std::nullopt, rho_max, rho_min,
                            solvers::MosekSolver::id(), std::nullopt, rho_tol,
                            &rho_sol_new, &r_sol_new, nullptr);
  EXPECT_GT(rho_sol_new, rho_sol);
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

GTEST_TEST(NegateIndeterminates, Test) {
  symbolic::Variable x("x");
  symbolic::Variable y("y");
  const symbolic::Polynomial p(x * x + 2 * x * y + 3 + 3 * x * y * y +
                               y * y * y);
  const symbolic::Polynomial q = internal::NegateIndeterminates(p);
  const symbolic::Polynomial q_expected(-x * -x + 2 * (-x) * (-y) + 3 +
                                        3 * (-x) * (-y) * (-y) +
                                        (-y) * (-y) * (-y));
  EXPECT_PRED2(symbolic::test::PolyEqualAfterExpansion, q, q_expected);
}

GTEST_TEST(NewFreePolynomialNoConstantOrLinear, Test) {
  solvers::MathematicalProgram prog;
  auto x = prog.NewIndeterminates<2>();
  const symbolic::Variables x_set(x);

  auto check = [&prog, &x_set](int degree,
                               symbolic::internal::DegreeType degree_type,
                               int num_monomials) {
    const auto p = internal::NewFreePolynomialNoConstantOrLinear(
        &prog, x_set, degree, "p", degree_type);
    EXPECT_EQ(p.monomial_to_coefficient_map().size(), num_monomials);
    for (const auto& [monomial, _] : p.monomial_to_coefficient_map()) {
      EXPECT_GE(monomial.total_degree(), 2);
      EXPECT_LE(monomial.total_degree(), degree);
      if (degree_type == symbolic::internal::DegreeType::kOdd) {
        EXPECT_EQ(monomial.total_degree() % 2, 1);
      } else if (degree_type == symbolic::internal::DegreeType::kEven) {
        EXPECT_EQ(monomial.total_degree() % 2, 0);
      }
    }
  };

  check(0, symbolic::internal::DegreeType::kAny, 0);
  check(2, symbolic::internal::DegreeType::kAny, 3);
  check(2, symbolic::internal::DegreeType::kOdd, 0);
  check(3, symbolic::internal::DegreeType::kOdd, 4);
  check(3, symbolic::internal::DegreeType::kEven, 3);
  check(3, symbolic::internal::DegreeType::kAny, 7);
  check(4, symbolic::internal::DegreeType::kAny, 12);
  check(4, symbolic::internal::DegreeType::kEven, 8);
  check(4, symbolic::internal::DegreeType::kOdd, 4);
}

TEST_F(SimpleLinearSystemTest, ControlLyapunov) {
  // Test ControlLyapunov::ConstructLagrangianProgram
  // and ControlLyapunov::ConstructLyapunovProgram
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  symbolic::Polynomial V;
  bool symmetric_dynamics = true;
  InitializeWithLQR(symmetric_dynamics, &V, &f, &G);
  V *= 0.1;
  VectorX<symbolic::Polynomial> state_constraints(0);
  ControlLyapunov dut(x_, f, G, u_vertices_, state_constraints);
  const int lambda0_degree = 0;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> p_degrees = {};
  const double deriv_eps = 0.01;
  symbolic::Polynomial lambda0;
  MatrixX<symbolic::Variable> lambda0_gram;
  VectorX<symbolic::Polynomial> l;
  std::vector<MatrixX<symbolic::Variable>> l_grams;
  VectorX<symbolic::Polynomial> p;
  symbolic::Polynomial vdot_sos;
  VectorX<symbolic::Monomial> vdot_monomials;
  MatrixX<symbolic::Variable> vdot_gram;
  auto prog_lagrangian = dut.ConstructLagrangianProgram(
      V, deriv_eps, lambda0_degree, l_degrees, p_degrees, &lambda0,
      &lambda0_gram, &l, &l_grams, &p, &vdot_sos, &vdot_monomials, &vdot_gram);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result_lagrangian =
      solvers::Solve(*prog_lagrangian, std::nullopt, solver_options);
  ASSERT_TRUE(result_lagrangian.is_success());
  // Check the positivity of Gram matrices.
  const auto lambda0_gram_sol = result_lagrangian.GetSolution(lambda0_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(lambda0_gram_sol));
  const int num_u_vertices = u_vertices_.cols();
  std::vector<Eigen::MatrixXd> l_grams_sol(num_u_vertices);
  for (int i = 0; i < num_u_vertices; ++i) {
    l_grams_sol[i] = result_lagrangian.GetSolution(l_grams[i]);
    EXPECT_TRUE(math::IsPositiveDefinite(l_grams_sol[i]));
  }
  const Eigen::MatrixXd vdot_gram_sol =
      result_lagrangian.GetSolution(vdot_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(vdot_gram_sol));

  // Now check if the vdot sos is computed correctly.
  symbolic::Polynomial lambda0_sol = result_lagrangian.GetSolution(lambda0);
  symbolic::Polynomial vdot_sos_expected =
      (1 + lambda0_sol) *
      symbolic::Polynomial(x_.cast<symbolic::Expression>().dot(x_)) * (V - 1);
  const auto dVdx = V.Jacobian(x_);
  const symbolic::Polynomial dVdx_times_f = dVdx.dot(f);
  VectorX<symbolic::Polynomial> l_sol(num_u_vertices);
  for (int i = 0; i < num_u_vertices; ++i) {
    l_sol(i) = result_lagrangian.GetSolution(l(i));
    vdot_sos_expected -= l_sol(i) * (dVdx_times_f + deriv_eps * V +
                                     dVdx.dot(G * u_vertices_.col(i)));
  }
  VectorX<symbolic::Polynomial> p_sol;
  GetPolynomialSolutions(result_lagrangian, p, 0, &p_sol);
  vdot_sos_expected -= p_sol.dot(state_constraints);
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               result_lagrangian.GetSolution(vdot_sos), vdot_sos_expected,
               1E-5);
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               result_lagrangian.GetSolution(vdot_sos),
               vdot_monomials.dot(vdot_gram_sol * vdot_monomials), 1E-5);
  for (int i = 0; i < vdot_monomials.rows(); ++i) {
    EXPECT_GT(vdot_monomials(i).total_degree(), 0);
  }

  // Now search for the Lyapunov given Lagrangian.
  const double positivity_eps = 0.0001;
  const int positivity_d = 1;
  const std::vector<int> positivity_eq_lagrangian_degrees{};
  symbolic::Polynomial V_new;
  VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
  const int V_degree = V.TotalDegree();
  auto prog_lyapunov = dut.ConstructLyapunovProgram(
      lambda0_sol, l_sol, V_degree, positivity_eps, positivity_d,
      positivity_eq_lagrangian_degrees, p_degrees, deriv_eps, &V_new,
      &positivity_eq_lagrangian, &p);
  for (const auto& binding : prog_lyapunov->linear_equality_constraints()) {
    binding.evaluator()->RemoveTinyCoefficient(1E-12);
  }
  solvers::MosekSolver mosek_solver;
  const auto result_lyapunov =
      mosek_solver.Solve(*prog_lyapunov, std::nullopt, solver_options);
  ASSERT_TRUE(result_lyapunov.is_success());

  // V(0) = 0
  const symbolic::Polynomial V_new_sol = result_lyapunov.GetSolution(V_new);
  EXPECT_EQ(V_new_sol.monomial_to_coefficient_map().count(symbolic::Monomial()),
            0);

  symbolic::Polynomial V_init = V;
  symbolic::Polynomial V_sol;
  const Eigen::Vector2d x_star = Eigen::Vector2d::Zero();
  const Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  int r_degree = V_degree - 2;
  const std::vector<int> ellipsoid_c_lagrangian_degrees{};
  ControlLyapunov::SearchOptions search_options;
  search_options.backoff_scale = 0.01;
  search_options.bilinear_iterations = 15;
  search_options.lyap_tiny_coeff_tol = 1E-10;
  const double rho_min = 0.01;
  const double rho_max = 1;
  const double rho_tol = 0.001;
  double rho_sol;
  ControlLyapunov::RhoBisectionOption rho_bisection_option(rho_min, rho_max,
                                                           rho_tol);

  symbolic::Polynomial r_sol;
  VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
  VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;

  dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
             positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
             ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S, r_degree,
             search_options, rho_bisection_option, &V_sol, &lambda0_sol, &l_sol,
             &r_sol, &p_sol, &positivity_eq_lagrangian_sol, &rho_sol,
             &ellipsoid_c_lagrangian_sol);
}

TEST_F(SimpleLinearSystemTest, ControlLyapunovConstructLagrangianProgram) {
  // Test ControlLyapunov::ConstructLagrangianProgram which maximizes rho.
  Vector2<symbolic::Polynomial> f;
  Matrix2<symbolic::Polynomial> G;
  symbolic::Polynomial V;
  bool symmetric_dynamics = true;
  InitializeWithLQR(symmetric_dynamics, &V, &f, &G);
  V *= 0.1;
  VectorX<symbolic::Polynomial> state_constraints(0);
  ControlLyapunov dut(x_, f, G, u_vertices_, state_constraints);
  const symbolic::Polynomial lambda0(1);
  const int d_degree = 1;
  const std::vector<int> l_degrees{2, 2, 2, 2};
  const std::vector<int> p_degrees{};
  const double deriv_eps = 0.01;
  VectorX<symbolic::Polynomial> l;
  std::vector<MatrixX<symbolic::Variable>> l_grams;
  VectorX<symbolic::Polynomial> p;
  symbolic::Variable rho;
  symbolic::Polynomial vdot_sos;
  VectorX<symbolic::Monomial> vdot_monomials;
  MatrixX<symbolic::Variable> vdot_gram;
  auto prog = dut.ConstructLagrangianProgram(
      V, lambda0, d_degree, l_degrees, p_degrees, deriv_eps, &l, &l_grams, &p,
      &rho, &vdot_sos, &vdot_monomials, &vdot_gram);
  const auto result = solvers::Solve(*prog);
  ASSERT_TRUE(result.is_success());
  const auto vdot_gram_sol = result.GetSolution(vdot_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(vdot_gram_sol));
  const double rho_sol = result.GetSolution(rho);
  using std::pow;
  symbolic::Polynomial vdot_sos_expected{
      (1 + lambda0.ToExpression()) *
          pow(x_.cast<symbolic::Expression>().dot(x_), d_degree) *
          (V.ToExpression() - rho_sol),
      symbolic::Variables(x_)};
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x_);
  const RowVectorX<symbolic::Polynomial> dVdx_times_G = dVdx * G;
  for (int i = 0; i < u_vertices_.cols(); ++i) {
    const symbolic::Polynomial l_sol = result.GetSolution(l(i));
    EXPECT_TRUE(math::IsPositiveDefinite(result.GetSolution(l_grams[i])));
    vdot_sos_expected -= l_sol * (dVdx.dot(f) + deriv_eps * V +
                                  dVdx_times_G.dot(u_vertices_.col(i)));
  }
  VectorX<symbolic::Polynomial> p_sol;
  GetPolynomialSolutions(result, p, 0, &p_sol);
  vdot_sos_expected -= p_sol.dot(state_constraints);
  EXPECT_PRED3(symbolic::test::PolynomialEqual, vdot_sos_expected.Expand(),
               result.GetSolution(vdot_sos).Expand(), 1E-5);
  EXPECT_GT(rho_sol, 1);
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
