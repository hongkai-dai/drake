#include "drake/systems/analysis/clf_cbf_utils.h"

#include <limits>

#include <gtest/gtest.h>

#include "drake/common/temp_directory.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/test/quadrotor2d.h"

namespace drake {
namespace systems {
namespace analysis {

const double kInf = std::numeric_limits<double>::infinity();

// Check if the ellipsoid {x | (x-x*)ᵀS(x-x*) <= d} in the
// sub-level set {x | f(x) <= 0}
void CheckEllipsoidInSublevelSet(
    const Eigen::Ref<const VectorX<symbolic::Variable>> x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, double d,
    const symbolic::Polynomial& f) {
  // Check if any point within the ellipsoid also satisfies V(x)<=1.
  // A point on the boundary of ellipsoid (x−x*)ᵀS(x−x*)=d
  // can be writeen as x=√d*L⁻ᵀ*u+x*
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
        std::sqrt(d) * qr_solver.solve(u_samples.col(i)) + x_star;
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
  double d_sol;
  symbolic::Polynomial s_sol;
  MaximizeInnerEllipsoidSize(x, x_star, S, V - 1, t, s_degree,
                             solvers::MosekSolver::id(), std::nullopt,
                             backoff_scale, &d_sol, &s_sol);
  const double tol = 1E-5;
  EXPECT_NEAR(d_sol, 0.5, tol);

  CheckEllipsoidInSublevelSet(x, x_star, S, d_sol, V - 1);
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
    // max d
    // s.t (1+t(x))((x-x*)ᵀS(x-x*)-d) - s(x)(V(x)-1) is sos
    //     s(x) is sos
    const symbolic::Polynomial t(0);
    const int s_degree = 2;
    const double backoff_scale = 0.05;
    double d_sol;
    symbolic::Polynomial s_sol;
    solvers::SolverOptions solver_options;
    solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 0);
    // I am really surprised that this SOS finds a solution with d > 0. AFAIK,
    // t(x) is constant, hence (1+t(x))((x-x*)ᵀS(x-x*)-d) is a degree 2
    // polynomial, while -s(x)*(V(x)-1) has much higher degree (>6) with
    // negative leading terms. The resulting polynomial cannot be sos.
    MaximizeInnerEllipsoidSize(x, x_star, S, V - 1, t, s_degree,
                               solvers::MosekSolver::id(), solver_options,
                               backoff_scale, &d_sol, &s_sol);
    CheckEllipsoidInSublevelSet(x, x_star, S, d_sol, V - 1);
  }

  {
    // Test the bisection approach
    const int r_degree = 2;
    const double size_max = 4;
    const double size_min = 0.2;
    const double size_tol = 0.1;
    double d_sol_wo_c;
    symbolic::Polynomial r_sol;
    VectorX<symbolic::Polynomial> c_lagrangian_sol;
    MaximizeInnerEllipsoidSize(
        x, x_star, S, V - 1, std::nullopt, r_degree, std::nullopt, size_max,
        size_min, solvers::MosekSolver::id(), std::nullopt, size_tol,
        &d_sol_wo_c, &r_sol, &c_lagrangian_sol);
    CheckEllipsoidInSublevelSet(x, x_star, S, d_sol_wo_c, V - 1);
    // Check if r_sol is sos.
    Eigen::Matrix2Xd x_check = Eigen::Matrix2Xd::Random(2, 100);
    const symbolic::Polynomial sos_cond =
        1 - V + r_sol * internal::EllipsoidPolynomial(x, x_star, S, d_sol_wo_c);
    for (int i = 0; i < x_check.cols(); ++i) {
      symbolic::Environment env;
      env.insert(x, x_check.col(i));
      EXPECT_GE(r_sol.Evaluate(env), 0.);
      EXPECT_GE(sos_cond.Evaluate(env), 0.);
    }

    // Consider the algrbraic set x(0) + x(1) == 1
    const Vector1<symbolic::Polynomial> c(
        symbolic::Polynomial(x(0) + x(1) - 1));
    const std::vector<int> c_lagrangian_degrees{{3}};
    double d_sol_w_c;
    MaximizeInnerEllipsoidSize(x, x_star, S, V - 1, c, r_degree,
                               c_lagrangian_degrees, size_max, size_min,
                               solvers::MosekSolver::id(), std::nullopt,
                               size_tol, &d_sol_w_c, &r_sol, &c_lagrangian_sol);
    EXPECT_GT(d_sol_w_c, d_sol_wo_c);
    // Now sample many points on the plane x(0) + x(1) == 1, if they are within
    // the ellipsoid, then they should satisfy V(x) <= 1.
    const Eigen::VectorXd x0_samples = Eigen::VectorXd::LinSpaced(500, -10, 10);
    Eigen::Matrix2Xd x_samples(2, x0_samples.rows());
    x_samples.row(0) = x0_samples.transpose();
    x_samples.row(1) =
        Eigen::RowVectorXd::Ones(x0_samples.rows()) - x_samples.row(0);
    const Eigen::VectorXd V_samples = V.EvaluateIndeterminates(x, x_samples);
    for (int i = 0; i < x0_samples.rows(); ++i) {
      const Eigen::Vector2d x_sample = x_samples.col(i);
      if ((x_sample - x_star).dot(S * (x_sample - x_star)) <= d_sol_w_c) {
        EXPECT_LE(V_samples(i), 1);
      }
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
  const int V_degree = 2;
  const int positivity_eps = 0.1;
  const int d = 1;
  VectorX<symbolic::Polynomial> state_constraints(0);
  std::vector<int> c_lagrangian_degrees{};
  VectorX<symbolic::Polynomial> c_lagrangian;

  auto prog = FindCandidateLyapunov(x, V_degree, positivity_eps, d,
                                    state_constraints, c_lagrangian_degrees,
                                    x_val, xdot_val, &V, &c_lagrangian);
  const auto result = solvers::Solve(*prog);
  ASSERT_TRUE(result.is_success());
  const auto V_sol = result.GetSolution(V);
  const Eigen::VectorXd V_val = V_sol.EvaluateIndeterminates(x, x_val);
  EXPECT_TRUE((V_val.array() >=
               positivity_eps *
                   (x_val.array() * x_val.array()).colwise().sum().transpose())
                  .all());
  EXPECT_TRUE((V_val.array() <= 1 + 1E-5).all());
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

GTEST_TEST(FindCandidateRegionalLyapunov, Test) {
  // Find a candidate Lyapunov for 2D quadrotor with trigonometric dynamics and
  // LQR controller.
  QuadrotorPlant<double> quadrotor2d;
  Eigen::Matrix<double, 7, 1> lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 1, 10, 10, 10;
  const Eigen::MatrixXd lqr_Q = lqr_Q_diag.asDiagonal();
  const Eigen::Matrix2d lqr_R = 10 * Eigen::Matrix2d::Identity();
  const auto lqr_result = SynthesizeTrigLqr(lqr_Q, lqr_R);
  Eigen::Matrix<symbolic::Variable, 7, 1> x;
  for (int i = 0; i < 7; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const double thrust_equilibrium = EquilibriumThrust(quadrotor2d);
  const Vector2<symbolic::Expression> u_lqr =
      -lqr_result.K * x +
      Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium);
  const Eigen::Matrix<symbolic::Expression, 7, 1> dynamics_expr =
      TrigDynamics<symbolic::Expression>(quadrotor2d,
                                         x.cast<symbolic::Expression>(), u_lqr);
  Eigen::Matrix<symbolic::Polynomial, 7, 1> dynamics;
  for (int i = 0; i < 7; ++i) {
    dynamics(i) = symbolic::Polynomial(dynamics_expr(i));
  }

  const int V_degree = 2;
  const double positivity_eps = 0.01;
  const int d = 1;
  const double deriv_eps = 0.0001;
  const Vector1<symbolic::Polynomial> state_eq_constraints(
      StateEqConstraint(x));
  const std::vector<int> positivity_ceq_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_ceq_lagrangian_degrees{2};
  const Vector1<symbolic::Polynomial> state_ineq_constraints(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x) - 0.0001));
  const std::vector<int> positivity_cin_lagrangian_degrees{{V_degree - 2}};
  const std::vector<int> derivative_cin_lagrangian_degrees{{2}};
  symbolic::Polynomial V;
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian;
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian;
  VectorX<symbolic::Polynomial> derivative_cin_lagrangian;
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian;
  symbolic::Polynomial positivity_sos_condition;
  symbolic::Polynomial derivative_sos_condition;
  auto prog = FindCandidateRegionalLyapunov(
      x, dynamics, std::nullopt /*dynamics_denominator */, V_degree,
      positivity_eps, d, deriv_eps, state_eq_constraints,
      positivity_ceq_lagrangian_degrees, derivative_ceq_lagrangian_degrees,
      state_ineq_constraints, positivity_cin_lagrangian_degrees,
      derivative_cin_lagrangian_degrees, &V, &positivity_cin_lagrangian,
      &positivity_ceq_lagrangian, &derivative_cin_lagrangian,
      &derivative_ceq_lagrangian, &positivity_sos_condition,
      &derivative_sos_condition);
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result = solvers::Solve(*prog, std::nullopt, solver_options);
  ASSERT_TRUE(result.is_success());
  const symbolic::Polynomial V_sol = result.GetSolution(V);
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian_sol;
  GetPolynomialSolutions(result, positivity_cin_lagrangian, 0,
                         &positivity_cin_lagrangian_sol);
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian_sol;
  GetPolynomialSolutions(result, positivity_ceq_lagrangian, 0,
                         &positivity_ceq_lagrangian_sol);
  const symbolic::Polynomial positivity_sos_condition_expected =
      V_sol -
      positivity_eps *
          symbolic::Polynomial(pow(x.cast<symbolic::Expression>().dot(x), d)) +
      positivity_cin_lagrangian_sol.dot(state_ineq_constraints) -
      positivity_ceq_lagrangian_sol.dot(state_eq_constraints);
  EXPECT_PRED2(
      symbolic::test::PolyEqual,
      result.GetSolution(positivity_sos_condition)
          .RemoveTermsWithSmallCoefficients(1E-5),
      positivity_sos_condition_expected.RemoveTermsWithSmallCoefficients(1E-5));

  VectorX<symbolic::Polynomial> derivative_cin_lagrangian_sol;
  GetPolynomialSolutions(result, derivative_cin_lagrangian, 0,
                         &derivative_cin_lagrangian_sol);
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian_sol;
  GetPolynomialSolutions(result, derivative_ceq_lagrangian, 0,
                         &derivative_ceq_lagrangian_sol);
  const symbolic::Polynomial derivative_sos_condition_expected =
      -V_sol.Jacobian(x).dot(dynamics) - deriv_eps * V_sol +
      derivative_cin_lagrangian_sol.dot(state_ineq_constraints) -
      derivative_ceq_lagrangian_sol.dot(state_eq_constraints);
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               result.GetSolution(derivative_sos_condition),
               derivative_sos_condition_expected, 1E-5);
}

GTEST_TEST(Meshgrid, Test) {
  std::vector<Eigen::VectorXd> x;
  x.push_back(Eigen::Vector2d(1, 2));
  x.push_back(Eigen::Vector3d(3, 4, 5));
  x.push_back(Eigen::Vector4d(6, 7, 8, 9));
  const auto mesh = Meshgrid(x);
  EXPECT_EQ(mesh.rows(), 3);
  EXPECT_EQ(mesh.cols(), 2 * 3 * 4);
  Eigen::MatrixXd mesh_expected(3, 2 * 3 * 4);
  int pt_count = 0;
  for (int i = 0; i < x[0].rows(); ++i) {
    for (int j = 0; j < x[1].rows(); ++j) {
      for (int k = 0; k < x[2].rows(); ++k) {
        mesh_expected.col(pt_count++) =
            Eigen::Vector3d(x[0](i), x[1](j), x[2](k));
      }
    }
  }
  EXPECT_TRUE(CompareMatrices(mesh, mesh_expected));
}

GTEST_TEST(SaveLoadPolynomial, Test) {
  const symbolic::Variable x0("x0");
  const symbolic::Variable x1("x1");
  const symbolic::Variables x_set{{x0, x1}};

  const std::string file = temp_directory() + "polynomial.txt";

  const symbolic::Polynomial p0{x0 * x1};
  Save(p0, file);
  EXPECT_PRED2(symbolic::test::PolyEqual, p0, Load(x_set, file));

  const symbolic::Polynomial p1{2 * x0 * x0 + 3 * x1};
  Save(p1, file);
  EXPECT_PRED2(symbolic::test::PolyEqual, p1, Load(x_set, file));

  const symbolic::Polynomial p2{3 * x0 + 4 * x1 + 2 * x0 * x1 +
                                3 * x0 * x1 * x1 + 1};
  Save(p2, file);
  EXPECT_PRED2(symbolic::test::PolyEqual, p2, Load(x_set, file));
}

GTEST_TEST(NewFreePolynomialPassOrigin, Test) {
  Vector3<symbolic::Variable> x;
  for (int i = 0; i < 3; ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(x);
  const symbolic::Variables x_set{x};
  const symbolic::internal::DegreeType degree_type{
      symbolic::internal::DegreeType::kAny};

  // Test with no_linear_term_variables being empty.
  const auto p1 =
      NewFreePolynomialPassOrigin(&prog, x_set, 2, "p", degree_type, {});
  EXPECT_EQ(p1.monomial_to_coefficient_map().size(), 9);
  for (const auto& [monomial, coeff] : p1.monomial_to_coefficient_map()) {
    EXPECT_LE(monomial.total_degree(), 2);
    EXPECT_GT(monomial.total_degree(), 0);
  }

  // Test with no_linear_term_variables being {x2}.
  const auto p2 = NewFreePolynomialPassOrigin(&prog, x_set, 2, "p", degree_type,
                                              symbolic::Variables({x(2)}));
  EXPECT_EQ(p2.monomial_to_coefficient_map().size(), 8);
  for (const auto& [monomial, coeff] : p2.monomial_to_coefficient_map()) {
    EXPECT_LE(monomial.total_degree(), 2);
    EXPECT_GT(monomial.total_degree(), 0);
    EXPECT_NE(monomial, symbolic::Monomial(x(2)));
  }

  // Test with no_linear_term_variables being {x0, x2}.
  const auto p3 = NewFreePolynomialPassOrigin(
      &prog, x_set, 2, "p", degree_type, symbolic::Variables({x(0), x(2)}));
  EXPECT_EQ(p3.monomial_to_coefficient_map().size(), 7);
  for (const auto& [monomial, coeff] : p3.monomial_to_coefficient_map()) {
    EXPECT_LE(monomial.total_degree(), 2);
    EXPECT_GT(monomial.total_degree(), 0);
    EXPECT_NE(monomial, symbolic::Monomial(x(2)));
    EXPECT_NE(monomial, symbolic::Monomial(x(0)));
  }
}

GTEST_TEST(FindNoLinearTermVariables, Test) {
  Vector4<symbolic::Variable> x;
  for (int i = 0; i < x.rows(); ++i) {
    x(i) = symbolic::Variable("x" + std::to_string(i));
  }
  const symbolic::Variables x_set(x);

  const Vector2<symbolic::Polynomial> p1(
      symbolic::Polynomial(x(0) * x(0) + 2 * x(1)),
      symbolic::Polynomial(x(0) * x(2) + x(3)));
  const symbolic::Variables no_linear_term_variables1 =
      FindNoLinearTermVariables(x_set, p1);
  EXPECT_EQ(no_linear_term_variables1.size(), 2);
  EXPECT_TRUE(no_linear_term_variables1.include(x(0)));
  EXPECT_TRUE(no_linear_term_variables1.include(x(2)));

  // p2 doesn't contain all variables.
  const Vector2<symbolic::Polynomial> p2(
      symbolic::Polynomial(x(2) * x(2) + x(3)), symbolic::Polynomial(x(0)));
  const symbolic::Variables no_linear_term_variables2 =
      FindNoLinearTermVariables(x_set, p2);
  EXPECT_EQ(no_linear_term_variables2.size(), 2);
  EXPECT_TRUE(no_linear_term_variables2.include(x(1)));
  EXPECT_TRUE(no_linear_term_variables2.include(x(2)));
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
