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
  const VectorX<symbolic::Polynomial> state_constraints(0);
  const double beta_minus = -1;
  const double beta_plus = 2;
  const ControlBarrier dut(f_, G_, std::nullopt, x_, beta_minus, beta_plus,
                           unsafe_regions, u_vertices, state_constraints);

  const symbolic::Polynomial h_init(1 - x_(0) * x_(0) - x_(1) * x_(1));
  const double deriv_eps = 0.1;
  const int lambda0_degree = 2;
  const int lambda1_degree = 2;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> state_constraints_lagrangian_degrees{};
  const std::optional<int> hdot_a_degree = std::nullopt;
  auto lagrangian_ret = dut.ConstructLagrangianProgram(
      h_init, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      state_constraints_lagrangian_degrees, hdot_a_degree);
  auto result_lagrangian = solvers::Solve(*(lagrangian_ret.prog));
  ASSERT_TRUE(result_lagrangian.is_success());
  const Eigen::MatrixXd lambda0_gram_sol =
      result_lagrangian.GetSolution(lagrangian_ret.lambda0_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(lambda0_gram_sol));
  symbolic::Polynomial lambda0_sol =
      result_lagrangian.GetSolution(lagrangian_ret.lambda0);
  symbolic::Polynomial lambda1_sol =
      result_lagrangian.GetSolution(*(lagrangian_ret.lambda1));
  symbolic::Polynomial hdot_sos_expected =
      (1 + lambda0_sol) * (beta_minus - h_init) -
      lambda1_sol * (beta_plus - h_init);
  VectorX<symbolic::Polynomial> l_sol(u_vertices.cols());
  RowVectorX<symbolic::Polynomial> dhdx = h_init.Jacobian(x_);
  for (int i = 0; i < u_vertices.cols(); ++i) {
    EXPECT_TRUE(math::IsPositiveDefinite(
        result_lagrangian.GetSolution(lagrangian_ret.l_grams[i])));
    l_sol(i) = result_lagrangian.GetSolution(lagrangian_ret.l[i]);
    hdot_sos_expected -= l_sol[i] * (-deriv_eps * h_init - dhdx.dot(f_) -
                                     dhdx.dot(G_ * u_vertices.col(i)));
  }
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               result_lagrangian.GetSolution(lagrangian_ret.hdot_sos).Expand(),
               hdot_sos_expected.Expand(), 1E-5);
  const Eigen::MatrixXd hdot_gram_sol =
      result_lagrangian.GetSolution(lagrangian_ret.hdot_gram);
  EXPECT_TRUE(math::IsPositiveDefinite(hdot_gram_sol));
  EXPECT_PRED3(symbolic::test::PolynomialEqual, hdot_sos_expected.Expand(),
               lagrangian_ret.hdot_monomials.dot(hdot_gram_sol *
                                                 lagrangian_ret.hdot_monomials),
               1E-5);

  const int t_degree = 0;
  const std::vector<int> s_degrees = {0, 0};
  const std::vector<int> unsafe_state_constraints_lagrangian_degrees{};
  const std::optional<int> unsafe_a_degrees = std::nullopt;
  auto unsafe_ret = dut.ConstructUnsafeRegionProgram(
      h_init, 0, t_degree, s_degrees,
      unsafe_state_constraints_lagrangian_degrees, unsafe_a_degrees);
  const auto result_unsafe = solvers::Solve(*(unsafe_ret.prog));
  ASSERT_TRUE(result_unsafe.is_success());
  const symbolic::Polynomial t_sol = result_unsafe.GetSolution(unsafe_ret.t);
  EXPECT_TRUE(
      math::IsPositiveDefinite(result_unsafe.GetSolution(unsafe_ret.t_gram)));
  VectorX<symbolic::Polynomial> s_sol(unsafe_ret.s.rows());
  EXPECT_EQ(unsafe_ret.s.rows(), unsafe_regions[0].rows());
  for (int i = 0; i < unsafe_ret.s.rows(); ++i) {
    s_sol(i) = result_unsafe.GetSolution(unsafe_ret.s(i));
    EXPECT_TRUE(math::IsPositiveDefinite(
        result_unsafe.GetSolution(unsafe_ret.s_grams[i])));
  }
  symbolic::Polynomial unsafe_sos_poly_expected =
      (1 + t_sol) * -h_init + s_sol.dot(unsafe_regions[0]);
  EXPECT_PRED3(symbolic::test::PolynomialEqual,
               unsafe_sos_poly_expected.Expand(),
               result_unsafe.GetSolution(unsafe_ret.sos_poly).Expand(), 1E-5);
  EXPECT_TRUE(math::IsPositiveDefinite(
      result_unsafe.GetSolution(unsafe_ret.sos_poly_gram)));

  // Now search for barrier given Lagrangians.
  Eigen::MatrixXd verified_safe_states;
  Eigen::MatrixXd unverified_candidate_states;
  SplitCandidateStates(h_init, x_, candidate_safe_states, &verified_safe_states,
                       &unverified_candidate_states);

  const std::vector<int> hdot_state_constraints_lagrangian_degrees{};
  const int h_degree = 2;
  lambda0_sol = lambda0_sol.RemoveTermsWithSmallCoefficients(1e-10);
  const double eps = 1E-3;
  auto barrier_ret = dut.ConstructBarrierProgram(
      lambda0_sol, lambda1_sol, l_sol,
      hdot_state_constraints_lagrangian_degrees, hdot_a_degree, {t_sol},
      {unsafe_state_constraints_lagrangian_degrees}, h_degree, deriv_eps,
      {s_degrees}, {unsafe_a_degrees});
  RemoveTinyCoeff(barrier_ret.prog.get(), 1E-10);
  auto result_barrier = solvers::Solve(*(barrier_ret.prog));
  ASSERT_TRUE(result_barrier.is_success());
  auto h_sol = result_barrier.GetSolution(barrier_ret.h);
  // Check sos for unsafe regions.
  EXPECT_EQ(barrier_ret.s.size(), unsafe_regions.size());
  for (int i = 0; i < static_cast<int>(barrier_ret.s.size()); ++i) {
    EXPECT_EQ(barrier_ret.s[i].rows(), unsafe_regions[i].rows());
    s_sol.resize(barrier_ret.s[i].rows());
    for (int j = 0; j < barrier_ret.s[i].rows(); ++j) {
      EXPECT_TRUE(math::IsPositiveDefinite(
          result_barrier.GetSolution(barrier_ret.s_grams[i][j])));
      s_sol[j] = result_barrier.GetSolution(barrier_ret.s[i](j));
    }
    unsafe_sos_poly_expected =
        (1 + t_sol) * -h_sol + s_sol.dot(unsafe_regions[i]);
    EXPECT_PRED3(
        symbolic::test::PolynomialEqual, unsafe_sos_poly_expected.Expand(),
        result_barrier.GetSolution(barrier_ret.unsafe_sos_polys[i]).Expand(),
        1E-5);
    EXPECT_TRUE(math::IsPositiveDefinite(
        result_barrier.GetSolution(barrier_ret.unsafe_sos_poly_grams[i])));
  }
  {
    // Add cost to maximize min(h(x), 0) on sampled states.
    // Check h_sol on verified_safe_states;
    auto prog_cost1 = barrier_ret.prog->Clone();
    dut.AddBarrierProgramCost(prog_cost1.get(), barrier_ret.h,
                              verified_safe_states, unverified_candidate_states,
                              eps, false /* minimize_max */);
    result_barrier = solvers::Solve(*prog_cost1);
    h_sol = result_barrier.GetSolution(barrier_ret.h);
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
  }
  {
    // Add cost to maximize min(h(x)) within some ellipsoids.
    auto prog_cost2 = barrier_ret.prog->Clone();
    std::vector<ControlBarrier::Ellipsoid> ellipsoids;
    std::vector<ControlBarrier::EllipsoidBisectionOption>
        ellipsoid_bisection_options;
    ellipsoids.emplace_back(Eigen::Vector2d(0.1, 0.5),
                            Eigen::Matrix2d::Identity(), 0.5, 0,
                            std::vector<int>());
    ellipsoid_bisection_options.emplace_back(0.1, 1, 0.1);
    ellipsoids.emplace_back(Eigen::Vector2d(0.1, -0.5),
                            Eigen::Matrix2d::Identity(), 0.3, 0,
                            std::vector<int>());
    ellipsoid_bisection_options.emplace_back(0.1, 1, 0.1);
    std::vector<symbolic::Polynomial> r;
    VectorX<symbolic::Variable> rho;
    std::vector<VectorX<symbolic::Polynomial>>
        ellipsoids_state_constraints_lagrangian;
    dut.AddBarrierProgramCost(prog_cost2.get(), barrier_ret.h, ellipsoids, &r,
                              &rho, &ellipsoids_state_constraints_lagrangian);
    Eigen::MatrixXd h_monomial_vals;
    VectorX<symbolic::Variable> h_coeff_vars;
    Eigen::Vector2d x_anchor(0.1, 0.2);
    EvaluatePolynomial(barrier_ret.h, x_, x_anchor, &h_monomial_vals,
                       &h_coeff_vars);
    prog_cost2->AddLinearConstraint(h_monomial_vals.row(0), -kInf, 100,
                                    h_coeff_vars);
    result_barrier = solvers::Solve(*prog_cost2);
    ASSERT_TRUE(result_barrier.is_success());
    h_sol = result_barrier.GetSolution(barrier_ret.h);
    const auto rho_sol = result_barrier.GetSolution(rho);
    for (int i = 0; i < static_cast<int>(ellipsoids.size()); ++i) {
      CheckEllipsoidInSublevelSet(x_, ellipsoids[i].c, ellipsoids[i].S,
                                  ellipsoids[i].d, rho_sol(i) - h_sol);
    }
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
  const VectorX<symbolic::Polynomial> state_constraints(0);
  const double beta_minus = -1;
  const double beta_plus = 10;
  const ControlBarrier dut(f_, G_, std::nullopt, x_, beta_minus, beta_plus,
                           unsafe_regions, u_vertices, state_constraints);

  const symbolic::Polynomial h_init(1 - x_(0) * x_(0) - x_(1) * x_(1));
  const int h_degree = 2;
  const double deriv_eps = 0.1;
  const int lambda0_degree = 2;
  const int lambda1_degree = 2;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_state_constraints_lagrangian_degrees{};
  const std::vector<int> t_degree = {0};
  const std::vector<std::vector<int>> s_degrees = {{0, 0}};
  const std::vector<std::vector<int>>
      unsafe_state_constraints_lagrangian_degrees = {{}};

  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  std::vector<std::variant<ControlBarrier::EllipsoidBisectionOption,
                           ControlBarrier::EllipsoidMaximizeOption>>
      ellipsoid_options;
  ellipsoids.emplace_back(Eigen::Vector2d(0.1, 0.2),
                          Eigen::Matrix2d::Identity(), 0, 0,
                          std::vector<int>());
  ellipsoid_options.push_back(
      ControlBarrier::EllipsoidBisectionOption(0, 2, 0.01));
  ellipsoids.emplace_back(Eigen::Vector2d(0.5, -0.9),
                          Eigen::Matrix2d::Identity(), 0, 0,
                          std::vector<int>());
  ellipsoid_options.push_back(
      ControlBarrier::EllipsoidBisectionOption(0, 2, 0.01));
  ellipsoids.emplace_back(Eigen::Vector2d(0.5, -1.9),
                          Eigen::Matrix2d::Identity(), 0, 0,
                          std::vector<int>());
  ellipsoid_options.push_back(
      ControlBarrier::EllipsoidBisectionOption(0, 2, 0.01));
  const Eigen::Vector2d x_anchor(0.3, 0.5);
  const double h_x_anchor_max = h_init.EvaluateIndeterminates(x_, x_anchor)(0);

  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lsol_tiny_coeff_tol = 1E-8;
  search_options.barrier_tiny_coeff_tol = 1E-10;
  search_options.barrier_step_solver_options = solvers::SolverOptions();
  search_options.barrier_step_solver = solvers::CsdpSolver::id();
  // search_options.barrier_step_solver_options->SetOption(
  //     solvers::CommonSolverOption::kPrintToConsole, 1);

  symbolic::Polynomial h_sol;
  symbolic::Polynomial lambda0_sol;
  symbolic::Polynomial lambda1_sol;
  VectorX<symbolic::Polynomial> l_sol;
  VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  std::vector<VectorX<symbolic::Polynomial>>
      unsafe_state_constraints_lagrangian;

  const auto search_ret = dut.Search(
      h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      hdot_state_constraints_lagrangian_degrees, t_degree, s_degrees,
      unsafe_state_constraints_lagrangian_degrees, x_anchor, h_x_anchor_max,
      search_options, &ellipsoids, &ellipsoid_options);
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
