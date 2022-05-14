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
