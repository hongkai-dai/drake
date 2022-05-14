#include "drake/systems/analysis/control_barrier.h"

#include <limits.h>

#include "control_barrier.h"

#include "drake/common/text_logging.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/sos_basis_generator.h"
namespace drake {
namespace systems {
namespace analysis {
namespace {
void AddHdotSosConstraint(
    solvers::MathematicalProgram* prog,
    const std::vector<std::array<symbolic::Polynomial, 2>>& l,
    const symbolic::Polynomial& dhdx_times_Gi, const symbolic::Polynomial& b_i,
    std::vector<MatrixX<symbolic::Variable>>* gram,
    std::vector<VectorX<symbolic::Monomial>>* monomials) {
  const symbolic::Polynomial p0 =
      (1 + l[0][0]) * (dhdx_times_Gi - b_i) - l[0][1] * dhdx_times_Gi;
  const symbolic::Polynomial p1 =
      (1 + l[1][0]) * (-dhdx_times_Gi - b_i) + l[1][1] * dhdx_times_Gi;
  gram->resize(2);
  monomials->resize(2);
  std::tie((*gram)[0], (*monomials)[0]) = prog->AddSosConstraint(
      p0, solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Hd0");
  std::tie((*gram)[1], (*monomials)[1]) = prog->AddSosConstraint(
      p1, solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Hd1");
}

void AddHdotSosConstraint(
    solvers::MathematicalProgram* prog,
    const std::vector<std::vector<std::array<symbolic::Polynomial, 2>>>& l,
    const RowVectorX<symbolic::Polynomial>& dhdx,
    const MatrixX<symbolic::Polynomial>& G,
    const VectorX<symbolic::Polynomial>& b,
    ControlBarrierBoxInputBound::HdotSosConstraintReturn* hdot_sos_constraint) {
  const int nu = static_cast<int>(l.size());
  DRAKE_DEMAND(G.cols() == nu);
  DRAKE_DEMAND(b.rows() == nu);
  for (int i = 0; i < nu; ++i) {
    const symbolic::Polynomial dhdx_times_Gi = dhdx.dot(G.col(i));
    AddHdotSosConstraint(prog, l[i], dhdx_times_Gi, b(i),
                         &(hdot_sos_constraint->grams[i]),
                         &(hdot_sos_constraint->monomials[i]));
  }
}
}  // namespace

ControlBarrierBoxInputBound::ControlBarrierBoxInputBound(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    std::vector<VectorX<symbolic::Polynomial>> unsafe_regions)
    : f_{f},
      G_{G},
      nx_{static_cast<int>(f.rows())},
      nu_{static_cast<int>(G_.cols())},
      x_{x},
      x_set_{x_},
      candidate_safe_states_{candidate_safe_states},
      unsafe_regions_{std::move(unsafe_regions)} {
  DRAKE_DEMAND(G_.rows() == nx_);
  DRAKE_DEMAND(candidate_safe_states_.rows() == nx_);
}

ControlBarrierBoxInputBound::HdotSosConstraintReturn::HdotSosConstraintReturn(
    int nu)
    : monomials{static_cast<size_t>(nu)}, grams{static_cast<size_t>(nu)} {
  for (int i = 0; i < nu; ++i) {
    monomials[i].resize(2);
    grams[i].resize(2);
  }
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrierBoxInputBound::ConstructLagrangianAndBProgram(
    const symbolic::Polynomial& h,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 2>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    std::vector<std::vector<std::array<symbolic::Polynomial, 2>>>* lagrangians,
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 2>>>*
        lagrangian_grams,
    VectorX<symbolic::Polynomial>* b, symbolic::Variable* deriv_eps,
    HdotSosConstraintReturn* hdot_sos_constraint) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  DRAKE_DEMAND(static_cast<int>(b_degrees.size()) == nu_);
  lagrangians->resize(nu_);
  lagrangian_grams->resize(nu_);
  // Add Lagrangian decision variables.
  const int num_hdot_sos = 2;
  for (int i = 0; i < nu_; ++i) {
    (*lagrangians)[i].resize(num_hdot_sos);
    (*lagrangian_grams)[i].resize(num_hdot_sos);
    for (int j = 0; j < num_hdot_sos; ++j) {
      (*lagrangians)[i][j][0] = l_given[i][j];

      DRAKE_DEMAND(lagrangian_degrees[i][j][1] % 2 == 0);
      std::tie((*lagrangians)[i][j][1], (*lagrangian_grams)[i][j][1]) =
          prog->NewSosPolynomial(x_set_, lagrangian_degrees[i][j][1]);
    }
  }

  *deriv_eps = prog->NewContinuousVariables<1>("eps")(0);

  const RowVectorX<symbolic::Polynomial> dhdx = h.Jacobian(x_);
  // Since we will add the constraint -∂h/∂x*f(x) - εh = ∑ᵢ bᵢ(x), we know that
  // the highest degree of b should be at least degree(∂h/∂x*f(x) + εh).
  const symbolic::Polynomial dhdx_times_f = (dhdx * f_)(0);
  if (*std::max_element(b_degrees.begin(), b_degrees.end()) <
      std::max(dhdx_times_f.TotalDegree(), h.TotalDegree())) {
    throw std::invalid_argument("The degree of b is too low.");
  }
  b->resize(nu_);
  for (int i = 0; i < nu_; ++i) {
    (*b)(i) =
        prog->NewFreePolynomial(x_set_, b_degrees[i], "b" + std::to_string(i));
  }
  // Add the constraint -∂h/∂x*f(x) - εh = ∑ᵢ bᵢ(x)
  prog->AddEqualityConstraintBetweenPolynomials(
      b->sum(), -dhdx_times_f - (*deriv_eps) * h);
  // Add the constraint
  // (1+lᵢ₀₀(x))(∂h/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)∂h/∂xGᵢ(x) is sos
  // (1+lᵢ₁₀(x))(−∂h/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)∂h/∂xGᵢ(x) is sos
  AddHdotSosConstraint(prog.get(), *lagrangians, dhdx, G_, *b,
                       hdot_sos_constraint);
  return prog;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
