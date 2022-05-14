#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"

namespace drake {
namespace systems {
namespace analysis {

/**
 * For a control-affine system ẋ = f(x) + G(x)u subject to input limit -1 <= u
 * <= 1 (the entries in f(x) and G(x) are polynomials of x), we synthesize a
 * control barrier function h(x).
 * h(x) satisfies the condition
 * ḣ(x) = maxᵤ ∂h/∂x*f(x) + ∂h/∂x*G(x)u ≥ −ε h(x) ∀x
 * This is equivalent to
 * ∂h/∂x*f(x) + |∂h/∂x*G(x)|₁ ≥ −ε h(x)
 */
class ControlBarrierBoxInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlBarrierBoxInputBound);
  /**
   * @param candidate_safe_states Each column is a candidate safe state.
   * @param unsafe_regions unsafe_regions[i]<=0 describes the i'th unsafe
   * region.
   */
  ControlBarrierBoxInputBound(
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
      std::vector<VectorX<symbolic::Polynomial>> unsafe_regions);

  struct HdotSosConstraintReturn {
    HdotSosConstraintReturn(int nu);

    std::vector<std::vector<VectorX<symbolic::Monomial>>> monomials;
    std::vector<std::vector<MatrixX<symbolic::Variable>>> grams;
  };

  /**
   * Given the control barrier function h(x) and Lagrangian muliplier lᵢⱼ₀(x),
   * search for b(x) and Lagrangian multiplier lᵢⱼ₁(x) satisfying the constraint
   * <pre>
   * ∑ᵢ bᵢ(x) = −∂h/∂x*f(x)−εh(x)
   * (1+lᵢ₀₀(x))(∂h/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)∂h/∂xGᵢ(x) is sos
   * (1+lᵢ₁₀(x))(−∂h/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)∂h/∂xGᵢ(x) is sos
   * </pre>
   * @param deriv_eps ε in the documentation above.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianAndBProgram(
      const symbolic::Polynomial& h,
      const std::vector<std::vector<symbolic::Polynomial>>& l_given,
      const std::vector<std::vector<std::array<int, 2>>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      std::vector<std::vector<std::array<symbolic::Polynomial, 2>>>*
          lagrangians,
      std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 2>>>*
          lagrangian_grams,
      VectorX<symbolic::Polynomial>* b, symbolic::Variable* deriv_eps,
      HdotSosConstraintReturn* hdot_sos_constraint) const;

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  int nx_;
  int nu_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  Eigen::MatrixXd candidate_safe_states_;
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions_;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
