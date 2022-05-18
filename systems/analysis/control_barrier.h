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
 * For a control affine system with dynamics ẋ = f(x) + G(x)u where both f(x)
 * and G(x) are polynomials of x. The system has bounds on the input u as u∈P,
 * where P is a bounded polytope. The unsafe region is given as set x ∈ 𝒳ᵤ. we
 * want to find a control barrier function for this system as h(x). The control
 * barrier function should satisfy the condition
 *
 *     h(x) <= 0 ∀ x ∈ 𝒳ᵤ                                 (1)
 *     ∀ x satisfying h(x) > −1, ∃u ∈ P, s.t. ḣ > −ε h    (2)
 *
 * Suppose 𝒳ᵤ is defined as the union of polynomial sub-level sets, namely 𝒳ᵤ =
 * 𝒳ᵤ¹ ∪ ... ∪ 𝒳ᵤᵐ, where each 𝒳ᵤʲ = { x | pⱼ(x)≤ 0} where pⱼ(x) is a vector of
 * polynomials. Condition (1) can be imposed through the following sos condition
 * <pre>
 * -h(x) + sⱼ(x)ᵀpⱼ(x) is sos
 * sⱼ is sos.
 * </pre>
 *
 * Condition (2) is the same as ḣ ≤ −εh ⇒ h(x)≤−1
 * We will verify this condition via sum-of-squares optimization, namely
 * <pre>
 * (1+λ₀(x))(−1 − h(x)) −∑ᵢ lᵢ(x)(−εh − ∂h/∂xf(x)−∂h/∂xG(x)uⁱ) is sos
 * λ₀(x), lᵢ(x) is sos
 * </pre>
 */
class SearchControlBarrier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SearchControlBarrier)

  SearchControlBarrier(
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
      std::vector<VectorX<symbolic::Polynomial>> unsafe_regions,
      const Eigen::Ref<const Eigen::MatrixXd>& u_vertices);

  /**
   * A helper function to add the constraint
   * (1+λ₀(x))(-1-h(x)) − ∑ᵢ lᵢ(x)*(-∂h/∂x*f(x)-ε*h - ∂h/∂x*G(x)*uᵢ) is sos.
   * @param[out] monomials The monomial basis of this sos constraint.
   * @param[out] gram The Gram matrix of this sos constraint.
   */
  void AddControlBarrierConstraint(solvers::MathematicalProgram* prog,
                                   const symbolic::Polynomial& lambda0,
                                   const VectorX<symbolic::Polynomial>& l,
                                   const symbolic::Polynomial& h,
                                   double deriv_eps,
                                   symbolic::Polynomial* hdot_poly,
                                   VectorX<symbolic::Monomial>* monomials,
                                   MatrixX<symbolic::Variable>* gram) const;

  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianProgram(
      const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
      const std::vector<int>& l_degrees, symbolic::Polynomial* lambda0,
      MatrixX<symbolic::Variable>* lambda0_gram,
      VectorX<symbolic::Polynomial>* l,
      std::vector<MatrixX<symbolic::Variable>>* l_grams,
      symbolic::Polynomial* hdot_sos,
      VectorX<symbolic::Monomial>* hdot_monomials,
      MatrixX<symbolic::Variable>* hdot_gram) const;

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  int nx_;
  int nu_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  Eigen::MatrixXd candidate_safe_states_;
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions_;
  Eigen::MatrixXd u_vertices_;
};

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
