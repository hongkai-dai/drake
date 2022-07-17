#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * Find an upper bound of the value function.
 * where the dynamics is ẋ = f(x)/d(x) + G(x)/d(x) * u
 * J is an upper bound of the value function if it satisfies
 * ∀x, ∃u, such that l(x) + 0.5*uᵀRu + ∂J/∂x(f(x)+G(x)u)/d(x) ≤ 0
 * We do this iteratively as
 * 1. Start with some u = π(x), solve the problem
 *    min J
 *    s.t −(l(x) + 0.5π(x)ᵀRπ(x) + ∂J/∂x*(f(x)+G(x)π(x))/d(x)) is sos
 * 2. Take the solution from step 1, compute π(x) = −R⁻¹*∂J/∂x*G(x)*n(x)
 */
class HjbUpper {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HjbUpper)

  HjbUpper(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
           symbolic::Polynomial l, const Eigen::Ref<const Eigen::MatrixXd>& R,
           const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
           const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
           const std::optional<symbolic::Polynomial>& dynamics_denominator,
           const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
               state_eq_constraints);

  /**
   * Construct a program to search for J.
   *
   * min ∑ᵢ J(xⁱ)
   * s.t −((l(x) + 0.5π(x)ᵀRπ(x))*n(x) + ∂J/∂x(f(x)+G(x)π(x))/n(x))
   *        + r(x) * cin(x) >= 0
   *     r(x) is sos
   *     J(0) = 0
   *
   * We suppose π(x) = πₙ(x)/d(x)
   * Then the first constraint is written as
   * -(l(x)d(x)² + 0.5πₙᵀ(x)Rπₙ(x) + ∂J/∂x(f(x)d(x) + G(x)πₙ(x)))
   *       + r(x)*cin(x) is sos
   *
   * where cin(x) <= 0 defines the region of interest on state x.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructJupperProgram(
      int J_degree, const Eigen::Ref<const Eigen::MatrixXd>& x_samples,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& policy_numerator,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& cin,
      const std::vector<int>& r_degrees,
      const std::vector<int>& state_constraints_lagrangian_degrees,
      symbolic::Polynomial* J, VectorX<symbolic::Polynomial>* r,
      VectorX<symbolic::Polynomial>* state_constraints_lagrangian) const;

  /**
   * Compute the numerator of the policy π(x) = −R⁻¹*∂J/∂x*G(x)/n(x)
   */
  VectorX<symbolic::Polynomial> ComputePolicyNumerator(
      const symbolic::Polynomial& J) const;

 private:
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  symbolic::Polynomial l_;
  Eigen::MatrixXd R_;
  Eigen::MatrixXd R_inv_;
  Eigen::LLT<Eigen::MatrixXd> llt_R_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  std::optional<symbolic::Polynomial> dynamics_denominator_;
  VectorX<symbolic::Polynomial> state_eq_constraints_;
};

class HjbController : public LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HjbController)

  HjbController(
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& policy_numerator,
      symbolic::Polynomial policy_denominator);

  ~HjbController() {}

  const OutputPort<double>& control_output_port() const {
    return this->get_output_port(control_output_index_);
  }

  const InputPort<double>& x_input_port() const {
    return this->get_input_port(x_input_index_);
  }

  void CalcControl(const Context<double>& context,
                   BasicVector<double>* output) const;

 private:
  VectorX<symbolic::Variable> x_;
  VectorX<symbolic::Polynomial> policy_numerator_;
  symbolic::Polynomial policy_denominator_;
  OutputPortIndex control_output_index_;
  InputPortIndex x_input_index_;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
