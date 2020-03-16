#pragma once

#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * For a control affine system with dynamics ẋ = f(x) + G(x)u where both f(x)
 * and G(x) are polynomials of x. The system has bounds on the input u as u∈P,
 * where P is a bounded polytope, we want to find a control Lyapunov function
 * (and region of attraction) for this system as V(x). The control Lyapunov
 * function should satisfy the condition
 * V(x) > 0 ∀ x ≠ x*                                     (1)
 * V(x*) = 0                                             (2)
 * ∀ x satisfying V(x) ≤ ρ, ∃ u ∈ P s.t V̇ < 0            (3)
 * These conditions prove that the sublevel set V(x) ≤ ρ is a region of
 * attraction, that starting from any state within this ROA, there exists
 * control actions that can stabilize the system to x*. Note that
 * V̇(x, u) = ∂V/∂x*f(x)+∂V/∂x*G(x)u. As we assumed that the bounds on the input
 * u is a polytope P. If we write the vertices of P as uᵢ, i = 1, ..., N, since
 * V̇ is a linear function of u, the minimal of min V̇, subject to u ∈ P is
 * obtained in one of the vertices of P. Hence the condition
 * ∃ u ∈ P s.t V̇(x, u) < 0
 * is equivalent to min_i V̇(x, uᵢ) < 0
 * We don't know which vertex gives us the minimal, but we can say if the
 * minimal is obtained at the i'th vertex (namely V̇(x, uᵢ)≤ V̇(x, uⱼ)∀ j≠ i),
 * then the minimal has to be negative. Mathematically this means
 * ∀ i, if V̇(x, uᵢ) ≤ dot{V}(x, uⱼ)∀ j≠ i, then V̇(x, uᵢ)< 0
 * As a result, condition (3) is equivalent to the following condition
 * for each i = 1, ..., N
 * V(x) ≤ ρ, V̇(x, uᵢ) ≤ V̇(x, uⱼ) => V̇(x, uᵢ)<0            (4)
 * We will impose condition (1) and (4) as sum-of-squares constraints.
 */
class SearchControlLyapunov {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SearchControlLyapunov)

  /**
   * @param f The dynamics of the system is ẋ = f(x) + G(x)u
   * @param G The dynamics of the system is ẋ = f(x) + G(x)u
   * @param x_equilibrium The equilibrium state.
   * @param u_vertices An nᵤ * K matrix. u_vertices.col(i) is the i'th vertex
   * of the polytope as the bounds on the control action.
   */
  SearchControlLyapunov(
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const Eigen::Ref<const Eigen::VectorXd>& x_equilibrium,
      const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x);

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  Eigen::VectorXd x_equilibrium_;
  Eigen::MatrixXd u_vertices_;
  // The indeterminates as the state.
  VectorX<symbolic::Variable> x_;
};

/**
 * Given control Lyapunov function V, formulates the SOS program to search for
 * the Lagrangian multiplier l and m in
 * -V̇(x, uᵢ) - εV(x) - ∑ⱼ lᵢⱼ(x)(V̇(x, uⱼ)-V̇(x, uᵢ)) - mᵢ(x)(ρ-V(x)) is SOS ∀ i
 * lᵢⱼ(x) is SOS, mᵢ(x) is SOS.
 */
class SearchControlLyapunovLagrangian {
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
