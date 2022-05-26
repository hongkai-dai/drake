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
 * (1 + t(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
 * t(x) is sos
 * sⱼ(x) is sos.
 * </pre>
 *
 * Condition (2) is the same as ḣ ≤ −εh ⇒ h(x)≤−1
 * We will verify this condition via sum-of-squares optimization, namely
 * <pre>
 * (1+λ₀(x))(−1 − h(x)) −∑ᵢ lᵢ(x)(−εh − ∂h/∂xf(x)−∂h/∂xG(x)uⁱ) is sos
 * λ₀(x), lᵢ(x) is sos
 * </pre>
 */
class ControlBarrier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlBarrier)

  ControlBarrier(const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                 const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
                 const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
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

  /**
   * Given the CBF h(x), constructs the program to find the Lagrangian λ₀(x) and
   * lᵢ(x)
   * <pre>
   * (1+λ₀(x))(−1 − h(x)) −∑ᵢ lᵢ(x)(−εh − ∂h/∂xf(x)−∂h/∂xG(x)uⁱ) is sos
   * λ₀(x), lᵢ(x) is sos
   * </pre>
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianProgram(
      const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
      const std::vector<int>& l_degrees, symbolic::Polynomial* lambda0,
      MatrixX<symbolic::Variable>* lambda0_gram,
      VectorX<symbolic::Polynomial>* l,
      std::vector<MatrixX<symbolic::Variable>>* l_grams,
      symbolic::Polynomial* hdot_sos,
      VectorX<symbolic::Monomial>* hdot_monomials,
      MatrixX<symbolic::Variable>* hdot_gram) const;

  /**
   * Given h(x), find the Lagrangian t(x) and sⱼ(x) to prove that the j'th
   * unsafe region is within the sub-level set {x | h(x)<=0}
   * <pre>
   * (1 + t(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
   * t(x) is sos
   * sⱼ(x) is sos.
   * </pre>
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructUnsafeRegionProgram(
      const symbolic::Polynomial& h, int region_index, int t_degree,
      const std::vector<int>& s_degrees, symbolic::Polynomial* t,
      MatrixX<symbolic::Variable>* t_gram, VectorX<symbolic::Polynomial>* s,
      std::vector<MatrixX<symbolic::Variable>>* s_grams,
      symbolic::Polynomial* sos_poly,
      MatrixX<symbolic::Variable>* sos_poly_gram) const;

  /**
   * Given Lagrangian multipliers λ₀(x), l(x), t(x), find the control barrier
   * function through
   * <pre>
   * max ∑ᵢ min(h(xⁱ), eps),   xⁱ ∈ unverified_candidate_states
   * s.t (1+λ₀(x))(−1 − h(x)) −∑ᵢlᵢ(x)(−εh − ∂h/∂xf(x)−∂h/∂xG(x)uⁱ) is sos
   *     (1 + t(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
   *     sⱼ(x) is sos.
   *     h(xʲ) >= 0, xʲ ∈ verified_safe_states
   * </pre>
   * where eps in the objective is a small positive constant.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructBarrierProgram(
      const symbolic::Polynomial& lambda0,
      const VectorX<symbolic::Polynomial>& l,
      const std::vector<symbolic::Polynomial>& t, int h_degree,
      double deriv_eps, const std::vector<std::vector<int>>& s_degrees,
      symbolic::Polynomial* h, symbolic::Polynomial* hdot_sos,
      MatrixX<symbolic::Variable>* hdot_sos_gram,
      std::vector<VectorX<symbolic::Polynomial>>* s,
      std::vector<std::vector<MatrixX<symbolic::Variable>>>* s_grams,
      std::vector<symbolic::Polynomial>* unsafe_sos_polys,
      std::vector<MatrixX<symbolic::Variable>>* unsafe_sos_poly_grams) const;

  /**
   * Add the cost
   * max ∑ᵢ min(h(xⁱ), eps),   xⁱ ∈ unverified_candidate_states
   * and the constraint
   * h(xʲ) >= 0, xʲ ∈ verified_safe_states
   * to prog.
   */
  void AddBarrierProgramCost(solvers::MathematicalProgram* prog,
                             const symbolic::Polynomial& h,
                             const Eigen::MatrixXd& verified_safe_states,
                             const Eigen::MatrixXd& unverified_candidate_states,
                             double eps) const;

  /**
   * An ellipsoid as
   * (x−c)ᵀS(x−c) ≤ ρ
   */
  struct Ellipsoid {
    Ellipsoid(const Eigen::Ref<const Eigen::VectorXd>& m_c,
              const Eigen::Ref<const Eigen::MatrixXd>& m_S, double m_rho,
              double m_rho_min, double m_rho_max, double m_rho_tol,
              int m_r_degree)
        : c{m_c},
          S{m_S},
          rho{m_rho},
          rho_min{m_rho_min},
          rho_max{m_rho_max},
          rho_tol{m_rho_tol},
          r_degree{m_r_degree} {
      DRAKE_DEMAND(c.rows() == S.rows());
      DRAKE_DEMAND(c.rows() == S.cols());
      DRAKE_DEMAND(rho_min <= rho);
      DRAKE_DEMAND(rho_max >= rho);
      DRAKE_DEMAND(rho_tol > 0);
    }

    Eigen::VectorXd c;
    Eigen::MatrixXd S;
    double rho;
    double rho_min;
    double rho_max;
    double rho_tol;
    int r_degree;
  };

  /**
   * Maximize the minimal value of h(x) within the ellipsoids.
   * Add the cost max ∑ᵢ dᵢ
   * with constraint
   * h(x)-dᵢ - rᵢ(x) * (ρᵢ−(x−cᵢ)ᵀS(x−cᵢ)) is sos.
   * rᵢ(x) is sos.
   */
  void AddBarrierProgramCost(solvers::MathematicalProgram* prog,
                             const symbolic::Polynomial& h,
                             const std::vector<Ellipsoid>& inner_ellipsoids,
                             std::vector<symbolic::Polynomial>* r,
                             VectorX<symbolic::Variable>* d) const;

  struct SearchOptions {
    solvers::SolverId barrier_step_solver{solvers::MosekSolver::id()};
    solvers::SolverId lagrangian_step_solver{solvers::MosekSolver::id()};
    int bilinear_iterations{10};
    double backoff_scale{0.};
    std::optional<solvers::SolverOptions> lagrangian_step_solver_options{
        std::nullopt};
    std::optional<solvers::SolverOptions> barrier_step_solver_options{
        std::nullopt};
    // Small coefficient in the constraints can cause numerical issues. We will
    // set the coefficient of linear constraints smaller than these tolerance to
    // 0.
    double barrier_tiny_coeff_tol = 0;
    double lagrangian_tiny_coeff_tol = 0;

    // The solution to these polynomials might contain terms with tiny
    // coefficient, due to numerical roundoff error coming from the solver. We
    // remove terms in the polynomial with tiny coefficients.
    double hsol_tiny_coeff_tol = 0;
    double lsol_tiny_coeff_tol = 0;
  };

  /**
   * @param x_anchor When searching for the barrier function h(x), we will
   * require h(x_anchor) <= h_init(x_anchor) to prevent scaling the barrier
   * function to infinity. This is because any positive scaling of a barrier
   * function is still a barrier function with the same verified safe set.
   * @pre h_init(x_anchor) > 0
   */
  void Search(const symbolic::Polynomial& h_init, int h_degree,
              double deriv_eps, int lambda0_degree,
              const std::vector<int>& l_degrees,
              const std::vector<int>& t_degree,
              const std::vector<std::vector<int>>& s_degrees,
              const std::vector<ControlBarrier::Ellipsoid>& ellipsoids,
              const Eigen::Ref<const Eigen::VectorXd>& x_anchor,
              const SearchOptions& search_options, symbolic::Polynomial* h_sol,
              symbolic::Polynomial* lambda0_sol,
              VectorX<symbolic::Polynomial>* l_sol,
              std::vector<symbolic::Polynomial>* t_sol,
              std::vector<VectorX<symbolic::Polynomial>>* s_sol) const;

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  int nx_;
  int nu_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
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
