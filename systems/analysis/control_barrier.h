#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/systems/framework/leaf_system.h"

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
 *     ∀ x satisfying β⁻ < h(x) < β⁺, ∃u ∈ P, s.t. ḣ > −ε h    (2)
 *
 * where β⁻< 0 and β⁺>0
 * Suppose 𝒳ᵤ is defined as the union of polynomial sub-level sets, namely 𝒳ᵤ =
 * 𝒳ᵤ¹ ∪ ... ∪ 𝒳ᵤᵐ, where each 𝒳ᵤʲ = { x | pⱼ(x)≤ 0} where pⱼ(x) is a vector of
 * polynomials. Condition (1) can be imposed through the following sos condition
 * <pre>
 * (1 + t(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
 * t(x) is sos
 * sⱼ(x) is sos.
 * </pre>
 *
 * Condition (2) is the same as h(x) ≤ β⁺ and ḣ ≤ −εh ⇒ h(x) ≤ β⁻
 * We will verify this condition via sum-of-squares optimization, namely
 * <pre>
 * (1+λ₀(x))(β⁻−h(x)) −∑ᵢlᵢ(x)(−εh−∂h/∂xf(x)−∂h/∂xG(x)uⁱ) −λ₁(x)(β⁺−h(x)) is sos
 * λ₀(x), λ₁(x), lᵢ(x) is sos
 * </pre>
 *
 * @param beta_plus β⁺ in the documentation above. if beta_plus is std::nullopt,
 * then we set β⁺=infinity, and ignore λ₁(x)
 */
class ControlBarrier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlBarrier)

  ControlBarrier(const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                 const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
                 std::optional<symbolic::Polynomial> dynamics_denominator,
                 const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                 double beta_minus, std::optional<double> beta_plus,
                 std::vector<VectorX<symbolic::Polynomial>> unsafe_regions,
                 const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
                 const Eigen::Ref<const VectorX<symbolic::Polynomial>>&
                     state_eq_constraints);

  /**
   * A helper function to add the constraint
   * (1+λ₀(x))(β⁻−h(x)) −∑ᵢlᵢ(x)(−εh−∂h/∂xf(x)−∂h/∂xG(x)uⁱ) −λ₁(x)(β⁺−h(x)) +
   * a(x)is sos
   * @param[out] monomials The monomial basis of this sos constraint.
   * @param[out] gram The Gram matrix of this sos constraint.
   */
  void AddControlBarrierConstraint(
      solvers::MathematicalProgram* prog, const symbolic::Polynomial& lambda0,
      const std::optional<symbolic::Polynomial>& lambda1,
      const VectorX<symbolic::Polynomial>& l,
      const VectorX<symbolic::Polynomial>& state_constraints_lagrangian,
      const symbolic::Polynomial& h, double deriv_eps,
      const std::optional<symbolic::Polynomial>& a,
      symbolic::Polynomial* hdot_poly, VectorX<symbolic::Monomial>* monomials,
      MatrixX<symbolic::Variable>* gram) const;

  struct LagrangianReturn {
    LagrangianReturn()
        : prog{std::make_unique<solvers::MathematicalProgram>()} {}

    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr.
    LagrangianReturn(LagrangianReturn&&) = default;
    LagrangianReturn& operator=(LagrangianReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    symbolic::Polynomial lambda0;
    MatrixX<symbolic::Variable> lambda0_gram;
    std::optional<symbolic::Polynomial> lambda1;
    std::optional<MatrixX<symbolic::Variable>> lambda1_gram;
    VectorX<symbolic::Polynomial> l;
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    VectorX<symbolic::Polynomial> state_constraints_lagrangian;
    std::optional<symbolic::Polynomial> a;
    std::optional<MatrixX<symbolic::Variable>> a_gram;
    symbolic::Polynomial hdot_sos;
    VectorX<symbolic::Monomial> hdot_monomials;
    MatrixX<symbolic::Variable> hdot_gram;
  };

  /**
   * Given the CBF h(x), constructs the program to find the Lagrangian λ₀(x) and
   * lᵢ(x)
   * <pre>
   * (1+λ₀(x))(β⁻−h(x)) −∑ᵢlᵢ(x)(−εh−∂h/∂xf(x)−∂h/∂xG(x)uⁱ) −λ₁(x)(β⁺−h(x))
   *     + a(x)is sos
   *  λ₀(x), λ₁(x), lᵢ(x), a(x) is sos
   * </pre>
   * a(x) is the slack variable to relax the roblem such that this optimization
   * is always feasible.
   * @param a_degree The degree of the slack polynomial a(x). If
   * a_degree=std::nullopt, then we use a(x)=0.
   */
  LagrangianReturn ConstructLagrangianProgram(
      const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
      std::optional<int> lambda1_degree, const std::vector<int>& l_degrees,
      const std::vector<int>& state_constraints_lagrangian_degrees,
      std::optional<int> a_degree) const;

  struct UnsafeReturn {
    UnsafeReturn() : prog{std::make_unique<solvers::MathematicalProgram>()} {}

    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr.
    UnsafeReturn(UnsafeReturn&&) = default;
    UnsafeReturn& operator=(UnsafeReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    symbolic::Polynomial t;
    MatrixX<symbolic::Variable> t_gram;
    VectorX<symbolic::Polynomial> s;
    std::vector<MatrixX<symbolic::Variable>> s_grams;
    VectorX<symbolic::Polynomial> state_constraints_lagrangian;
    std::optional<symbolic::Polynomial> a;
    std::optional<MatrixX<symbolic::Variable>> a_gram;
    symbolic::Polynomial sos_poly;
    MatrixX<symbolic::Variable> sos_poly_gram;
  };

  /**
   * Given h(x), find the Lagrangian tⱼ(x) and sⱼ(x) to prove that the j'th
   * unsafe region is within the sub-level set {x | h(x)<=0}
   * <pre>
   * (1 + tⱼ(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
   * t(x) is sos
   * sⱼ(x) is sos.
   * </pre>
   */
  UnsafeReturn ConstructUnsafeRegionProgram(
      const symbolic::Polynomial& h, int region_index, int t_degree,
      const std::vector<int>& s_degrees,
      const std::vector<int>& state_constraints_lagrangian_degrees,
      std::optional<int> a_degree) const;

  struct BarrierReturn {
    BarrierReturn() : prog{std::make_unique<solvers::MathematicalProgram>()} {}

    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr.
    BarrierReturn(BarrierReturn&&) = default;
    BarrierReturn& operator=(BarrierReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    symbolic::Polynomial h;
    symbolic::Polynomial hdot_sos;
    MatrixX<symbolic::Variable> hdot_sos_gram;
    VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian;
    std::optional<symbolic::Polynomial> hdot_a;
    std::optional<MatrixX<symbolic::Variable>> hdot_a_gram;
    std::vector<VectorX<symbolic::Polynomial>> s;
    std::vector<std::vector<MatrixX<symbolic::Variable>>> s_grams;
    std::vector<symbolic::Polynomial> unsafe_sos_polys;
    std::vector<MatrixX<symbolic::Variable>> unsafe_sos_poly_grams;
    std::vector<VectorX<symbolic::Polynomial>>
        unsafe_state_constraints_lagrangian;
    std::vector<std::optional<symbolic::Polynomial>> unsafe_a;
    std::vector<std::optional<MatrixX<symbolic::Variable>>> unsafe_a_gram;
  };

  /**
   * Given Lagrangian multipliers λ₀(x), l(x), t(x), find the control barrier
   * function through
   * <pre>
   * Find h(x), sⱼ(x)
   * s.t (1+λ₀(x))(β⁻−h(x)) −∑ᵢlᵢ(x)(−εh−∂h/∂xf(x)−∂h/∂xG(x)uⁱ)
   *             − λ₁(x)(β⁺−h(x)) is sos
   *     (1 + tⱼ(x))*(-h(x)) + sⱼ(x)ᵀpⱼ(x) is sos
   *     sⱼ(x) is sos.
   *     h(xʲ) >= 0, xʲ ∈ verified_safe_states
   * </pre>
   * where eps in the objective is a small positive constant.
   * @note We haven't imposed any cost on this program yet (even when we have
   * the slack polynomial hdot_a and unsafe_a).
   */
  BarrierReturn ConstructBarrierProgram(
      const symbolic::Polynomial& lambda0,
      const std::optional<symbolic::Polynomial>& lambda1,
      const VectorX<symbolic::Polynomial>& l,
      const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
      std::optional<int> hdot_a_degree,
      const std::vector<symbolic::Polynomial>& t,
      const std::vector<std::vector<int>>&
          unsafe_state_constraints_lagrangian_degrees,
      int h_degree, double deriv_eps,
      const std::vector<std::vector<int>>& s_degrees,
      const std::vector<std::optional<int>>& unsafe_a_degrees) const;

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
   * (x−c)ᵀS(x−c) ≤ d
   */
  struct Ellipsoid {
    Ellipsoid(const Eigen::Ref<const Eigen::VectorXd>& m_c,
              const Eigen::Ref<const Eigen::MatrixXd>& m_S, double m_d,
              int m_r_degree, std::vector<int> m_eq_lagrangian_degrees)
        : c{m_c},
          S{m_S},
          d{m_d},
          r_degree{m_r_degree},
          eq_lagrangian_degrees{std::move(m_eq_lagrangian_degrees)} {
      DRAKE_DEMAND(c.rows() == S.rows());
      DRAKE_DEMAND(c.rows() == S.cols());
    }

    Eigen::VectorXd c;
    Eigen::MatrixXd S;
    double d;
    int r_degree;
    std::vector<int> eq_lagrangian_degrees;
  };

  struct EllipsoidBisectionOption {
    EllipsoidBisectionOption(double m_d_min, double m_d_max, double m_d_tol)
        : d_min{m_d_min}, d_max{m_d_max}, d_tol{m_d_tol} {
      DRAKE_DEMAND(d_max >= d_min);
      DRAKE_DEMAND(d_tol >= 0);
    }
    double d_min;
    double d_max;
    double d_tol;
  };

  /**
   * Maximize the minimal value of h(x) within the ellipsoids.
   * Add the cost max ∑ᵢ ρᵢ
   * with constraint
   * h(x)-ρᵢ - rᵢ(x) * (dᵢ−(x−cᵢ)ᵀS(x−cᵢ)) is sos.
   * rᵢ(x) is sos.
   */
  void AddBarrierProgramCost(solvers::MathematicalProgram* prog,
                             const symbolic::Polynomial& h,
                             const std::vector<Ellipsoid>& inner_ellipsoids,
                             std::vector<symbolic::Polynomial>* r,
                             VectorX<symbolic::Variable>* rho,
                             std::vector<VectorX<symbolic::Polynomial>>*
                                 ellipsoids_state_constraints_lagrangian) const;

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

  struct SearchReturn {
    symbolic::Polynomial h;
    symbolic::Polynomial lambda0;
    std::optional<symbolic::Polynomial> lambda1;
    VectorX<symbolic::Polynomial> l;
    VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian;
    std::vector<symbolic::Polynomial> t;
    std::vector<VectorX<symbolic::Polynomial>> s;
    std::vector<VectorX<symbolic::Polynomial>>
        unsafe_state_constraints_lagrangian;
  };

  /**
   * @param x_anchor When searching for the barrier function h(x), we will
   * require h(x_anchor) <= h_init(x_anchor) to prevent scaling the barrier
   * function to infinity. This is because any positive scaling of a barrier
   * function is still a barrier function with the same verified safe set.
   * @pre h_init(x_anchor) > 0
   * @param[in/out] ellipsoids The ellipsoids contained inside the safe region.
   */
  SearchReturn Search(
      const symbolic::Polynomial& h_init, int h_degree, double deriv_eps,
      int lambda0_degree, std::optional<int> lambda1_degree,
      const std::vector<int>& l_degrees,
      const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
      const std::vector<int>& t_degree,
      const std::vector<std::vector<int>>& s_degrees,
      const std::vector<std::vector<int>>&
          unsafe_state_constraints_lagrangian_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_anchor,
      const SearchOptions& search_options,
      std::vector<ControlBarrier::Ellipsoid>* ellipsoids,
      std::vector<EllipsoidBisectionOption>* ellipsoid_bisection_options) const;

  struct SearchLagrangianReturn {
    bool success;
    symbolic::Polynomial lambda0;
    std::optional<symbolic::Polynomial> lambda1;
    VectorX<symbolic::Polynomial> l;
    VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian;
    std::optional<symbolic::Polynomial> hdot_a;
    std::optional<Eigen::MatrixXd> hdot_a_gram;
    std::vector<symbolic::Polynomial> t;
    std::vector<VectorX<symbolic::Polynomial>> s;
    std::vector<VectorX<symbolic::Polynomial>>
        unsafe_state_constraints_lagrangian;
    std::vector<std::optional<symbolic::Polynomial>> unsafe_a;
    std::vector<std::optional<Eigen::MatrixXd>> unsafe_a_grams;
  };
  /**
   * Search Lagrangian multiplier λ₀(x), l(x), t(x), s(x) to prove that h(x) is
   * a valid CBF, whose super-level set doesn't contain any unsafe regions.
   * @return success Returns true if the Lagrangian multipliers are found.
   */
  SearchLagrangianReturn SearchLagrangian(
      const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
      std::optional<int> lambda1_degree, const std::vector<int>& l_degrees,
      const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
      std::optional<int> hdot_a_degree, const std::vector<int>& t_degree,
      const std::vector<std::vector<int>>& s_degrees,
      const std::vector<std::vector<int>>&
          unsafe_state_constraints_lagrangian_degrees,
      const std::vector<std::optional<int>>& unsafe_a_degrees,
      const SearchOptions& search_options) const;

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  std::optional<symbolic::Polynomial> dynamics_denominator_;
  int nx_;
  int nu_;
  double beta_minus_;
  std::optional<double> beta_plus_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions_;
  Eigen::MatrixXd u_vertices_;
  VectorX<symbolic::Polynomial> state_eq_constraints_;
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

class CbfController : public systems::LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CbfController)

  CbfController(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
                std::optional<symbolic::Polynomial> dynamics_denominator,
                symbolic::Polynomial cbf, double deriv_eps);

  virtual ~CbfController() {}

  void CalcCbf(const Context<double>& context,
               BasicVector<double>* output) const;

  void CalcControl(const Context<double>& context,
                   BasicVector<double>* output) const {
    DoCalcControl(context, output);
  }

  const InputPort<double>& x_input_port() const {
    return this->get_input_port(x_input_index_);
  }

  const OutputPort<double>& cbf_output_port() const {
    return this->get_output_port(cbf_output_index_);
  }

  const OutputPort<double>& control_output_port() const {
    return this->get_output_port(control_output_index_);
  }

 protected:
  const VectorX<symbolic::Variable>& x() const { return x_; }

  const VectorX<symbolic::Polynomial>& f() const { return f_; }

  const MatrixX<symbolic::Polynomial>& G() const { return G_; }

  const std::optional<symbolic::Polynomial>& dynamics_denominator() const {
    return dynamics_denominator_;
  }

  const symbolic::Polynomial& cbf() const { return cbf_; }

  double deriv_eps() const { return deriv_eps_; }

  const symbolic::Polynomial& dhdx_times_f() const { return dhdx_times_f_; }

  const RowVectorX<symbolic::Polynomial>& dhdx_times_G() const {
    return dhdx_times_G_;
  }

 private:
  virtual void DoCalcControl(const Context<double>& context,
                             BasicVector<double>* output) const = 0;

  VectorX<symbolic::Variable> x_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  std::optional<symbolic::Polynomial> dynamics_denominator_;
  symbolic::Polynomial cbf_;
  double deriv_eps_;
  symbolic::Polynomial dhdx_times_f_;
  RowVectorX<symbolic::Polynomial> dhdx_times_G_;
  InputPortIndex x_input_index_;
  OutputPortIndex control_output_index_;
  OutputPortIndex cbf_output_index_;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
