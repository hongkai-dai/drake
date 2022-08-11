#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/solvers/csdp_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * For a control affine system with dynamics ẋ = f(x) + G(x)u where both f(x)
 * and G(x) are polynomials of x. The system has bounds on the input u as u∈P,
 * where P is a bounded polytope, we want to find a control Lyapunov function
 * (and region of attraction) for this system as V(x). The control Lyapunov
 * function should satisfy the condition
 *
 *     V(x) > 0 ∀ x ≠ 0                                     (1)
 *     V(0) = 0                                             (2)
 *     ∀ x satisfying V(x) ≤ ρ, ∃ u ∈ P s.t V̇ < 0            (3)
 *
 * These conditions prove that the sublevel set V(x) ≤ ρ is a region of
 * attraction, that starting from any state within this ROA, there exists
 * control actions that can stabilize the system to x*. In other word, we want
 * to prove the condition that minᵤ V̇ ≥ −εV ⇒ V≥ρ or x = 0 Note that V̇(x, u) =
 * ∂V/∂x*f(x)+∂V/∂x*G(x)u. As we assumed that the bounds on the input u is a
 * polytope P. If we write the vertices of P as uᵢ, i = 1, ..., N, since V̇ is a
 * linear function of u, the minimal of min V̇, subject to u ∈ P is obtained in
 * one of the vertices of P. Hence the condition minᵤ V̇ ≥ −εV
 *
 * is equivalent to
 *
 *      V̇(x, uᵢ) ≥ −εV ∀i
 *
 * Using s-procedure, the condition we want to prove is that
 * V(x) >= 0
 * V(0) = 0
 * (1+λ(x))xᵀx(V−ρ) − ∑ᵢ lᵢ(x)(∂V/∂x*f(x)+∂V/∂x*G(x)uᵢ+εV) >= 0
 * λ(x), lᵢ(x) >= 0
 * We will search for such V, λ(x), lᵢ(x) through bilinear alternation.
 */
class ControlLyapunov {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlLyapunov)

  /**
   * @param f The dynamics of the system is ẋ = f(x) + G(x)u
   * @param G The dynamics of the system is ẋ = f(x) + G(x)u
   * @param dynamics_denominator If not nullopt, then the dynamics is ẋ =
   * f(x)/d(x) + G(x)/d(x) * u where d(x) = dynamics_denominator; otherwise d(x)
   * = 1.
   * @param u_vertices An nᵤ * K matrix. u_vertices.col(i) is the i'th vertex
   * of the polytope as the bounds on the control action.
   * @param state_constraints The additional equality constraints on the system
   * state x. For example if the state contains the quaternion z, then we have
   * the unit length constraint on quaternion zᵀz−1 = 0
   */
  ControlLyapunov(
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const std::optional<symbolic::Polynomial>& dynamics_denominator,
      const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_constraints);

  /**
   * A helper function to add the constraint
   * (1+λ₀(x))(xᵀx)ᵈ(V−ρ) − ∑ᵢ lᵢ(x)*(∂V/∂x*f(x)+ε*V + ∂V/∂x*G(x)*uᵢ)-p(x)ᵀc(x)
   * + a(x) is sos. where c(x) is state_constraint, p(x) is its Lagrangian
   * multipliers.
   * @param[in] a The slack polynomial a(x) in the documentation above. if
   * a=std::nullopt, then a(x) = 0.
   * @param[out] monomials The monomial basis of this sos constraint.
   * @param[out] gram The Gram matrix of this sos constraint.
   */
  template <typename T>
  void AddControlLyapunovConstraint(
      solvers::MathematicalProgram* prog, const VectorX<symbolic::Variable>& x,
      const symbolic::Polynomial& lambda0, int d_degree,
      const VectorX<symbolic::Polynomial>& l, const symbolic::Polynomial& V,
      const T& rho, const Eigen::MatrixXd& u_vertices, double deriv_eps,
      const VectorX<symbolic::Polynomial>& p,
      const std::optional<symbolic::Polynomial>& a,
      symbolic::Polynomial* vdot_poly, VectorX<symbolic::Monomial>* monomials,
      MatrixX<symbolic::Variable>* gram) const;

  struct LagrangianReturn {
    LagrangianReturn()
        : prog{std::make_unique<solvers::MathematicalProgram>()} {}
    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr
    LagrangianReturn(LagrangianReturn&&) = default;
    LagrangianReturn& operator=(LagrangianReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    symbolic::Polynomial lambda0;
    MatrixX<symbolic::Variable> lambda0_gram;
    VectorX<symbolic::Polynomial> l;
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    VectorX<symbolic::Polynomial> p;
    symbolic::Polynomial vdot_sos;
    VectorX<symbolic::Monomial> vdot_monomials;
    MatrixX<symbolic::Variable> vdot_gram;
  };

  /**
   * Given the control Lyapunov function V, constructs an optimization program
   * to search for the Lagrangians.
   * <pre>
   * find λ₀(x), l(x), p(x)
   * s.t (1+λ₀(x))xᵀx(V−ρ) − ∑ᵢlᵢ(x)*(∂V/∂x*f(x)+ε*V+∂V/∂x*G(x)*sᵢ)
   *             - p(x)ᵀc(x) is sos
   *      λ₀(x), l(x) is sos
   * </pre>
   * where c(x) is the state equality constraints (like unit quaternion
   * constraint).
   */
  LagrangianReturn ConstructLagrangianProgram(
      const symbolic::Polynomial& V, double rho, double deriv_eps,
      int lambda0_degree, const std::vector<int>& l_degrees,
      const std::vector<int>& p_degrees) const;

  struct LagrangianMaxRhoReturn {
    LagrangianMaxRhoReturn()
        : prog{std::make_unique<solvers::MathematicalProgram>()} {}
    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr.
    LagrangianMaxRhoReturn(LagrangianMaxRhoReturn&&) = default;
    LagrangianMaxRhoReturn& operator=(LagrangianMaxRhoReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    VectorX<symbolic::Polynomial> l;
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    VectorX<symbolic::Polynomial> p;
    symbolic::Variable rho;
    symbolic::Polynomial vdot_sos;
    VectorX<symbolic::Monomial> vdot_monomials;
    MatrixX<symbolic::Variable> vdot_gram;
  };

  /*
   * Given λ₀(x) and V(x), constructs the following optimization problem
   * <pre>
   * max ρ
   * s.t (1+λ₀(x))(xᵀx)ᵈ(V(x) − ρ) − ∑ᵢ lᵢ(x)(∂V/∂x*f(x) + ε*V+∂V/∂xG(x)uⁱ)
   *              - p(x)ᵀc(x)is sos.
   * lᵢ(x) is sos.
   * </pre>
   * The decision variables are ρ, l(x), p(x)
   */
  LagrangianMaxRhoReturn ConstructLagrangianProgram(
      const symbolic::Polynomial& V, const symbolic::Polynomial& lambda0,
      int d_degree, const std::vector<int>& l_degrees,
      const std::vector<int>& p_degrees, double deriv_eps) const;

  struct LyapunovReturn {
    LyapunovReturn() : prog{std::make_unique<solvers::MathematicalProgram>()} {}
    // Add the move constructor and move assignment operator so that we can
    // return this struct which has a unique_ptr.
    LyapunovReturn(LyapunovReturn&&) = default;
    LyapunovReturn& operator=(LyapunovReturn&&) = default;

    std::unique_ptr<solvers::MathematicalProgram> prog;
    symbolic::Polynomial V;
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    VectorX<symbolic::Polynomial> p;
  };
  /**
   * Given λ₀(x) and l(x), construct a mathematical program
   * <pre>
   * find V(x), q(x), p(x)
   * V - ε₁*(xᵀx)ᵈ¹ - q(x) * c(x) is sos
   * (1+λ₀(x))(xᵀx)ᵈ²(V(x) − ρ) − ∑ᵢ lᵢ(x)(∂V/∂x*f(x) + ε₂*V+∂V/∂xG(x)uⁱ)
   *              - p(x)ᵀc(x)is sos.
   * </pre>
   * @param[out] positivity_eq_lagrangian q(x) in the documentation above.
   */
  LyapunovReturn ConstructLyapunovProgram(
      const symbolic::Polynomial& lambda0,
      const VectorX<symbolic::Polynomial>& l, int V_degree, double rho,
      double positivity_eps, int positivity_d,
      const std::vector<int>& positivity_eq_lagrangian_degrees,
      const std::vector<int>& p_degrees, double deriv_eps) const;

  struct SearchOptions {
    solvers::SolverId lyap_step_solver{solvers::MosekSolver::id()};
    solvers::SolverId ellipsoid_step_solver{solvers::MosekSolver::id()};
    solvers::SolverId lagrangian_step_solver{solvers::MosekSolver::id()};
    int bilinear_iterations{10};
    // Stop when the improvement on ellipsoid d is below this tolerance.
    double d_converge_tol{1E-5};
    // Back off in each steps.
    double lyap_step_backoff_scale{0.};
    std::optional<solvers::SolverOptions> lagrangian_step_solver_options{
        std::nullopt};
    std::optional<solvers::SolverOptions> lyap_step_solver_options{
        std::nullopt};
    std::optional<solvers::SolverOptions> ellipsoid_step_solver_options{
        std::nullopt};
    // Small coefficient in the constraints can cause numerical issues. We will
    // set the coefficient of linear constraints smaller than these tolerance to
    // 0.
    double lyap_tiny_coeff_tol = 0;
    double lagrangian_tiny_coeff_tol = 0;

    // The solution to these polynomials might contain terms with tiny
    // coefficient, due to numerical roundoff error coming from the solver. We
    // remove terms in the polynomial with tiny coefficients.
    double Vsol_tiny_coeff_tol = 0;
    double lsol_tiny_coeff_tol = 0;

    double rho = {1.};

    // Name of the txt file to store the clf in each iteration of Search().
    std::optional<std::string> save_clf_file;
  };

  /**
   * We can search for the inscribed ellipsoid {x|(x−x*)ᵀS(x−x*) <=d} with the
   * largest d through bisection.
   */
  struct EllipsoidBisectionOption {
    EllipsoidBisectionOption(double m_size_min, double m_size_max,
                             double m_size_tol)
        : size_min{m_size_min}, size_max{m_size_max}, size_tol{m_size_tol} {}
    double size_min;
    double size_max;
    double size_tol;
  };

  /**
   * We can search for the inscribed ellipsoid {x|(x−x*)ᵀS(x−x*) <=d} with the
   * largest d through optimization
   * max d
   * s.t (1+t(x))((x-x*)ᵀS(x-x*)-d) - s(x)*(V(x)-ρ) is sos
   *     s(x) is sos
   */
  struct EllipsoidMaximizeOption {
    EllipsoidMaximizeOption(symbolic::Polynomial m_t, int m_s_degree,
                            double m_backoff_scale)
        : t{std::move(m_t)},
          s_degree{m_s_degree},
          backoff_scale{m_backoff_scale} {}
    symbolic::Polynomial t;
    int s_degree;
    double backoff_scale;
  };

  struct SearchResult {
    bool success;
    symbolic::Polynomial V;
    VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
    symbolic::Polynomial lambda0;
    VectorX<symbolic::Polynomial> l;
    VectorX<symbolic::Polynomial> p;
    SearchResultDetails search_result_details;
  };

  struct SearchWithEllipsoidResult : public SearchResult {
    symbolic::Polynomial r;
    double d;
    VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
  };

  /**
   * Use bilinear alternation to grow the region-of-attraction of the control
   * Lyapunov function (CLF).
   * @param ellipsoid_c_lagrangian_degrees. The degrees of the Lagrangian
   * multiplier for state_constraints(x) = 0 when searching for the maximal
   * ellipsoid contained in the sub-level set { x | V(x) <= ρ}
   */
  SearchWithEllipsoidResult Search(
      const symbolic::Polynomial& V_init, int lambda0_degree,
      const std::vector<int>& l_degrees, int V_degree, double positivity_eps,
      int positivity_d,
      const std::vector<int>& positivity_eq_lagrangian_degrees,
      const std::vector<int>& p_degrees,
      const std::vector<int>& ellipsoid_c_lagrangian_degrees, double deriv_eps,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int r_degree,
      const SearchOptions& search_options,
      const std::variant<EllipsoidBisectionOption, EllipsoidMaximizeOption>&
          ellipsoid_option) const;

  /**
   * Search the control Lyapunov V with the objective
   * min ∑ᵢ V(xⁱ)
   * or
   * min maxᵢ V(xⁱ)
   * where xⁱ is the i'th sampled state.
   * The constraints are
   * <pre>
   * V - ε₁*(xᵀx)ᵈ¹ - r(x) * c(x) is sos
   * (1+λ₀(x))(xᵀx)ᵈ²(V(x) − ρ) − ∑ᵢ lᵢ(x)(∂V/∂x*f(x) + ε₂*V+∂V/∂xG(x)uⁱ)
   *              - p(x)ᵀc(x)is sos.
   * </pre>
   * @param positivity_eps ε₁ in the documentation above.
   * @param positivity_d d1 in the documentation above.
   * @param positivity_eq_lagrangian_degrees Degrees of r(x).
   * @param deriv_eps ε₂ in the documentation above.
   * @param in_roa_samples. For each column of in_roa_samples, we will impose
   * the constraint V(in_roa_samples.col(i)) <= rho.
   */
  SearchResult Search(const symbolic::Polynomial& V_init, int lambda0_degree,
                      const std::vector<int>& l_degrees, int V_degree,
                      double positivity_eps, int positivity_d,
                      const std::vector<int>& positivity_eq_lagrangian_degrees,
                      const std::vector<int>& p_degrees, double deriv_eps,
                      const Eigen::Ref<const Eigen::MatrixXd>& x_samples,
                      const std::optional<Eigen::MatrixXd>& in_roa_samples,
                      bool minimize_max,
                      const SearchOptions& search_options) const;

  [[nodiscard]] const VectorX<symbolic::Variable>& x() const { return x_; }

  [[nodiscard]] const VectorX<symbolic::Polynomial>& f() const { return f_; }

  [[nodiscard]] const MatrixX<symbolic::Polynomial>& G() const { return G_; }

  [[nodiscard]] const Eigen::MatrixXd& u_vertices() const {
    return u_vertices_;
  }

  bool SearchLagrangian(const symbolic::Polynomial& V, double rho,
                        int lambda0_degree, const std::vector<int>& l_degrees,
                        const std::vector<int>& p_degrees, double deriv_eps,
                        const SearchOptions& search_options,
                        std::optional<bool> always_write_sol,
                        symbolic::Polynomial* lambda0,
                        VectorX<symbolic::Polynomial>* l,
                        VectorX<symbolic::Polynomial>* p) const;

  /**
   * Fix V, find rho through binary search such that V(x)<=rho satisfies the
   * control Lyapunov condition. Note that we don't check the positivity of V.
   * @return found_rho A flag whether rho is found or not.
   */
  bool FindRhoBinarySearch(const symbolic::Polynomial& V, double rho_min,
                           double rho_max, double rho_tol, int lambda0_degree,
                           const std::vector<int>& l_degrees,
                           const std::vector<int>& p_degrees, double deriv_eps,
                           const SearchOptions& search_options, double* rho_sol,
                           symbolic::Polynomial* lambda0,
                           VectorX<symbolic::Polynomial>* l,
                           VectorX<symbolic::Polynomial>* p) const;

 private:
  // The indeterminates as the state.
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  symbolic::Polynomial dynamics_denominator_;
  Eigen::MatrixXd u_vertices_;
  VectorX<symbolic::Polynomial> state_constraints_;
};

/**
 * Compute V̇(x, u) = ∂V/∂x * (f(x)/n(x)+G(x)/n(x)*u)
 * @param dynamics_numerator n(x) in the documentation above. If
 * dynamics_numerator=std::nullopt, then n(x) = 1.
 */
class VdotCalculator {
 public:
  VdotCalculator(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                 const symbolic::Polynomial& V,
                 const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                 const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
                 const std::optional<symbolic::Polynomial>& dynamics_numerator,
                 const Eigen::Ref<const Eigen::MatrixXd>& u_vertices);

  symbolic::Polynomial Calc(const Eigen::Ref<const Eigen::VectorXd>& u) const;

  /**
   * Compute min ∂V/∂x*f(x)+∂V/∂x * G(x)*u
   *             u in ConvexHull(u_vertices)
   * for each x.
   * @param x_vals A batch of x values. x_vals.col(i) is the i'th sample value
   * of x.
   */
  Eigen::VectorXd CalcMin(
      const Eigen::Ref<const Eigen::MatrixXd>& x_vals) const;

 private:
  VectorX<symbolic::Variable> x_;
  std::optional<symbolic::Polynomial> dynamics_numerator_;
  Eigen::MatrixXd u_vertices_;
  symbolic::Polynomial dVdx_times_f_;
  RowVectorX<symbolic::Polynomial> dVdx_times_G_;
};

/**
 * We need to impose the constraint
 * (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >= 0
 * (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V) >= 0
 * If this dynamics is symmetric (namely f(x) = -f(-x), G(x) = G(-x)), then we
 * only impose the first constraint. We store the monomial basis and the Gram
 * matrix of these two SOS constraint.
 */
struct VdotSosConstraintReturn {
  VdotSosConstraintReturn(int nu, bool m_symmetric_dynamics);

  /**
   * Compute the i'th pair of SOS constraint (the polynomial on the left
   * handside)
   * (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 −
   * V)>=0
   *
   * (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) -
   * lᵢ₁₂(x)*(1 − V)>=0
   * @param i Compute the i'th pair.
   * @param result The result after solving the program.
   */
  std::vector<symbolic::Polynomial> ComputeSosConstraint(
      int i, const solvers::MathematicalProgramResult& result) const;

  symbolic::Polynomial ComputeSosConstraint(
      int i, int j, const solvers::MathematicalProgramResult& result) const;

  bool symmetric_dynamics;

  // monomials[i][0] / grams[i][0] is the monomial basis / gram matrix for the
  // constraint (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) -
  // lᵢ₀₂(x)*(1 − V) >= 0 monomials[i][1] / grams[i][1] is the monomial basis /
  // gram matrix for the constraint (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) +
  // lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V) >= 0
  std::vector<std::vector<VectorX<symbolic::Monomial>>> monomials;
  std::vector<std::vector<MatrixX<symbolic::Variable>>> grams;
};

/**
 * This is the Lagrangian step in ControlLaypunovBoxInputBound. The control
 * Lyapunov function V, together with b are fixed, and we search for the
 * Lagrangian multipliers
 * lᵢ₀₀(x), lᵢ₀₁(x), lᵢ₀₂(x), lᵢ₁₀(x), lᵢ₁₁(x), lᵢ₁₂(x) satisfying the
 * constraints
 *
 *     (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >=
 *     0
 *     (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)
 *     >= 0
 *     lᵢ₀₀(x) >= 0, lᵢ₀₁(x) >= 0, lᵢ₀₂(x) >= 0,
 *     lᵢ₁₀(x) >= 0, lᵢ₁₁(x) >= 0, lᵢ₁₂(x) >= 0
 *     for i = 0, ..., nᵤ-1
 *
 * We will create 2*nᵤ SOS problems for each i = 0, ..., nᵤ - 1. Note that for
 * each i we can solve two programs separately.
 */
class SearchLagrangianGivenVBoxInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SearchLagrangianGivenVBoxInputBound)

  /**
   * @param x The state as the indeterminates.
   * @param lagrangian_degrees lagrangian_degrees[i][j][k] is the degree of the
   * Lagrangian multiplier lᵢⱼₖ(x).
   */
  SearchLagrangianGivenVBoxInputBound(
      symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
      MatrixX<symbolic::Polynomial> G, bool symmetric_dynamics,
      VectorX<symbolic::Polynomial> b, VectorX<symbolic::Variable> x,
      std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees);

  /**
   * If j = 0, then this is the program
   *
   *     (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V)
   * >=
   *     0
   *     lᵢ₀₀(x) >= 0, lᵢ₀₁(x) >= 0, lᵢ₀₂(x) >= 0,
   *  if j = 1, then this is the program
   *     (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)
   *     >= 0
   *     lᵢ₁₀(x) >= 0, lᵢ₁₁(x) >= 0, lᵢ₁₂(x) >= 0
   */
  const solvers::MathematicalProgram& prog(int i, int j) const {
    return *(progs_[i][j]);
  }

  /**
   * Get the mutable version of prog().
   */
  solvers::MathematicalProgram* get_mutable_prog(int i, int j) {
    return progs_[i][j].get();
  }

  int nu() const { return nu_; }

  const symbolic::Polynomial& V() const { return V_; }

  const VectorX<symbolic::Variable>& x() const { return x_; }

  const VectorX<symbolic::Polynomial>& f() const { return f_; }

  const MatrixX<symbolic::Polynomial>& G() const { return G_; }

  const VectorX<symbolic::Polynomial>& b() const { return b_; }

  const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>&
  lagrangians() const {
    return l_;
  }

  const std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>&
  lagrangian_grams() const {
    return lagrangian_grams_;
  }

  const VdotSosConstraintReturn& vdot_sos_constraint() const {
    return vdot_sos_constraint_;
  }

 private:
  symbolic::Polynomial V_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  bool symmetric_dynamics_;
  VectorX<symbolic::Polynomial> b_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  std::vector<std::vector<std::unique_ptr<solvers::MathematicalProgram>>>
      progs_;
  int nu_;
  int nx_;
  std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l_;
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees_;
  std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>
      lagrangian_grams_;
  VdotSosConstraintReturn vdot_sos_constraint_;
};

/**
 * Search a control Lyapunov function (together with its region of attraction)
 * for a control affine system with box-shaped input limits. Namely the system
 * dynamics is ẋ = f(x) + G(x)u where the input bounds are -1 <= u <= 1 and the
 * goal state being x=0. If we denote the Lyapunov function as V(x), then the
 * control Lyapunov condition is
 *
 *     if x ≠ 0     V(x) > 0                                (1a)
 *     V(0) = 0                                             (1b)
 *     -ε₂V >= minᵤ V̇(x, u) = minᵤ ∂V/∂x*f(x) + ∂V/∂x * G(x)u    (2)
 *
 * where ε₂ is a small positive constant, that proves the system is
 * exponentially stable with convegence rate ε₂.
 * since minᵤ ∂V/∂x*f(x) + ∂V/∂x G(x)u = ∂V/∂x*f(x) - |∂V/∂x G(x)|₁
 * when -1 <= u <= 1, where |∂V/∂x G(x)|₁ is the 1-norm of ∂V/∂x G(x).
 * we know the condition (2) is equivalent to
 *
 *     |∂V/∂x G(x)|₁ >= ∂V/∂x*f(x) + ε₂V                          (3)
 *
 * Note that ∂V/∂x G(x) is a vector of size nᵤ, where nᵤ is the input size.
 * Condition (3) is equivalent to
 *
 *     ∃ bᵢ(x), such that ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
 *     bᵢ(x) <= |∂V/∂x * Gᵢ(x)|,
 *
 * where Gᵢ(x) is the i'th column of the matrix G(x).
 * We know that bᵢ(x) <= |∂V/∂x * Gᵢ(x)| if and only if
 *
 *     when ∂V/∂x * Gᵢ(x) > 0, then bᵢ(x) <= ∂V/∂x * Gᵢ(x)
 *     when ∂V/∂x * Gᵢ(x) <= 0, then bᵢ(x) <= -∂V/∂x * Gᵢ(x)
 *
 * So to impose the constraint bᵢ(x) <= |∂V/∂x * Gᵢ(x)|, we introduce the
 * Lagrangian multiplier lᵢ₀₀(x), lᵢ₀₁(x), lᵢ₁₀(x), lᵢ₁₁(x)with the constraint
 *
 *     (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x))−lᵢ₀₁(x)*∂V/∂x*Gᵢ(x)>=0
 *     (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x))+lᵢ₁₁(x)*∂V/∂x*Gᵢ(x)>=0
 *     lᵢ₀₀(x) >= 0, lᵢ₀₁(x) >= 0, lᵢ₁₀(x) >= 0, lᵢ₁₁(x) >= 0
 *
 * To summarize, in order to prove the control Lyapunov function with region
 * of attraction V(x) ≤ 1, we impose the following constraint
 *
 *    V(x) ≥ ε₁xᵀx
 *    V(0) = 0
 *    ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
 *    (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >=
 * 0 (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)>= 0
 *     lᵢ₀₀(x) >= 0, lᵢ₀₁(x) >= 0, lᵢ₀₂(x) >= 0,
 *     lᵢ₁₀(x) >= 0, lᵢ₁₁(x) >= 0, lᵢ₁₂(x) >= 0
 *
 * We will use bilinear alternation to search for the control Lyapunov
 * function V, the Lagrangian multipliers and the slack polynomials b(x).
 *
 * During bilinear alternation, our goal is to maximize the region-of-attraction
 * (ROA). We measure the size of the ROA by an inner ellipsoid (x-x*)ᵀS(x-x*)≤ρ,
 * where the shape of the ellipsoid (S) and the center of the ellipsoid x* are
 * both given, and we want to maximize ρ.
 */
class ControlLyapunovBoxInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlLyapunovBoxInputBound)

  /**
   * @param positivity_eps ε₁ in the documentation above, to enforce the
   * positivity constraint V(x) > 0.
   */
  ControlLyapunovBoxInputBound(
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      double positivity_eps);

  struct SearchReturn {
    symbolic::Polynomial V;
    VectorX<symbolic::Polynomial> b;
    double deriv_eps;
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l;
    symbolic::Polynomial ellipsoid_lagrangian;
    // The ellipsoid is {x | (x-x*)S(x-x*) <= d}
    double d;
  };

  struct SearchOptions {
    solvers::SolverId lyap_step_solver{solvers::MosekSolver::id()};
    solvers::SolverId lagrangian_step_solver{solvers::MosekSolver::id()};
    int bilinear_iterations{10};
    // Stop when the improvement on d is below this tolerance.
    double d_converge_tol{1E-5};
    // Back off in each steps.
    double backoff_scale{0.};
    std::optional<solvers::SolverOptions> lagrangian_step_solver_options{
        std::nullopt};
    std::optional<solvers::SolverOptions> lyap_step_solver_options{
        std::nullopt};
    // Small coefficient in the constraints can cause numerical issues. We will
    // set the coefficient smaller than these tolerance to 0.
    double lyap_tiny_coeff_tol = 0;
    double lagrangian_tiny_coeff_tol = 0;

    // The solution to these polynomials might contain terms with tiny
    // coefficient, due to numerical roundoff error coming from the solver. We
    // remove terms in the polynomial with tiny coefficients.
    double Vsol_tiny_coeff_tol = 0;
    double lsol_tiny_coeff_tol = 0;

    // If set to true, then in step 3 we search for Lagrangian and b (namely
    // step 1) while fixing lᵢⱼ₀ to l_given; otherwise in step 3 we search for
    // Lagrangian (including lᵢⱼ₀) but not b.
    bool search_l_and_b = false;
  };

  /**
   * Given V_init(x), lᵢ₀₀(x) and lᵢ₁₀(x), we search the control Lyapunov
   * function and maximize the ROA through the following process
   * 1. Fix V_init(x), lᵢ₀₀(x), lᵢ₁₀(x), search for lᵢ₀₁(x)lᵢ₀₂(x), lᵢ₁₁(x),
   * lᵢ₁₂(x), s(x), bᵢ(x).
   * 2. Fix Lagrangian multipliers lᵢ₀₀(x),...,lᵢ₁₂(x) s(x), search V(x) and
   *    bᵢ(x).
   * 3. Fix V(x), bᵢ(x), and search for Lagrangian multipliers
   *    lᵢ₀₀(x),...,lᵢ₁₂(x), s(x). Go to step 2.
   *
   * This function maximizes volume of the inner ellipsoid
   * {x | (x−x*)ᵀS(x−x*)≤ρ}, the condition of the ellipsoid being within the
   * sub-level set {x | V(x) <= 1} is the existence of polynomials t(x) and s(x)
   *
   *     (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1) is sos.
   *     t(x), s(x) is sos.
   */
  SearchReturn Search(
      const symbolic::Polynomial& V_init,
      const std::vector<std::vector<symbolic::Polynomial>>& l_given,
      const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
      const symbolic::Polynomial& t_given, int V_degree, double deriv_eps_lower,
      double deriv_eps_upper, const SearchOptions& options) const;

  /**
   * We can search for the inscribed ellipsoid {x|(x−x*)ᵀS(x−x*) <=d} with the
   * largest d through bisection.
   */
  struct EllipsoidBisectionOption {
    EllipsoidBisectionOption(double m_size_min, double m_size_max,
                             double m_size_tol)
        : size_min{m_size_min}, size_max{m_size_max}, size_tol{m_size_tol} {}
    double size_min;
    double size_max;
    double size_tol;
  };

  /**
   * Overloaded Search function.
   * The condition of the ellipsoid {x | (x−x*)ᵀS(x−x*)≤ρ} being inside the
   * sub-level set {x | V(x) <= 1} is the existence of the polynomial r(x)
   *
   *     1−V(x) − r(x)(ρ−(x−x*)ᵀS(x−x*)) is sos.
   *     r(x) is sos.
   */
  SearchReturn Search(
      const symbolic::Polynomial& V_init,
      const std::vector<std::vector<symbolic::Polynomial>>& l_given,
      const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int r_degree, int V_degree,
      double deriv_eps_lower, double deriv_eps_upper,
      const SearchOptions& options,
      const EllipsoidBisectionOption& ellipsoid_bisection_option) const;

  // Step 1 in Search() function.
  void SearchLagrangianAndB(
      const symbolic::Polynomial& V,
      const std::vector<std::vector<symbolic::Polynomial>>& l_given,
      const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
      const std::vector<int>& b_degrees, double deriv_eps_lower,
      double deriv_eps_upper, const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double lsol_tiny_coeff_tol, double* deriv_eps,
      VectorX<symbolic::Polynomial>* b,
      std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* l) const;

  // Step 2 in Search() function.
  // The objective is to maximize d in the ellipsoid.
  void SearchLyapunov(
      const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
      const std::vector<int>& b_degrees, int V_degree, double deriv_eps,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& s,
      const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double backoff_scale, double lyap_tiny_coeff_tol,
      double Vsol_tiny_coeff_tol, symbolic::Polynomial* V,
      VectorX<symbolic::Polynomial>* b, double* d) const;

  // Overloaded step 2 of Search() function.
  // Given the ellipsoid {x|(x−x*)ᵀS(x−x*) <=d}, the goal is to minimize the
  // maximal value of V(x) within the ellipsoid. This maximal value is denoted
  // by rho.
  void SearchLyapunov(
      const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
      const std::vector<int>& b_degrees, int V_degree, double deriv_eps,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, double d, int r_degree,
      const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double backoff_scale, double lyap_tiny_coeff_tol,
      double Vsol_tiny_coeff_tol, symbolic::Polynomial* V,
      VectorX<symbolic::Polynomial>* b, symbolic::Polynomial* r,
      double* rho_sol) const;

  // Step 3 in Search() function.
  void SearchLagrangian(
      const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& b,
      const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
      const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double lagrangian_tiny_coeff_tol, double lsol_tiny_coeff_tol,
      std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* l) const;

  /**
   * Given the control Lyapunov function candidate V, together with the
   * Lagrangian multipliers lᵢ₀₀(x), lᵢ₁₀(x), search for b and Lagrangian
   * multipliers lᵢ₀₁(x), lᵢ₀₂(x), lᵢ₁₁(x), lᵢ₁₂(x) satisfying the following
   * constraint
   * <pre>
   * ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
   * (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1−V) >= 0
   * (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1−V) >= 0
   * lᵢ₀₁(x) >= 0, lᵢ₀₂(x) >= 0, lᵢ₁₁(x) >= 0, lᵢ₁₂(x) >= 0
   * </pre>
   *
   * The variables are ε₂, b(x), lᵢ₀₁(x), lᵢ₀₂(x), lᵢ₁₁(x), lᵢ₁₂(x)
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLagrangianAndBProgram(
      const symbolic::Polynomial& V,
      const std::vector<std::vector<symbolic::Polynomial>>& l_given,
      const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
      const std::vector<int>& b_degrees, bool symmetric_dynamics,
      std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>*
          lagrangians,
      std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>*
          lagrangian_grams,
      symbolic::Variable* deriv_eps, VectorX<symbolic::Polynomial>* b,
      VdotSosConstraintReturn* vdot_sos_constraint) const;

  /**
   * For u bounded in a unit box -1 <= u <= 1.
   * Given the Lagrangian multiplier, find the Lyapunov function V and slack
   * polynomials b, satisfying the condition
   *
   *     V(x) >= ε₁xᵀx
   *     V(0) = 0
   *     ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
   *     (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V)
   * >=
   *     0
   *     (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)
   *     >= 0
   * where lᵢⱼₖ(x) are all given.
   */
  std::unique_ptr<solvers::MathematicalProgram> ConstructLyapunovProgram(
      const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
      bool symmetric_dynamics, double deriv_eps, int V_degree,
      const std::vector<int>& b_degrees, symbolic::Polynomial* V,
      MatrixX<symbolic::Expression>* positivity_constraint_gram,
      VectorX<symbolic::Monomial>* positivity_constraint_monomial,
      VectorX<symbolic::Polynomial>* b,
      VdotSosConstraintReturn* vdot_sos_constraint) const;

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  bool symmetric_dynamics_;
  // The indeterminates as the state.
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  double positivity_eps_;
  int nx_;
  int nu_;
};

/**
 * Computes the control action through the QP
 * min (u−u*)ᵀRᵤ(u−u*) + k*∂V/∂x*(f(x)/n(x)+G(x)/n(x)*u)
 * s.t ∂V/∂x*(f(x)/n(x)+G(x)/n(x)*u)≤ −εV
 *     Aᵤ*u ≤ bᵤ
 */
class ClfController : public LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ClfController)

  /**
   * @param dynamics_numerator. n(x) in the documentation above. If
   * dynamics_numerator = std::nullopt, then n(x)=1.
   * @param u_star If u_star=nullptr, then we use u in the previous step as u*
   * @param vdot_cost k in the documentation above.
   */
  ClfController(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
                const std::optional<symbolic::Polynomial>& dynamics_numerator,

                symbolic::Polynomial V, double deriv_eps,
                const Eigen::Ref<const Eigen::MatrixXd>& Au,
                const Eigen::Ref<const Eigen::VectorXd>& bu,
                const std::optional<Eigen::VectorXd>& u_star,
                const Eigen::Ref<const Eigen::MatrixXd>& Ru, double Vdot_cost);

  ~ClfController() {}

  const OutputPortIndex& control_output_index() const {
    return control_output_index_;
  }

  const OutputPortIndex& clf_output_index() const { return clf_output_index_; }

  const InputPortIndex& x_input_index() const { return x_input_index_; }

  void CalcControl(const Context<double>& context,
                   BasicVector<double>* output) const;

  /**
   * Compute CLF V(x).
   */
  void CalcClf(const Context<double>& context,
               BasicVector<double>* output) const;

 private:
  VectorX<symbolic::Variable> x_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  std::optional<symbolic::Polynomial> dynamics_numerator_;
  symbolic::Polynomial V_;
  double deriv_eps_;
  Eigen::MatrixXd Au_;
  Eigen::VectorXd bu_;
  mutable std::optional<Eigen::VectorXd> u_star_;
  Eigen::MatrixXd Ru_;
  double vdot_cost_;
  symbolic::Polynomial dVdx_times_f_;
  RowVectorX<symbolic::Polynomial> dVdx_times_G_;
  OutputPortIndex control_output_index_;
  OutputPortIndex clf_output_index_;
  InputPortIndex x_input_index_;
};

class ControlLyapunovNoInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ControlLyapunovNoInputBound)

  /**
   * @param f The dynamics of the system is ẋ = f(x) + G(x)u
   * @param G The dynamics of the system is ẋ = f(x) + G(x)u
   * @param dynamics_denominator If not nullopt, then the dynamics is ẋ =
   * f(x)/d(x) + G(x)/d(x) * u where d(x) = dynamics_denominator; otherwise d(x)
   * = 1.
   * @param state_constraints The additional equality constraints on the system
   * state x. For example if the state contains the quaternion z, then we have
   * the unit length constraint on quaternion zᵀz−1 = 0
   */
  ControlLyapunovNoInputBound(
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const std::optional<symbolic::Polynomial>& dynamics_denominator,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_constraints);

  /**
   * Add the constraint that
   * eq(x) = 0
   * ineq(x) <= 0
   * and ∂V/∂x*G(x)=0
   * implies
   * ∂V/∂x*f(x)+εV<=0
   */
  void AddControlLyapunovConstraint(
      solvers::MathematicalProgram* prog, const symbolic::Polynomial& lambda0,
      const VectorX<symbolic::Polynomial>& l, const symbolic::Polynomial& V,
      double deriv_eps, const VectorX<symbolic::Polynomial>& eq_lagrangian,
      const VectorX<symbolic::Polynomial>& ineq_constraints,
      const VectorX<symbolic::Polynomial>& ineq_lagrangian,
      symbolic::Polynomial* vdot_sos,
      VectorX<symbolic::Monomial>* vdot_sos_monomials,
      MatrixX<symbolic::Variable>* vdot_sos_gram) const;

  bool SearchLagrangian(
      const symbolic::Polynomial& V, int lambda0_degree,
      const std::vector<int>& l_degrees, double deriv_eps,
      const std::vector<int>& eq_lagrangian_degrees,
      const VectorX<symbolic::Polynomial>& ineq_constraints,
      const std::vector<int>& ineq_lagrangian_degrees,
      const solvers::SolverOptions& solver_options, double lsol_tiny_coeff_tol,
      symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
      VectorX<symbolic::Polynomial>* eq_lagrangian_sol,
      VectorX<symbolic::Polynomial>* ineq_lagrangian_sol) const;

 private:
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  std::optional<symbolic::Polynomial> dynamics_denominator_;
  VectorX<symbolic::Polynomial> state_constraints_;
};

namespace internal {
/**
 * Returns if the dynamics is symmetric, namely f(x) = -f(-x) and G(x) = G(-x),
 * which implies the time derivative at (-x, -u) is -ẋ.
 */
bool IsDynamicsSymmetric(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G);

/** Return q(x) = p(-x)
 */
symbolic::Polynomial NegateIndeterminates(const symbolic::Polynomial& p);

/** Create a polynomial the coefficient for constant and linear terms to be 0.
 */
symbolic::Polynomial NewFreePolynomialNoConstantOrLinear(
    solvers::MathematicalProgram* prog,
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name, symbolic::internal::DegreeType degree_type);

void PrintPsdConstraintStat(const solvers::MathematicalProgram& prog,
                            const solvers::MathematicalProgramResult& result);
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
