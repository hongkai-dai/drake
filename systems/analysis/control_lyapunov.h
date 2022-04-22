#pragma once

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
 * where P is a bounded polytope, we want to find a control Lyapunov function
 * (and region of attraction) for this system as V(x). The control Lyapunov
 * function should satisfy the condition
 *
 *     V(x) > 0 ∀ x ≠ x*                                     (1)
 *     V(x*) = 0                                             (2)
 *     ∀ x satisfying V(x) ≤ ρ, ∃ u ∈ P s.t V̇ < 0            (3)
 *
 * These conditions prove that the sublevel set V(x) ≤ ρ is a region of
 * attraction, that starting from any state within this ROA, there exists
 * control actions that can stabilize the system to x*. Note that
 * V̇(x, u) = ∂V/∂x*f(x)+∂V/∂x*G(x)u. As we assumed that the bounds on the input
 * u is a polytope P. If we write the vertices of P as uᵢ, i = 1, ..., N, since
 * V̇ is a linear function of u, the minimal of min V̇, subject to u ∈ P is
 * obtained in one of the vertices of P. Hence the condition
 *
 *     ∃ u ∈ P s.t V̇(x, u) < 0
 *
 * is equivalent to
 *
 *      min_i V̇(x, uᵢ) < 0
 *
 * We don't know which vertex gives us the minimal, but we can say if the
 * minimal is obtained at the i'th vertex (namely V̇(x, uᵢ)≤ V̇(x, uⱼ)∀ j≠ i),
 * then the minimal has to be negative. Mathematically this means
 * ∀ i, if V̇(x, uᵢ) ≤ V̇(x, uⱼ)∀ uⱼ∈ Neighbour(uᵢ), then V̇(x, uᵢ)< 0
 * where Neighbour(uᵢ) is the set of vertices on polytope P neighbouring uᵢ.
 * As a result, condition (3) is equivalent to the following condition
 * for each i = 1, ..., N
 *
 *     V(x) ≤ ρ, V̇(x, uᵢ) ≤ V̇(x, uⱼ) => V̇(x, uᵢ)<0            (4)
 *
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
   * @param neighbouring_vertices neighbouring_vertices[i] are the indices of
   * the vertices neighbouring u_vertices.col(i)
   */
  SearchControlLyapunov(
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
      const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
      const Eigen::Ref<const Eigen::VectorXd>& x_equilibrium,
      const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
      std::map<int, std::set<int>> neighbouring_vertices,
      const Eigen::Ref<const VectorX<symbolic::Variable>>& x);

 private:
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  Eigen::VectorXd x_equilibrium_;
  Eigen::MatrixXd u_vertices_;
  std::map<int, std::set<int>> neighbouring_vertices_;
  // The indeterminates as the state.
  VectorX<symbolic::Variable> x_;
};

/**
 * Compute V̇(x, u) = ∂V/∂x * (f(x)+G(x)u)
 */
class VdotCalculator {
 public:
  VdotCalculator(const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                 const symbolic::Polynomial& V,
                 const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
                 const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G);

  symbolic::Polynomial Calc(const Eigen::Ref<const Eigen::VectorXd>& u) const;

 private:
  symbolic::Polynomial dVdx_times_f_;
  RowVectorX<symbolic::Polynomial> dVdx_times_G_;
};

/**
 * We need to impose the constraint
 * (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
 * (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
 * We store the monomial basis and the Gram matrix of these two SOS constraint.
 */
struct VdotSosConstraintReturn {
  VdotSosConstraintReturn(int nu)
      : monomials{static_cast<size_t>(nu)}, grams{static_cast<size_t>(nu)} {}

  /**
   * Compute the i'th pair of SOS constraint (the polynomial on the left
   * handside)
   * (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
   * (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
   * @param i Compute the i'th pair.
   * @param result The result after solving the program.
   */
  std::array<symbolic::Polynomial, 2> ComputeSosConstraint(
      int i, const solvers::MathematicalProgramResult& result) const;

  /**
   * If j == 0, compute the SOS constraint (the polynomial on the left handside)
   * (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
   * if j == 1, compute the polynomial on the left handside
   * (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
   * @param i Compute the i'th pair.
   * @param j
   * @param result The result after solving the program.
   */
  symbolic::Polynomial ComputeSosConstraint(
      int i, int j, const solvers::MathematicalProgramResult& result) const;

  // monomials[i][0] is the monomial basis for the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // monomials[i][1] is the monomial basis for the constraint
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  std::vector<std::array<VectorX<symbolic::Monomial>, 2>> monomials;
  // grams[i][0] is the Gram matrix for the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // grams[i][1] is the Gram matrix for the constraint
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  std::vector<std::array<MatrixX<symbolic::Variable>, 2>> grams;
};

/**
 * For u bounded in a unit box -1 <= u <= 1.
 * Given the control Lyapunov function candidate V, together with the
 * Lagrangian multipliers lᵢ₁(x), lᵢ₂(x), search for b and Lagrangian
 * multipliers lᵢ₃(x), lᵢ₄(x), lᵢ₅(x), lᵢ₆(x), satisfying the following
 * constraint
 *
 *     ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
 *     (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >=
 * 0
 *     (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
 *     lᵢ₃(x) >= 0, lᵢ₄(x) >= 0, lᵢ₅(x) >= 0, lᵢ₆(x) >= 0
 *
 * The variables are ε₂, b(x), lₖ₃(x), lₖ₄(x), lₖ₅(x), lₖ₆(x)
 * This is the starting step of the search, where we can get a good guess
 * of V(x), lₖ₁(x), lₖ₂(x).
 */
class SearchLagrangianAndBGivenVBoxInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SearchLagrangianAndBGivenVBoxInputBound)

  /**
   * @param l_given l_given[i][0] is lᵢ₁(x), l_given[i][1] is lᵢ₂(x).
   * @param lagrangian_degrees lagrangian_degrees[i][j] is the degree of the
   * Lagrangian multiplier lᵢⱼ₊₁(x).
   * @param b_degrees b_degrees[i] is the degree of the polynomial b(i).
   */
  SearchLagrangianAndBGivenVBoxInputBound(
      symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
      MatrixX<symbolic::Polynomial> G,
      const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
      std::vector<std::array<int, 6>> lagrangian_degrees,
      std::vector<int> b_degrees, VectorX<symbolic::Variable> x);

  const std::vector<std::array<symbolic::Polynomial, 6>>& lagrangians() const {
    return l_;
  }

  /**
   * The gram matrix of each lagrangian. Note that lagrangian_grams()[i][0]
   * and lagrangian_grams()[i][1] are empty.
   */
  const std::vector<std::array<MatrixX<symbolic::Variable>, 6>>&
  lagrangian_grams() const {
    return lagrangian_grams_;
  }

  const solvers::MathematicalProgram& prog() const { return prog_; }

  solvers::MathematicalProgram* get_mutable_prog() { return &prog_; }

  const symbolic::Variable& deriv_eps() const { return deriv_eps_; }

  const VectorX<symbolic::Polynomial>& b() const { return b_; }

  const VdotSosConstraintReturn& vdot_sos_constraint() const {
    return vdot_sos_constraint_;
  }

  /**
   * The return struct in AddEllipsoidInRoaConstraint
   */
  struct EllipsoidInRoaReturn {
    // The size of the ellipoid.
    symbolic::Variable rho;
    // The monomial of the lagrangian multiplier s(x).
    symbolic::Polynomial s;
    // The Gram matrix of the lagrangian multiplier s(x).
    MatrixX<symbolic::Variable> s_gram;
    // The monomials of the constraint (1+t(x))((x−x*)ᵀS(x−x*)−ρ) −
    // s(x)(V(x)−1)
    VectorX<symbolic::Monomial> constraint_monomials;
    // The Gram matrix of the constraint
    // (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1)
    MatrixX<symbolic::Variable> constraint_gram;
  };

 private:
  // Step 1 of Search() function. Search for Lagrangian multipliers and b.
  void SearchLagrangianAndB(
      const symbolic::Polynomial& V,
      const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
      const std::vector<std::array<int, 6>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
      const symbolic::Polynomial& t, double deriv_eps_lower,
      double deriv_eps_upper, const solvers::SolverId& solver_id,
      const solvers::SolverOptions& solver_options, double backoff_scale,
      double* deriv_eps, VectorX<symbolic::Polynomial>* b,
      std::vector<std::array<symbolic::Polynomial, 6>>* l, double* rho) const;

  solvers::MathematicalProgram prog_;
  symbolic::Polynomial V_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  int nx_{};
  int nu_{};
  std::vector<std::array<symbolic::Polynomial, 6>> l_;
  std::vector<std::array<int, 6>> lagrangian_degrees_;
  std::vector<std::array<MatrixX<symbolic::Variable>, 6>> lagrangian_grams_;
  std::vector<int> b_degrees_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  VectorX<symbolic::Polynomial> b_;
  symbolic::Variable deriv_eps_;

  VdotSosConstraintReturn vdot_sos_constraint_;
};

/**
 * For u bounded in a unit box -1 <= u <= 1.
 * Given the Lagrangian multiplier, find the Lyapunov function V and slack
 * polynomials b, satisfying the condition
 *
 *     V(x) >= ε₁xᵀx
 *     V(0) = 0
 *     ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
 *     (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >=
 * 0 (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
 * where lᵢⱼ(x) are all given.
 */
class SearchLyapunovGivenLagrangianBoxInputBound {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(SearchLyapunovGivenLagrangianBoxInputBound)

  /**
   * For a control affine system with 0 being the goal state, we find the
   * Lyapunov function given Lagrangian multipliers.
   * @param f The dynamics is ẋ = f(x)+G(x)u
   * @param G The dynamics is ẋ = f(x)+G(x)u
   * @param V_monomial The monomial basis m(x) for Lyapunov function V(x) =
   * m(x)ᵀQm(x) where Q is a PSD matrix.
   * @param positivity_eps ε₁ in the documentation above. Used to constrain
   * V(x) to be a positive definite function.
   * @param deriv_eps ε₂ in the documentation above. The rate of exponential
   * convergence.
   * @param l_given l_given[i][j] is lᵢⱼ in the documentation above.
   * @param b_degrees b_degrees[i] is the degree of the polynomial bᵢ(x).
   * @param x The indeterminates for the state.
   */
  SearchLyapunovGivenLagrangianBoxInputBound(
      VectorX<symbolic::Polynomial> f, MatrixX<symbolic::Polynomial> G,
      const Eigen::Ref<const VectorX<symbolic::Monomial>>& V_monomial,
      double positivity_eps, double deriv_eps,
      std::vector<std::array<symbolic::Polynomial, 6>> l_given,
      const std::vector<int>& b_degrees, VectorX<symbolic::Variable> x);

  const solvers::MathematicalProgram& prog() const { return prog_; }

  solvers::MathematicalProgram* get_mutable_prog() { return &prog_; }

  const VectorX<symbolic::Polynomial>& b() const { return b_; }

  const symbolic::Polynomial& V() const { return V_; }

  /**
   * The Gram matrix of the positivity constraint
   * V(x) >= ε₁xᵀx
   */
  const MatrixX<symbolic::Variable>& positivity_constraint_gram() const {
    return positivity_constraint_gram_;
  }

  /**
   * The monomial basis m(x) for the positivity constraint
   * V(x) - ε₁xᵀx = m(x)ᵀQm(x)
   */
  const VectorX<symbolic::Monomial>& positivity_constraint_monomial() const {
    return positivity_constraint_monomial_;
  }

  /**
   * The return struct in AddEllipsoidInRoaConstraint
   */
  struct EllipsoidInRoaReturn {
    // The size of the ellipoid.
    symbolic::Variable rho;
    // The monomials of the constraint (1+t(x))((x−x*)ᵀS(x−x*)−ρ) −
    // s(x)(V(x)−1)
    VectorX<symbolic::Monomial> constraint_monomials;
    // The Gram matrix of the constraint
    // (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1)
    MatrixX<symbolic::Variable> constraint_gram;
  };

  /**
   * Add the constraint that an ellipsoid {x|(x−x*)ᵀS(x−x*)<=ρ} is within the
   * region-of-attraction (ROA) {x | V(x) <= 1}.
   * We enforce the constraint
   * (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1) is sos
   * @param x_star The center of the ellipsoid.
   * @param S The shape of the ellipsoid.
   * @param t The Lagrangian multipler t(x).
   * @param s The Lagrangian multiplier s(x).
   */
  EllipsoidInRoaReturn AddEllipsoidInRoaConstraint(
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& t,
      const symbolic::Polynomial& s);

 private:
  solvers::MathematicalProgram prog_;
  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  double deriv_eps_;
  std::vector<std::array<symbolic::Polynomial, 6>> l_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  symbolic::Polynomial V_;
  VectorX<symbolic::Monomial> positivity_constraint_monomial_;
  MatrixX<symbolic::Variable> positivity_constraint_gram_;
  int nx_;
  int nu_;
  VectorX<symbolic::Polynomial> b_;
  VdotSosConstraintReturn vdot_sos_constraint_;
};

/**
 * This is the Lagrangian step in ControlLaypunovBoxInputBound. The control
 * Lyapunov function V, together with b are fixed, and we search for the
 * Lagrangian multipliers
 * lᵢ₁(x), lᵢ₂(x), lᵢ₃(x), lᵢ₄(x), lᵢ₅(x), lᵢ₆(x) satisfying the constraints
 *
 *     (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) ≥ 0
 *     (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) ≥ 0
 *     lᵢ₁(x) >= 0, lᵢ₃(x) >= 0, lᵢ₅(x) >= 0,
 *     lᵢ₂(x) >= 0, lᵢ₄(x) >= 0, lᵢ₆(x) >= 0
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
   * @param lagrangian_degrees lagrangian_degrees[i][j] is the degree of the
   * Lagrangian multiplier lᵢⱼ₊₁(x).
   */
  SearchLagrangianGivenVBoxInputBound(
      symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
      MatrixX<symbolic::Polynomial> G, VectorX<symbolic::Polynomial> b,
      VectorX<symbolic::Variable> x,
      std::vector<std::array<int, 6>> lagrangian_degrees);

  /**
   * If j = 0, then this is the program
   *
   *     (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) ≥ 0
   *     lᵢ₁(x) >= 0, lᵢ₃(x) >= 0, lᵢ₅(x) >= 0,
   *
   * If j = 1, then this is the program
   *
   *     (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1−V) ≥ 0
   *     lᵢ₂(x) >= 0, lᵢ₄(x) >= 0, lᵢ₆(x) >= 0
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

  const std::vector<std::array<symbolic::Polynomial, 6>>& lagrangians() const {
    return l_;
  }

  const std::vector<std::array<MatrixX<symbolic::Variable>, 6>>&
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
  VectorX<symbolic::Polynomial> b_;
  VectorX<symbolic::Variable> x_;
  symbolic::Variables x_set_;
  std::vector<std::array<std::unique_ptr<solvers::MathematicalProgram>, 2>>
      progs_;
  int nu_;
  int nx_;
  std::vector<std::array<symbolic::Polynomial, 6>> l_;
  std::vector<std::array<int, 6>> lagrangian_degrees_;
  std::vector<std::array<MatrixX<symbolic::Variable>, 6>> lagrangian_grams_;
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
 * Lagrangian multiplier lᵢ₁(x), lᵢ₂(x),lᵢ₃(x), lᵢ₄(x)with the constraint
 *
 *     (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x) − bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x)>=0
 *     (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x) − bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x)>=0
 *     lᵢ₁(x) >= 0, lᵢ₂(x) >= 0, lᵢ₃(x) >= 0, lᵢ₄(x) >= 0
 *
 * To summarize, in order to prove the control Lyapunov function with region
 * of attraction V(x) ≤ 1, we impose the following constraint
 *
 *    V(x) ≥ ε₁xᵀx
 *    V(0) = 0
 *    ∂V/∂x*f(x) + ε₂V = ∑ᵢ bᵢ(x)
 *    (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) ≥ 0
 *    (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) ≥ 0
 *     lᵢ₁(x) ≥ 0, lᵢ₂(x) ≥ 0,
 *     lᵢ₃(x) ≥ 0, lᵢ₄(x) ≥ 0, lᵢ₅(x) ≥ 0, lᵢ₆(x) ≥ 0
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
    std::vector<std::array<symbolic::Polynomial, 6>> l;
    symbolic::Polynomial s;
    double rho;
  };

  struct SearchOptions {
    solvers::SolverId lyap_step_solver{solvers::MosekSolver::id()};
    solvers::SolverId lagrangian_step_solver{solvers::MosekSolver::id()};
    int bilinear_iterations{10};
    // Stop when the improvement on rho is below this tolerance.
    double rho_converge_tol{1E-5};
    // Back off in each steps.
    double backoff_scale{0.};
    std::optional<solvers::SolverOptions> lagrangian_step_solver_options{
        std::nullopt};
    std::optional<solvers::SolverOptions> lyap_step_solver_options{
        std::nullopt};
  };

  /**
   * Given V_init(x) and lᵢ₀(x), lᵢ₁(x), we search the control Lyapunov function
   * and maximize the ROA through the following process
   * 1. Fix V_init(x), lᵢ₁(x), lᵢ₂(x), search for lᵢ₃(x),...,lᵢ₆(x), s(x),
   *    bᵢ(x).
   * 2. Fix Lagrangian multipliers lᵢ₁(x),..., lᵢ₆(x), s(x), search V(x) and
   *    bᵢ(x).
   * 3. Fix V(x), bᵢ(x), and search for Lagrangian multipliers
   *    lᵢ₁(x),..., lᵢ₆(x), s(x). Go to step 2.
   *
   * @param V_monomial The monomial basis m(x) of the Lyapunov function V(x) =
   * m(x)ᵀQm(x).
   */
  SearchReturn Search(
      const symbolic::Polynomial& V_init,
      const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
      const std::vector<std::array<int, 6>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
      const symbolic::Polynomial& t_given,
      const Eigen::Ref<const VectorX<symbolic::Monomial>>& V_monomial,
      double deriv_eps_lower, double deriv_eps_upper,
      const SearchOptions& options) const;

 private:
  // Step 1 in Search() function.
  void SearchLagrangianAndB(
      const symbolic::Polynomial& V,
      const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
      const std::vector<std::array<int, 6>>& lagrangian_degrees,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
      const symbolic::Polynomial& t, double deriv_eps_lower,
      double deriv_eps_upper, const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double backoff_scale, double* deriv_eps, VectorX<symbolic::Polynomial>* b,
      std::vector<std::array<symbolic::Polynomial, 6>>* l, double* rho,
      symbolic::Polynomial* s) const;

  // Step 2 in Search() function.
  void SearchLyapunov(
      const std::vector<std::array<symbolic::Polynomial, 6>>& l,
      const std::vector<int>& b_degrees,
      const Eigen::Ref<const VectorX<symbolic::Monomial>>& V_monomial,
      double positivity_eps, double deriv_eps,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& s,
      const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double backoff_scale, symbolic::Polynomial* V,
      VectorX<symbolic::Polynomial>* b, double* rho) const;

  // Step 3 in Search() function.
  void SearchLagrangian(
      const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& b,
      const std::vector<std::array<int, 6>>& lagrangian_degrees,
      const Eigen::Ref<const Eigen::VectorXd>& x_star,
      const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
      const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
      const std::optional<solvers::SolverOptions>& solver_options,
      double backoff_scale, std::vector<std::array<symbolic::Polynomial, 6>>* l,
      symbolic::Polynomial* s, double* rho) const;

  VectorX<symbolic::Polynomial> f_;
  MatrixX<symbolic::Polynomial> G_;
  // The indeterminates as the state.
  VectorX<symbolic::Variable> x_;
  double positivity_eps_;
};

/**
 * Find the largest inscribed ellipsoid {x | (x-x*)ᵀS(x-x*) <= ρ} in the
 * sub-level set {x | V(x)<= 1}. Solve the following problem on the variable
 * s(x), ρ
 * max ρ
 * s.t (1+t(x))((x-x*)ᵀS(x-x*)-ρ) - s(x)(V(x)-1) is sos
 *     s(x) is sos
 */
void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& V,
    const symbolic::Polynomial& t, int s_degree,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double* rho_sol, symbolic::Polynomial* s_sol);

namespace internal {
/** The ellipsoid polynomial (x−x*)ᵀS(x−x*)−ρ
 */
template <typename RhoType>
symbolic::Polynomial EllipsoidPolynomial(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho);

/**
 * Compute the monomial basis for a given set of variables up to a certain
 * degree, but remove the constant term 1 from the basis.
 */
VectorX<symbolic::Monomial> ComputeMonomialBasisNoConstant(
    const symbolic::Variables& vars, int degree);
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
