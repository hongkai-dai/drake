#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * For a polynomial whose coefficients are just variables, evaluate the
 * polynomial on a batch of indeterminate values.
 * return (coeff_mat, v) such that coeff_mat.row(i) * v is evaluating p with
 * indeterminates x = x_vals.col(i)
 */
void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        Eigen::MatrixXd* coeff_mat,
                        VectorX<symbolic::Variable>* v);

void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        VectorX<symbolic::Expression>* p_vals);

/**
 * Evaluate h(x) at x = candidate_safe_states, split candidate_safe_states to
 * positive_states and negative_states, such that
 * h(positive_states) >= 0 and h(negative_states) < 0
 */
void SplitCandidateStates(
    const symbolic::Polynomial& h,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    Eigen::MatrixXd* positive_states, Eigen::MatrixXd* negative_states);

/**
 * Remove coefficient with absolute value <= zero_tol.
 */
void RemoveTinyCoeff(solvers::MathematicalProgram* prog, double zero_tol);

solvers::MathematicalProgramResult SearchWithBackoff(
    solvers::MathematicalProgram* prog, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale);

/**
 * Find the largest inscribed ellipsoid {x | (x-x*)ᵀS(x-x*) <= ρ} in the
 * sub-level set {x | f(x)<= 0}. Solve the following problem on the variable
 * s(x), ρ
 * max ρ
 * s.t (1+t(x))((x-x*)ᵀS(x-x*)-ρ) - s(x)*f(x) is sos
 *     s(x) is sos
 */
void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    const symbolic::Polynomial& t, int s_degree,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double* rho_sol, symbolic::Polynomial* s_sol);

/**
 * Find the largest inscribed ellipsoid {x | (x-x*)ᵀS(x-x*) <= ρ} in the
 * sub-level set {x | f(x) <= 0}. Solve the following problem on the variable
 * r(x) through bisecting ρ
 *
 *     max ρ
 *     s.t -f(x) - r(x)*(ρ-(x-x*)ᵀS(x-x*)) is sos
 *         r(x) is sos.
 *
 * Optionally we can specify that we only care about the containment in the
 * algebraic set {x | c(x) = 0}. Namely the intersection of the ellipsoid and
 * the algebraic set {x | c(x) = 0} is contained inside the sub-level set {x |
 * f(x) <= 0}., and we solve the following problem on the variable r(x), t(x)
 * through bisecting ρ
 *
 *     max ρ
 *     s.t -f(x) - r(x)*(ρ-(x-x*)ᵀS(x-x*)) -t(x)ᵀ*c(x) is sos
 *         r(x) is sos.
 */
void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    const std::optional<VectorX<symbolic::Polynomial>>& c, int r_degree,
    const std::optional<std::vector<int>>& c_lagrangian_degrees, double rho_max,
    double rho_min, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options, double rho_tol,
    double* rho_sol, symbolic::Polynomial* r_sol,
    VectorX<symbolic::Polynomial>* c_lagrangian_sol);

void GetPolynomialSolutions(const solvers::MathematicalProgramResult& result,
                            const VectorX<symbolic::Polynomial>& p,
                            double zero_coeff_tol,
                            VectorX<symbolic::Polynomial>* p_sol);

/**
 * Compute the monomial basis for a given set of variables up to a certain
 * degree, but remove the constant term 1 from the basis.
 */
VectorX<symbolic::Monomial> ComputeMonomialBasisNoConstant(
    const symbolic::Variables& vars, int degree,
    symbolic::internal::DegreeType degree_type);

/**
 * Creates a new sos polynomial p(x) which satisfies p(0) = 0
 */
void NewSosPolynomialPassOrigin(solvers::MathematicalProgram* prog,
                                const symbolic::Variables& indeterminates,
                                int degree,
                                symbolic::internal::DegreeType degree_type,
                                symbolic::Polynomial* p,
                                VectorX<symbolic::Monomial>* monomial_basis,
                                MatrixX<symbolic::Expression>* gram);

/**
 * Constructs a program to find Lyapunov candidate V that satisfy the following
 * constraints
 * <pre>
 * min ∑ᵢVdot(xⁱ)
 * s.t V is sos
 *     V(0) = 0
 *     Vdot(xⁱ) <= 0
 *     V(xⁱ) <= 1
 * </pre>
 * @param x The indeterminates
 * @param V_degree The total degree of V
 * @param x_vals The sampled value of x, x_vals.col(i) is xⁱ
 * @param xdot_vals xdot_vals.col(i) is the dynamics derivative at xⁱ
 */
std::unique_ptr<solvers::MathematicalProgram> FindCandidateLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x, int V_degree,
    const Eigen::Ref<const Eigen::MatrixXd>& x_val,
    const Eigen::Ref<const Eigen::MatrixXd>& xdot_val, symbolic::Polynomial* V,
    MatrixX<symbolic::Expression>* V_gram);

/**
 * Each x[i] contains the coordinate along one dimension, returns the matrix
 * such that each column of the matrix corresponds to one point in the meshgrid.
 */
Eigen::MatrixXd Meshgrid(const std::vector<Eigen::VectorXd>& x);

void Save(const symbolic::Polynomial& p, const std::string& file_name);

[[nodiscard]] symbolic::Polynomial Load(
    const symbolic::Variables& indeterminates, const std::string& file_name);

/**
 * Construct a nonlinear optimization problem to search for x with the maximal
 * Vdot max s s.t ∂V/∂x*(f(x)+G(x)uⁱ)≥ s where uⁱ is the i'th vertex in
 * u_vertices.
 */
std::unique_ptr<solvers::MathematicalProgram> ConstructMaxVdotProgram(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Polynomial& V,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    symbolic::Variable* max_Vdot);

namespace internal {
/** The ellipsoid polynomial (x−x*)ᵀS(x−x*)−ρ
 */
template <typename RhoType>
symbolic::Polynomial EllipsoidPolynomial(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho);
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
