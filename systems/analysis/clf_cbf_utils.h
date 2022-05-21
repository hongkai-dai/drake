#pragma once

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
 */
void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    int r_degree, double rho_max, double rho_min,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options, double rho_tol,
    double* rho_sol, symbolic::Polynomial* r_sol);

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
