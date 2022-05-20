#pragma once

#include "drake/common/symbolic.h"
#include "drake/solvers/mathematical_program.h"
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
}  // namespace analysis
}  // namespace systems
}  // namespace drake
