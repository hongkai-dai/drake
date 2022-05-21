#include "drake/systems/analysis/clf_cbf_utils.h"

#include <limits>

#include "drake/common/text_logging.h"
#include "drake/solvers/choose_best_solver.h"
namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();
void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        Eigen::MatrixXd* coeff_mat,
                        VectorX<symbolic::Variable>* v) {
  const int num_terms = p.monomial_to_coefficient_map().size();
  coeff_mat->resize(x_vals.cols(), num_terms);
  v->resize(num_terms);
  int v_count = 0;
  for (const auto& [monomial, coeff] : p.monomial_to_coefficient_map()) {
    DRAKE_DEMAND(symbolic::is_variable(coeff));
    (*v)(v_count) = symbolic::get_variable(coeff);
    coeff_mat->col(v_count) = monomial.Evaluate(x, x_vals);
    v_count++;
  }
}

void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        VectorX<symbolic::Expression>* p_vals) {
  const int num_terms = p.monomial_to_coefficient_map().size();
  p_vals->resize(x_vals.cols());
  Eigen::MatrixXd monomial_vals(x_vals.cols(), num_terms);
  VectorX<symbolic::Expression> coeffs(num_terms);
  int coeff_count = 0;
  for (const auto& [monomial, coeff] : p.monomial_to_coefficient_map()) {
    coeffs(coeff_count) = coeff;
    monomial_vals.col(coeff_count) = monomial.Evaluate(x, x_vals);
    coeff_count++;
  }
  *p_vals = monomial_vals * coeffs;
}

void SplitCandidateStates(
    const symbolic::Polynomial& h,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    Eigen::MatrixXd* positive_states, Eigen::MatrixXd* negative_states) {
  DRAKE_DEMAND(x.rows() == candidate_safe_states.rows());
  positive_states->resize(candidate_safe_states.rows(),
                          candidate_safe_states.cols());
  negative_states->resize(candidate_safe_states.rows(),
                          candidate_safe_states.cols());
  const Eigen::VectorXd h_vals =
      h.EvaluateIndeterminates(x, candidate_safe_states);
  int positive_states_count = 0;
  int negative_states_count = 0;
  for (int i = 0; i < candidate_safe_states.cols(); ++i) {
    if (h_vals(i) >= 0) {
      positive_states->col(positive_states_count++) =
          candidate_safe_states.col(i);
    } else {
      negative_states->col(negative_states_count++) =
          candidate_safe_states.col(i);
    }
  }
  positive_states->conservativeResize(positive_states->rows(),
                                      positive_states_count);
  negative_states->conservativeResize(negative_states->rows(),
                                      negative_states_count);
}

void RemoveTinyCoeff(solvers::MathematicalProgram* prog, double zero_tol) {
  if (zero_tol > 0) {
    for (const auto& binding : prog->linear_equality_constraints()) {
      binding.evaluator()->RemoveTinyCoefficient(zero_tol);
    }
    for (const auto& binding : prog->linear_constraints()) {
      binding.evaluator()->RemoveTinyCoefficient(zero_tol);
    }
  }
}

solvers::MathematicalProgramResult SearchWithBackoff(
    solvers::MathematicalProgram* prog, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale) {
  DRAKE_DEMAND(prog->linear_costs().size() == 1u);
  DRAKE_DEMAND(prog->quadratic_costs().size() == 0u);
  auto solver = solvers::MakeSolver(solver_id);
  solvers::MathematicalProgramResult result;
  solver->Solve(*prog, std::nullopt, solver_options, &result);
  if (!result.is_success()) {
    drake::log()->error("Failed before backoff\n");
    return result;
  }
  // PrintPsdConstraintStat(*prog, result);
  DRAKE_DEMAND(backoff_scale >= 0 && backoff_scale <= 1);
  if (backoff_scale > 0) {
    drake::log()->info("backoff");
    auto cost = prog->linear_costs()[0];
    prog->RemoveCost(cost);
    const double cost_val = result.get_optimal_cost();
    const double cost_ub = cost_val >= 0 ? (1 + backoff_scale) * cost_val
                                         : (1 - backoff_scale) * cost_val;
    prog->AddLinearConstraint(cost.evaluator()->a(), -kInf,
                              cost_ub - cost.evaluator()->b(),
                              cost.variables());
    solver->Solve(*prog, std::nullopt, solver_options, &result);
    if (!result.is_success()) {
      drake::log()->error("Backoff failed\n");
      return result;
    }
    // PrintPsdConstraintStat(*prog, result);
  }
  return result;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
