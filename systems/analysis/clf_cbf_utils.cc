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
      auto bnd = binding.evaluator()->lower_bound();
      for (int i = 0; i < bnd.rows(); ++i) {
        if (std::abs(bnd(i)) < zero_tol) {
          bnd(i) = 0;
        }
      }
      binding.evaluator()->set_bounds(bnd, bnd);
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
    drake::log()->error("Failed before backoff with solution_result={}\n",
                        result.get_solution_result());
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

void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    const symbolic::Polynomial& t, int s_degree,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double* rho_sol, symbolic::Polynomial* s_sol) {
  solvers::MathematicalProgram prog;
  prog.AddIndeterminates(x);
  auto rho = prog.NewContinuousVariables<1>("rho")(0);
  const symbolic::Variables x_set(x);
  symbolic::Polynomial s;
  if (s_degree == 0) {
    const auto s_constant = prog.NewContinuousVariables<1>("s_constant")(0);
    prog.AddBoundingBoxConstraint(0, kInf, s_constant);
    s = symbolic::Polynomial({{symbolic::Monomial(), s_constant}});
  } else {
    std::tie(s, std::ignore) = prog.NewSosPolynomial(x_set, s_degree);
  }

  const symbolic::Polynomial ellipsoid_poly =
      internal::EllipsoidPolynomial<symbolic::Variable>(x, x_star, S, rho);
  const symbolic::Polynomial sos_poly = (1 + t) * ellipsoid_poly - s * f;
  prog.AddSosConstraint(sos_poly);
  prog.AddLinearCost(-rho);
  const auto result =
      SearchWithBackoff(&prog, solver_id, solver_options, backoff_scale);
  DRAKE_DEMAND(result.is_success());
  *rho_sol = result.GetSolution(rho);
  *s_sol = result.GetSolution(s);
}

void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    int r_degree, double rho_max, double rho_min,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options, double rho_tol,
    double* rho_sol, symbolic::Polynomial* r_sol) {
  DRAKE_DEMAND(rho_max > rho_min);
  DRAKE_DEMAND(rho_tol > 0);
  const symbolic::Polynomial ellipsoid_quadratic =
      internal::EllipsoidPolynomial(x, x_star, S, 0.);
  auto is_feasible = [&x, &f, &r_degree, &solver_id, &solver_options,
                      &ellipsoid_quadratic, r_sol](double rho) {
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    symbolic::Polynomial r;
    std::tie(r, std::ignore) =
        prog.NewSosPolynomial(symbolic::Variables(x), r_degree);
    prog.AddSosConstraint(-f - r * (rho - ellipsoid_quadratic));
    auto solver = solvers::MakeSolver(solver_id);
    solvers::MathematicalProgramResult result;
    solver->Solve(prog, std::nullopt, solver_options, &result);
    if (result.is_success()) {
      *r_sol = result.GetSolution(r);
      return true;
    } else {
      return false;
    }
  };

  if (!is_feasible(rho_min)) {
    drake::log()->error("MaximizeEllipsoidRho: rho_min={} is infeasible",
                        rho_min);
  }
  if (is_feasible(rho_max)) {
    *rho_sol = rho_max;
    return;
  }
  while (rho_max - rho_min > rho_tol) {
    const double rho_mid = (rho_max + rho_min) / 2;
    if (is_feasible(rho_mid)) {
      rho_min = rho_mid;
    } else {
      rho_max = rho_mid;
    }
  }
  *rho_sol = rho_min;
}

namespace internal {
template <typename RhoType>
symbolic::Polynomial EllipsoidPolynomial(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho) {
  // The ellipsoid polynomial (x−x*)ᵀS(x−x*)−ρ
  symbolic::Polynomial::MapType ellipsoid_poly_map;
  // Add constant term x*ᵀ*S*x* - ρ
  ellipsoid_poly_map.emplace(symbolic::Monomial(),
                             x_star.dot(S * x_star) - rho);
  const Eigen::VectorXd S_times_x_star = (S + S.transpose()) / 2 * x_star;
  for (int i = 0; i < x.rows(); ++i) {
    // Add S(i, i) * x(i)²
    ellipsoid_poly_map.emplace(symbolic::Monomial(x(i), 2), S(i, i));
    // Add -2 * x_starᵀ * S.col(i) * x(i)
    ellipsoid_poly_map.emplace(symbolic::Monomial(x(i)),
                               -2 * S_times_x_star(i));
    for (int j = i + 1; j < x.rows(); ++j) {
      // Add 2*S(i, j) * x(i) * x(j)
      ellipsoid_poly_map.emplace(symbolic::Monomial({{x(i), 1}, {x(j), 1}}),
                                 S(i, j) + S(j, i));
    }
  }
  return symbolic::Polynomial{ellipsoid_poly_map};
}

// Explicit instantiation
template symbolic::Polynomial EllipsoidPolynomial<double>(
    const Eigen::Ref<const VectorX<symbolic::Variable>>&,
    const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::MatrixXd>&, const double&);
template symbolic::Polynomial EllipsoidPolynomial<symbolic::Variable>(
    const Eigen::Ref<const VectorX<symbolic::Variable>>&,
    const Eigen::Ref<const Eigen::VectorXd>&,
    const Eigen::Ref<const Eigen::MatrixXd>&, const symbolic::Variable&);
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
