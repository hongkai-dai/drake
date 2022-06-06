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

void GetPolynomialSolutions(const solvers::MathematicalProgramResult& result,
                            const VectorX<symbolic::Polynomial>& p,
                            double zero_coeff_tol,
                            VectorX<symbolic::Polynomial>* p_sol) {
  p_sol->resize(p.rows());
  for (int i = 0; i < p_sol->rows(); ++i) {
    (*p_sol)(i) = result.GetSolution(p(i));
    if (zero_coeff_tol > 0) {
      (*p_sol)(i) =
          (*p_sol)(i).RemoveTermsWithSmallCoefficients(zero_coeff_tol);
    }
  }
}

VectorX<symbolic::Monomial> ComputeMonomialBasisNoConstant(
    const symbolic::Variables& vars, int degree,
    symbolic::internal::DegreeType degree_type) {
  const auto m = symbolic::internal::ComputeMonomialBasis<Eigen::Dynamic>(
      vars, degree, degree_type);
  VectorX<symbolic::Monomial> ret(m.rows());
  int index = 0;
  for (int i = 0; i < m.rows(); ++i) {
    if (m(i).total_degree() > 0) {
      ret(index) = m(i);
      index++;
    }
  }
  ret.conservativeResize(index);
  return ret;
}

// Create a new sos polynomial p(x) which satisfies p(0)=0
void NewSosPolynomialPassOrigin(solvers::MathematicalProgram* prog,
                                const symbolic::Variables& indeterminates,
                                int degree,
                                symbolic::internal::DegreeType degree_type,
                                symbolic::Polynomial* p,
                                VectorX<symbolic::Monomial>* monomial_basis,
                                MatrixX<symbolic::Expression>* gram) {
  switch (degree_type) {
    case symbolic::internal::DegreeType::kAny: {
      *monomial_basis = ComputeMonomialBasisNoConstant(
          indeterminates, degree / 2, symbolic::internal::DegreeType::kAny);
      MatrixX<symbolic::Variable> gram_var;
      std::tie(*p, gram_var) = prog->NewSosPolynomial(
          *monomial_basis,
          solvers::MathematicalProgram::NonnegativePolynomial::kSos);
      *gram = gram_var.cast<symbolic::Expression>();
      break;
    }
    case symbolic::internal::DegreeType::kEven: {
      symbolic::Polynomial p_even, p_odd;
      const VectorX<symbolic::Monomial> monomial_basis_even =
          ComputeMonomialBasisNoConstant(indeterminates, degree / 2,
                                         symbolic::internal::DegreeType::kEven);
      const VectorX<symbolic::Monomial> monomial_basis_odd =
          ComputeMonomialBasisNoConstant(indeterminates, degree / 2,
                                         symbolic::internal::DegreeType::kOdd);
      MatrixX<symbolic::Expression> gram_even, gram_odd;
      std::tie(p_even, gram_even) = prog->NewSosPolynomial(monomial_basis_even);
      std::tie(p_odd, gram_odd) = prog->NewSosPolynomial(monomial_basis_odd);
      monomial_basis->resize(monomial_basis_even.rows() +
                             monomial_basis_odd.rows());
      *monomial_basis << monomial_basis_even, monomial_basis_odd;
      gram->resize(monomial_basis->rows(), monomial_basis->rows());
      gram->topLeftCorner(gram_even.rows(), gram_even.cols()) = gram_even;
      gram->bottomRightCorner(gram_odd.rows(), gram_odd.cols()) = gram_odd;
      *p = p_even + p_odd;
      break;
    }
    default: {
      throw std::runtime_error("sos polynomial cannot be odd order.");
    }
  }
}

std::unique_ptr<solvers::MathematicalProgram> FindCandidateLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x, int V_degree,
    const Eigen::Ref<const Eigen::MatrixXd>& x_val,
    const Eigen::Ref<const Eigen::MatrixXd>& xdot_val, symbolic::Polynomial* V,
    MatrixX<symbolic::Expression>* V_gram) {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x);
  const symbolic::Variables x_set(x);
  DRAKE_DEMAND(x_val.rows() == x.rows());
  DRAKE_DEMAND(xdot_val.rows() == x.rows());
  DRAKE_DEMAND(xdot_val.cols() == x_val.cols());
  VectorX<symbolic::Monomial> V_monomials;
  NewSosPolynomialPassOrigin(prog.get(), x_set, V_degree,
                             symbolic::internal::DegreeType::kAny, V,
                             &V_monomials, V_gram);
  // Now add the constraint V(xⁱ)<=1
  const int num_samples = x_val.cols();
  Eigen::MatrixXd V_monomials_vals(V_monomials.rows(), num_samples);
  for (int i = 0; i < V_monomials.rows(); ++i) {
    V_monomials_vals.row(i) = V_monomials(i).Evaluate(x, x_val).transpose();
  }
  for (int i = 0; i < num_samples; ++i) {
    prog->AddLinearConstraint((*V_gram * (V_monomials_vals.col(i) *
                                          V_monomials_vals.col(i).transpose()))
                                  .trace(),
                              -kInf, 1);
  }
  // Now add the constraint Vdot(xⁱ) <= 0
  const RowVectorX<symbolic::Polynomial> dVdx = V->Jacobian(x);
  symbolic::Expression cost;
  for (int i = 0; i < num_samples; ++i) {
    symbolic::Environment env;
    env.insert(x, x_val.col(i));
    const symbolic::Expression dVdx_i =
        (dVdx.dot(xdot_val.col(i))).ToExpression().EvaluatePartial(env);
    cost += dVdx_i;
    prog->AddLinearConstraint(dVdx_i, -kInf, 0);
  }
  prog->AddLinearCost(cost);
  return prog;
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
