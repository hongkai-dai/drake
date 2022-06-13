#include "drake/systems/analysis/clf_cbf_utils.h"

#include <fstream>
#include <iostream>
#include <limits>

#include <fmt/format.h>

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
    const std::optional<VectorX<symbolic::Polynomial>>& c, int r_degree,
    const std::optional<std::vector<int>>& c_lagrangian_degrees, double rho_max,
    double rho_min, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options, double rho_tol,
    double* rho_sol, symbolic::Polynomial* r_sol,
    VectorX<symbolic::Polynomial>* c_lagrangian_sol) {
  DRAKE_DEMAND(rho_max > rho_min);
  DRAKE_DEMAND(rho_tol > 0);
  const symbolic::Polynomial ellipsoid_quadratic =
      internal::EllipsoidPolynomial(x, x_star, S, 0.);
  const symbolic::Variables x_set{x};
  auto is_feasible = [&x, &x_set, &f, &c, &r_degree, &c_lagrangian_degrees,
                      &solver_id, &solver_options, &ellipsoid_quadratic, r_sol,
                      c_lagrangian_sol](double rho) {
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    symbolic::Polynomial r;
    std::tie(r, std::ignore) = prog.NewSosPolynomial(x_set, r_degree);
    symbolic::Polynomial sos_condition = -f - r * (rho - ellipsoid_quadratic);
    VectorX<symbolic::Polynomial> c_lagrangian(0);
    if (c.has_value() && c->rows() > 0) {
      c_lagrangian.resize(c->rows());
      for (int i = 0; i < c->rows(); ++i) {
        c_lagrangian(i) =
            prog.NewFreePolynomial(x_set, c_lagrangian_degrees.value()[i]);
      }
      sos_condition -= c_lagrangian.dot(*c);
    }
    prog.AddSosConstraint(sos_condition);
    auto solver = solvers::MakeSolver(solver_id);
    solvers::MathematicalProgramResult result;
    solver->Solve(prog, std::nullopt, solver_options, &result);
    if (result.is_success()) {
      *r_sol = result.GetSolution(r);
      if (c.has_value() && c->rows() > 0) {
        GetPolynomialSolutions(result, c_lagrangian, 0, c_lagrangian_sol);
      } else {
        if (c_lagrangian_sol != nullptr) {
          c_lagrangian_sol->resize(0);
        }
      }
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
  Eigen::MatrixXd A_V_samples;
  VectorX<symbolic::Variable> V_decision_variables;
  Eigen::VectorXd b_V_samples;
  V->EvaluateWithAffineCoefficients(x, x_val, &A_V_samples,
                                    &V_decision_variables, &b_V_samples);
  // Now add the constraint V(xⁱ)<=1
  prog->AddLinearConstraint(
      A_V_samples, Eigen::VectorXd::Constant(A_V_samples.rows(), -kInf),
      Eigen::VectorXd::Ones(A_V_samples.rows()) - b_V_samples,
      V_decision_variables);

  // Now add the constraint Vdot(xⁱ) <= 0
  VectorX<symbolic::Variable> xdot_vars(x.rows());
  VectorX<symbolic::Polynomial> xdot_poly(x.rows());
  for (int i = 0; i < xdot_vars.rows(); ++i) {
    xdot_vars(i) = symbolic::Variable("xd" + std::to_string(i));
    xdot_poly(i) = symbolic::Polynomial(xdot_vars(i));
  }
  const RowVectorX<symbolic::Polynomial> dVdx = V->Jacobian(x);
  const symbolic::Polynomial Vdot = dVdx.dot(xdot_poly);
  Eigen::MatrixXd A_Vdot_samples;
  VectorX<symbolic::Variable> Vdot_decision_variables;
  Eigen::VectorXd b_Vdot_samples;
  VectorX<symbolic::Variable> x_xdot(2 * x.rows());
  x_xdot << x, xdot_vars;
  Eigen::MatrixXd x_xdot_vals(2 * x.rows(), x_val.cols());
  x_xdot_vals.topRows(x.rows()) = x_val;
  x_xdot_vals.bottomRows(x.rows()) = xdot_val;
  Vdot.EvaluateWithAffineCoefficients(x_xdot, x_xdot_vals, &A_Vdot_samples,
                                      &Vdot_decision_variables,
                                      &b_Vdot_samples);
  // Add the cost min ∑ᵢ Vdot(xⁱ)
  prog->AddLinearConstraint(
      A_Vdot_samples, Eigen::VectorXd::Constant(A_Vdot_samples.rows(), -kInf),
      -b_Vdot_samples, Vdot_decision_variables);
  prog->AddLinearCost(A_Vdot_samples.colwise().sum(), b_Vdot_samples.sum(),
                      Vdot_decision_variables);
  return prog;
}

namespace {
Eigen::MatrixXd MeshgridHelper(const std::vector<Eigen::VectorXd>& x,
                               int stop) {
  DRAKE_DEMAND(x.size() >= 2);
  DRAKE_DEMAND(stop >= 1 && stop < static_cast<int>(x.size()));
  if (stop == 1) {
    Eigen::MatrixXd ret(2, x[0].rows() * x[1].rows());
    int pt_count = 0;
    for (int i = 0; i < x[0].rows(); ++i) {
      for (int j = 0; j < x[1].rows(); ++j) {
        ret.col(pt_count++) = Eigen::Vector2d(x[0](i), x[1](j));
      }
    }
    return ret;
  } else {
    const Eigen::MatrixXd ret_prev = MeshgridHelper(x, stop - 1);
    Eigen::MatrixXd ret(ret_prev.rows() + 1, ret_prev.cols() * x[stop].rows());
    for (int i = 0; i < ret_prev.cols(); ++i) {
      for (int j = 0; j < x[stop].rows(); ++j) {
        ret.block(0, x[stop].rows() * i + j, ret_prev.rows(), 1) =
            ret_prev.col(i);
      }
      ret.block(ret_prev.rows(), x[stop].rows() * i, 1, x[stop].rows()) =
          x[stop].transpose();
    }
    return ret;
  }
}
}  // namespace

Eigen::MatrixXd Meshgrid(const std::vector<Eigen::VectorXd>& x) {
  return MeshgridHelper(x, static_cast<int>(x.size()) - 1);
}

void Save(const symbolic::Polynomial& p, const std::string& file_name) {
  std::ofstream outfile;
  outfile.open(file_name, std::ios::out);
  std::unordered_map<symbolic::Variable::Id, int> var_to_index;
  int indeterminate_count = 0;
  for (const auto& x : p.indeterminates()) {
    var_to_index.emplace(x.get_id(), indeterminate_count);
    indeterminate_count++;
  }
  outfile << indeterminate_count << "\n";
  for (const auto& [monomial, coeff] : p.monomial_to_coefficient_map()) {
    DRAKE_DEMAND(symbolic::is_constant(coeff));
    std::string term;
    term.append(fmt::format("{:.10f} ", symbolic::get_constant_value(coeff)));
    for (const auto& [var, degree] : monomial.get_powers()) {
      term.append(
          fmt::format("{} {}, ", var_to_index.at(var.get_id()), degree));
    }
    term.append("\n");
    outfile << term;
  }
  outfile.close();
}

symbolic::Polynomial Load(const symbolic::Variables& indeterminates,
                          const std::string& file_name) {
  std::ifstream infile;
  infile.open(file_name);

  std::string line;
  std::getline(infile, line);
  std::stringstream ss(line);
  int indeterminate_count;
  ss >> indeterminate_count;
  if (indeterminate_count != static_cast<int>(indeterminates.size())) {
    throw std::runtime_error(
        fmt::format("Load: expect {} indeterminates, but got {}",
                    indeterminate_count, indeterminates.size()));
  }
  std::unordered_map<int, symbolic::Variable> index_to_var;
  indeterminate_count = 0;
  for (const auto& var : indeterminates) {
    index_to_var.emplace(indeterminate_count, var);
    indeterminate_count++;
  }
  symbolic::Polynomial::MapType monomial_to_coeff_map;
  while (std::getline(infile, line)) {
    size_t last = 0;
    size_t next = line.find(" ", last);
    const double coeff = std::stod(line.substr(last, next - last));
    last = next + 1;
    std::map<symbolic::Variable, int> monomial_powers;
    while ((next = line.find(",", last)) != std::string::npos) {
      const std::string var_power_pair = line.substr(last, next - last);
      std::stringstream ss_pair(var_power_pair);
      int var_index;
      int degree;
      ss_pair >> var_index >> degree;
      monomial_powers.emplace(index_to_var.at(var_index), degree);
      last = next + 1;
    }
    const symbolic::Monomial monomial{monomial_powers};
    monomial_to_coeff_map.emplace(monomial, coeff);
  }
  infile.close();
  return symbolic::Polynomial(monomial_to_coeff_map);
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
