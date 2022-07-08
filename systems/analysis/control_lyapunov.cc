#include "drake/systems/analysis/control_lyapunov.h"

#include <limits.h>

#include <iostream>

#include "drake/common/text_logging.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/solve.h"
#include "drake/solvers/sos_basis_generator.h"
#include "drake/systems/analysis/clf_cbf_utils.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

namespace {
[[maybe_unused]] void PrintPolynomial(const symbolic::Polynomial& p) {
  for (const auto& [monomial, coeff] : p.monomial_to_coefficient_map()) {
    std::cout << monomial << " --> " << coeff.Expand() << "\n";
  }
}

[[maybe_unused]] double SmallestCoeff(const symbolic::Polynomial& p) {
  double ret = kInf;
  for (const auto& [_, coeff] : p.monomial_to_coefficient_map()) {
    DRAKE_DEMAND(symbolic::is_constant(coeff));
    const double coeff_val = symbolic::get_constant_value(coeff);
    if (std::abs(coeff_val) < std::abs(ret)) {
      ret = coeff_val;
    }
  }
  return ret;
}

// add the constraint that the intersection of the ellipsoid and the algebraic
// set c(x) is within the sub-level set {x | V(x)<=d},
// Namely
// d−V(x) − r(x)(ρ−(x−x*)ᵀS(x−x*)) - t(x) * c(x) is sos
// and r(x) is sos.
template <typename T>
void AddEllipsoidInRoaConstraint(
    solvers::MathematicalProgram* prog, const VectorX<symbolic::Variable>& x,
    const T& d, const symbolic::Polynomial& V,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, double rho, int r_degree,
    const std::optional<VectorX<symbolic::Polynomial>>& c,
    const std::optional<std::vector<int>>& c_lagrangian_degrees,
    symbolic::Polynomial* r, VectorX<symbolic::Polynomial>* c_lagrangian) {
  const symbolic::Variables x_set(x);
  std::tie(*r, std::ignore) = prog->NewSosPolynomial(x_set, r_degree);
  const symbolic::Polynomial ellipsoid_poly =
      internal::EllipsoidPolynomial(x, x_star, S, rho);
  symbolic::Polynomial sos_condition =
      symbolic::Polynomial({{symbolic::Monomial(), d}}) - V +
      (*r) * ellipsoid_poly;
  if (c.has_value() && c->rows() > 0) {
    DRAKE_DEMAND(c_lagrangian_degrees.has_value() &&
                 static_cast<int>(c_lagrangian_degrees->size()) == c->rows());
    c_lagrangian->resize(c->rows());
    for (int i = 0; i < c->rows(); ++i) {
      (*c_lagrangian)(i) =
          prog->NewFreePolynomial(x_set, (*c_lagrangian_degrees)[i]);
    }
    sos_condition -= c_lagrangian->dot(*c);
  }
  prog->AddSosConstraint(sos_condition);
}

}  // namespace

ControlLyapunov::ControlLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const std::optional<symbolic::Polynomial>& dynamics_denominator,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_constraints)
    : x_{x},
      x_set_{x_},
      f_{f},
      G_{G},
      dynamics_denominator_{
          dynamics_denominator.value_or(symbolic::Polynomial(1))},
      u_vertices_{u_vertices},
      state_constraints_{state_constraints} {
  DRAKE_ASSERT(f.rows() == G.rows());
  const symbolic::Variables x_set(x_);
  for (int i = 0; i < f.rows(); ++i) {
    DRAKE_DEMAND(f(i).indeterminates().IsSubsetOf(x_set));
    for (int j = 0; j < G.cols(); ++j) {
      DRAKE_DEMAND(G(i, j).indeterminates().IsSubsetOf(x_set));
    }
  }
  const int num_u = G.cols();
  DRAKE_DEMAND(u_vertices.rows() == num_u);
}

void ControlLyapunov::AddControlLyapunovConstraint(
    solvers::MathematicalProgram* prog, const VectorX<symbolic::Variable>& x,
    const symbolic::Polynomial& lambda0, const VectorX<symbolic::Polynomial>& l,
    const symbolic::Polynomial& V, double rho,
    const Eigen::MatrixXd& u_vertices, double deriv_eps,
    const VectorX<symbolic::Polynomial>& p, symbolic::Polynomial* vdot_poly,
    VectorX<symbolic::Monomial>* monomials,
    MatrixX<symbolic::Variable>* gram) const {
  // First compute the polynomial xᵀx
  symbolic::Polynomial::MapType x_square_map;
  for (int i = 0; i < x.rows(); ++i) {
    x_square_map.emplace(symbolic::Monomial(x(i), 2), 1);
  }
  const symbolic::Polynomial x_square{x_square_map};
  *vdot_poly = (1 + lambda0) * x_square * (V - rho) * dynamics_denominator_;
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  const symbolic::Polynomial dVdx_times_f = dVdx.dot(f_);
  *vdot_poly -=
      l.sum() * (dVdx_times_f + deriv_eps * V * dynamics_denominator_);
  *vdot_poly -= (dVdx * G_ * u_vertices).dot(l);
  if (state_constraints_.rows() > 0) {
    *vdot_poly -= p.dot(state_constraints_);
  }
  std::tie(*gram, *monomials) = prog->AddSosConstraint(
      *vdot_poly, solvers::MathematicalProgram::NonnegativePolynomial::kSos,
      "Vd");
  if (state_constraints_.rows() == 0) {
    // If there is no state constraints, then because V(x) >= 0 and V(0) = 0, we
    // know that the monomial basis of V cannot contain 1, and this constraint
    // cannot contain 1 in its basis either.
    for (int i = 0; i < monomials->rows(); ++i) {
      if ((*monomials)(i).total_degree() == 0) {
        throw std::runtime_error(
            "This polynomial should not contain a constant term");
      }
    }
  }
}

std::unique_ptr<solvers::MathematicalProgram>
ControlLyapunov::ConstructLagrangianProgram(
    const symbolic::Polynomial& V, double rho, double deriv_eps,
    int lambda0_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& p_degrees, symbolic::Polynomial* lambda0,
    MatrixX<symbolic::Variable>* lambda0_gram, VectorX<symbolic::Polynomial>* l,
    std::vector<MatrixX<symbolic::Variable>>* l_grams,
    VectorX<symbolic::Polynomial>* p, symbolic::Polynomial* vdot_sos,
    VectorX<symbolic::Monomial>* vdot_monomials,
    MatrixX<symbolic::Variable>* vdot_gram) const {
  // Make sure that V(0) = 0
  auto it_V_constant =
      V.monomial_to_coefficient_map().find(symbolic::Monomial());
  if (it_V_constant != V.monomial_to_coefficient_map().end()) {
    if (!symbolic::is_constant(it_V_constant->second) &&
        symbolic::get_constant_value(it_V_constant->second)) {
      throw std::runtime_error(
          "ConstructLagrangianProgram: V should have no constant term.");
    }
  }
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  // Now construct Lagrangian multipliers.
  std::tie(*lambda0, *lambda0_gram) =
      prog->NewSosPolynomial(x_set_, lambda0_degree);
  const int num_u_vertices = u_vertices_.cols();
  l->resize(num_u_vertices);
  l_grams->resize(num_u_vertices);
  for (int i = 0; i < num_u_vertices; ++i) {
    std::tie((*l)(i), (*l_grams)[i]) =
        prog->NewSosPolynomial(x_set_, l_degrees[i]);
  }
  // Construct the Lagrangian multiplier p(x) for the state constraints.
  p->resize(state_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(p_degrees.size()) == state_constraints_.rows());
  for (int i = 0; i < p->rows(); ++i) {
    (*p)(i) = prog->NewFreePolynomial(x_set_, p_degrees[i], "P");
  }
  this->AddControlLyapunovConstraint(prog.get(), x_, *lambda0, *l, V, rho,
                                     u_vertices_, deriv_eps, *p, vdot_sos,
                                     vdot_monomials, vdot_gram);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlLyapunov::ConstructLagrangianProgram(
    const symbolic::Polynomial& V, const symbolic::Polynomial& lambda0,
    int d_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& p_degrees, double deriv_eps,
    VectorX<symbolic::Polynomial>* l,
    std::vector<MatrixX<symbolic::Variable>>* l_grams,
    VectorX<symbolic::Polynomial>* p, symbolic::Variable* rho,
    symbolic::Polynomial* vdot_sos, VectorX<symbolic::Monomial>* vdot_monomials,
    MatrixX<symbolic::Variable>* vdot_gram) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  *rho = prog->NewContinuousVariables<1>("rho")(0);
  l->resize(u_vertices_.cols());
  l_grams->resize(u_vertices_.cols());
  DRAKE_DEMAND(static_cast<int>(l_degrees.size()) == u_vertices_.cols());
  for (int i = 0; i < u_vertices_.cols(); ++i) {
    std::tie((*l)(i), (*l_grams)[i]) = prog->NewSosPolynomial(
        x_set_, l_degrees[i],
        solvers::MathematicalProgram::NonnegativePolynomial::kSos,
        fmt::format("L{}", i));
  }
  // Constructs Lagrangian multiplier p(x) for state constraints.
  p->resize(state_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(p_degrees.size()) == state_constraints_.rows());
  for (int i = 0; i < p->rows(); ++i) {
    (*p)(i) = prog->NewFreePolynomial(x_set_, p_degrees[i], "P");
  }
  // construct the polynomial (xᵀx)ᵈ
  DRAKE_DEMAND(d_degree >= 1);
  using std::pow;
  const symbolic::Polynomial x_square_power(
      pow(x_.cast<symbolic::Expression>().dot(x_), d_degree));
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x_);
  const RowVectorX<symbolic::Polynomial> dVdx_times_G = dVdx * G_;
  *vdot_sos =
      (symbolic::Polynomial(1) + lambda0) * x_square_power *
          (V - symbolic::Polynomial({{symbolic::Monomial(), *rho}})) *
          dynamics_denominator_ -
      l->sum() * (dVdx.dot(f_) + deriv_eps * V * dynamics_denominator_) -
      l->dot(dVdx_times_G * u_vertices_) - p->dot(state_constraints_);
  std::tie(*vdot_gram, *vdot_monomials) = prog->AddSosConstraint(*vdot_sos);
  prog->AddLinearCost(-Vector1d::Ones(), 0, Vector1<symbolic::Variable>(rho));
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlLyapunov::ConstructLyapunovProgram(
    const symbolic::Polynomial& lambda0, const VectorX<symbolic::Polynomial>& l,
    int V_degree, double rho, double positivity_eps, int positivity_d,
    const std::vector<int>& positivity_eq_lagrangian_degrees,
    const std::vector<int>& p_degrees, double deriv_eps,
    symbolic::Polynomial* V,
    VectorX<symbolic::Polynomial>* positivity_eq_lagrangian,
    VectorX<symbolic::Polynomial>* p) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  VectorX<symbolic::Monomial> V_monomial;

  // TODO(hongkai.dai): if the dynamics is symmetric, then use an even V.
  const symbolic::internal::DegreeType V_degree_type{
      symbolic::internal::DegreeType::kAny};
  // Since V(0) = 0, V(x) can't have a constant term.
  // If state_constraints is empty, then V(x) >= 0 ∀x, hence V(x) can't have a
  // linear term either.
  // If state_constraints doesn't have a linear term in variable x(i), then V(x)
  // can't have the term x(i) either.
  if (state_constraints_.rows() == 0) {
    *V = internal::NewFreePolynomialNoConstantOrLinear(
        prog.get(), x_set_, V_degree, "V", V_degree_type);
  } else {
    const symbolic::Variables no_linear_term_variables =
        FindNoLinearTermVariables(x_set_, state_constraints_);
    unused(no_linear_term_variables);
    // Not sure why, but if I use no_linear_term_variables in the line below,
    // quadrotor2d_trig_clf_demo says it is infeasible to find V, although
    // feasible V really doesn't have linear terms for these variables.
    *V = NewFreePolynomialPassOrigin(prog.get(), x_set_, V_degree, "V",
                                     V_degree_type, symbolic::Variables());
  }
  symbolic::Polynomial positivity_sos_condition = *V;
  DRAKE_DEMAND(positivity_d >= 0);
  if (positivity_eps != 0) {
    positivity_sos_condition -=
        positivity_eps *
        symbolic::Polynomial(
            pow(x_.cast<symbolic::Expression>().dot(x_), positivity_d));
  }
  positivity_eq_lagrangian->resize(state_constraints_.rows());
  for (int i = 0; i < state_constraints_.rows(); ++i) {
    (*positivity_eq_lagrangian)(i) =
        prog->NewFreePolynomial(x_set_, positivity_eq_lagrangian_degrees[i]);
  }
  positivity_sos_condition -= positivity_eq_lagrangian->dot(state_constraints_);
  prog->AddSosConstraint(positivity_sos_condition);

  symbolic::Polynomial vdot_poly;
  VectorX<symbolic::Monomial> vdot_monomials;
  MatrixX<symbolic::Variable> vdot_gram;
  // Constructs Lagrangian multiplier p(x) for state constraints.
  p->resize(state_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(p_degrees.size()) == state_constraints_.rows());
  for (int i = 0; i < p->rows(); ++i) {
    (*p)(i) = prog->NewFreePolynomial(x_set_, p_degrees[i], "P");
  }
  this->AddControlLyapunovConstraint(prog.get(), x_, lambda0, l, *V, rho,
                                     u_vertices_, deriv_eps, *p, &vdot_poly,
                                     &vdot_monomials, &vdot_gram);
  return prog;
}

bool ControlLyapunov::SearchLagrangian(
    const symbolic::Polynomial& V, double rho, int lambda0_degree,
    const std::vector<int>& l_degrees, const std::vector<int>& p_degrees,
    double deriv_eps, const ControlLyapunov::SearchOptions& search_options,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    VectorX<symbolic::Polynomial>* p_sol) const {
  symbolic::Polynomial lambda0;
  MatrixX<symbolic::Variable> lambda0_gram;
  VectorX<symbolic::Polynomial> l;
  std::vector<MatrixX<symbolic::Variable>> l_grams;
  VectorX<symbolic::Polynomial> p;
  symbolic::Polynomial vdot_sos;
  VectorX<symbolic::Monomial> vdot_monomials;
  MatrixX<symbolic::Variable> vdot_gram;
  auto prog_lagrangian = this->ConstructLagrangianProgram(
      V, rho, deriv_eps, lambda0_degree, l_degrees, p_degrees, &lambda0,
      &lambda0_gram, &l, &l_grams, &p, &vdot_sos, &vdot_monomials, &vdot_gram);
  RemoveTinyCoeff(prog_lagrangian.get(),
                  search_options.lagrangian_tiny_coeff_tol);
  solvers::MathematicalProgramResult result_lagrangian;
  drake::log()->info("Search Lagrangian");
  drake::log()->info("Smallest coeff {}", SmallestCoeff(*prog_lagrangian));
  solvers::MakeSolver(search_options.lagrangian_step_solver)
      ->Solve(*prog_lagrangian, std::nullopt,
              search_options.lagrangian_step_solver_options,
              &result_lagrangian);
  if (result_lagrangian.is_success()) {
    // internal::PrintPsdConstraintStat(*prog_lagrangian,
    // result_lagrangian);
    *lambda0_sol = result_lagrangian.GetSolution(lambda0);
    if (search_options.lsol_tiny_coeff_tol > 0) {
      *lambda0_sol = lambda0_sol->RemoveTermsWithSmallCoefficients(
          search_options.lsol_tiny_coeff_tol);
    }
    GetPolynomialSolutions(result_lagrangian, l,
                           search_options.lsol_tiny_coeff_tol, l_sol);
    GetPolynomialSolutions(result_lagrangian, p,
                           search_options.lsol_tiny_coeff_tol, p_sol);
  } else {
    drake::log()->error("Failed to find Lagrangian.");
    return false;
  }
  return true;
}

bool ControlLyapunov::FindRhoBinarySearch(
    const symbolic::Polynomial& V, double rho_min, double rho_max,
    double rho_tol, int lambda0_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& p_degrees, double deriv_eps,
    const SearchOptions& search_options, double* rho_sol,
    symbolic::Polynomial* lambda0, VectorX<symbolic::Polynomial>* l,
    VectorX<symbolic::Polynomial>* p) const {
  auto is_rho_feasible = [this, &V, lambda0_degree, &l_degrees, &p_degrees,
                          deriv_eps, &search_options, lambda0, l,
                          p](double rho) {
    return this->SearchLagrangian(V, rho, lambda0_degree, l_degrees, p_degrees,
                                  deriv_eps, search_options, lambda0, l, p);
  };
  DRAKE_DEMAND(rho_max >= rho_min);
  DRAKE_DEMAND(rho_tol > 0);
  if (is_rho_feasible(rho_max)) {
    *rho_sol = rho_max;
    return true;
  }
  if (!is_rho_feasible(rho_min)) {
    *rho_sol = -kInf;
    return false;
  }
  while (rho_max - rho_min > rho_tol) {
    const double rho_mid = (rho_max + rho_min) / 2;
    drake::log()->info(fmt::format("rho_max={}, rho_min={}, rho_mid={}",
                                   rho_max, rho_min, rho_mid));
    if (is_rho_feasible(rho_mid)) {
      rho_min = rho_mid;
    } else {
      rho_max = rho_mid;
    }
  }
  *rho_sol = rho_min;
  return true;
}

void ControlLyapunov::Search(
    const symbolic::Polynomial& V_init, int lambda0_degree,
    const std::vector<int>& l_degrees, int V_degree, double positivity_eps,
    int positivity_d, const std::vector<int>& positivity_eq_lagrangian_degrees,
    const std::vector<int>& p_degrees,
    const std::vector<int>& ellipsoid_c_lagrangian_degrees, double deriv_eps,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int r_degree,
    const SearchOptions& search_options,
    const RhoBisectionOption& rho_bisection_option, symbolic::Polynomial* V_sol,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    symbolic::Polynomial* r_sol, VectorX<symbolic::Polynomial>* p_sol,
    VectorX<symbolic::Polynomial>* positivity_eq_lagrangian_sol,
    double* rho_sol,
    VectorX<symbolic::Polynomial>* ellipsoid_c_lagrangian_sol) const {
  int iter_count = 0;
  double rho_prev = 0;
  *V_sol = V_init;
  while (iter_count < search_options.bilinear_iterations) {
    {
      MaximizeInnerEllipsoidRho(
          x_, x_star, S, *V_sol - search_options.rho, state_constraints_, r_degree,
          ellipsoid_c_lagrangian_degrees, rho_bisection_option.rho_max,
          rho_bisection_option.rho_min, search_options.ellipsoid_step_solver,
          search_options.ellipsoid_step_solver_options,
          rho_bisection_option.rho_tol, rho_sol, r_sol,
          ellipsoid_c_lagrangian_sol);
      drake::log()->info("iter {}, ellipsoid rho={}", iter_count, *rho_sol);
      if (*rho_sol - rho_prev < search_options.rho_converge_tol) {
        return;
      } else {
        rho_prev = *rho_sol;
      }
    }

    const bool found_lagrangian = SearchLagrangian(
        *V_sol, search_options.rho, lambda0_degree, l_degrees, p_degrees,
        deriv_eps, search_options, lambda0_sol, l_sol, p_sol);
    if (!found_lagrangian) {
      return;
    }

    {
      // Solve the program to find new V given Lagrangians.
      MatrixX<symbolic::Expression> V_gram;
      symbolic::Polynomial V_search;
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
      VectorX<symbolic::Polynomial> p;
      auto prog_lyapunov = this->ConstructLyapunovProgram(
          *lambda0_sol, *l_sol, V_degree, search_options.rho, positivity_eps,
          positivity_d, positivity_eq_lagrangian_degrees, p_degrees, deriv_eps,
          &V_search, &positivity_eq_lagrangian, &p);
      const auto d = prog_lyapunov->NewContinuousVariables<1>("d");
      symbolic::Polynomial r;
      VectorX<symbolic::Polynomial> c_lagrangian;
      AddEllipsoidInRoaConstraint(
          prog_lyapunov.get(), x_, d(0), V_search, x_star, S, *rho_sol,
          r_degree, state_constraints_, ellipsoid_c_lagrangian_degrees, &r,
          &c_lagrangian);
      prog_lyapunov->AddBoundingBoxConstraint(-kInf, search_options.rho, d);
      prog_lyapunov->AddLinearCost(Vector1d(1), 0, d);
      RemoveTinyCoeff(prog_lyapunov.get(), search_options.lyap_tiny_coeff_tol);
      drake::log()->info("Search Lyapunov, Lyapunov program smallest coeff: {}",
                         SmallestCoeff(*prog_lyapunov));
      const auto result_lyapunov = SearchWithBackoff(
          prog_lyapunov.get(), search_options.lyap_step_solver,
          search_options.lyap_step_solver_options,
          search_options.backoff_scale);
      if (result_lyapunov.is_success()) {
        *V_sol = result_lyapunov.GetSolution(V_search);
        *r_sol = result_lyapunov.GetSolution(r);
        if (search_options.Vsol_tiny_coeff_tol > 0) {
          *V_sol = V_sol->RemoveTermsWithSmallCoefficients(
              search_options.Vsol_tiny_coeff_tol);
        }
        if (search_options.save_clf_file.has_value()) {
          Save(*V_sol, search_options.save_clf_file.value());
        }
        GetPolynomialSolutions(result_lyapunov, p,
                               search_options.lsol_tiny_coeff_tol, p_sol);
        GetPolynomialSolutions(result_lyapunov, c_lagrangian,
                               search_options.lsol_tiny_coeff_tol,
                               ellipsoid_c_lagrangian_sol);
        GetPolynomialSolutions(result_lyapunov, positivity_eq_lagrangian,
                               search_options.lsol_tiny_coeff_tol,
                               positivity_eq_lagrangian_sol);
        drake::log()->info("d = {}", result_lyapunov.GetSolution(d(0)));
      } else {
        drake::log()->error("Failed to find Lyapunov");
        return;
      }
    }
    iter_count++;
  }
}

void ControlLyapunov::Search(
    const symbolic::Polynomial& V_init, int lambda0_degree,
    const std::vector<int>& l_degrees, int V_degree, double positivity_eps,
    int positivity_d, const std::vector<int>& positivity_eq_lagrangian_degrees,
    const std::vector<int>& p_degrees, double deriv_eps,
    const Eigen::Ref<const Eigen::MatrixXd>& x_samples, bool minimize_max,
    const SearchOptions& search_options, symbolic::Polynomial* V_sol,
    VectorX<symbolic::Polynomial>* positivity_eq_lagrangian_sol,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    VectorX<symbolic::Polynomial>* p_sol) const {
  DRAKE_DEMAND(x_samples.rows() == x_.rows());
  int iter_count = 0;
  double prev_cost = kInf;
  *V_sol = V_init;
  while (iter_count < search_options.bilinear_iterations) {
    drake::log()->info("Iteration {}", iter_count);
    const bool found_lagrangian = SearchLagrangian(
        *V_sol, search_options.rho, lambda0_degree, l_degrees, p_degrees,
        deriv_eps, search_options, lambda0_sol, l_sol, p_sol);
    if (!found_lagrangian) {
      return;
    }

    {
      // Given the Lagrangian, find V to minimize the cost.
      symbolic::Polynomial V_search;
      VectorX<symbolic::Polynomial> positivity_eq_lagrangian;
      VectorX<symbolic::Polynomial> p;
      auto prog_lyapunov = this->ConstructLyapunovProgram(
          *lambda0_sol, *l_sol, V_degree, search_options.rho, positivity_eps,
          positivity_d, positivity_eq_lagrangian_degrees, p_degrees, deriv_eps,
          &V_search, &positivity_eq_lagrangian, &p);
      OptimizePolynomialAtSamples(prog_lyapunov.get(), V_search, x_, x_samples,
                                  minimize_max
                                      ? OptimizePolynomialMode::kMinimizeMaximal
                                      : OptimizePolynomialMode::kMinimizeAverage);
      RemoveTinyCoeff(prog_lyapunov.get(), search_options.lyap_tiny_coeff_tol);
      drake::log()->info("Search Lyapunov, Lyapunov program smallest coeff: {}",
                         SmallestCoeff(*prog_lyapunov));
      const auto result_lyapunov = SearchWithBackoff(
          prog_lyapunov.get(), search_options.lyap_step_solver,
          search_options.lyap_step_solver_options,
          search_options.backoff_scale);
      if (result_lyapunov.is_success()) {
        *V_sol = result_lyapunov.GetSolution(V_search);
        if (search_options.Vsol_tiny_coeff_tol > 0) {
          *V_sol = V_sol->RemoveTermsWithSmallCoefficients(
              search_options.Vsol_tiny_coeff_tol);
        }
        if (search_options.save_clf_file.has_value()) {
          Save(*V_sol, search_options.save_clf_file.value());
        }
        GetPolynomialSolutions(result_lyapunov, positivity_eq_lagrangian,
                               search_options.lsol_tiny_coeff_tol,
                               positivity_eq_lagrangian_sol);
        GetPolynomialSolutions(result_lyapunov, p,
                               search_options.lsol_tiny_coeff_tol, p_sol);
        double cost;
        const Eigen::VectorXd V_samples =
            V_sol->EvaluateIndeterminates(x_, x_samples);
        if (minimize_max) {
          cost = V_samples.maxCoeff();
        } else {
          cost = V_samples.sum();
        }
        drake::log()->info("Optimal cost = {}", cost);
        if (prev_cost - cost < search_options.rho_converge_tol) {
          return;
        }
        prev_cost = cost;
      } else {
        *V_sol = result_lyapunov.GetSolution(V_search);
        drake::log()->error("Failed to find Lyapunov");
        return;
      }
    }
    iter_count++;
  }
}

VdotCalculator::VdotCalculator(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Polynomial& V,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const std::optional<symbolic::Polynomial>& dynamics_numerator,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices)
    : x_{x}, dynamics_numerator_{dynamics_numerator}, u_vertices_{u_vertices} {
  DRAKE_DEMAND(u_vertices_.rows() == G.cols());
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  dVdx_times_f_ = (dVdx * f)(0);
  dVdx_times_G_ = dVdx * G;
}

symbolic::Polynomial VdotCalculator::Calc(
    const Eigen::Ref<const Eigen::VectorXd>& u) const {
  return dVdx_times_f_ + (dVdx_times_G_ * u)(0);
}

Eigen::VectorXd VdotCalculator::CalcMin(
    const Eigen::Ref<const Eigen::MatrixXd>& x_vals) const {
  DRAKE_DEMAND(x_vals.rows() == x_.rows());
  Eigen::VectorXd ret = dVdx_times_f_.EvaluateIndeterminates(x_, x_vals);
  Eigen::MatrixXd dVdx_times_G_val(x_vals.cols(), dVdx_times_G_.cols());
  for (int i = 0; i < dVdx_times_G_.cols(); ++i) {
    dVdx_times_G_val.col(i) =
        dVdx_times_G_(i).EvaluateIndeterminates(x_, x_vals);
  }
  ret += (dVdx_times_G_val * u_vertices_).rowwise().minCoeff();
  if (dynamics_numerator_.has_value()) {
    const Eigen::VectorXd numerator_val =
        dynamics_numerator_->EvaluateIndeterminates(x_, x_vals);
    ret = (ret.array() / numerator_val.array()).matrix();
  }
  return ret;
}

void CheckDynamicsInput(const symbolic::Polynomial& V,
                        const VectorX<symbolic::Polynomial>& f,
                        const MatrixX<symbolic::Polynomial>& G,
                        const symbolic::Variables& x_set) {
  DRAKE_DEMAND(V.indeterminates().IsSubsetOf(x_set));
  DRAKE_DEMAND(f.rows() == static_cast<long>(x_set.size()));
  DRAKE_DEMAND(f.rows() == G.rows());
  for (int i = 0; i < static_cast<int>(f.rows()); ++i) {
    DRAKE_DEMAND(f[i].indeterminates().IsSubsetOf(x_set));
    for (int j = 0; j < static_cast<int>(G.cols()); ++j) {
      DRAKE_DEMAND(G(i, j).indeterminates().IsSubsetOf(x_set));
    }
  }
}

namespace {
// For a specific i
// if j = 0, add the constraint
// (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >= 0
// if j = 1, add the constraint
// (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V) >= 0
void AddControlLyapunovBoxInputBoundConstraints(
    solvers::MathematicalProgram* prog, int j,
    const std::array<symbolic::Polynomial, 3>& l_ij,
    const symbolic::Polynomial& V, const symbolic::Polynomial& dVdx_times_Gi,
    const symbolic::Polynomial& b_i, MatrixX<symbolic::Variable>* gram,
    VectorX<symbolic::Monomial>* monomials) {
  DRAKE_DEMAND(j == 0 || j == 1);
  const symbolic::Polynomial p =
      j == 0 ? (l_ij[0] + 1) * (dVdx_times_Gi - b_i) - l_ij[1] * dVdx_times_Gi -
                   l_ij[2] * (1 - V)
             : (l_ij[0] + 1) * (-dVdx_times_Gi - b_i) +
                   l_ij[1] * dVdx_times_Gi - l_ij[2] * (1 - V);
  std::tie(*gram, *monomials) = prog->AddSosConstraint(
      p, solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Vd");
}

// Add the constraint
// (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >= 0
// (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V) >= 0
// for all i = 0, ..., nᵤ-1
void AddControlLyapunovBoxInputBoundConstraints(
    solvers::MathematicalProgram* prog,
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const symbolic::Polynomial& V, const RowVectorX<symbolic::Polynomial>& dVdx,
    const VectorX<symbolic::Polynomial>& b,
    const MatrixX<symbolic::Polynomial>& G,
    VdotSosConstraintReturn* vdot_sos_constraint) {
  const int nu = G.cols();
  for (int i = 0; i < nu; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = dVdx.dot(G.col(i));
    const int num_vdot_sos = vdot_sos_constraint->symmetric_dynamics ? 1 : 2;
    for (int j = 0; j < num_vdot_sos; ++j) {
      AddControlLyapunovBoxInputBoundConstraints(
          prog, j, l[i][j], V, dVdx_times_Gi, b(i),
          &(vdot_sos_constraint->grams[i][j]),
          &(vdot_sos_constraint->monomials[i][j]));
    }
  }
}

// Add the constraint
// (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−V_level) is sos
template <typename RhoType>
std::pair<MatrixX<symbolic::Variable>, VectorX<symbolic::Monomial>>
AddEllipsoidInRoaConstraintHelper(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& t,
    const VectorX<symbolic::Variable>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho,
    const symbolic::Polynomial& s, const symbolic::Polynomial& V, double V_level) {
  const symbolic::Polynomial ellipsoid_poly =
      internal::EllipsoidPolynomial<RhoType>(x, x_star, S, rho);
  const symbolic::Polynomial sos_poly = (1 + t) * ellipsoid_poly - s * (V - V_level);
  return prog->AddSosConstraint(sos_poly);
}
}  // namespace

VdotSosConstraintReturn::VdotSosConstraintReturn(int nu,
                                                 bool m_symmetric_dynamics)
    : symmetric_dynamics{m_symmetric_dynamics},
      monomials{static_cast<size_t>(nu)},
      grams{static_cast<size_t>(nu)} {
  const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
  for (int i = 0; i < nu; ++i) {
    monomials[i].resize(num_vdot_sos);
    grams[i].resize(num_vdot_sos);
  }
}

std::vector<symbolic::Polynomial> VdotSosConstraintReturn::ComputeSosConstraint(
    int i, const solvers::MathematicalProgramResult& result) const {
  std::vector<symbolic::Polynomial> ret(symmetric_dynamics ? 1 : 2);
  for (int j = 0; j < static_cast<int>(ret.size()); ++j) {
    const Eigen::MatrixXd gram_sol = result.GetSolution(grams[i][j]);
    ret[j] = monomials[i][j].dot(gram_sol * monomials[i][j]);
  }
  return ret;
}

symbolic::Polynomial VdotSosConstraintReturn::ComputeSosConstraint(
    int i, int j, const solvers::MathematicalProgramResult& result) const {
  const Eigen::MatrixXd gram_sol = result.GetSolution(grams[i][j]);
  symbolic::Polynomial ret = monomials[i][j].dot(gram_sol * monomials[i][j]);
  return ret;
}

namespace {
[[maybe_unused]] symbolic::Polynomial
ComputePolynomialFromMonomialBasisAndGramMatrix(
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const Eigen::Ref<const MatrixX<symbolic::Variable>>& gram) {
  // TODO(hongkai.dai & soonho.kong): ideally we should compute p in one line as
  // monomial_basis.dot(gramian * monomial_basis). But as explained in #10200,
  // this one line version is too slow, so we use this double for loop to
  // compute the matrix product by hand. I will revert to the one line version
  // when it is fast.
  symbolic::Polynomial p{};
  for (int i = 0; i < gram.rows(); ++i) {
    p.AddProduct(gram(i, i), pow(monomial_basis(i), 2));
    for (int j = i + 1; j < gram.cols(); ++j) {
      p.AddProduct(2 * gram(i, j), monomial_basis(i) * monomial_basis(j));
    }
  }
  return p;
}

}  // namespace

SearchLagrangianGivenVBoxInputBound::SearchLagrangianGivenVBoxInputBound(
    symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
    MatrixX<symbolic::Polynomial> G, bool symmetric_dynamics,
    VectorX<symbolic::Polynomial> b, VectorX<symbolic::Variable> x,
    std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees)
    : V_{std::move(V)},
      f_{std::move(f)},
      G_{std::move(G)},
      symmetric_dynamics_{symmetric_dynamics},
      b_{std::move(b)},
      x_{std::move(x)},
      x_set_{x_},
      progs_{static_cast<size_t>(G_.cols())},
      nu_{static_cast<int>(G_.cols())},
      nx_{static_cast<int>(f.rows())},
      l_{static_cast<size_t>(nu_)},
      lagrangian_degrees_{std::move(lagrangian_degrees)},
      lagrangian_grams_{static_cast<size_t>(nu_)},
      vdot_sos_constraint_{nu_, symmetric_dynamics_} {
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(b_.rows() == nu_);
  const int num_vdot_sos = symmetric_dynamics_ ? 1 : 2;
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(b_(i).indeterminates().IsSubsetOf(x_set_));
    progs_[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      progs_[i][j] = std::make_unique<solvers::MathematicalProgram>();
      progs_[i][j]->AddIndeterminates(x_);
    }
  }
  for (int i = 0; i < nu_; ++i) {
    l_[i].resize(num_vdot_sos);
    lagrangian_grams_[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      for (int k = 0; k < 2; ++k) {
        std::tie(l_[i][j][k], lagrangian_grams_[i][j][k]) =
            progs_[i][j]->NewSosPolynomial(x_set_,
                                           lagrangian_degrees_[i][j][k]);
      }
      // k == 2
      DRAKE_DEMAND(lagrangian_degrees_[i][j][2] % 2 == 0);

      const VectorX<symbolic::Monomial> l_monomial_basis =
          ComputeMonomialBasisNoConstant(x_set_,
                                         lagrangian_degrees_[i][j][2] / 2,
                                         symbolic::internal::DegreeType::kAny);
      std::tie(l_[i][j][2], lagrangian_grams_[i][j][2]) =
          progs_[i][j]->NewSosPolynomial(l_monomial_basis);
    }
  }

  const auto dVdx = V_.Jacobian(x_);
  // Now impose the constraint
  // (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V) >= 0
  // (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)>= 0
  for (int i = 0; i < nu_; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = dVdx.dot(G_.col(i));
    for (int j = 0; j < num_vdot_sos; ++j) {
      AddControlLyapunovBoxInputBoundConstraints(
          progs_[i][j].get(), j, l_[i][j], V_, dVdx_times_Gi, b_(i),
          &(vdot_sos_constraint_.grams[i][j]),
          &(vdot_sos_constraint_.monomials[i][j]));
    }
  }
}

ControlLyapunovBoxInputBound::ControlLyapunovBoxInputBound(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    double positivity_eps)
    : f_{f},
      G_{G},
      symmetric_dynamics_{internal::IsDynamicsSymmetric(f_, G_)},
      x_{x},
      x_set_{x_},
      positivity_eps_{positivity_eps},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())} {}

void ControlLyapunovBoxInputBound::SearchLagrangianAndB(
    const symbolic::Polynomial& V,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees, double deriv_eps_lower,
    double deriv_eps_upper, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double lsol_tiny_coeff_tol, double* deriv_eps_sol,
    VectorX<symbolic::Polynomial>* b_sol,
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* l_sol)
    const {
  // Check if V(0) = 0
  const auto it_V_constant =
      V.monomial_to_coefficient_map().find(symbolic::Monomial());
  if (it_V_constant != V.monomial_to_coefficient_map().end()) {
    DRAKE_DEMAND(symbolic::is_constant(it_V_constant->second) &&
                 symbolic::get_constant_value(it_V_constant->second) == 0.);
  }

  std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> lagrangians;
  std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>
      lagrangian_grams;
  VectorX<symbolic::Polynomial> b;
  symbolic::Variable deriv_eps_var;
  VdotSosConstraintReturn vdot_sos_constraint(nu_, symmetric_dynamics_);
  auto prog = ConstructLagrangianAndBProgram(
      V, l_given, lagrangian_degrees, b_degrees, symmetric_dynamics_,
      &lagrangians, &lagrangian_grams, &deriv_eps_var, &b,
      &vdot_sos_constraint);

  prog->AddBoundingBoxConstraint(deriv_eps_lower, deriv_eps_upper,
                                 deriv_eps_var);
  auto solver = solvers::MakeSolver(solver_id);
  solvers::MathematicalProgramResult result_searcher;
  solver->Solve(*prog, std::nullopt, solver_options, &result_searcher);
  if (!result_searcher.is_success()) {
    log()->error("Failed to find Lagrangian and b(x)");
  }
  // PrintPsdConstraintStat(searcher.prog(), result_searcher);
  DRAKE_DEMAND(result_searcher.is_success());
  *deriv_eps_sol = result_searcher.GetSolution(deriv_eps_var);
  b_sol->resize(nu_);
  l_sol->resize(nu_);
  const int num_vdot_sos = symmetric_dynamics_ ? 1 : 2;
  for (int i = 0; i < nu_; ++i) {
    (*b_sol)(i) = result_searcher.GetSolution(b(i));
    if (lsol_tiny_coeff_tol > 0) {
      (*b_sol)(i) =
          (*b_sol)(i).RemoveTermsWithSmallCoefficients(lsol_tiny_coeff_tol);
    }
    (*l_sol)[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      (*l_sol)[i][j][0] = l_given[i][j];
      for (int k = 1; k < 3; ++k) {
        (*l_sol)[i][j][k] = result_searcher.GetSolution(lagrangians[i][j][k]);
        if (lsol_tiny_coeff_tol > 0) {
          (*l_sol)[i][j][k] =
              (*l_sol)[i][j][k].RemoveTermsWithSmallCoefficients(
                  lsol_tiny_coeff_tol);
        }
      }
    }
  }
}

void ControlLyapunovBoxInputBound::SearchLyapunov(
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const std::vector<int>& b_degrees, int V_degree, double deriv_eps,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& s,
    const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double lyap_tiny_coeff_tol,
    double Vsol_tiny_coeff_tol, symbolic::Polynomial* V_sol,
    VectorX<symbolic::Polynomial>* b_sol, double* rho_sol) const {
  symbolic::Polynomial V;
  MatrixX<symbolic::Expression> positivity_constraint_gram;
  VectorX<symbolic::Monomial> positivity_constraint_monomial;
  VectorX<symbolic::Polynomial> b;
  VdotSosConstraintReturn vdot_sos_constraint(nu_, symmetric_dynamics_);
  auto prog = this->ConstructLyapunovProgram(
      l, symmetric_dynamics_, deriv_eps, V_degree, b_degrees, &V,
      &positivity_constraint_gram, &positivity_constraint_monomial, &b,
      &vdot_sos_constraint);
  const auto rho = prog->NewContinuousVariables<1>("rho")(0);
  auto [ellipsoid_constraint_gram, ellipsoid_constraint_monomials] =
      AddEllipsoidInRoaConstraintHelper<symbolic::Variable>(
          prog.get(), t, x_, x_star, S, rho, s, V, 1);
  prog->AddLinearCost(-rho);
  RemoveTinyCoeff(prog.get(), lyap_tiny_coeff_tol);
  drake::log()->info("Smallest coeff in Lyapunov program: {}",
                     SmallestCoeff(*prog));
  const auto result =
      SearchWithBackoff(prog.get(), solver_id, solver_options, backoff_scale);
  *rho_sol = result.GetSolution(rho);
  *V_sol = result.GetSolution(V);
  if (Vsol_tiny_coeff_tol > 0) {
    *V_sol = V_sol->RemoveTermsWithSmallCoefficients(Vsol_tiny_coeff_tol);
  }
  const int nu = G_.cols();
  b_sol->resize(nu);
  for (int i = 0; i < nu; ++i) {
    (*b_sol)(i) = result.GetSolution(b(i));
    if (Vsol_tiny_coeff_tol > 0) {
      (*b_sol)(i) =
          (*b_sol)(i).RemoveTermsWithSmallCoefficients(Vsol_tiny_coeff_tol);
    }
  }
}

void ControlLyapunovBoxInputBound::SearchLyapunov(
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const std::vector<int>& b_degrees, int V_degree, double deriv_eps,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, double rho, int r_degree,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double lyap_tiny_coeff_tol,
    double Vsol_tiny_coeff_tol, symbolic::Polynomial* V_sol,
    VectorX<symbolic::Polynomial>* b_sol, symbolic::Polynomial* r_sol,
    double* d_sol) const {
  symbolic::Polynomial V;
  MatrixX<symbolic::Expression> positivity_constraint_gram;
  VectorX<symbolic::Monomial> positivity_constraint_monomial;
  VectorX<symbolic::Polynomial> b;
  VdotSosConstraintReturn vdot_sos_constraint(nu_, symmetric_dynamics_);
  auto prog = this->ConstructLyapunovProgram(
      l, symmetric_dynamics_, deriv_eps, V_degree, b_degrees, &V,
      &positivity_constraint_gram, &positivity_constraint_monomial, &b,
      &vdot_sos_constraint);
  // Now add the constraint that the ellipsoid is within the sub-level set {x |
  // V(x)<=d} Namely d−V(x) − r(x)(ρ−(x−x*)ᵀS(x−x*)) is sos and r(x) is sos.
  const symbolic::Variable d_var = prog->NewContinuousVariables<1>("d")(0);
  // 0 <= d <= 1
  prog->AddBoundingBoxConstraint(0, 1, d_var);
  symbolic::Polynomial r;
  AddEllipsoidInRoaConstraint(prog.get(), x_, d_var, V, x_star, S, rho,
                              r_degree, std::nullopt, std::nullopt, &r,
                              nullptr);
  // The objective is to minimize d.
  prog->AddLinearCost(Vector1d::Ones(), 0, Vector1<symbolic::Variable>(d_var));
  RemoveTinyCoeff(prog.get(), lyap_tiny_coeff_tol);
  drake::log()->info("Smallest coeff in Lyapunov program: {}",
                     SmallestCoeff(*prog));
  const auto result =
      SearchWithBackoff(prog.get(), solver_id, solver_options, backoff_scale);
  // internal::PrintPsdConstraintStat(*prog, result);
  DRAKE_DEMAND(result.is_success());
  *V_sol = result.GetSolution(V);
  if (Vsol_tiny_coeff_tol > 0) {
    *V_sol = V_sol->RemoveTermsWithSmallCoefficients(Vsol_tiny_coeff_tol);
  }
  b_sol->resize(b.rows());
  for (int i = 0; i < b.rows(); ++i) {
    (*b_sol)(i) = result.GetSolution(b(i));
    if (Vsol_tiny_coeff_tol) {
      (*b_sol)(i) =
          (*b_sol)(i).RemoveTermsWithSmallCoefficients(Vsol_tiny_coeff_tol);
    }
  }
  *r_sol = result.GetSolution(r);
  *d_sol = result.GetSolution(d_var);
}

void ControlLyapunovBoxInputBound::SearchLagrangian(
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& b,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double lagrangian_tiny_coeff_tol, double lsol_tiny_coeff_tol,
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* l) const {
  SearchLagrangianGivenVBoxInputBound searcher(V, f_, G_, symmetric_dynamics_,
                                               b, x_, lagrangian_degrees);
  auto solver = solvers::MakeSolver(solver_id);
  const int nu = G_.cols();
  l->resize(nu);
  const int num_vdot_sos = symmetric_dynamics_ ? 1 : 2;
  for (int i = 0; i < nu; ++i) {
    (*l)[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      drake::log()->info("smallest coeff in Lagrangian program ({}, {}), {}", i,
                         j, SmallestCoeff(searcher.prog(i, j)));
      if (lagrangian_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(searcher.get_mutable_prog(i, j),
                        lagrangian_tiny_coeff_tol);
      }
      solvers::MathematicalProgramResult result_searcher;
      solver->Solve(searcher.prog(i, j), std::nullopt, solver_options,
                    &result_searcher);
      if (!result_searcher.is_success()) {
        drake::log()->error("Failed to find Lagrangian for i={}, j={}", i, j);
      }
      // PrintPsdConstraintStat(searcher.prog(i, j), result_searcher);
      DRAKE_DEMAND(result_searcher.is_success());
      for (int k = 0; k < 3; ++k) {
        (*l)[i][j][k] =
            result_searcher.GetSolution(searcher.lagrangians()[i][j][k]);
        if (lsol_tiny_coeff_tol > 0) {
          (*l)[i][j][k] = (*l)[i][j][k].RemoveTermsWithSmallCoefficients(
              lsol_tiny_coeff_tol);
        }
      }
    }
  }
}

ControlLyapunovBoxInputBound::SearchReturn ControlLyapunovBoxInputBound::Search(
    const symbolic::Polynomial& V_init,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t_given, int V_degree, double deriv_eps_lower,
    double deriv_eps_upper, const SearchOptions& options) const {
  // First search for b and lagrangians.
  SearchReturn ret;
  SearchLagrangianAndB(
      V_init, l_given, lagrangian_degrees, b_degrees, deriv_eps_lower,
      deriv_eps_upper, options.lagrangian_step_solver,
      options.lagrangian_step_solver_options, options.lsol_tiny_coeff_tol,
      &(ret.deriv_eps), &(ret.b), &(ret.l));

  // Solve a separate program to find the inscribed ellipsoid.
  MaximizeInnerEllipsoidRho(
      x_, x_star, S, V_init - 1, t_given, s_degree,
      options.lagrangian_step_solver, options.lagrangian_step_solver_options,
      options.backoff_scale, &(ret.rho), &(ret.ellipsoid_lagrangian));

  int iter = 0;
  bool converged = false;
  while (iter < options.bilinear_iterations && !converged) {
    double rho_new;
    drake::log()->info("search Lyapunov");
    SearchLyapunov(ret.l, b_degrees, V_degree, ret.deriv_eps, x_star, S,
                   ret.ellipsoid_lagrangian, t_given, options.lyap_step_solver,
                   options.lyap_step_solver_options, options.backoff_scale,
                   options.lyap_tiny_coeff_tol, options.Vsol_tiny_coeff_tol,
                   &(ret.V), &(ret.b), &rho_new);
    if (options.search_l_and_b) {
      drake::log()->info("search Lagrangian");
      SearchLagrangianAndB(
          ret.V, l_given, lagrangian_degrees, b_degrees, deriv_eps_lower,
          deriv_eps_upper, options.lagrangian_step_solver,
          options.lagrangian_step_solver_options, options.lsol_tiny_coeff_tol,
          &(ret.deriv_eps), &(ret.b), &(ret.l));
    }
    drake::log()->info("search Lagrangian");
    SearchLagrangian(ret.V, ret.b, lagrangian_degrees,
                     options.lagrangian_step_solver,
                     options.lagrangian_step_solver_options,
                     options.lagrangian_tiny_coeff_tol,
                     options.lsol_tiny_coeff_tol, &(ret.l));
    // Solve a separate program to find the innner ellipsoid.
    MaximizeInnerEllipsoidRho(
        x_, x_star, S, ret.V - 1, t_given, s_degree,
        options.lagrangian_step_solver, options.lagrangian_step_solver_options,
        options.backoff_scale, &rho_new, &(ret.ellipsoid_lagrangian));
    drake::log()->info("iter: {}, rho: {}", iter, rho_new);
    if (rho_new - ret.rho < options.rho_converge_tol) {
      converged = true;
    }
    ret.rho = rho_new;
    iter += 1;
  }
  return ret;
}

ControlLyapunovBoxInputBound::SearchReturn ControlLyapunovBoxInputBound::Search(
    const symbolic::Polynomial& V_init,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int r_degree, int V_degree,
    double deriv_eps_lower, double deriv_eps_upper,
    const SearchOptions& options,
    const RhoBisectionOption& rho_bisection_option) const {
  SearchReturn ret;
  drake::log()->info("search Lagrangian and b.");
  // First search for b and Lagrangians.
  SearchLagrangianAndB(
      V_init, l_given, lagrangian_degrees, b_degrees, deriv_eps_lower,
      deriv_eps_upper, options.lagrangian_step_solver,
      options.lagrangian_step_solver_options, options.lsol_tiny_coeff_tol,
      &(ret.deriv_eps), &(ret.b), &(ret.l));
  // Solve a separate program to find the inscribed ellipsoid.
  const bool found_inner_ellipsoid = MaximizeInnerEllipsoidRho(
      x_, x_star, S, V_init - 1, std::nullopt, r_degree, std::nullopt,
      rho_bisection_option.rho_max, rho_bisection_option.rho_min,
      options.lagrangian_step_solver, options.lagrangian_step_solver_options,
      rho_bisection_option.rho_tol, &(ret.rho), &(ret.ellipsoid_lagrangian),
      nullptr);
  DRAKE_DEMAND(found_inner_ellipsoid);

  int iter = 0;
  bool converged = false;
  while (iter < options.bilinear_iterations && !converged) {
    double d;
    drake::log()->info("search Lyapunov");
    SearchLyapunov(ret.l, b_degrees, V_degree, ret.deriv_eps, x_star, S,
                   ret.rho, r_degree, options.lyap_step_solver,
                   options.lyap_step_solver_options, options.backoff_scale,
                   options.lyap_tiny_coeff_tol, options.Vsol_tiny_coeff_tol,
                   &(ret.V), &(ret.b), &(ret.ellipsoid_lagrangian), &d);
    // Solve a separate program to find the inner ellipsoid.
    double rho_new;
    MaximizeInnerEllipsoidRho(
        x_, x_star, S, ret.V - 1, std::nullopt, r_degree, std::nullopt,
        rho_bisection_option.rho_max, rho_bisection_option.rho_min,
        options.lagrangian_step_solver, options.lagrangian_step_solver_options,
        rho_bisection_option.rho_tol, &rho_new, &(ret.ellipsoid_lagrangian),
        nullptr);
    drake::log()->info("iter: {}, rho: {}", iter, rho_new);
    if (rho_new - ret.rho < options.rho_converge_tol) {
      converged = true;
    }
    ret.rho = rho_new;
    drake::log()->info("d={}", d);

    if (options.search_l_and_b) {
      drake::log()->info("search Lagrangian and b");
      SearchLagrangianAndB(
          ret.V, l_given, lagrangian_degrees, b_degrees, deriv_eps_lower,
          deriv_eps_upper, options.lagrangian_step_solver,
          options.lagrangian_step_solver_options, options.lsol_tiny_coeff_tol,
          &(ret.deriv_eps), &(ret.b), &(ret.l));
    } else {
      drake::log()->info("search Lagrangian");
      SearchLagrangian(ret.V, ret.b, lagrangian_degrees,
                       options.lagrangian_step_solver,
                       options.lagrangian_step_solver_options,
                       options.lagrangian_tiny_coeff_tol,
                       options.lsol_tiny_coeff_tol, &(ret.l));
    }
    iter += 1;
  }
  return ret;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlLyapunovBoxInputBound::ConstructLagrangianAndBProgram(
    const symbolic::Polynomial& V,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees, bool symmetric_dynamics,
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* lagrangians,
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 3>>>*
        lagrangian_grams,
    symbolic::Variable* deriv_eps, VectorX<symbolic::Polynomial>* b,
    VdotSosConstraintReturn* vdot_sos_constraint) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  DRAKE_DEMAND(static_cast<int>(b_degrees.size()) == nu_);

  if (symmetric_dynamics && !V.IsEven()) {
    throw std::runtime_error(
        "For symmetric dynamics, V should be an even function.");
  }

  const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
  lagrangians->resize(nu_);
  lagrangian_grams->resize(nu_);
  // Add Lagrangian decision variables.
  for (int i = 0; i < nu_; ++i) {
    (*lagrangians)[i].resize(num_vdot_sos);
    (*lagrangian_grams)[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      // k == 0
      DRAKE_DEMAND(l_given[i][j].TotalDegree() == lagrangian_degrees[i][j][0]);
      (*lagrangians)[i][j][0] = l_given[i][j];

      // k == 1
      DRAKE_DEMAND(lagrangian_degrees[i][j][1] % 2 == 0);
      std::tie((*lagrangians)[i][j][1], (*lagrangian_grams)[i][j][1]) =
          prog->NewSosPolynomial(x_set_, lagrangian_degrees[i][j][1]);

      // k == 2, l_[i][j][2] doesn't have 1 in its monomial basis.
      DRAKE_DEMAND(lagrangian_degrees[i][j][2] % 2 == 0);
      const VectorX<symbolic::Monomial> l_monomial_basis =
          ComputeMonomialBasisNoConstant(x_set_,
                                         lagrangian_degrees[i][j][2] / 2,
                                         symbolic::internal::DegreeType::kAny);
      std::tie((*lagrangians)[i][j][2], (*lagrangian_grams)[i][j][2]) =
          prog->NewSosPolynomial(l_monomial_basis);
    }
  }
  *deriv_eps = prog->NewContinuousVariables<1>("deriv_eps")(0);

  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x_);
  // Since we will add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x), we know that
  // the highest degree of b should be at least degree(∂V/∂x*f(x) + εV).
  const symbolic::Polynomial dVdx_times_f = (dVdx * f_)(0);
  if (*std::max_element(b_degrees.begin(), b_degrees.end()) <
      std::max(dVdx_times_f.TotalDegree(), V.TotalDegree())) {
    throw std::invalid_argument("The degree of b is too low.");
  }
  // Add free polynomial b
  const symbolic::internal::DegreeType b_degree_type =
      symmetric_dynamics ? symbolic::internal::DegreeType::kEven
                         : symbolic::internal::DegreeType::kAny;
  b->resize(nu_);
  for (int i = 0; i < nu_; ++i) {
    (*b)(i) = NewFreePolynomialPassOrigin(prog.get(), x_set_, b_degrees[i],
                                          "b" + std::to_string(i),
                                          b_degree_type, symbolic::Variables{});
  }
  // Add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  prog->AddEqualityConstraintBetweenPolynomials(
      b->sum(), dVdx_times_f + (*deriv_eps) * V);
  // Add the constraint
  // (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 −V)>=0
  // (lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) -
  // lᵢ₁₂(x)*(1 − V)>=0
  AddControlLyapunovBoxInputBoundConstraints(prog.get(), *lagrangians, V, dVdx,
                                             *b, G_, vdot_sos_constraint);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlLyapunovBoxInputBound::ConstructLyapunovProgram(
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    bool symmetric_dynamics, double deriv_eps, int V_degree,
    const std::vector<int>& b_degrees, symbolic::Polynomial* V,
    MatrixX<symbolic::Expression>* positivity_constraint_gram,
    VectorX<symbolic::Monomial>* positivity_constraint_monomial,
    VectorX<symbolic::Polynomial>* b,
    VdotSosConstraintReturn* vdot_sos_constraint) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  DRAKE_DEMAND(V_degree >= 0 && V_degree % 2 == 0);
  if (positivity_eps_ == 0 && !symmetric_dynamics) {
    const VectorX<symbolic::Monomial> V_monomial =
        ComputeMonomialBasisNoConstant(x_set_, V_degree / 2,
                                       symbolic::internal::DegreeType::kAny);
    std::tie(*V, *positivity_constraint_gram) =
        prog->NewSosPolynomial(V_monomial);
    *positivity_constraint_monomial = V_monomial;
  } else {
    const symbolic::internal::DegreeType V_degree_type =
        symmetric_dynamics ? symbolic::internal::DegreeType::kEven
                           : symbolic::internal::DegreeType::kAny;
    // We know that V(0) = 0 and V(x) >= ε₁xᵀx, hence V(x) has no constant or
    // linear terms.
    *V = internal::NewFreePolynomialNoConstantOrLinear(
        prog.get(), x_set_, V_degree, "V_coeff", V_degree_type);
    // quadratic_poly_map stores the mapping for the polynomial
    // ε₁xᵀx
    symbolic::Polynomial::MapType quadratic_poly_map;
    for (int i = 0; i < nx_; ++i) {
      quadratic_poly_map.emplace(symbolic::Monomial(x_(i), 2), positivity_eps_);
    }
    symbolic::Polynomial V_minus_eps;
    NewSosPolynomialPassOrigin(prog.get(), x_set_, V_degree, V_degree_type,
                               &V_minus_eps, positivity_constraint_monomial,
                               positivity_constraint_gram);
    prog->AddEqualityConstraintBetweenPolynomials(
        *V - symbolic::Polynomial(quadratic_poly_map), V_minus_eps);
  }
  // ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  const symbolic::internal::DegreeType b_degree_type =
      symmetric_dynamics ? symbolic::internal::DegreeType::kEven
                         : symbolic::internal::DegreeType::kAny;
  b->resize(nu_);
  for (int i = 0; i < nu_; ++i) {
    (*b)(i) =
        NewFreePolynomialPassOrigin(prog.get(), x_set_, b_degrees[i], "b_coeff",
                                    b_degree_type, symbolic::Variables{});
  }
  const RowVectorX<symbolic::Polynomial> dVdx = V->Jacobian(x_);
  prog->AddEqualityConstraintBetweenPolynomials(
      (dVdx * f_)(0) + deriv_eps * (*V), b->sum());
  // Add the constraint
  // (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V)>=0
  //(lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)>=0
  AddControlLyapunovBoxInputBoundConstraints(prog.get(), l, *V, dVdx, *b, G_,
                                             vdot_sos_constraint);
  return prog;
}

ClfController::ClfController(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const std::optional<symbolic::Polynomial>& dynamics_numerator,
    symbolic::Polynomial V, double deriv_eps,
    const Eigen::Ref<const Eigen::MatrixXd>& Au,
    const Eigen::Ref<const Eigen::VectorXd>& bu,
    const std::optional<Eigen::VectorXd>& u_star,
    const Eigen::Ref<const Eigen::MatrixXd>& Ru, double vdot_cost)
    : LeafSystem<double>(),
      x_{x},
      f_{f},
      G_{G},
      dynamics_numerator_{dynamics_numerator},
      V_{std::move(V)},
      deriv_eps_{deriv_eps},
      Au_{Au},
      bu_{bu},
      u_star_{u_star},
      Ru_{Ru},
      vdot_cost_{vdot_cost} {
  const int nx = f_.rows();
  const int nu = G_.cols();
  DRAKE_DEMAND(x_.rows() == nx);
  DRAKE_DEMAND(Au_.cols() == nu);
  DRAKE_DEMAND(Au_.rows() == bu_.rows());
  if (u_star_.has_value()) {
    DRAKE_DEMAND(u_star_->rows() == nu);
  }
  DRAKE_DEMAND(Ru_.rows() == nu && Ru_.cols() == nu);
  const RowVectorX<symbolic::Polynomial> dVdx = V_.Jacobian(x_);
  dVdx_times_f_ = dVdx.dot(f_);
  dVdx_times_G_ = dVdx * G_;

  x_input_index_ = this->DeclareVectorInputPort("x", nx).get_index();

  control_output_index_ =
      this->DeclareVectorOutputPort("control", nu, &ClfController::CalcControl)
          .get_index();

  clf_output_index_ =
      this->DeclareVectorOutputPort("clf", 1, &ClfController::CalcClf)
          .get_index();
}

void ClfController::CalcClf(const Context<double>& context,
                            BasicVector<double>* output) const {
  const Eigen::VectorXd x_val =
      this->get_input_port(x_input_index_).Eval(context);
  symbolic::Environment env;
  env.insert(x_, x_val);
  Eigen::VectorBlock<VectorX<double>> clf_vec = output->get_mutable_value();
  clf_vec(0) = V_.Evaluate(env);
}

void ClfController::CalcControl(const Context<double>& context,
                                BasicVector<double>* output) const {
  const Eigen::VectorXd x_val =
      this->get_input_port(x_input_index_).Eval(context);
  symbolic::Environment env;
  env.insert(x_, x_val);

  solvers::MathematicalProgram prog;
  const int nu = G_.cols();
  auto u = prog.NewContinuousVariables(nu, "u");
  prog.AddLinearConstraint(Au_, Eigen::VectorXd::Constant(bu_.rows(), -kInf),
                           bu_, u);
  prog.AddQuadraticErrorCost(Ru_, *u_star_, u);

  const double dVdx_times_f_val = dVdx_times_f_.Evaluate(env);
  Eigen::RowVectorXd dVdx_times_G_val(nu);
  for (int i = 0; i < nu; ++i) {
    dVdx_times_G_val(i) = dVdx_times_G_(i).Evaluate(env);
  }
  const double V_val = V_.Evaluate(env);
  // dVdx * G * u + dVdx * f <= -eps * V * n(x)
  const double dynamics_numerator_val =
      dynamics_numerator_.has_value() ? dynamics_numerator_->Evaluate(env) : 1;
  prog.AddLinearConstraint(
      dVdx_times_G_val, -kInf,
      -deriv_eps_ * V_val * dynamics_numerator_val - dVdx_times_f_val, u);
  prog.AddLinearCost(dVdx_times_G_val / dynamics_numerator_val * vdot_cost_,
                     dVdx_times_f_val / dynamics_numerator_val * vdot_cost_, u);
  const auto result = solvers::Solve(prog);
  if (!result.is_success()) {
    drake::log()->error("ClfController fails at t={} with x={}, V={}",
                        context.get_time(), x_val.transpose(), V_val);
    DRAKE_DEMAND(result.is_success());
  }
  const Eigen::VectorXd u_val = result.GetSolution(u);
  output->get_mutable_value() = u_val;
}

namespace internal {
bool IsDynamicsSymmetric(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G) {
  for (int i = 0; i < f.rows(); ++i) {
    if (!f(i).IsOdd()) {
      return false;
    }
    for (int j = 0; j < G.cols(); ++j) {
      if (!G(i, j).IsEven()) {
        return false;
      }
    }
  }
  return true;
}

symbolic::Polynomial NegateIndeterminates(const symbolic::Polynomial& p) {
  // Flips the sign of the odd-degree monomials.
  symbolic::Polynomial::MapType monomial_to_coeff_map =
      p.monomial_to_coefficient_map();
  for (auto& [monomial, coeff] : monomial_to_coeff_map) {
    if (monomial.total_degree() % 2 == 1) {
      coeff *= -1;
    }
  }
  return symbolic::Polynomial(std::move(monomial_to_coeff_map));
}

// Create a polynomial the coefficient for constant and linear terms to be 0.
symbolic::Polynomial NewFreePolynomialNoConstantOrLinear(
    solvers::MathematicalProgram* prog,
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name, symbolic::internal::DegreeType degree_type) {
  if (degree <= 1 ||
      (degree_type == symbolic::internal::DegreeType::kOdd && degree == 2)) {
    return symbolic::Polynomial();
  }
  const VectorX<symbolic::Monomial> m =
      symbolic::internal::ComputeMonomialBasis<Eigen::Dynamic>(
          indeterminates, degree, degree_type);
  int monomial_count = 0;
  VectorX<symbolic::Monomial> m_degree_gt_1(m.rows());
  for (int i = 0; i < m.rows(); ++i) {
    if (m(i).total_degree() > 1) {
      m_degree_gt_1(monomial_count++) = m(i);
    }
  }
  m_degree_gt_1.conservativeResize(monomial_count);
  const VectorX<symbolic::Variable> coeffs =
      prog->NewContinuousVariables(m_degree_gt_1.size(), coeff_name);
  symbolic::Polynomial::MapType poly_map;
  for (int i = 0; i < coeffs.rows(); ++i) {
    poly_map.emplace(m_degree_gt_1(i), coeffs(i));
  }
  return symbolic::Polynomial(poly_map);
}

[[maybe_unused]] void PrintPsdConstraintStat(
    const solvers::MathematicalProgram& prog,
    const solvers::MathematicalProgramResult& result) {
  for (const auto& psd_constraint : prog.positive_semidefinite_constraints()) {
    const Eigen::MatrixXd psd_sol = Eigen::Map<Eigen::MatrixXd>(
        result.GetSolution(psd_constraint.variables()).data(),
        psd_constraint.evaluator()->matrix_rows(),
        psd_constraint.evaluator()->matrix_rows());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(psd_sol);
    drake::log()->info(
        "rows: {}, min eigen: {}, cond number: {}", psd_sol.rows(),
        es.eigenvalues().minCoeff(),
        es.eigenvalues().maxCoeff() / es.eigenvalues().minCoeff());
    const bool dual_result_enabled =
        result.get_solver_id() == solvers::MosekSolver::id();
    if (dual_result_enabled) {
      const Eigen::MatrixXd psd_dual =
          math::ToSymmetricMatrixFromLowerTriangularColumns(
              result.GetDualSolution(psd_constraint));
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_dual(psd_dual);
      drake::log()->info(
          "dual min eigen: {}, cond number {}, maxCoeff {}",
          es_dual.eigenvalues().minCoeff(),
          es_dual.eigenvalues().maxCoeff() / es_dual.eigenvalues().minCoeff(),
          psd_dual.array().abs().maxCoeff());
    }
  }
}
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
