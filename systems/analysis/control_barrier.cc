#include "drake/systems/analysis/control_barrier.h"

#include <limits.h>

#include "drake/common/text_logging.h"
#include "drake/math/matrix_util.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/sos_basis_generator.h"
#include "drake/systems/analysis/clf_cbf_utils.h"
namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();
namespace {
void AddHdotSosConstraint(
    solvers::MathematicalProgram* prog,
    const std::vector<std::array<symbolic::Polynomial, 2>>& l,
    const symbolic::Polynomial& dhdx_times_Gi, const symbolic::Polynomial& b_i,
    std::vector<MatrixX<symbolic::Variable>>* gram,
    std::vector<VectorX<symbolic::Monomial>>* monomials) {
  const symbolic::Polynomial p0 =
      (1 + l[0][0]) * (dhdx_times_Gi - b_i) - l[0][1] * dhdx_times_Gi;
  const symbolic::Polynomial p1 =
      (1 + l[1][0]) * (-dhdx_times_Gi - b_i) + l[1][1] * dhdx_times_Gi;
  gram->resize(2);
  monomials->resize(2);
  std::tie((*gram)[0], (*monomials)[0]) = prog->AddSosConstraint(
      p0, solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Hd0");
  std::tie((*gram)[1], (*monomials)[1]) = prog->AddSosConstraint(
      p1, solvers::MathematicalProgram::NonnegativePolynomial::kSos, "Hd1");
}

void AddHdotSosConstraint(
    solvers::MathematicalProgram* prog,
    const std::vector<std::vector<std::array<symbolic::Polynomial, 2>>>& l,
    const RowVectorX<symbolic::Polynomial>& dhdx,
    const MatrixX<symbolic::Polynomial>& G,
    const VectorX<symbolic::Polynomial>& b,
    ControlBarrierBoxInputBound::HdotSosConstraintReturn* hdot_sos_constraint) {
  const int nu = static_cast<int>(l.size());
  DRAKE_DEMAND(G.cols() == nu);
  DRAKE_DEMAND(b.rows() == nu);
  for (int i = 0; i < nu; ++i) {
    const symbolic::Polynomial dhdx_times_Gi = dhdx.dot(G.col(i));
    AddHdotSosConstraint(prog, l[i], dhdx_times_Gi, b(i),
                         &(hdot_sos_constraint->grams[i]),
                         &(hdot_sos_constraint->monomials[i]));
  }
}
}  // namespace

ControlBarrier::ControlBarrier(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    std::optional<symbolic::Polynomial> dynamics_denominator,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    std::vector<VectorX<symbolic::Polynomial>> unsafe_regions,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_eq_constraints)
    : f_{f},
      G_{G},
      dynamics_denominator_{std::move(dynamics_denominator)},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      x_{x},
      x_set_{x},
      unsafe_regions_{std::move(unsafe_regions)},
      u_vertices_{u_vertices},
      state_eq_constraints_{state_eq_constraints} {
  DRAKE_DEMAND(G_.rows() == nx_);
  DRAKE_DEMAND(u_vertices_.rows() == nu_);
}
void ControlBarrier::AddControlBarrierConstraint(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& lambda0,
    const VectorX<symbolic::Polynomial>& l,
    const VectorX<symbolic::Polynomial>& state_constraints_lagrangian,
    const symbolic::Polynomial& h, double deriv_eps,
    symbolic::Polynomial* hdot_poly, VectorX<symbolic::Monomial>* monomials,
    MatrixX<symbolic::Variable>* gram) const {
  *hdot_poly = (1 + lambda0) * (-1 - h);
  const RowVectorX<symbolic::Polynomial> dhdx = h.Jacobian(x_);
  const symbolic::Polynomial dhdx_times_f = dhdx.dot(f_);
  *hdot_poly -=
      l.sum() *
      (-dhdx_times_f -
       deriv_eps * h * dynamics_denominator_.value_or(symbolic::Polynomial(1)));
  *hdot_poly += (dhdx * G_ * u_vertices_).dot(l);
  DRAKE_DEMAND(state_eq_constraints_.rows() ==
               state_constraints_lagrangian.rows());
  *hdot_poly -= state_constraints_lagrangian.dot(state_eq_constraints_);
  std::tie(*gram, *monomials) = prog->AddSosConstraint(
      *hdot_poly, solvers::MathematicalProgram::NonnegativePolynomial::kSos,
      "hd");
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructLagrangianProgram(
    const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
    const std::vector<int>& l_degrees,
    const std::vector<int>& state_constraints_lagrangian_degrees,
    symbolic::Polynomial* lambda0, MatrixX<symbolic::Variable>* lambda0_gram,
    VectorX<symbolic::Polynomial>* l,
    std::vector<MatrixX<symbolic::Variable>>* l_grams,
    VectorX<symbolic::Polynomial>* state_constraints_lagrangian,
    symbolic::Polynomial* hdot_sos, VectorX<symbolic::Monomial>* hdot_monomials,
    MatrixX<symbolic::Variable>* hdot_gram) const {
  DRAKE_DEMAND(static_cast<int>(l_degrees.size()) == u_vertices_.cols());
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
  state_constraints_lagrangian->resize(state_eq_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(state_constraints_lagrangian_degrees.size()) ==
               state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    (*state_constraints_lagrangian)(i) =
        prog->NewFreePolynomial(x_set_, state_constraints_lagrangian_degrees[i],
                                "le" + std::to_string(i));
  }
  this->AddControlBarrierConstraint(prog.get(), *lambda0, *l,
                                    *state_constraints_lagrangian, h, deriv_eps,
                                    hdot_sos, hdot_monomials, hdot_gram);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructUnsafeRegionProgram(
    const symbolic::Polynomial& h, int region_index, int t_degree,
    const std::vector<int>& s_degrees,
    const std::vector<int>& unsafe_state_constraints_lagrangian_degrees,
    symbolic::Polynomial* t, MatrixX<symbolic::Variable>* t_gram,
    VectorX<symbolic::Polynomial>* s,
    std::vector<MatrixX<symbolic::Variable>>* s_grams,
    VectorX<symbolic::Polynomial>* unsafe_state_constraints_lagrangian,
    symbolic::Polynomial* sos_poly,
    MatrixX<symbolic::Variable>* sos_poly_gram) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  std::tie(*t, *t_gram) = prog->NewSosPolynomial(
      x_set_, t_degree,
      solvers::MathematicalProgram::NonnegativePolynomial::kSos, "T");
  s->resize(unsafe_regions_[region_index].rows());
  s_grams->resize(s->rows());
  for (int i = 0; i < s->rows(); ++i) {
    std::tie((*s)(i), (*s_grams)[i]) = prog->NewSosPolynomial(
        x_set_, s_degrees[i],
        solvers::MathematicalProgram::NonnegativePolynomial::kSos, "S");
  }
  DRAKE_DEMAND(
      static_cast<int>(unsafe_state_constraints_lagrangian_degrees.size()) ==
      state_eq_constraints_.rows());
  unsafe_state_constraints_lagrangian->resize(state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    (*unsafe_state_constraints_lagrangian)(i) = prog->NewFreePolynomial(
        x_set_, unsafe_state_constraints_lagrangian_degrees[i]);
  }
  *sos_poly = (1 + *t) * (-h) + s->dot(unsafe_regions_[region_index]) -
              unsafe_state_constraints_lagrangian->dot(state_eq_constraints_);
  std::tie(*sos_poly_gram, std::ignore) = prog->AddSosConstraint(*sos_poly);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructBarrierProgram(
    const symbolic::Polynomial& lambda0, const VectorX<symbolic::Polynomial>& l,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<symbolic::Polynomial>& t,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    int h_degree, double deriv_eps,
    const std::vector<std::vector<int>>& s_degrees, symbolic::Polynomial* h,
    symbolic::Polynomial* hdot_sos, MatrixX<symbolic::Variable>* hdot_sos_gram,
    std::vector<VectorX<symbolic::Polynomial>>* s,
    std::vector<std::vector<MatrixX<symbolic::Variable>>>* s_grams,
    std::vector<symbolic::Polynomial>* unsafe_sos_polys,
    std::vector<MatrixX<symbolic::Variable>>* unsafe_sos_poly_grams) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  *h = prog->NewFreePolynomial(x_set_, h_degree, "H");
  VectorX<symbolic::Monomial> hdot_monomials;
  VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian(
      state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    hdot_state_constraints_lagrangian(i) = prog->NewFreePolynomial(
        x_set_, hdot_state_constraints_lagrangian_degrees[i]);
  }
  // Add the constraint on hdot.
  this->AddControlBarrierConstraint(
      prog.get(), lambda0, l, hdot_state_constraints_lagrangian, *h, deriv_eps,
      hdot_sos, &hdot_monomials, hdot_sos_gram);
  // Add the constraint that the unsafe region has h <= 0
  const int num_unsafe_regions = static_cast<int>(unsafe_regions_.size());
  DRAKE_DEMAND(static_cast<int>(t.size()) == num_unsafe_regions);
  DRAKE_DEMAND(static_cast<int>(s_degrees.size()) == num_unsafe_regions);
  s->resize(num_unsafe_regions);
  s_grams->resize(num_unsafe_regions);
  std::vector<VectorX<symbolic::Polynomial>>
      unsafe_state_constraints_lagrangian(num_unsafe_regions);
  unsafe_sos_polys->resize(num_unsafe_regions);
  unsafe_sos_poly_grams->resize(num_unsafe_regions);
  for (int i = 0; i < num_unsafe_regions; ++i) {
    const int num_unsafe_polys = static_cast<int>(unsafe_regions_[i].rows());
    (*s)[i].resize(num_unsafe_polys);
    (*s_grams)[i].resize(num_unsafe_polys);
    DRAKE_DEMAND(static_cast<int>(s_degrees[i].size()) == num_unsafe_polys);
    for (int j = 0; j < num_unsafe_polys; ++j) {
      std::tie((*s)[i](j), (*s_grams)[i][j]) = prog->NewSosPolynomial(
          x_set_, s_degrees[i][j],
          solvers::MathematicalProgram::NonnegativePolynomial::kSos,
          fmt::format("S{},{}", i, j));
    }
    unsafe_state_constraints_lagrangian[i].resize(state_eq_constraints_.rows());
    for (int j = 0; j < state_eq_constraints_.rows(); ++j) {
      unsafe_state_constraints_lagrangian[i](j) = prog->NewFreePolynomial(
          x_set_, unsafe_state_constraints_lagrangian_degrees[i][j]);
    }
    (*unsafe_sos_polys)[i] =
        (1 + t[i]) * (-(*h)) + (*s)[i].dot(unsafe_regions_[i]) -
        unsafe_state_constraints_lagrangian[i].dot(state_eq_constraints_);
    std::tie((*unsafe_sos_poly_grams)[i], std::ignore) =
        prog->AddSosConstraint((*unsafe_sos_polys)[i]);
  }

  return prog;
}

void ControlBarrier::AddBarrierProgramCost(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& h,
    const Eigen::MatrixXd& verified_safe_states,
    const Eigen::MatrixXd& unverified_candidate_states, double eps) const {
  // Add the constraint that the verified states all have h(x) >= 0
  Eigen::MatrixXd h_monomial_vals;
  VectorX<symbolic::Variable> h_coeff_vars;
  EvaluatePolynomial(h, x_, verified_safe_states, &h_monomial_vals,
                     &h_coeff_vars);
  prog->AddLinearConstraint(
      h_monomial_vals, Eigen::VectorXd::Zero(h_monomial_vals.rows()),
      Eigen::VectorXd::Constant(h_monomial_vals.rows(), kInf), h_coeff_vars);
  // Add the objective to maximize sum min(h(xʲ), eps) for xʲ in
  // unverified_candidate_states
  EvaluatePolynomial(h, x_, unverified_candidate_states, &h_monomial_vals,
                     &h_coeff_vars);
  auto h_unverified_min0 =
      prog->NewContinuousVariables(unverified_candidate_states.cols());
  prog->AddBoundingBoxConstraint(-kInf, eps, h_unverified_min0);
  // Add constraint h_unverified_min0 <= h(xʲ)
  Eigen::MatrixXd A_unverified(
      unverified_candidate_states.cols(),
      unverified_candidate_states.cols() + h_monomial_vals.cols());
  A_unverified << Eigen::MatrixXd::Identity(unverified_candidate_states.cols(),
                                            unverified_candidate_states.cols()),
      -h_monomial_vals;
  prog->AddLinearConstraint(
      A_unverified, Eigen::VectorXd::Constant(A_unverified.rows(), -kInf),
      Eigen::VectorXd::Zero(A_unverified.rows()),
      {h_unverified_min0, h_coeff_vars});
  prog->AddLinearCost(-Eigen::VectorXd::Ones(h_unverified_min0.rows()), 0,
                      h_unverified_min0);
}

void ControlBarrier::AddBarrierProgramCost(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& h,
    const std::vector<Ellipsoid>& inner_ellipsoids,
    std::vector<symbolic::Polynomial>* r, VectorX<symbolic::Variable>* d,
    std::vector<VectorX<symbolic::Polynomial>>* state_constraints_lagrangian)
    const {
  r->resize(inner_ellipsoids.size());
  *d = prog->NewContinuousVariables(static_cast<int>(inner_ellipsoids.size()),
                                    "d");
  state_constraints_lagrangian->resize(inner_ellipsoids.size());
  for (int i = 0; i < static_cast<int>(inner_ellipsoids.size()); ++i) {
    std::tie((*r)[i], std::ignore) = prog->NewSosPolynomial(
        x_set_, inner_ellipsoids[i].r_degree,
        solvers::MathematicalProgram::NonnegativePolynomial::kSos, "R");
    DRAKE_DEMAND(
        static_cast<int>(
            inner_ellipsoids[i].state_constraints_lagrangian_degrees.size()) ==
        state_eq_constraints_.rows());
    (*state_constraints_lagrangian)[i].resize(state_eq_constraints_.rows());
    for (int j = 0; j < state_eq_constraints_.rows(); ++j) {
      (*state_constraints_lagrangian)[i](j) = prog->NewFreePolynomial(
          x_set_, inner_ellipsoids[i].state_constraints_lagrangian_degrees[j]);
    }
    prog->AddSosConstraint(
        h - (*d)(i) +
        (*r)[i] * internal::EllipsoidPolynomial(x_, inner_ellipsoids[i].c,
                                                inner_ellipsoids[i].S,
                                                inner_ellipsoids[i].rho) -
        (*state_constraints_lagrangian)[i].dot(state_eq_constraints_));
  }
  prog->AddLinearCost(-Eigen::VectorXd::Ones(d->rows()), 0, *d);
  prog->AddBoundingBoxConstraint(0, kInf, *d);
}

void ControlBarrier::Search(
    const symbolic::Polynomial& h_init, int h_degree, double deriv_eps,
    int lambda0_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<int>& t_degree,
    const std::vector<std::vector<int>>& s_degrees,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    const std::vector<ControlBarrier::Ellipsoid>& ellipsoids,
    const Eigen::Ref<const Eigen::VectorXd>& x_anchor,
    const SearchOptions& search_options, symbolic::Polynomial* h_sol,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    VectorX<symbolic::Polynomial>* hdot_state_constraints_lagrangian,
    std::vector<symbolic::Polynomial>* t_sol,
    std::vector<VectorX<symbolic::Polynomial>>* s_sol,
    std::vector<VectorX<symbolic::Polynomial>>*
        unsafe_state_constraints_lagrangian) const {
  *h_sol = h_init;
  double h_at_x_anchor{};
  {
    symbolic::Environment env;
    env.insert(x_, x_anchor);
    h_at_x_anchor = h_init.Evaluate(env);
    if (h_at_x_anchor <= 0) {
      throw std::runtime_error(fmt::format(
          "ControlBarrier::Search(): h_init(x_anchor) = {}, should be > 0",
          h_at_x_anchor));
    }
  }

  int iter_count = 0;

  std::vector<ControlBarrier::Ellipsoid> inner_ellipsoids;
  std::list<ControlBarrier::Ellipsoid> uncovered_ellipsoids{ellipsoids.begin(),
                                                            ellipsoids.end()};
  while (iter_count < search_options.bilinear_iterations) {
    const bool found_lagrangian = SearchLagrangian(
        *h_sol, deriv_eps, lambda0_degree, l_degrees,
        hdot_state_constraints_lagrangian_degrees, t_degree, s_degrees,
        unsafe_state_constraints_lagrangian_degrees, search_options,
        lambda0_sol, l_sol, hdot_state_constraints_lagrangian, t_sol, s_sol,
        unsafe_state_constraints_lagrangian);
    if (!found_lagrangian) {
      return;
    }

    // Maximize the inner ellipsoids.
    drake::log()->info("Find maximal inner ellipsoids");
    // For each inner ellipsoid, compute rho.
    for (auto& ellipsoid : inner_ellipsoids) {
      double rho_sol;
      symbolic::Polynomial r_sol;
      VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
      MaximizeInnerEllipsoidRho(
          x_, ellipsoid.c, ellipsoid.S, -(*h_sol), state_eq_constraints_,
          ellipsoid.r_degree, ellipsoid.state_constraints_lagrangian_degrees,
          ellipsoid.rho_max, ellipsoid.rho,
          search_options.lagrangian_step_solver,
          search_options.lagrangian_step_solver_options, ellipsoid.rho_tol,
          &rho_sol, &r_sol, &ellipsoid_c_lagrangian_sol);
      drake::log()->info("rho {}", rho_sol);
      ellipsoid.rho = rho_sol;
      ellipsoid.rho_min = rho_sol;
    }
    // First determine if the ellipsoid center is within the super-level set
    // {x|h(x)>=0}, if yes, then find rho for the ellipsoid.
    for (auto it = uncovered_ellipsoids.begin();
         it != uncovered_ellipsoids.end();) {
      symbolic::Environment env;
      env.insert(x_, it->c);
      if (h_sol->Evaluate(env) > 0) {
        double rho_sol;
        symbolic::Polynomial r_sol;
        VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
        MaximizeInnerEllipsoidRho(
            x_, it->c, it->S, -(*h_sol), state_eq_constraints_, it->r_degree,
            it->state_constraints_lagrangian_degrees, it->rho_max, it->rho_min,
            search_options.lagrangian_step_solver,
            search_options.lagrangian_step_solver_options, it->rho_tol,
            &rho_sol, &r_sol, &ellipsoid_c_lagrangian_sol);
        inner_ellipsoids.emplace_back(it->c, it->S, rho_sol, rho_sol,
                                      it->rho_max, it->rho_tol, it->r_degree,
                                      it->state_constraints_lagrangian_degrees);
        drake::log()->info("rho {}", rho_sol);
        it = uncovered_ellipsoids.erase(it);
      } else {
        ++it;
      }
    }

    {
      // Now search for the barrier function.
      symbolic::Polynomial h;
      std::vector<VectorX<symbolic::Polynomial>> s;
      std::vector<std::vector<MatrixX<symbolic::Variable>>> s_grams;
      std::vector<symbolic::Polynomial> unsafe_sos_polys;
      std::vector<MatrixX<symbolic::Variable>> unsafe_sos_poly_grams;
      symbolic::Polynomial hdot_sos;
      MatrixX<symbolic::Variable> hdot_gram;
      auto prog_barrier = this->ConstructBarrierProgram(
          *lambda0_sol, *l_sol, hdot_state_constraints_lagrangian_degrees,
          *t_sol, unsafe_state_constraints_lagrangian_degrees, h_degree,
          deriv_eps, s_degrees, &h, &hdot_sos, &hdot_gram, &s, &s_grams,
          &unsafe_sos_polys, &unsafe_sos_poly_grams);
      std::vector<symbolic::Polynomial> r;
      VectorX<symbolic::Variable> d;
      std::vector<VectorX<symbolic::Polynomial>>
          ellipsoid_state_constraints_lagrangian;
      this->AddBarrierProgramCost(prog_barrier.get(), h, inner_ellipsoids, &r,
                                  &d, &ellipsoid_state_constraints_lagrangian);
      // To prevent scaling h arbitrarily to infinity, we constrain
      // h(x_anchor)
      // <= h_init(x_anchor).
      {
        Eigen::MatrixXd h_monomial_vals;
        VectorX<symbolic::Variable> h_coeff_vars;
        EvaluatePolynomial(h, x_, x_anchor, &h_monomial_vals, &h_coeff_vars);
        prog_barrier->AddLinearConstraint(h_monomial_vals.row(0), -kInf,
                                          h_at_x_anchor, h_coeff_vars);
      }

      if (search_options.barrier_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(prog_barrier.get(),
                        search_options.barrier_tiny_coeff_tol);
      }
      drake::log()->info("Search barrier");
      const auto result_barrier = SearchWithBackoff(
          prog_barrier.get(), search_options.barrier_step_solver,
          search_options.barrier_step_solver_options,
          search_options.backoff_scale);
      if (result_barrier.is_success()) {
        *h_sol = result_barrier.GetSolution(h);
        if (search_options.hsol_tiny_coeff_tol > 0) {
          *h_sol = h_sol->RemoveTermsWithSmallCoefficients(
              search_options.hsol_tiny_coeff_tol);
        }
        drake::log()->info("d: {}", result_barrier.GetSolution(d).transpose());
        s_sol->resize(s.size());
        for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
          GetPolynomialSolutions(result_barrier, s[i],
                                 search_options.hsol_tiny_coeff_tol,
                                 &(*s_sol)[i]);
        }
      } else {
        drake::log()->error("Failed to find the barrier.");
        return;
      }
    }
    iter_count++;
  }
}

bool ControlBarrier::SearchLagrangian(
    const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
    const std::vector<int>& l_degrees,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<int>& t_degree,
    const std::vector<std::vector<int>>& s_degrees,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    const ControlBarrier::SearchOptions& search_options,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    VectorX<symbolic::Polynomial>* hdot_state_constraints_lagrangian_sol,
    std::vector<symbolic::Polynomial>* t_sol,
    std::vector<VectorX<symbolic::Polynomial>>* s_sol,
    std::vector<VectorX<symbolic::Polynomial>>*
        unsafe_state_constraints_lagrangian_sol) const {
  {
    symbolic::Polynomial lambda0;
    MatrixX<symbolic::Variable> lambda0_gram;
    VectorX<symbolic::Polynomial> l;
    std::vector<MatrixX<symbolic::Variable>> l_grams;
    VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian;
    symbolic::Polynomial hdot_sos;
    VectorX<symbolic::Monomial> hdot_monomials;
    MatrixX<symbolic::Variable> hdot_gram;
    auto prog_lagrangian = this->ConstructLagrangianProgram(
        h, deriv_eps, lambda0_degree, l_degrees,
        hdot_state_constraints_lagrangian_degrees, &lambda0, &lambda0_gram, &l,
        &l_grams, &hdot_state_constraints_lagrangian, &hdot_sos,
        &hdot_monomials, &hdot_gram);
    if (search_options.lagrangian_tiny_coeff_tol > 0) {
      RemoveTinyCoeff(prog_lagrangian.get(),
                      search_options.lagrangian_tiny_coeff_tol);
    }
    auto lagrangian_solver =
        solvers::MakeSolver(search_options.lagrangian_step_solver);
    solvers::MathematicalProgramResult result_lagrangian;
    drake::log()->info("search Lagrangian");
    lagrangian_solver->Solve(*prog_lagrangian, std::nullopt,
                             search_options.lagrangian_step_solver_options,
                             &result_lagrangian);
    if (result_lagrangian.is_success()) {
      *lambda0_sol = result_lagrangian.GetSolution(lambda0);
      GetPolynomialSolutions(result_lagrangian, l,
                             search_options.lsol_tiny_coeff_tol, l_sol);
      GetPolynomialSolutions(result_lagrangian,
                             hdot_state_constraints_lagrangian,
                             search_options.lsol_tiny_coeff_tol,
                             hdot_state_constraints_lagrangian_sol);
    } else {
      drake::log()->error("Failed to find Lagrangian");
      return false;
    }
  }

  {
    // Find Lagrangian multiplier for each unsafe region.
    t_sol->resize(unsafe_regions_.size());
    s_sol->resize(unsafe_regions_.size());
    unsafe_state_constraints_lagrangian_sol->resize(unsafe_regions_.size());
    for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
      symbolic::Polynomial t;
      VectorX<symbolic::Polynomial> s;
      MatrixX<symbolic::Variable> t_gram;
      std::vector<MatrixX<symbolic::Variable>> s_grams;
      VectorX<symbolic::Polynomial> unsafe_state_constraints_lagrangian;
      symbolic::Polynomial unsafe_sos_poly;
      MatrixX<symbolic::Variable> unsafe_sos_poly_gram;

      auto prog_unsafe = this->ConstructUnsafeRegionProgram(
          h, i, t_degree[i], s_degrees[i],
          unsafe_state_constraints_lagrangian_degrees[i], &t, &t_gram, &s,
          &s_grams, &unsafe_state_constraints_lagrangian, &unsafe_sos_poly,
          &unsafe_sos_poly_gram);
      if (search_options.lagrangian_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(prog_unsafe.get(),
                        search_options.lagrangian_tiny_coeff_tol);
      }
      drake::log()->info("Search Lagrangian multiplier for unsafe region {}",
                         i);
      solvers::MathematicalProgramResult result_unsafe;
      auto lagrangian_solver =
          solvers::MakeSolver(search_options.lagrangian_step_solver);
      lagrangian_solver->Solve(*prog_unsafe, std::nullopt,
                               search_options.lagrangian_step_solver_options,
                               &result_unsafe);
      if (result_unsafe.is_success()) {
        (*t_sol)[i] = result_unsafe.GetSolution(t);
        if (search_options.lsol_tiny_coeff_tol > 0) {
          (*t_sol)[i] = (*t_sol)[i].RemoveTermsWithSmallCoefficients(
              search_options.lsol_tiny_coeff_tol);
        }
        GetPolynomialSolutions(result_unsafe, s,
                               search_options.lsol_tiny_coeff_tol,
                               &((*s_sol)[i]));
        GetPolynomialSolutions(
            result_unsafe, unsafe_state_constraints_lagrangian,
            search_options.lsol_tiny_coeff_tol,
            &((*unsafe_state_constraints_lagrangian_sol)[i]));
      } else {
        drake::log()->error(
            "Cannot find Lagrangian multipler for unsafe region {}", i);
        return false;
      }
    }
  }
  return true;
}

ControlBarrierBoxInputBound::ControlBarrierBoxInputBound(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    std::vector<VectorX<symbolic::Polynomial>> unsafe_regions)
    : f_{f},
      G_{G},
      nx_{static_cast<int>(f.rows())},
      nu_{static_cast<int>(G_.cols())},
      x_{x},
      x_set_{x_},
      candidate_safe_states_{candidate_safe_states},
      unsafe_regions_{std::move(unsafe_regions)} {
  DRAKE_DEMAND(G_.rows() == nx_);
  DRAKE_DEMAND(candidate_safe_states_.rows() == nx_);
}

ControlBarrierBoxInputBound::HdotSosConstraintReturn::HdotSosConstraintReturn(
    int nu)
    : monomials{static_cast<size_t>(nu)}, grams{static_cast<size_t>(nu)} {
  for (int i = 0; i < nu; ++i) {
    monomials[i].resize(2);
    grams[i].resize(2);
  }
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrierBoxInputBound::ConstructLagrangianAndBProgram(
    const symbolic::Polynomial& h,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 2>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    std::vector<std::vector<std::array<symbolic::Polynomial, 2>>>* lagrangians,
    std::vector<std::vector<std::array<MatrixX<symbolic::Variable>, 2>>>*
        lagrangian_grams,
    VectorX<symbolic::Polynomial>* b, symbolic::Variable* deriv_eps,
    HdotSosConstraintReturn* hdot_sos_constraint) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  DRAKE_DEMAND(static_cast<int>(b_degrees.size()) == nu_);
  lagrangians->resize(nu_);
  lagrangian_grams->resize(nu_);
  // Add Lagrangian decision variables.
  const int num_hdot_sos = 2;
  for (int i = 0; i < nu_; ++i) {
    (*lagrangians)[i].resize(num_hdot_sos);
    (*lagrangian_grams)[i].resize(num_hdot_sos);
    for (int j = 0; j < num_hdot_sos; ++j) {
      (*lagrangians)[i][j][0] = l_given[i][j];

      DRAKE_DEMAND(lagrangian_degrees[i][j][1] % 2 == 0);
      std::tie((*lagrangians)[i][j][1], (*lagrangian_grams)[i][j][1]) =
          prog->NewSosPolynomial(x_set_, lagrangian_degrees[i][j][1]);
    }
  }

  *deriv_eps = prog->NewContinuousVariables<1>("eps")(0);

  const RowVectorX<symbolic::Polynomial> dhdx = h.Jacobian(x_);
  // Since we will add the constraint -∂h/∂x*f(x) - εh = ∑ᵢ bᵢ(x), we know
  // that the highest degree of b should be at least degree(∂h/∂x*f(x) + εh).
  const symbolic::Polynomial dhdx_times_f = (dhdx * f_)(0);
  if (*std::max_element(b_degrees.begin(), b_degrees.end()) <
      std::max(dhdx_times_f.TotalDegree(), h.TotalDegree())) {
    throw std::invalid_argument("The degree of b is too low.");
  }
  b->resize(nu_);
  for (int i = 0; i < nu_; ++i) {
    (*b)(i) =
        prog->NewFreePolynomial(x_set_, b_degrees[i], "b" + std::to_string(i));
  }
  // Add the constraint -∂h/∂x*f(x) - εh = ∑ᵢ bᵢ(x)
  prog->AddEqualityConstraintBetweenPolynomials(
      b->sum(), -dhdx_times_f - (*deriv_eps) * h);
  // Add the constraint
  // (1+lᵢ₀₀(x))(∂h/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)∂h/∂xGᵢ(x) is sos
  // (1+lᵢ₁₀(x))(−∂h/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)∂h/∂xGᵢ(x) is sos
  AddHdotSosConstraint(prog.get(), *lagrangians, dhdx, G_, *b,
                       hdot_sos_constraint);
  return prog;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
