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
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x, double beta_minus,
    std::optional<double> beta_plus,
    std::vector<VectorX<symbolic::Polynomial>> unsafe_regions,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_eq_constraints)
    : f_{f},
      G_{G},
      dynamics_denominator_{std::move(dynamics_denominator)},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      beta_minus_{beta_minus},
      beta_plus_{beta_plus},
      x_{x},
      x_set_{x},
      unsafe_regions_{std::move(unsafe_regions)},
      u_vertices_{u_vertices},
      state_eq_constraints_{state_eq_constraints} {
  DRAKE_DEMAND(beta_minus_ <= 0);
  if (beta_plus_.has_value()) {
    DRAKE_DEMAND(*beta_plus_ > 0);
  }
  DRAKE_DEMAND(G_.rows() == nx_);
  DRAKE_DEMAND(u_vertices_.rows() == nu_);
}

void ControlBarrier::AddControlBarrierConstraint(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& lambda0,
    const std::optional<symbolic::Polynomial>& lambda1,
    const VectorX<symbolic::Polynomial>& l,
    const VectorX<symbolic::Polynomial>& state_constraints_lagrangian,
    const symbolic::Polynomial& h, double deriv_eps,
    const std::optional<symbolic::Polynomial>& a,
    symbolic::Polynomial* hdot_poly, VectorX<symbolic::Monomial>* monomials,
    MatrixX<symbolic::Variable>* gram) const {
  *hdot_poly = (1 + lambda0) * (beta_minus_ - h);
  const RowVectorX<symbolic::Polynomial> dhdx = h.Jacobian(x_);
  const symbolic::Polynomial dhdx_times_f = dhdx.dot(f_);
  *hdot_poly -=
      l.sum() *
      (-dhdx_times_f -
       deriv_eps * h * dynamics_denominator_.value_or(symbolic::Polynomial(1)));
  *hdot_poly += (dhdx * G_ * u_vertices_).dot(l);
  if (beta_plus_.has_value()) {
    *hdot_poly -= *lambda1 * (*beta_plus_ - h);
  }
  DRAKE_DEMAND(state_eq_constraints_.rows() ==
               state_constraints_lagrangian.rows());
  *hdot_poly -= state_constraints_lagrangian.dot(state_eq_constraints_);
  if (a.has_value()) {
    *hdot_poly += a.value();
  }
  std::tie(*gram, *monomials) = prog->AddSosConstraint(
      *hdot_poly, solvers::MathematicalProgram::NonnegativePolynomial::kSos,
      "hd");
}

ControlBarrier::LagrangianReturn ControlBarrier::ConstructLagrangianProgram(
    const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
    std::optional<int> lambda1_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& state_constraints_lagrangian_degrees) const {
  DRAKE_DEMAND(static_cast<int>(l_degrees.size()) == u_vertices_.cols());
  LagrangianReturn ret{};
  ret.prog->AddIndeterminates(x_);
  // Now construct Lagrangian multipliers.
  std::tie(ret.lambda0, ret.lambda0_gram) =
      ret.prog->NewSosPolynomial(x_set_, lambda0_degree);
  if (beta_plus_.has_value()) {
    ret.lambda1.emplace(symbolic::Polynomial());
    ret.lambda1_gram.emplace(MatrixX<symbolic::Variable>());
    std::tie(*(ret.lambda1), *(ret.lambda1_gram)) =
        ret.prog->NewSosPolynomial(x_set_, *lambda1_degree);
  }
  const int num_u_vertices = u_vertices_.cols();
  ret.l.resize(num_u_vertices);
  ret.l_grams.resize(num_u_vertices);
  for (int i = 0; i < num_u_vertices; ++i) {
    std::tie(ret.l(i), ret.l_grams[i]) =
        ret.prog->NewSosPolynomial(x_set_, l_degrees[i]);
  }
  ret.state_constraints_lagrangian.resize(state_eq_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(state_constraints_lagrangian_degrees.size()) ==
               state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    ret.state_constraints_lagrangian(i) = ret.prog->NewFreePolynomial(
        x_set_, state_constraints_lagrangian_degrees[i],
        "le" + std::to_string(i));
  }
  this->AddControlBarrierConstraint(
      ret.prog.get(), ret.lambda0, ret.lambda1, ret.l,
      ret.state_constraints_lagrangian, h, deriv_eps, std::nullopt /* a */,
      &(ret.hdot_sos), &(ret.hdot_monomials), &(ret.hdot_gram));
  return ret;
}

ControlBarrier::UnsafeReturn ControlBarrier::ConstructUnsafeRegionProgram(
    const symbolic::Polynomial& h, int region_index, int t_degree,
    const std::vector<int>& s_degrees,
    const std::vector<int>& unsafe_state_constraints_lagrangian_degrees) const {
  ControlBarrier::UnsafeReturn ret{};
  ret.prog->AddIndeterminates(x_);
  std::tie(ret.t, ret.t_gram) = ret.prog->NewSosPolynomial(
      x_set_, t_degree,
      solvers::MathematicalProgram::NonnegativePolynomial::kSos, "T");
  ret.s.resize(unsafe_regions_[region_index].rows());
  ret.s_grams.resize(ret.s.rows());
  for (int i = 0; i < ret.s.rows(); ++i) {
    std::tie(ret.s(i), ret.s_grams[i]) = ret.prog->NewSosPolynomial(
        x_set_, s_degrees[i],
        solvers::MathematicalProgram::NonnegativePolynomial::kSos, "S");
  }
  DRAKE_DEMAND(
      static_cast<int>(unsafe_state_constraints_lagrangian_degrees.size()) ==
      state_eq_constraints_.rows());
  ret.state_constraints_lagrangian.resize(state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    ret.state_constraints_lagrangian(i) = ret.prog->NewFreePolynomial(
        x_set_, unsafe_state_constraints_lagrangian_degrees[i]);
  }
  ret.sos_poly = (1 + ret.t) * (-h) + ret.s.dot(unsafe_regions_[region_index]) -
                 ret.state_constraints_lagrangian.dot(state_eq_constraints_);
  std::tie(ret.sos_poly_gram, std::ignore) =
      ret.prog->AddSosConstraint(ret.sos_poly);
  return ret;
}

ControlBarrier::BarrierReturn ControlBarrier::ConstructBarrierProgram(
    const symbolic::Polynomial& lambda0,
    const std::optional<symbolic::Polynomial>& lambda1,
    const VectorX<symbolic::Polynomial>& l,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<symbolic::Polynomial>& t,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    int h_degree, double deriv_eps,
    const std::vector<std::vector<int>>& s_degrees) const {
  BarrierReturn ret{};
  ret.prog->AddIndeterminates(x_);
  ret.h = ret.prog->NewFreePolynomial(x_set_, h_degree, "H");
  VectorX<symbolic::Monomial> hdot_monomials;
  ret.hdot_state_constraints_lagrangian.resize(state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    ret.hdot_state_constraints_lagrangian(i) = ret.prog->NewFreePolynomial(
        x_set_, hdot_state_constraints_lagrangian_degrees[i]);
  }
  // Add the constraint on hdot.
  this->AddControlBarrierConstraint(ret.prog.get(), lambda0, lambda1, l,
                                    ret.hdot_state_constraints_lagrangian,
                                    ret.h, deriv_eps, std::nullopt /* a */,
                                    &(ret.hdot_sos), &hdot_monomials,
                                    &(ret.hdot_sos_gram));
  // Add the constraint that the unsafe region has h <= 0
  const int num_unsafe_regions = static_cast<int>(unsafe_regions_.size());
  DRAKE_DEMAND(static_cast<int>(t.size()) == num_unsafe_regions);
  DRAKE_DEMAND(static_cast<int>(s_degrees.size()) == num_unsafe_regions);
  ret.s.resize(num_unsafe_regions);
  ret.s_grams.resize(num_unsafe_regions);
  ret.unsafe_state_constraints_lagrangian.resize(num_unsafe_regions);
  ret.unsafe_sos_polys.resize(num_unsafe_regions);
  ret.unsafe_sos_poly_grams.resize(num_unsafe_regions);
  for (int i = 0; i < num_unsafe_regions; ++i) {
    const int num_unsafe_polys = static_cast<int>(unsafe_regions_[i].rows());
    ret.s[i].resize(num_unsafe_polys);
    ret.s_grams[i].resize(num_unsafe_polys);
    DRAKE_DEMAND(static_cast<int>(s_degrees[i].size()) == num_unsafe_polys);
    for (int j = 0; j < num_unsafe_polys; ++j) {
      std::tie(ret.s[i](j), ret.s_grams[i][j]) = ret.prog->NewSosPolynomial(
          x_set_, s_degrees[i][j],
          solvers::MathematicalProgram::NonnegativePolynomial::kSos,
          fmt::format("S{},{}", i, j));
    }
    ret.unsafe_state_constraints_lagrangian[i].resize(
        state_eq_constraints_.rows());
    for (int j = 0; j < state_eq_constraints_.rows(); ++j) {
      ret.unsafe_state_constraints_lagrangian[i](j) =
          ret.prog->NewFreePolynomial(
              x_set_, unsafe_state_constraints_lagrangian_degrees[i][j]);
    }
    ret.unsafe_sos_polys[i] =
        (1 + t[i]) * (-ret.h) + ret.s[i].dot(unsafe_regions_[i]) -
        ret.unsafe_state_constraints_lagrangian[i].dot(state_eq_constraints_);
    std::tie(ret.unsafe_sos_poly_grams[i], std::ignore) =
        ret.prog->AddSosConstraint(ret.unsafe_sos_polys[i]);
  }

  return ret;
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
    std::vector<symbolic::Polynomial>* r, VectorX<symbolic::Variable>* rho,
    std::vector<VectorX<symbolic::Polynomial>>* state_constraints_lagrangian)
    const {
  r->resize(inner_ellipsoids.size());
  *rho = prog->NewContinuousVariables(static_cast<int>(inner_ellipsoids.size()),
                                      "rho");
  state_constraints_lagrangian->resize(inner_ellipsoids.size());
  for (int i = 0; i < static_cast<int>(inner_ellipsoids.size()); ++i) {
    std::tie((*r)[i], std::ignore) = prog->NewSosPolynomial(
        x_set_, inner_ellipsoids[i].r_degree,
        solvers::MathematicalProgram::NonnegativePolynomial::kSos, "R");
    DRAKE_DEMAND(
        static_cast<int>(inner_ellipsoids[i].eq_lagrangian_degrees.size()) ==
        state_eq_constraints_.rows());
    (*state_constraints_lagrangian)[i].resize(state_eq_constraints_.rows());
    for (int j = 0; j < state_eq_constraints_.rows(); ++j) {
      (*state_constraints_lagrangian)[i](j) = prog->NewFreePolynomial(
          x_set_, inner_ellipsoids[i].eq_lagrangian_degrees[j]);
    }
    prog->AddSosConstraint(
        h - (*rho)(i) +
        (*r)[i] * internal::EllipsoidPolynomial(x_, inner_ellipsoids[i].c,
                                                inner_ellipsoids[i].S,
                                                inner_ellipsoids[i].d) -
        (*state_constraints_lagrangian)[i].dot(state_eq_constraints_));
  }
  prog->AddLinearCost(-Eigen::VectorXd::Ones(rho->rows()), 0, *rho);
  prog->AddBoundingBoxConstraint(0, kInf, *rho);
}

ControlBarrier::SearchReturn ControlBarrier::Search(
    const symbolic::Polynomial& h_init, int h_degree, double deriv_eps,
    int lambda0_degree, std::optional<int> lambda1_degree,
    const std::vector<int>& l_degrees,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<int>& t_degree,
    const std::vector<std::vector<int>>& s_degrees,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_anchor,
    const SearchOptions& search_options,
    std::vector<ControlBarrier::Ellipsoid>* ellipsoids,
    std::vector<EllipsoidBisectionOption>* ellipsoid_bisection_options) const {
  SearchReturn ret;
  ret.h = h_init;
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

  // inner_ellipsoid_flag[i] is true if and only if the center of ellipsoids[i]
  // is covered in the safe region h(x) >= 0.
  std::vector<bool> inner_ellipsoid_flag(ellipsoids->size(), false);
  DRAKE_DEMAND(ellipsoids->size() == ellipsoid_bisection_options->size());
  while (iter_count < search_options.bilinear_iterations) {
    const auto search_lagrangian_ret = SearchLagrangian(
        ret.h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
        hdot_state_constraints_lagrangian_degrees, t_degree, s_degrees,
        unsafe_state_constraints_lagrangian_degrees, search_options);
    if (!search_lagrangian_ret.success) {
      return ret;
    } else {
      ret.lambda0 = search_lagrangian_ret.lambda0;
      ret.lambda1 = search_lagrangian_ret.lambda1;
      ret.l = search_lagrangian_ret.l;
      ret.hdot_state_constraints_lagrangian =
          search_lagrangian_ret.hdot_state_constraints_lagrangian;
      ret.t = search_lagrangian_ret.t;
      ret.s = search_lagrangian_ret.s;
      ret.unsafe_state_constraints_lagrangian =
          search_lagrangian_ret.unsafe_state_constraints_lagrangian;
    }

    // Maximize the inner ellipsoids.
    // For each inner ellipsoid, compute d.
    for (int ellipsoid_idx = 0;
         ellipsoid_idx < static_cast<int>(ellipsoids->size());
         ellipsoid_idx++) {
      auto& ellipsoid = (*ellipsoids)[ellipsoid_idx];
      if (ret.h.EvaluateIndeterminates(x_, ellipsoid.c)(0) > 0) {
        inner_ellipsoid_flag[ellipsoid_idx] = true;
        double d_sol;
        symbolic::Polynomial r_sol;
        VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;
        auto& ellipsoid_bisection_option =
            (*ellipsoid_bisection_options)[ellipsoid_idx];
        MaximizeInnerEllipsoidSize(
            x_, ellipsoid.c, ellipsoid.S, -ret.h, state_eq_constraints_,
            ellipsoid.r_degree, ellipsoid.eq_lagrangian_degrees,
            ellipsoid_bisection_option.d_max, ellipsoid_bisection_option.d_min,
            search_options.lagrangian_step_solver,
            search_options.lagrangian_step_solver_options,
            ellipsoid_bisection_option.d_tol, &d_sol, &r_sol,
            &ellipsoid_c_lagrangian_sol);
        drake::log()->info("d {}", d_sol);
        ellipsoid.d = d_sol;
        ellipsoid_bisection_option.d_min = d_sol;
      } else {
        inner_ellipsoid_flag[ellipsoid_idx] = false;
      }
    }

    {
      // Now search for the barrier function.
      auto barrier_ret = this->ConstructBarrierProgram(
          ret.lambda0, ret.lambda1, ret.l,
          hdot_state_constraints_lagrangian_degrees, ret.t,
          unsafe_state_constraints_lagrangian_degrees, h_degree, deriv_eps,
          s_degrees);
      std::vector<symbolic::Polynomial> r;
      VectorX<symbolic::Variable> rho;
      std::vector<VectorX<symbolic::Polynomial>>
          ellipsoid_state_constraints_lagrangian;
      std::vector<Ellipsoid> inner_ellipsoids;
      for (int ellipsoid_idx = 0;
           ellipsoid_idx < static_cast<int>(ellipsoids->size());
           ++ellipsoid_idx) {
        if (inner_ellipsoid_flag[ellipsoid_idx]) {
          inner_ellipsoids.push_back((*ellipsoids)[ellipsoid_idx]);
        }
      }
      this->AddBarrierProgramCost(barrier_ret.prog.get(), barrier_ret.h,
                                  inner_ellipsoids, &r, &rho,
                                  &ellipsoid_state_constraints_lagrangian);
      // To prevent scaling h arbitrarily to infinity, we constrain
      // h(x_anchor)
      // <= h_init(x_anchor).
      {
        Eigen::MatrixXd h_monomial_vals;
        VectorX<symbolic::Variable> h_coeff_vars;
        EvaluatePolynomial(barrier_ret.h, x_, x_anchor, &h_monomial_vals,
                           &h_coeff_vars);
        barrier_ret.prog->AddLinearConstraint(h_monomial_vals.row(0), -kInf,
                                              h_at_x_anchor, h_coeff_vars);
      }

      if (search_options.barrier_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(barrier_ret.prog.get(),
                        search_options.barrier_tiny_coeff_tol);
      }
      drake::log()->info("Search barrier");
      const auto result_barrier = SearchWithBackoff(
          barrier_ret.prog.get(), search_options.barrier_step_solver,
          search_options.barrier_step_solver_options,
          search_options.backoff_scale);
      if (result_barrier.is_success()) {
        ret.h = result_barrier.GetSolution(barrier_ret.h);
        if (search_options.hsol_tiny_coeff_tol > 0) {
          ret.h = ret.h.RemoveTermsWithSmallCoefficients(
              search_options.hsol_tiny_coeff_tol);
        }
        GetPolynomialSolutions(result_barrier,
                               barrier_ret.hdot_state_constraints_lagrangian,
                               search_options.lsol_tiny_coeff_tol,
                               &(ret.hdot_state_constraints_lagrangian));
        drake::log()->info("min h(x) on ellipsoid: {}",
                           result_barrier.GetSolution(rho).transpose());
        ret.s.resize(barrier_ret.s.size());
        ret.unsafe_state_constraints_lagrangian.resize(unsafe_regions_.size());
        for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
          GetPolynomialSolutions(result_barrier, barrier_ret.s[i],
                                 search_options.lsol_tiny_coeff_tol,
                                 &(ret.s[i]));
          GetPolynomialSolutions(
              result_barrier,
              barrier_ret.unsafe_state_constraints_lagrangian[i],
              search_options.lsol_tiny_coeff_tol,
              &(ret.unsafe_state_constraints_lagrangian[i]));
        }
      } else {
        drake::log()->error("Failed to find the barrier.");
        return ret;
      }
    }
    iter_count++;
  }
  return ret;
}

ControlBarrier::SearchLagrangianReturn ControlBarrier::SearchLagrangian(
    const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
    std::optional<int> lambda1_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& hdot_state_constraints_lagrangian_degrees,
    const std::vector<int>& t_degree,
    const std::vector<std::vector<int>>& s_degrees,
    const std::vector<std::vector<int>>&
        unsafe_state_constraints_lagrangian_degrees,
    const ControlBarrier::SearchOptions& search_options) const {
  ControlBarrier::SearchLagrangianReturn search_lagrangian_ret;
  search_lagrangian_ret.success = true;
  {
    auto lagrangian_ret = this->ConstructLagrangianProgram(
        h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
        hdot_state_constraints_lagrangian_degrees);
    if (search_options.lagrangian_tiny_coeff_tol > 0) {
      RemoveTinyCoeff(lagrangian_ret.prog.get(),
                      search_options.lagrangian_tiny_coeff_tol);
    }
    auto lagrangian_solver =
        solvers::MakeSolver(search_options.lagrangian_step_solver);
    solvers::MathematicalProgramResult result_lagrangian;
    drake::log()->info("search Lagrangian");
    lagrangian_solver->Solve(*(lagrangian_ret.prog), std::nullopt,
                             search_options.lagrangian_step_solver_options,
                             &result_lagrangian);
    if (result_lagrangian.is_success()) {
      search_lagrangian_ret.lambda0 =
          result_lagrangian.GetSolution(lagrangian_ret.lambda0);
      GetPolynomialSolutions(result_lagrangian, lagrangian_ret.l,
                             search_options.lsol_tiny_coeff_tol,
                             &(search_lagrangian_ret.l));
      GetPolynomialSolutions(
          result_lagrangian, lagrangian_ret.state_constraints_lagrangian,
          search_options.lsol_tiny_coeff_tol,
          &(search_lagrangian_ret.hdot_state_constraints_lagrangian));
      if (beta_plus_.has_value()) {
        search_lagrangian_ret.lambda1 =
            result_lagrangian.GetSolution(*(lagrangian_ret.lambda1));
        if (search_options.lsol_tiny_coeff_tol > 0) {
          search_lagrangian_ret.lambda1 =
              search_lagrangian_ret.lambda1->RemoveTermsWithSmallCoefficients(
                  search_options.lsol_tiny_coeff_tol);
        }
      }
    } else {
      drake::log()->error("Failed to find Lagrangian");
      search_lagrangian_ret.success = false;
      return search_lagrangian_ret;
    }
  }

  {
    // Find Lagrangian multiplier for each unsafe region.
    //
    search_lagrangian_ret.t.resize(unsafe_regions_.size());
    search_lagrangian_ret.s.resize(unsafe_regions_.size());
    search_lagrangian_ret.unsafe_state_constraints_lagrangian.resize(
        unsafe_regions_.size());
    for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
      auto unsafe_ret = this->ConstructUnsafeRegionProgram(
          h, i, t_degree[i], s_degrees[i],
          unsafe_state_constraints_lagrangian_degrees[i]);
      if (search_options.lagrangian_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(unsafe_ret.prog.get(),
                        search_options.lagrangian_tiny_coeff_tol);
      }
      drake::log()->info("Search Lagrangian multiplier for unsafe region {}",
                         i);
      solvers::MathematicalProgramResult result_unsafe;
      auto lagrangian_solver =
          solvers::MakeSolver(search_options.lagrangian_step_solver);
      lagrangian_solver->Solve(*(unsafe_ret.prog), std::nullopt,
                               search_options.lagrangian_step_solver_options,
                               &result_unsafe);
      if (result_unsafe.is_success()) {
        search_lagrangian_ret.t[i] = result_unsafe.GetSolution(unsafe_ret.t);
        if (search_options.lsol_tiny_coeff_tol > 0) {
          search_lagrangian_ret.t[i] =
              search_lagrangian_ret.t[i].RemoveTermsWithSmallCoefficients(
                  search_options.lsol_tiny_coeff_tol);
        }
        GetPolynomialSolutions(result_unsafe, unsafe_ret.s,
                               search_options.lsol_tiny_coeff_tol,
                               &(search_lagrangian_ret.s[i]));
        GetPolynomialSolutions(
            result_unsafe, unsafe_ret.state_constraints_lagrangian,
            search_options.lsol_tiny_coeff_tol,
            &(search_lagrangian_ret.unsafe_state_constraints_lagrangian[i]));
      } else {
        drake::log()->error(
            "Cannot find Lagrangian multipler for unsafe region {}", i);
        search_lagrangian_ret.success = false;
        return search_lagrangian_ret;
      }
    }
  }
  return search_lagrangian_ret;
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

CbfController::CbfController(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    std::optional<symbolic::Polynomial> dynamics_denominator,
    symbolic::Polynomial cbf, double deriv_eps)
    : LeafSystem<double>(),
      x_{x},
      f_{f},
      G_{G},
      dynamics_denominator_{std::move(dynamics_denominator)},
      cbf_{std::move(cbf)},
      deriv_eps_{deriv_eps} {
  const int nx = f_.rows();
  const int nu = G_.cols();
  DRAKE_DEMAND(x_.rows() == nx);
  const RowVectorX<symbolic::Polynomial> dhdx = cbf_.Jacobian(x_);
  dhdx_times_f_ = dhdx.dot(f_);
  dhdx_times_G_ = dhdx * G_;

  x_input_index_ = this->DeclareVectorInputPort("x", nx).get_index();

  control_output_index_ =
      this->DeclareVectorOutputPort("control", nu, &CbfController::CalcControl)
          .get_index();

  cbf_output_index_ =
      this->DeclareVectorOutputPort("cbf", 1, &CbfController::CalcCbf)
          .get_index();
}

void CbfController::CalcCbf(const Context<double>& context,
                            BasicVector<double>* output) const {
  const Eigen::VectorXd x_val =
      this->get_input_port(x_input_index_).Eval(context);
  symbolic::Environment env;
  env.insert(x_, x_val);
  Eigen::VectorBlock<VectorX<double>> cbf_vec = output->get_mutable_value();
  cbf_vec(0) = cbf_.Evaluate(env);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
