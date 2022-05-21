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
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    std::vector<VectorX<symbolic::Polynomial>> unsafe_regions,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices)
    : f_{f},
      G_{G},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      x_{x},
      x_set_{x},
      candidate_safe_states_{candidate_safe_states},
      unsafe_regions_{std::move(unsafe_regions)},
      u_vertices_{u_vertices} {
  DRAKE_DEMAND(G_.rows() == nx_);
  DRAKE_DEMAND(candidate_safe_states_.rows() == nx_);
  DRAKE_DEMAND(u_vertices_.rows() == nu_);
}
void ControlBarrier::AddControlBarrierConstraint(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& lambda0,
    const VectorX<symbolic::Polynomial>& l, const symbolic::Polynomial& h,
    double deriv_eps, symbolic::Polynomial* hdot_poly,
    VectorX<symbolic::Monomial>* monomials,
    MatrixX<symbolic::Variable>* gram) const {
  *hdot_poly = (1 + lambda0) * (-1 - h);
  const RowVectorX<symbolic::Polynomial> dhdx = h.Jacobian(x_);
  const symbolic::Polynomial dhdx_times_f = dhdx.dot(f_);
  *hdot_poly -= l.sum() * (-dhdx_times_f - deriv_eps * h);
  *hdot_poly += (dhdx * G_ * u_vertices_).dot(l);
  std::tie(*gram, *monomials) = prog->AddSosConstraint(
      *hdot_poly, solvers::MathematicalProgram::NonnegativePolynomial::kSos,
      "hd");
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructLagrangianProgram(
    const symbolic::Polynomial& h, double deriv_eps, int lambda0_degree,
    const std::vector<int>& l_degrees, symbolic::Polynomial* lambda0,
    MatrixX<symbolic::Variable>* lambda0_gram, VectorX<symbolic::Polynomial>* l,
    std::vector<MatrixX<symbolic::Variable>>* l_grams,
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
  this->AddControlBarrierConstraint(prog.get(), *lambda0, *l, h, deriv_eps,
                                    hdot_sos, hdot_monomials, hdot_gram);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructUnsafeRegionProgram(
    const symbolic::Polynomial& h, int region_index, int t_degree,
    const std::vector<int>& s_degrees, symbolic::Polynomial* t,
    MatrixX<symbolic::Variable>* t_gram, VectorX<symbolic::Polynomial>* s,
    std::vector<MatrixX<symbolic::Variable>>* s_grams,
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
  *sos_poly = (1 + *t) * (-h) + s->dot(unsafe_regions_[region_index]);
  std::tie(*sos_poly_gram, std::ignore) = prog->AddSosConstraint(*sos_poly);
  return prog;
}

std::unique_ptr<solvers::MathematicalProgram>
ControlBarrier::ConstructBarrierProgram(
    const symbolic::Polynomial& lambda0, const VectorX<symbolic::Polynomial>& l,
    const std::vector<symbolic::Polynomial>& t, int h_degree, double deriv_eps,
    const std::vector<std::vector<int>>& s_degrees,
    const Eigen::MatrixXd& verified_safe_states,
    const Eigen::MatrixXd& unverified_candidate_states, double eps,
    symbolic::Polynomial* h, symbolic::Polynomial* hdot_sos,
    MatrixX<symbolic::Variable>* hdot_sos_gram,
    std::vector<VectorX<symbolic::Polynomial>>* s,
    std::vector<std::vector<MatrixX<symbolic::Variable>>>* s_grams,
    std::vector<symbolic::Polynomial>* unsafe_sos_polys,
    std::vector<MatrixX<symbolic::Variable>>* unsafe_sos_poly_grams) const {
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  *h = prog->NewFreePolynomial(x_set_, h_degree, "H");
  VectorX<symbolic::Monomial> hdot_monomials;
  // Add the constraint on hdot.
  this->AddControlBarrierConstraint(prog.get(), lambda0, l, *h, deriv_eps,
                                    hdot_sos, &hdot_monomials, hdot_sos_gram);
  // Add the constraint that the unsafe region has h <= 0
  const int num_unsafe_regions = static_cast<int>(unsafe_regions_.size());
  DRAKE_DEMAND(static_cast<int>(t.size()) == num_unsafe_regions);
  DRAKE_DEMAND(static_cast<int>(s_degrees.size()) == num_unsafe_regions);
  s->resize(num_unsafe_regions);
  s_grams->resize(num_unsafe_regions);
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
    (*unsafe_sos_polys)[i] =
        (1 + t[i]) * (-(*h)) + (*s)[i].dot(unsafe_regions_[i]);
    std::tie((*unsafe_sos_poly_grams)[i], std::ignore) =
        prog->AddSosConstraint((*unsafe_sos_polys)[i]);
  }

  // Add the constraint that the verified states all have h(x) >= 0
  Eigen::MatrixXd h_monomial_vals;
  VectorX<symbolic::Variable> h_coeff_vars;
  EvaluatePolynomial(*h, x_, verified_safe_states, &h_monomial_vals,
                     &h_coeff_vars);
  prog->AddLinearConstraint(
      h_monomial_vals, Eigen::VectorXd::Zero(h_monomial_vals.rows()),
      Eigen::VectorXd::Constant(h_monomial_vals.rows(), kInf), h_coeff_vars);
  // Add the objective to maximize sum min(h(xʲ), eps) for xʲ in
  // unverified_candidate_states
  EvaluatePolynomial(*h, x_, unverified_candidate_states, &h_monomial_vals,
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

  return prog;
}

void ControlBarrier::Search(
    const symbolic::Polynomial& h_init, int h_degree, double deriv_eps,
    int lambda0_degree, const std::vector<int>& l_degrees,
    const std::vector<int>& t_degree,
    const std::vector<std::vector<int>>& s_degrees,
    const SearchOptions& search_options, symbolic::Polynomial* h_sol,
    symbolic::Polynomial* lambda0_sol, VectorX<symbolic::Polynomial>* l_sol,
    std::vector<symbolic::Polynomial>* t_sol,
    std::vector<VectorX<symbolic::Polynomial>>* s_sol,
    Eigen::MatrixXd* verified_safe_states,
    Eigen::MatrixXd* unverified_candidate_states) const {
  *h_sol = h_init;
  SplitCandidateStates(*h_sol, x_, candidate_safe_states_, verified_safe_states,
                       unverified_candidate_states);
  drake::log()->info("Number of verified safe states: {}",
                     verified_safe_states->cols());
  int verified_safe_states_count = verified_safe_states->cols();

  int iter_count = 0;

  while (iter_count < search_options.bilinear_iterations) {
    {
      symbolic::Polynomial lambda0;
      MatrixX<symbolic::Variable> lambda0_gram;
      VectorX<symbolic::Polynomial> l;
      std::vector<MatrixX<symbolic::Variable>> l_grams;
      symbolic::Polynomial hdot_sos;
      VectorX<symbolic::Monomial> hdot_monomials;
      MatrixX<symbolic::Variable> hdot_gram;
      auto prog_lagrangian = this->ConstructLagrangianProgram(
          h_init, deriv_eps, lambda0_degree, l_degrees, &lambda0, &lambda0_gram,
          &l, &l_grams, &hdot_sos, &hdot_monomials, &hdot_gram);
      if (search_options.lagrangian_tiny_coeff_tol > 0) {
        RemoveTinyCoeff(prog_lagrangian.get(),
                        search_options.lagrangian_tiny_coeff_tol);
      }
      auto lagrangian_solver =
          solvers::MakeSolver(search_options.lagrangian_step_solver);
      solvers::MathematicalProgramResult result_lagrangian;
      drake::log()->info("Iter {}, search Lagrangian", iter_count);
      lagrangian_solver->Solve(*prog_lagrangian, std::nullopt,
                               search_options.lagrangian_step_solver_options,
                               &result_lagrangian);
      if (result_lagrangian.is_success()) {
        *lambda0_sol = result_lagrangian.GetSolution(lambda0);
        l_sol->resize(l.rows());
        for (int i = 0; i < l_sol->rows(); ++i) {
          (*l_sol)(i) = result_lagrangian.GetSolution(l(i));
          if (search_options.lsol_tiny_coeff_tol > 0) {
            (*l_sol)(i) = (*l_sol)(i).RemoveTermsWithSmallCoefficients(
                search_options.lsol_tiny_coeff_tol);
          }
        }
      } else {
        drake::log()->error("Failed to find Lagrangian");
        return;
      }
    }

    {
      // Find Lagrangian multiplier for each unsafe region.
      t_sol->resize(unsafe_regions_.size());
      s_sol->resize(unsafe_regions_.size());
      for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
        symbolic::Polynomial t;
        VectorX<symbolic::Polynomial> s;
        MatrixX<symbolic::Variable> t_gram;
        std::vector<MatrixX<symbolic::Variable>> s_grams;
        symbolic::Polynomial unsafe_sos_poly;
        MatrixX<symbolic::Variable> unsafe_sos_poly_gram;

        auto prog_unsafe = this->ConstructUnsafeRegionProgram(
            *h_sol, i, t_degree[i], s_degrees[i], &t, &t_gram, &s, &s_grams,
            &unsafe_sos_poly, &unsafe_sos_poly_gram);
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
          (*s_sol)[i].resize(s.rows());
          for (int j = 0; j < (*s_sol)[i].rows(); ++j) {
            (*s_sol)[i](j) = result_unsafe.GetSolution(s(j));
            if (search_options.lsol_tiny_coeff_tol > 0) {
              (*s_sol)[i](j) = (*s_sol)[i](j).RemoveTermsWithSmallCoefficients(
                  search_options.lsol_tiny_coeff_tol);
            }
          }
        } else {
          drake::log()->error(
              "Cannot find Lagrangian multipler for unsafe region {}", i);
          return;
        }
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
          *lambda0_sol, *l_sol, *t_sol, h_degree, deriv_eps, s_degrees,
          *verified_safe_states, *unverified_candidate_states,
          search_options.candidate_state_eps, &h, &hdot_sos, &hdot_gram, &s,
          &s_grams, &unsafe_sos_polys, &unsafe_sos_poly_grams);
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
        std::cout << "h: " << *h_sol << "\n";
        SplitCandidateStates(*h_sol, x_, candidate_safe_states_,
                             verified_safe_states, unverified_candidate_states);
        drake::log()->info("Number of verified safe states {}",
                           verified_safe_states->cols());
        if (verified_safe_states->cols() == verified_safe_states_count) {
          return;
        } else {
          verified_safe_states_count = verified_safe_states->cols();
        }
        s_sol->resize(s.size());
        for (int i = 0; i < static_cast<int>(unsafe_regions_.size()); ++i) {
          (*s_sol)[i].resize(s[i].rows());
          for (int j = 0; j < (*s_sol)[i].rows(); ++j) {
            (*s_sol)[i](j) = result_barrier.GetSolution(s[i](j));
            if (search_options.hsol_tiny_coeff_tol > 0) {
              (*s_sol)[i](j) = (*s_sol)[i](j).RemoveTermsWithSmallCoefficients(
                  search_options.hsol_tiny_coeff_tol);
            }
          }
        }
      } else {
        drake::log()->error("Failed to find the barrier.");
        return;
      }
    }
    iter_count++;
  }
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
