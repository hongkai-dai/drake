#include "drake/systems/analysis/control_lyapunov.h"

#include <limits.h>

#include "drake/solvers/choose_best_solver.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

SearchControlLyapunov::SearchControlLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const Eigen::VectorXd>& x_equilibrium,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    std::map<int, std::set<int>> neighbouring_vertices,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x)
    : f_{f},
      G_{G},
      x_equilibrium_{x_equilibrium},
      u_vertices_{u_vertices},
      neighbouring_vertices_{std::move(neighbouring_vertices)},
      x_{x} {
  DRAKE_ASSERT(f.rows() == G.rows());
  const symbolic::Variables x_set(x_);
  for (int i = 0; i < f.rows(); ++i) {
    DRAKE_DEMAND(f(i).indeterminates().IsSubsetOf(x_set));
    for (int j = 0; j < G.cols(); ++j) {
      DRAKE_DEMAND(G(i, j).indeterminates().IsSubsetOf(x_set));
    }
  }
  const int num_x = f.rows();
  const int num_u = G.cols();
  DRAKE_DEMAND(u_vertices.rows() == num_u);
  DRAKE_DEMAND(x_equilibrium.rows() == num_x);
}

VdotCalculator::VdotCalculator(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Polynomial& V,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G) {
  const RowVectorX<symbolic::Polynomial> dVdx = V.Jacobian(x);
  dVdx_times_f_ = (dVdx * f)(0);
  dVdx_times_G_ = dVdx * G;
}

symbolic::Polynomial VdotCalculator::Calc(
    const Eigen::Ref<const Eigen::VectorXd>& u) const {
  return dVdx_times_f_ + (dVdx_times_G_ * u)(0);
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
// Add the constraint
// (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
// (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
void AddControlLyapunovBoxInputBoundConstraints(
    solvers::MathematicalProgram* prog,
    const std::vector<std::array<symbolic::Polynomial, 6>>& l,
    const symbolic::Polynomial& V, const RowVectorX<symbolic::Polynomial>& dVdx,
    const VectorX<symbolic::Polynomial>& b,
    const MatrixX<symbolic::Polynomial>& G,
    VdotSosConstraintReturn* vdot_sos_constraint) {
  const int nu = G.cols();
  for (int i = 0; i < nu; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = (dVdx * G.col(i))(0);
    const symbolic::Polynomial p1 = (l[i][0] + 1) * (dVdx_times_Gi - b(i)) -
                                    l[i][2] * dVdx_times_Gi - l[i][4] * (1 - V);
    const symbolic::Polynomial p2 = (l[i][1] + 1) * (-dVdx_times_Gi - b(i)) +
                                    l[i][3] * dVdx_times_Gi - l[i][5] * (1 - V);
    std::tie(vdot_sos_constraint->grams[i][0],
             vdot_sos_constraint->monomials[i][0]) = prog->AddSosConstraint(p1);
    std::tie(vdot_sos_constraint->grams[i][1],
             vdot_sos_constraint->monomials[i][1]) = prog->AddSosConstraint(p2);
  }
}

// Add the constraint
// (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1) is sos
template <typename RhoType>

std::pair<MatrixX<symbolic::Variable>, VectorX<symbolic::Monomial>>
AddEllipsoidInRoaConstraintHelper(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& t,
    const VectorX<symbolic::Variable>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho,
    const symbolic::Polynomial& s, const symbolic::Polynomial& V) {
  // The ellipsoid polynomial (x−x*)ᵀS(x−x*)−ρ
  symbolic::Polynomial::MapType ellipsoid_poly_map;
  // Add constant term x*ᵀ*x* - ρ
  ellipsoid_poly_map.emplace(symbolic::Monomial(), x_star.dot(x_star) - rho);
  const Eigen::VectorXd S_times_x_star = S * x_star;
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
  const symbolic::Polynomial ellipsoid_poly{ellipsoid_poly_map};
  const symbolic::Polynomial sos_poly = (1 + t) * ellipsoid_poly - s * (V - 1);
  return prog->AddSosConstraint(sos_poly);
}
}  // namespace

std::array<symbolic::Polynomial, 2>
VdotSosConstraintReturn::ComputeSosConstraint(
    int i, const solvers::MathematicalProgramResult& result) const {
  std::array<symbolic::Polynomial, 2> ret;
  for (int j = 0; j < 2; ++j) {
    const Eigen::MatrixXd gram_sol = result.GetSolution(grams[i][j]);
    ret[j] = monomials[i][j].dot(gram_sol * monomials[i][j]);
  }
  return ret;
}

SearchLagrangianAndBGivenVBoxInputBound::
    SearchLagrangianAndBGivenVBoxInputBound(
        symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
        MatrixX<symbolic::Polynomial> G,
        const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
        std::vector<std::array<int, 6>> lagrangian_degrees,
        std::vector<int> b_degrees, VectorX<symbolic::Variable> x)
    : prog_{},
      V_{std::move(V)},
      f_{std::move(f)},
      G_{std::move(G)},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      l_{static_cast<size_t>(nu_)},
      lagrangian_degrees_{std::move(lagrangian_degrees)},
      lagrangian_grams_{static_cast<size_t>(nu_)},
      b_degrees_{std::move(b_degrees)},
      x_{std::move(x)},
      x_set_{x_},
      b_{nu_},
      vdot_sos_constraint_{nu_} {
  prog_.AddIndeterminates(x_);
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(static_cast<int>(b_degrees_.size()) == nu_);

  // Add Lagrangian decision variables.
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(l_given[i][0].TotalDegree() == lagrangian_degrees_[i][0]);
    DRAKE_DEMAND(l_given[i][1].TotalDegree() == lagrangian_degrees_[i][1]);
    l_[i][0] = l_given[i][0];
    l_[i][1] = l_given[i][1];
    for (int j = 2; j < 6; ++j) {
      std::tie(l_[i][j], lagrangian_grams_[i][j]) =
          prog_.NewSosPolynomial(x_set_, lagrangian_degrees_[i][j]);
    }
  }

  deriv_eps_ = prog_.NewContinuousVariables<1>("deriv_eps")(0);

  const RowVectorX<symbolic::Polynomial> dVdx = V_.Jacobian(x_);
  // Since we will add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x), we know that
  // the highest degree of b should be at least degree(∂V/∂x*f(x) + εV).
  const symbolic::Polynomial dVdx_times_f = (dVdx * f_)(0);
  if (*std::max_element(b_degrees_.begin(), b_degrees_.end()) <
      std::max(dVdx_times_f.TotalDegree(), V_.TotalDegree())) {
    throw std::invalid_argument("The degree of b is too low.");
  }

  // Add free polynomial b
  for (int i = 0; i < nu_; ++i) {
    b_(i) =
        prog_.NewFreePolynomial(x_set_, b_degrees_[i], "b" + std::to_string(i));
  }

  // Add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  prog_.AddEqualityConstraintBetweenPolynomials(b_.sum(),
                                                dVdx_times_f + deriv_eps_ * V_);
  // Add the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  AddControlLyapunovBoxInputBoundConstraints(&prog_, l_, V_, dVdx, b_, G_,
                                             &vdot_sos_constraint_);
}

SearchLagrangianAndBGivenVBoxInputBound::EllipsoidInRoaReturn
SearchLagrangianAndBGivenVBoxInputBound::AddEllipsoidInRoaConstraint(
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t) {
  EllipsoidInRoaReturn ret;
  ret.rho = prog_.NewContinuousVariables<1>("rho")(0);
  std::tie(ret.s, ret.s_gram) = prog_.NewSosPolynomial(x_set_, s_degree);
  std::tie(ret.constraint_gram, ret.constraint_monomials) =
      AddEllipsoidInRoaConstraintHelper<symbolic::Variable>(
          &prog_, t, x_, x_star, S, ret.rho, ret.s, V_);
  return ret;
}

SearchLyapunovGivenLagrangianBoxInputBound::
    SearchLyapunovGivenLagrangianBoxInputBound(
        VectorX<symbolic::Polynomial> f, MatrixX<symbolic::Polynomial> G,
        int V_degree, double positivity_eps, double deriv_eps,
        const Eigen::Ref<const Eigen::VectorXd>& x_des,
        std::vector<std::array<symbolic::Polynomial, 6>> l_given,
        const std::vector<int>& b_degrees, VectorX<symbolic::Variable> x)
    : prog_{},
      f_{std::move(f)},
      G_{std::move(G)},
      deriv_eps_{deriv_eps},
      l_{std::move(l_given)},
      x_{std::move(x)},
      x_set_{x_},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      b_{nu_},
      vdot_sos_constraint_{nu_} {
  prog_.AddIndeterminates(x_);
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(static_cast<int>(b_degrees.size()) == nu_);
  DRAKE_DEMAND(positivity_eps >= 0);
  // V(x) >= ε₁(x-x_des)ᵀ(x-x_des)
  if (positivity_eps == 0) {
    std::tie(V_, positivity_constraint_gram_) =
        prog_.NewSosPolynomial(x_set_, V_degree);
  } else {
    V_ = prog_.NewFreePolynomial(x_set_, V_degree);
    // quadratic_poly_map stores the mapping for the polynomial
    // ε₁(x-x_des)ᵀ(x-x_des)
    symbolic::Polynomial::MapType quadratic_poly_map;
    quadratic_poly_map.emplace(symbolic::Monomial(),
                               positivity_eps * x_des.dot(x_des));
    for (int i = 0; i < nx_; ++i) {
      quadratic_poly_map.emplace(symbolic::Monomial(x_(i), 2), positivity_eps);
      quadratic_poly_map.emplace(symbolic::Monomial(x_(i)),
                                 -2 * x_des(i) * positivity_eps);
    }
    std::tie(positivity_constraint_gram_, std::ignore) =
        prog_.AddSosConstraint(V_ - symbolic::Polynomial(quadratic_poly_map));
  }
  // Add the constraint V(x_des) = 0
  symbolic::Environment env_x_des;
  for (int i = 0; i < nx_; ++i) {
    env_x_des.insert(x_(i), x_des(i));
  }
  prog_.AddLinearEqualityConstraint(
      V_.ToExpression().EvaluatePartial(env_x_des), 0.);

  // ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  for (int i = 0; i < nu_; ++i) {
    b_(i) = prog_.NewFreePolynomial(x_set_, b_degrees[i]);
  }
  const RowVectorX<symbolic::Polynomial> dVdx = V_.Jacobian(x_);
  prog_.AddEqualityConstraintBetweenPolynomials(
      (dVdx * f_)(0) + deriv_eps_ * V_, b_.sum());
  // Add the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  AddControlLyapunovBoxInputBoundConstraints(&prog_, l_, V_, dVdx, b_, G_,
                                             &vdot_sos_constraint_);
}

SearchLyapunovGivenLagrangianBoxInputBound::EllipsoidInRoaReturn
SearchLyapunovGivenLagrangianBoxInputBound::AddEllipsoidInRoaConstraint(
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& t,
    const symbolic::Polynomial& s) {
  EllipsoidInRoaReturn ret;
  ret.rho = prog_.NewContinuousVariables<1>("rho")(0);
  std::tie(ret.constraint_gram, ret.constraint_monomials) =
      AddEllipsoidInRoaConstraintHelper<symbolic::Variable>(
          &prog_, t, x_, x_star, S, ret.rho, s, V_);
  return ret;
}

SearchLagrangianGivenVBoxInputBound::SearchLagrangianGivenVBoxInputBound(
    symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
    MatrixX<symbolic::Polynomial> G, VectorX<symbolic::Polynomial> b,
    VectorX<symbolic::Variable> x,
    std::vector<std::array<int, 6>> lagrangian_degrees)
    : V_{std::move(V)},
      f_{std::move(f)},
      G_{std::move(G)},
      b_{std::move(b)},
      x_{std::move(x)},
      x_set_{x_},
      prog_{},
      nu_{static_cast<int>(G_.cols())},
      nx_{static_cast<int>(f.rows())},
      l_{static_cast<size_t>(nu_)},
      lagrangian_degrees_{std::move(lagrangian_degrees)},
      lagrangian_grams_{static_cast<size_t>(nu_)},
      vdot_sos_constraint_{nu_} {
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(b_.rows() == nu_);
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(b_(i).indeterminates().IsSubsetOf(x_set_));
  }
  prog_.AddIndeterminates(x_);
  for (int i = 0; i < nu_; ++i) {
    for (int j = 0; j < 6; ++j) {
      std::tie(l_[i][j], lagrangian_grams_[i][j]) =
          prog_.NewSosPolynomial(x_set_, lagrangian_degrees_[i][j]);
    }
  }

  const auto dVdx = V_.Jacobian(x_);
  // Now impose the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  AddControlLyapunovBoxInputBoundConstraints(&prog_, l_, V_, dVdx, b_, G_,
                                             &vdot_sos_constraint_);
}

SearchLagrangianGivenVBoxInputBound::EllipsoidInRoaReturn
SearchLagrangianGivenVBoxInputBound::AddEllipsoidInRoaConstraint(
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t) {
  EllipsoidInRoaReturn ret;
  ret.rho = prog_.NewContinuousVariables<1>("rho")(0);
  std::tie(ret.s, ret.s_gram) = prog_.NewSosPolynomial(x_set_, s_degree);
  std::tie(ret.constraint_gram, ret.constraint_monomials) =
      AddEllipsoidInRoaConstraintHelper<symbolic::Variable>(
          &prog_, t, x_, x_star, S, ret.rho, ret.s, V_);
  return ret;
}

namespace {
solvers::MathematicalProgramResult SearchWithBackoff(
    solvers::MathematicalProgram* prog, const symbolic::Variable& rho,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale) {
  auto cost = prog->AddLinearCost(-rho);
  auto solver = solvers::MakeSolver(solver_id);
  solvers::MathematicalProgramResult result;
  solver->Solve(*prog, std::nullopt, solver_options, &result);
  DRAKE_DEMAND(result.is_success());
  DRAKE_DEMAND(backoff_scale >= 0 && backoff_scale <= 1);
  if (backoff_scale < 1) {
    std::cout << "backoff\n";
    prog->AddBoundingBoxConstraint(result.GetSolution(rho) * backoff_scale,
                                   kInf, rho);
    cost.evaluator()->UpdateCoefficients(Vector1d(0.));
    solver->Solve(*prog, std::nullopt, solver_options, &result);
    DRAKE_DEMAND(result.is_success());
  }
  return result;
}

}  // namespace

ControlLyapunovBoxInputBound::ControlLyapunovBoxInputBound(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const Eigen::VectorXd>& x_des,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    double positivity_eps)
    : f_{f}, G_{G}, x_des_{x_des}, x_{x}, positivity_eps_{positivity_eps} {}

void ControlLyapunovBoxInputBound::SearchLagrangianAndB(
    const symbolic::Polynomial& V,
    const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
    const std::vector<std::array<int, 6>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t, double deriv_eps_lower,
    double deriv_eps_upper, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double* deriv_eps, VectorX<symbolic::Polynomial>* b,
    std::vector<std::array<symbolic::Polynomial, 6>>* l, double* rho,
    symbolic::Polynomial* s) const {
  // Check if V(x_des) = 0
  symbolic::Environment env_x_des;
  for (int i = 0; i < G_.rows(); ++i) {
    env_x_des.insert(x_(i), x_des_(i));
  }
  DRAKE_DEMAND(std::abs(V.Evaluate(env_x_des)) < 1E-5);
  SearchLagrangianAndBGivenVBoxInputBound searcher(
      V, f_, G_, l_given, lagrangian_degrees, b_degrees, x_);
  const auto ellipsoid_ret =
      searcher.AddEllipsoidInRoaConstraint(x_star, S, s_degree, t);
  searcher.get_mutable_prog()->AddBoundingBoxConstraint(
      deriv_eps_lower, deriv_eps_upper, searcher.deriv_eps());
  const auto result =
      SearchWithBackoff(searcher.get_mutable_prog(), ellipsoid_ret.rho,
                        solver_id, solver_options, backoff_scale);
  *deriv_eps = result.GetSolution(searcher.deriv_eps());
  *rho = result.GetSolution(ellipsoid_ret.rho);
  const int nu = G_.cols();
  b->resize(nu);
  l->resize(nu);
  for (int i = 0; i < nu; ++i) {
    (*b)(i) = result.GetSolution(searcher.b()(i));
    for (int j = 0; j < 2; ++j) {
      (*l)[i][j] = l_given[i][j];
    }
    for (int j = 2; j < 6; ++j) {
      (*l)[i][j] = result.GetSolution(searcher.lagrangians()[i][j]);
    }
  }
  *s = result.GetSolution(ellipsoid_ret.s);
}

void ControlLyapunovBoxInputBound::SearchLyapunov(
    const std::vector<std::array<symbolic::Polynomial, 6>>& l,
    const std::vector<int>& b_degrees, int V_degree, double positivity_eps,
    double deriv_eps, const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& s,
    const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, symbolic::Polynomial* V,
    VectorX<symbolic::Polynomial>* b, double* rho) const {
  SearchLyapunovGivenLagrangianBoxInputBound searcher(
      f_, G_, V_degree, positivity_eps, deriv_eps, x_des_, l, b_degrees, x_);
  const auto ellipsoid_ret =
      searcher.AddEllipsoidInRoaConstraint(x_star, S, t, s);
  const auto result =
      SearchWithBackoff(searcher.get_mutable_prog(), ellipsoid_ret.rho,
                        solver_id, solver_options, backoff_scale);
  *rho = result.GetSolution(ellipsoid_ret.rho);
  *V = result.GetSolution(searcher.V());
  const int nu = G_.cols();
  b->resize(nu);
  for (int i = 0; i < nu; ++i) {
    (*b)(i) = result.GetSolution(searcher.b()(i));
  }
}

void ControlLyapunovBoxInputBound::SearchLagrangian(
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& b,
    const std::vector<std::array<int, 6>>& lagrangian_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, std::vector<std::array<symbolic::Polynomial, 6>>* l,
    symbolic::Polynomial* s, double* rho) const {
  SearchLagrangianGivenVBoxInputBound searcher(V, f_, G_, b, x_,
                                               lagrangian_degrees);
  const auto ellipsoid_ret =
      searcher.AddEllipsoidInRoaConstraint(x_star, S, s_degree, t);
  const auto result =
      SearchWithBackoff(searcher.get_mutable_prog(), ellipsoid_ret.rho,
                        solver_id, solver_options, backoff_scale);
  const int nu = G_.cols();
  l->resize(nu);
  for (int i = 0; i < nu; ++i) {
    for (int j = 0; j < 6; ++j) {
      (*l)[i][j] = result.GetSolution(searcher.lagrangians()[i][j]);
    }
  }
  *s = result.GetSolution(ellipsoid_ret.s);
  *rho = result.GetSolution(ellipsoid_ret.rho);
}

ControlLyapunovBoxInputBound::SearchReturn ControlLyapunovBoxInputBound::Search(
    const symbolic::Polynomial& V_init,
    const std::vector<std::array<symbolic::Polynomial, 2>>& l_given,
    const std::vector<std::array<int, 6>>& lagrangian_degrees,
    const std::vector<int>& b_degrees,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, int s_degree,
    const symbolic::Polynomial& t_given, int V_degree, double deriv_eps_lower,
    double deriv_eps_upper, const SearchOptions& options) const {
  // First search for b and lagrangians.
  SearchReturn ret;
  SearchLagrangianAndB(
      V_init, l_given, lagrangian_degrees, b_degrees, x_star, S, s_degree,
      t_given, deriv_eps_lower, deriv_eps_upper, options.lagrangian_step_solver,
      options.lagrangian_step_solver_options, options.backoff_scale,
      &(ret.deriv_eps), &(ret.b), &(ret.l), &(ret.rho), &(ret.s));

  int iter = 0;
  bool converged = false;
  while (iter < options.bilinear_iterations && !converged) {
    double rho_new;
    std::cout << "search Lyapunov\n";
    SearchLyapunov(ret.l, b_degrees, V_degree, positivity_eps_, ret.deriv_eps,
                   x_star, S, ret.s, t_given, options.lyap_step_solver,
                   options.lyap_step_solver_options, options.backoff_scale,
                   &(ret.V), &(ret.b), &rho_new);
    std::cout << "search Lagrangian\n";
    SearchLagrangian(ret.V, ret.b, lagrangian_degrees, x_star, S, s_degree,
                     t_given, options.lagrangian_step_solver,
                     options.lagrangian_step_solver_options,
                     options.backoff_scale, &(ret.l), &(ret.s), &rho_new);
    std::cout << "iter: " << iter << ", rho: " << rho_new << "\n";
    if (rho_new - ret.rho < options.rho_converge_tol) {
      converged = true;
    }
    ret.rho = rho_new;
    iter += 1;
  }
  return ret;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
