#include "drake/systems/analysis/control_lyapunov.h"

#include <limits.h>

#include "drake/common/text_logging.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/sos_basis_generator.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

namespace {
symbolic::Polynomial NewFreePolynomialNoConstant(
    solvers::MathematicalProgram* prog,
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name, symbolic::internal::DegreeType degree_type) {
  const VectorX<symbolic::Monomial> m =
      internal::ComputeMonomialBasisNoConstant(indeterminates, degree,
                                               degree_type);
  const VectorX<symbolic::Variable> coeffs =
      prog->NewContinuousVariables(m.size(), coeff_name);
  symbolic::Polynomial::MapType poly_map;
  for (int i = 0; i < coeffs.rows(); ++i) {
    poly_map.emplace(m(i), coeffs(i));
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
        fmt::format("rows: {}, min eigen: {}, cond number: {}\n",
                    psd_sol.rows(), es.eigenvalues().minCoeff(),
                    es.eigenvalues().maxCoeff() / es.eigenvalues().minCoeff()));
  }
}
}  // namespace

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
  const VectorX<symbolic::Monomial> monomial_with_1 =
      solvers::ConstructMonomialBasis(p);
  // Remove 1 from monomial_with_1.
  int monomial_size = 0;
  monomials->resize(monomial_with_1.rows());
  for (int i = 0; i < monomial_with_1.rows(); ++i) {
    if (monomial_with_1(i).total_degree() > 0) {
      (*monomials)(monomial_size) = monomial_with_1(i);
      monomial_size++;
    }
  }
  monomials->conservativeResize(monomial_size);
  *gram = prog->AddSosConstraint(p, *monomials);
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
// (1+t(x))((x−x*)ᵀS(x−x*)−ρ) − s(x)(V(x)−1) is sos
template <typename RhoType>
std::pair<MatrixX<symbolic::Variable>, VectorX<symbolic::Monomial>>
AddEllipsoidInRoaConstraintHelper(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& t,
    const VectorX<symbolic::Variable>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho,
    const symbolic::Polynomial& s, const symbolic::Polynomial& V) {
  const symbolic::Polynomial ellipsoid_poly =
      internal::EllipsoidPolynomial<RhoType>(x, x_star, S, rho);
  const symbolic::Polynomial sos_poly = (1 + t) * ellipsoid_poly - s * (V - 1);
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

SearchLagrangianAndBGivenVBoxInputBound::
    SearchLagrangianAndBGivenVBoxInputBound(
        symbolic::Polynomial V, VectorX<symbolic::Polynomial> f,
        MatrixX<symbolic::Polynomial> G, bool symmetric_dynamics,
        const std::vector<std::vector<symbolic::Polynomial>>& l_given,
        std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees,
        std::vector<int> b_degrees, VectorX<symbolic::Variable> x)
    : prog_{},
      V_{std::move(V)},
      f_{std::move(f)},
      G_{std::move(G)},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      symmetric_dynamics_{symmetric_dynamics},
      l_{static_cast<size_t>(nu_)},
      lagrangian_degrees_{std::move(lagrangian_degrees)},
      lagrangian_grams_{static_cast<size_t>(nu_)},
      b_degrees_{std::move(b_degrees)},
      x_{std::move(x)},
      x_set_{x_},
      b_{nu_},
      vdot_sos_constraint_{nu_, symmetric_dynamics_} {
  prog_.AddIndeterminates(x_);
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(static_cast<int>(b_degrees_.size()) == nu_);
  if (symmetric_dynamics_ && !V_.IsEven()) {
    throw std::runtime_error(
        "For symmetric dynamics, V should be an even function.");
  }

  const int num_vdot_sos = symmetric_dynamics_ ? 1 : 2;
  // Add Lagrangian decision variables.
  for (int i = 0; i < nu_; ++i) {
    l_[i].resize(num_vdot_sos);
    lagrangian_grams_[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      // k == 0
      DRAKE_DEMAND(l_given[i][j].TotalDegree() == lagrangian_degrees_[i][j][0]);
      l_[i][j][0] = l_given[i][j];

      // k == 1
      DRAKE_DEMAND(lagrangian_degrees_[i][j][1] % 2 == 0);
      std::tie(l_[i][j][1], lagrangian_grams_[i][j][1]) =
          prog_.NewSosPolynomial(x_set_, lagrangian_degrees_[i][j][1]);

      // k == 2, l_[i][j][2] doesn't have 1 in its monomial basis.
      DRAKE_DEMAND(lagrangian_degrees_[i][j][2] % 2 == 0);
      const VectorX<symbolic::Monomial> l_monomial_basis =
          internal::ComputeMonomialBasisNoConstant(
              x_set_, lagrangian_degrees_[i][j][2] / 2,
              symbolic::internal::DegreeType::kAny);
      std::tie(l_[i][j][2], lagrangian_grams_[i][j][2]) =
          prog_.NewSosPolynomial(l_monomial_basis);
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
  const symbolic::internal::DegreeType b_degree_type =
      symmetric_dynamics_ ? symbolic::internal::DegreeType::kEven
                          : symbolic::internal::DegreeType::kAny;
  for (int i = 0; i < nu_; ++i) {
    b_(i) = NewFreePolynomialNoConstant(&prog_, x_set_, b_degrees_[i],
                                        "b" + std::to_string(i), b_degree_type);
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
      *monomial_basis = internal::ComputeMonomialBasisNoConstant(
          indeterminates, degree / 2, symbolic::internal::DegreeType::kAny);
      MatrixX<symbolic::Variable> gram_var;
      std::tie(*p, gram_var) = prog->NewSosPolynomial(*monomial_basis);
      *gram = gram_var.cast<symbolic::Expression>();
      break;
    }
    case symbolic::internal::DegreeType::kEven: {
      symbolic::Polynomial p_even, p_odd;
      const VectorX<symbolic::Monomial> monomial_basis_even =
          internal::ComputeMonomialBasisNoConstant(
              indeterminates, degree / 2,
              symbolic::internal::DegreeType::kEven);
      const VectorX<symbolic::Monomial> monomial_basis_odd =
          internal::ComputeMonomialBasisNoConstant(
              indeterminates, degree / 2, symbolic::internal::DegreeType::kOdd);
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
}  // namespace

SearchLyapunovGivenLagrangianBoxInputBound::
    SearchLyapunovGivenLagrangianBoxInputBound(
        VectorX<symbolic::Polynomial> f, MatrixX<symbolic::Polynomial> G,
        bool symmetric_dynamics, int V_degree, double positivity_eps,
        double deriv_eps,
        std::vector<std::vector<std::array<symbolic::Polynomial, 3>>> l_given,
        const std::vector<int>& b_degrees, VectorX<symbolic::Variable> x)
    : prog_{},
      f_{std::move(f)},
      G_{std::move(G)},
      symmetric_dynamics_{symmetric_dynamics},
      deriv_eps_{deriv_eps},
      l_{std::move(l_given)},
      x_{std::move(x)},
      x_set_{x_},
      nx_{static_cast<int>(f_.rows())},
      nu_{static_cast<int>(G_.cols())},
      b_{nu_},
      vdot_sos_constraint_{nu_, symmetric_dynamics_} {
  prog_.AddIndeterminates(x_);
  CheckDynamicsInput(V_, f_, G_, x_set_);
  DRAKE_DEMAND(static_cast<int>(b_degrees.size()) == nu_);
  DRAKE_DEMAND(positivity_eps >= 0);
  DRAKE_DEMAND(V_degree >= 0 && V_degree % 2 == 0);
  // V(x) >= ε₁xᵀx
  if (positivity_eps == 0 && !symmetric_dynamics) {
    const VectorX<symbolic::Monomial> V_monomial =
        internal::ComputeMonomialBasisNoConstant(
            x_set_, V_degree / 2, symbolic::internal::DegreeType::kAny);
    std::tie(V_, positivity_constraint_gram_) =
        prog_.NewSosPolynomial(V_monomial);
    positivity_constraint_monomial_ = V_monomial;
  } else {
    const symbolic::internal::DegreeType V_degree_type =
        symmetric_dynamics_ ? symbolic::internal::DegreeType::kEven
                            : symbolic::internal::DegreeType::kAny;
    // We know that V(0) = 0 and V(x) >= ε₁xᵀx, hence V(x) has no constant or
    // linear terms.
    V_ = internal::NewFreePolynomialNoConstantOrLinear(
        &prog_, x_set_, V_degree, "V_coeff", V_degree_type);
    // quadratic_poly_map stores the mapping for the polynomial
    // ε₁xᵀx
    symbolic::Polynomial::MapType quadratic_poly_map;
    for (int i = 0; i < nx_; ++i) {
      quadratic_poly_map.emplace(symbolic::Monomial(x_(i), 2), positivity_eps);
    }
    symbolic::Polynomial V_minus_eps;
    NewSosPolynomialPassOrigin(&prog_, x_set_, V_degree, V_degree_type,
                               &V_minus_eps, &positivity_constraint_monomial_,
                               &positivity_constraint_gram_);
    prog_.AddEqualityConstraintBetweenPolynomials(
        V_ - symbolic::Polynomial(quadratic_poly_map), V_minus_eps);
  }

  // ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  const symbolic::internal::DegreeType b_degree_type =
      symmetric_dynamics_ ? symbolic::internal::DegreeType::kEven
                          : symbolic::internal::DegreeType::kAny;
  for (int i = 0; i < nu_; ++i) {
    b_(i) = NewFreePolynomialNoConstant(&prog_, x_set_, b_degrees[i], "b_coeff",
                                        b_degree_type);
  }
  const RowVectorX<symbolic::Polynomial> dVdx = V_.Jacobian(x_);
  prog_.AddEqualityConstraintBetweenPolynomials(
      (dVdx * f_)(0) + deriv_eps_ * V_, b_.sum());
  // Add the constraint
  // (lᵢ₀₀(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₀₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₀₂(x)*(1 − V)>=0
  //(lᵢ₁₀(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₁₁(x)*∂V/∂x*Gᵢ(x) - lᵢ₁₂(x)*(1 − V)>=0
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
          internal::ComputeMonomialBasisNoConstant(
              x_set_, lagrangian_degrees_[i][j][2] / 2,
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

namespace {
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
  }
  // PrintPsdConstraintStat(*prog, result);
  DRAKE_DEMAND(result.is_success());
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
    }
    // PrintPsdConstraintStat(*prog, result);
    DRAKE_DEMAND(result.is_success());
  }
  return result;
}

}  // namespace

ControlLyapunovBoxInputBound::ControlLyapunovBoxInputBound(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    double positivity_eps)
    : f_{f},
      G_{G},
      symmetric_dynamics_{internal::IsDynamicsSymmetric(f_, G_)},
      x_{x},
      positivity_eps_{positivity_eps} {}

void ControlLyapunovBoxInputBound::SearchLagrangianAndB(
    const symbolic::Polynomial& V,
    const std::vector<std::vector<symbolic::Polynomial>>& l_given,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const std::vector<int>& b_degrees, double deriv_eps_lower,
    double deriv_eps_upper, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double* deriv_eps, VectorX<symbolic::Polynomial>* b,
    std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>* l) const {
  // Check if V(0) = 0
  const auto it_V_constant =
      V.monomial_to_coefficient_map().find(symbolic::Monomial());
  if (it_V_constant != V.monomial_to_coefficient_map().end()) {
    DRAKE_DEMAND(symbolic::is_constant(it_V_constant->second) &&
                 symbolic::get_constant_value(it_V_constant->second) == 0.);
  }
  SearchLagrangianAndBGivenVBoxInputBound searcher(
      V, f_, G_, symmetric_dynamics_, l_given, lagrangian_degrees, b_degrees,
      x_);
  searcher.get_mutable_prog()->AddBoundingBoxConstraint(
      deriv_eps_lower, deriv_eps_upper, searcher.deriv_eps());
  auto solver = solvers::MakeSolver(solver_id);
  solvers::MathematicalProgramResult result_searcher;
  solver->Solve(searcher.prog(), std::nullopt, solver_options,
                &result_searcher);
  if (!result_searcher.is_success()) {
    log()->error("Failed to find Lagrangian and b(x)");
  }
  // PrintPsdConstraintStat(searcher.prog(), result_searcher);
  DRAKE_DEMAND(result_searcher.is_success());
  *deriv_eps = result_searcher.GetSolution(searcher.deriv_eps());
  const int nu = G_.cols();
  b->resize(nu);
  l->resize(nu);
  const int num_vdot_sos = symmetric_dynamics_ ? 1 : 2;
  for (int i = 0; i < nu; ++i) {
    (*b)(i) = result_searcher.GetSolution(searcher.b()(i));
    (*l)[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      (*l)[i][j][0] = l_given[i][j];
      for (int k = 1; k < 3; ++k) {
        (*l)[i][j][k] =
            result_searcher.GetSolution(searcher.lagrangians()[i][j][k]);
      }
    }
  }
}

void ControlLyapunovBoxInputBound::SearchLyapunov(
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const std::vector<int>& b_degrees, int V_degree, double positivity_eps,
    double deriv_eps, const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& s,
    const symbolic::Polynomial& t, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, symbolic::Polynomial* V,
    VectorX<symbolic::Polynomial>* b, double* rho) const {
  SearchLyapunovGivenLagrangianBoxInputBound searcher(
      f_, G_, symmetric_dynamics_, V_degree, positivity_eps, deriv_eps, l,
      b_degrees, x_);
  const auto ellipsoid_ret =
      searcher.AddEllipsoidInRoaConstraint(x_star, S, t, s);
  searcher.get_mutable_prog()->AddLinearCost(-ellipsoid_ret.rho);
  const auto result = SearchWithBackoff(searcher.get_mutable_prog(), solver_id,
                                        solver_options, backoff_scale);
  *rho = result.GetSolution(ellipsoid_ret.rho);
  *V = result.GetSolution(searcher.V());
  const int nu = G_.cols();
  b->resize(nu);
  for (int i = 0; i < nu; ++i) {
    (*b)(i) = result.GetSolution(searcher.b()(i));
  }
}

void ControlLyapunovBoxInputBound::SearchLyapunov(
    const std::vector<std::vector<std::array<symbolic::Polynomial, 3>>>& l,
    const std::vector<int>& b_degrees, int V_degree, double positivity_eps,
    double deriv_eps, const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, double rho, int r_degree,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, symbolic::Polynomial* V,
    VectorX<symbolic::Polynomial>* b, symbolic::Polynomial* r_sol,
    double* d_sol) const {
  SearchLyapunovGivenLagrangianBoxInputBound searcher(
      f_, G_, symmetric_dynamics_, V_degree, positivity_eps, deriv_eps, l,
      b_degrees, x_);
  // Now add the constraint that the ellipsoid is within the sub-level set {x |
  // V(x)<=d} Namely d−V(x) − r(x)(ρ−(x−x*)ᵀS(x−x*)) is sos and r(x) is sos.
  const symbolic::Variables x_set(x_);
  symbolic::Polynomial r;
  std::tie(r, std::ignore) =
      searcher.get_mutable_prog()->NewSosPolynomial(x_set, r_degree);
  const symbolic::Polynomial ellipsoid_poly =
      internal::EllipsoidPolynomial(x_, x_star, S, rho);
  const symbolic::Variable d_var =
      searcher.get_mutable_prog()->NewContinuousVariables<1>("d")(0);
  // 0 <= d <= 1
  searcher.get_mutable_prog()->AddBoundingBoxConstraint(0, 1, d_var);
  searcher.get_mutable_prog()->AddSosConstraint(
      symbolic::Polynomial({{symbolic::Monomial(), d_var}}) - searcher.V() +
      r * ellipsoid_poly);
  // The objective is to minimize d.
  searcher.get_mutable_prog()->AddLinearCost(
      Vector1d::Ones(), 0, Vector1<symbolic::Variable>(d_var));
  const auto result = SearchWithBackoff(searcher.get_mutable_prog(), solver_id,
                                        solver_options, backoff_scale);
  *V = result.GetSolution(searcher.V());
  b->resize(searcher.b().rows());
  for (int i = 0; i < searcher.b().rows(); ++i) {
    (*b)(i) = result.GetSolution(searcher.b()(i));
  }
  *r_sol = result.GetSolution(r);
  *d_sol = result.GetSolution(d_var);
}

void ControlLyapunovBoxInputBound::SearchLagrangian(
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& b,
    const std::vector<std::vector<std::array<int, 3>>>& lagrangian_degrees,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
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
  SearchLagrangianAndB(V_init, l_given, lagrangian_degrees, b_degrees,
                       deriv_eps_lower, deriv_eps_upper,
                       options.lagrangian_step_solver,
                       options.lagrangian_step_solver_options, &(ret.deriv_eps),
                       &(ret.b), &(ret.l));

  // Solve a separate program to find the inscribed ellipsoid.
  MaximizeInnerEllipsoidRho(
      x_, x_star, S, V_init, t_given, s_degree, options.lagrangian_step_solver,
      options.lagrangian_step_solver_options, options.backoff_scale, &(ret.rho),
      &(ret.ellipsoid_lagrangian));

  int iter = 0;
  bool converged = false;
  while (iter < options.bilinear_iterations && !converged) {
    double rho_new;
    drake::log()->info("search Lyapunov");
    SearchLyapunov(ret.l, b_degrees, V_degree, positivity_eps_, ret.deriv_eps,
                   x_star, S, ret.ellipsoid_lagrangian, t_given,
                   options.lyap_step_solver, options.lyap_step_solver_options,
                   options.backoff_scale, &(ret.V), &(ret.b), &rho_new);
    drake::log()->info("search Lagrangian");
    SearchLagrangian(ret.V, ret.b, lagrangian_degrees,
                     options.lagrangian_step_solver,
                     options.lagrangian_step_solver_options, &(ret.l));
    // Solve a separate program to find the innner ellipsoid.
    MaximizeInnerEllipsoidRho(
        x_, x_star, S, ret.V, t_given, s_degree, options.lagrangian_step_solver,
        options.lagrangian_step_solver_options, options.backoff_scale, &rho_new,
        &(ret.ellipsoid_lagrangian));
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
  SearchLagrangianAndB(V_init, l_given, lagrangian_degrees, b_degrees,
                       deriv_eps_lower, deriv_eps_upper,
                       options.lagrangian_step_solver,
                       options.lagrangian_step_solver_options, &(ret.deriv_eps),
                       &(ret.b), &(ret.l));
  // Solve a separate program to find the inscribed ellipsoid.
  MaximizeInnerEllipsoidRho(
      x_, x_star, S, V_init, r_degree, rho_bisection_option.rho_max,
      rho_bisection_option.rho_min, options.lagrangian_step_solver,
      options.lagrangian_step_solver_options, rho_bisection_option.rho_tol,
      &(ret.rho), &(ret.ellipsoid_lagrangian));

  int iter = 0;
  bool converged = false;
  while (iter < options.bilinear_iterations && !converged) {
    double d;
    drake::log()->info("search Lyapunov");
    SearchLyapunov(ret.l, b_degrees, V_degree, positivity_eps_, ret.deriv_eps,
                   x_star, S, ret.rho, r_degree, options.lyap_step_solver,
                   options.lyap_step_solver_options, options.backoff_scale,
                   &(ret.V), &(ret.b), &(ret.ellipsoid_lagrangian), &d);
    drake::log()->info("d={}", d);
    drake::log()->info("search Lagrangian");
    SearchLagrangian(ret.V, ret.b, lagrangian_degrees,
                     options.lagrangian_step_solver,
                     options.lagrangian_step_solver_options, &(ret.l));
    // Solve a separate program to find the inner ellipsoid.
    double rho_new;
    MaximizeInnerEllipsoidRho(
        x_, x_star, S, ret.V, r_degree, rho_bisection_option.rho_max,
        rho_bisection_option.rho_min, options.lagrangian_step_solver,
        options.lagrangian_step_solver_options, rho_bisection_option.rho_tol,
        &rho_new, &(ret.ellipsoid_lagrangian));
    drake::log()->info("iter: {}, rho: {}", iter, rho_new);
    if (rho_new - ret.rho < options.rho_converge_tol) {
      converged = true;
    }
    ret.rho = rho_new;
    iter += 1;
  }
  return ret;
}

void MaximizeInnerEllipsoidRho(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& V,
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

  AddEllipsoidInRoaConstraintHelper(&prog, t, x, x_star, S, rho, s, V);
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
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& V,
    int r_degree, double rho_max, double rho_min,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options, double rho_tol,
    double* rho_sol, symbolic::Polynomial* r_sol) {
  DRAKE_DEMAND(rho_max > rho_min);
  DRAKE_DEMAND(rho_tol > 0);
  const symbolic::Polynomial ellipsoid_quadratic =
      internal::EllipsoidPolynomial(x, x_star, S, 0.);
  auto is_feasible = [&x, &V, &r_degree, &solver_id, &solver_options,
                      &ellipsoid_quadratic, r_sol](double rho) {
    solvers::MathematicalProgram prog;
    prog.AddIndeterminates(x);
    symbolic::Polynomial r;
    std::tie(r, std::ignore) =
        prog.NewSosPolynomial(symbolic::Variables(x), r_degree);
    prog.AddSosConstraint(1 - V - r * (rho - ellipsoid_quadratic));
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
