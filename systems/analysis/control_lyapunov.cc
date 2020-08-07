#include "drake/systems/analysis/control_lyapunov.h"

namespace drake {
namespace systems {
namespace analysis {
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
MaximizeEpsGivenVBoxInputBound::MaximizeEpsGivenVBoxInputBound(
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
      b_degrees_{std::move(b_degrees)},
      x_{std::move(x)},
      b_{nu_},
      constraint_grams_{static_cast<size_t>(nu_)} {
  const symbolic::Variables x_set(x_);
  prog_.AddIndeterminates(x_);
  CheckDynamicsInput(V_, f_, G_, x_set);
  DRAKE_DEMAND(static_cast<int>(b_degrees_.size()) == nu_);

  // Add Lagrangian decision variables.
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(l_given[i][0].TotalDegree() == lagrangian_degrees_[i][0]);
    DRAKE_DEMAND(l_given[i][1].TotalDegree() == lagrangian_degrees_[i][1]);
    l_[i][0] = l_given[i][0];
    l_[i][1] = l_given[i][1];
    for (int j = 2; j < 6; ++j) {
      MatrixX<symbolic::Variable> lagrangian_gram;
      std::tie(l_[i][j], lagrangian_gram) =
          prog_.NewSosPolynomial(x_set, lagrangian_degrees_[i][j]);
    }
  }

  eps_ = prog_.NewContinuousVariables<1>("eps")(0);

  const RowVectorX<symbolic::Polynomial> dVdx = V_.Jacobian(x_);
  // Since we will add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x), we know that
  // the highest degree of b should be at least degree(∂V/∂x*f(x) + εV).
  const symbolic::Polynomial dVdx_times_f = (dVdx * f_)(0);
  if (*std::max_element(b_degrees_.begin(), b_degrees_.end()) <
      std::max(dVdx_times_f.TotalDegree(), V.TotalDegree())) {
    throw std::invalid_argument("The degree of b is too low.");
  }

  // Add free polynomial b
  for (int i = 0; i < nu_; ++i) {
    b_(i) = prog_.NewFreePolynomial(x_set, b_degrees_[i], "b");
  }

  // Add the constraint ∂V/∂x*f(x) + εV = ∑ᵢ bᵢ(x)
  prog_.AddEqualityConstraintBetweenPolynomials(b_.sum(),
                                                dVdx_times_f + eps_ * V_);
  // Add the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  for (int i = 0; i < nu_; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = (dVdx * G_.col(i))(0);
    const symbolic::Polynomial p1 = (l_[i][0] + 1) * (dVdx_times_Gi - b_(i)) -
                                    l_[i][2] * dVdx_times_Gi -
                                    l_[i][4] * (1 - V_);
    const symbolic::Polynomial p2 = (l_[i][1] + 1) * (-dVdx_times_Gi - b_(i)) +
                                    l_[i][3] * dVdx_times_Gi -
                                    l_[i][5] * (1 - V_);
    constraint_grams_[i][0] = prog_.AddSosConstraint(p1);
    constraint_grams_[i][1] = prog_.AddSosConstraint(p2);
  }

  // prog_.AddLinearCost(-eps_);
  prog_.AddLinearConstraint(eps_ >= 3);
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
      prog_{},
      nu_{static_cast<int>(G_.cols())},
      nx_{static_cast<int>(f.rows())},
      l_{static_cast<size_t>(nu_)},
      lagrangian_degrees_{std::move(lagrangian_degrees)},
      lagrangian_grams_{static_cast<size_t>(nu_)} {
  const symbolic::Variables x_set{x_};
  CheckDynamicsInput(V_, f_, G_, x_set);
  DRAKE_DEMAND(b.rows() == nu_);
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(b_(i).indeterminates().IsSubsetOf(x_set));
  }
  prog_.AddIndeterminates(x_);
  for (int i = 0; i < nu_; ++i) {
    for (int j = 0; j < 6; ++j) {
      std::tie(l_[i][j], lagrangian_grams_[i][j]) =
          prog_.NewSosPolynomial(x_set, lagrangian_degrees_[i][j]);
    }
  }

  const auto dVdx = V_.Jacobian(x_);
  // Now impose the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  for (int i = 0; i < nu_; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = dVdx * G_.col(i);
    const symbolic::Polynomial p1 =
        (l_[i][0] + 1) * (dVdx_times_Gi - b_(i)) -
        l_[i][2] * dVdx_times_Gi * l_[i][4] * (1 - V_);
    const symbolic::Polynomial p2 = (l_[i][1] + 1) * (-dVdx_times_Gi - b_(i)) +
                                    l_[i][3] * dVdx_times_Gi -
                                    l_[i][5] * (1 - V_);
    prog_.AddSosConstraint(p1);
    prog_.AddSosConstraint(p2);
  }
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
