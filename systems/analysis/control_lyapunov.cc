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
  DRAKE_DEMAND(G.rows() == nx_);
  DRAKE_DEMAND(b.rows() == nu_);
  const symbolic::Variables x_set{x_};
  DRAKE_DEMAND(V.indeterminates().IsSubsetOf(x_set));
  for (int i = 0; i < nx_; ++i) {
    DRAKE_DEMAND(f_(i).indeterminates().IsSubsetOf(x_set));
    for (int j = 0; j < nu_; ++j) {
      DRAKE_DEMAND(G_(i, j).indeterminates().IsSubsetOf(x_set));
    }
  }
  for (int i = 0; i < nu_; ++i) {
    DRAKE_DEMAND(b(i).indeterminates().IsSubsetOf(x_set));
  }
  prog_.AddIndeterminates(x_);
  for (int i = 0; i < nu_; ++i) {
    for (int j = 0; j < 6; ++j) {
      std::tie(l_[i][j], lagrangian_grams_[i][j]) =
          prog_.NewSosPolynomial(x_set, lagrangian_degrees_[i][j]);
    }
  }

  const auto dVdx = V.Jacobian(x);
  // Now impose the constraint
  // (lᵢ₁(x)+1)(∂V/∂x*Gᵢ(x)−bᵢ(x)) − lᵢ₃(x)*∂V/∂x*Gᵢ(x) - lᵢ₅(x)*(1 − V) >= 0
  // (lᵢ₂(x)+1)(−∂V/∂x*Gᵢ(x)−bᵢ(x)) + lᵢ₄(x)*∂V/∂x*Gᵢ(x) - lᵢ₆(x)*(1 − V) >= 0
  for (int i = 0; i < nu_; ++i) {
    const symbolic::Polynomial dVdx_times_Gi = dVdx * G.col(i);
    const symbolic::Polynomial p1 =
        (l_[i][0] + 1) * (dVdx_times_Gi - b(i)) -
        l_[i][2] * dVdx_times_Gi * l_[i][4] * (1 - V);
    const symbolic::Polynomial p2 = (l_[i][1] + 1) * (-dVdx_times_Gi - b(i)) +
                                    l_[i][3] * dVdx_times_Gi -
                                    l_[i][5] * (1 - V);
    prog_.AddSosConstraint(p1);
    prog_.AddSosConstraint(p2);
  }
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
