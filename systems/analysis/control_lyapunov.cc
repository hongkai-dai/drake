#include "drake/systems/analysis/control_lyapunov.h"

namespace drake {
namespace systems {
namespace analysis {
SearchControlLyapunov::SearchControlLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const Eigen::Ref<const Eigen::VectorXd>& x_equilibrium,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x)
    : f_{f},
      G_{G},
      x_equilibrium_{x_equilibrium},
      u_vertices_{u_vertices},
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
}  // namespace analysis
}  // namespace systems
}  // namespace drake
