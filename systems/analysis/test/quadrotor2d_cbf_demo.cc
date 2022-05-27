#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_barrier.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/analysis/test/quadrotor2d.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {

const double kInf = std::numeric_limits<double>::infinity();

int DoMain() {
  Quadrotor quadrotor;

  // Synthesize an LQR controller.
  const Vector6d x_des = Vector6d::Zero();
  const double thrust_equilibrium = quadrotor.mass * quadrotor.gravity / 2;
  const Eigen::Vector2d u_des = quadrotor.NormalizeU(
      Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium));
  const auto x_des_ad = math::InitializeAutoDiff(x_des);
  Vector6<AutoDiffXd> f_des;
  Eigen::Matrix<AutoDiffXd, 6, 2> G_des;
  quadrotor.ControlAffineDynamics<AutoDiffXd>(x_des_ad, &f_des, &G_des);
  const auto xdot_des = f_des + G_des * u_des;
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10, 10;
  const Eigen::Matrix<double, 6, 6> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      math::ExtractGradient(xdot_des), math::ExtractValue(G_des), lqr_Q,
      Eigen::Matrix2d::Identity() * 10);

  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable(fmt::format("x{}", i));
  }

  Vector6<symbolic::Polynomial> f;
  Eigen::Matrix<symbolic::Polynomial, 6, 2> G;
  quadrotor.PolynomialControlAffineDynamics(x, &f, &G);
  for (int i = 0; i < 6; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    for (int j = 0; j < 2; ++j) {
      G(i, j) = G(i, j).RemoveTermsWithSmallCoefficients(1E-6);
    }
  }

  // The unsafe region is the ground and the ceiling.
  std::vector<VectorX<symbolic::Polynomial>> unsafe_regions(2);
  unsafe_regions[0].resize(1);
  unsafe_regions[0](0) = symbolic::Polynomial(x(1) + 0.3);
  unsafe_regions[1].resize(1);
  unsafe_regions[1](0) = symbolic::Polynomial(0.5 - x(1));

  Eigen::Matrix<double, 2, 4> u_vertices;
  // clang-format off
  u_vertices << 1, 1, -1, -1,
                1, -1, 1, -1;
  // clang-format on
  const ControlBarrier dut(f, G, x, unsafe_regions, u_vertices);

  // Construct h_init;
  const symbolic::Polynomial h_init(
      0.1 - x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  const int h_degree = 2;
  const double deriv_eps = 0.1;
  const int lambda0_degree = 4;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> t_degree = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{0}, {0}};
  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  ellipsoids.emplace_back(Vector6d::Zero(), Matrix6<double>::Identity(), 0., 0.,
                          2, 0.001, 0);
  const Vector6d x_anchor = Vector6d::Zero();
  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lsol_tiny_coeff_tol = 1E-8;
  search_options.lagrangian_tiny_coeff_tol = 1E-8;
  search_options.barrier_tiny_coeff_tol = 1E-8;
  symbolic::Polynomial h_sol;
  symbolic::Polynomial lambda0_sol;
  VectorX<symbolic::Polynomial> l_sol;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  dut.Search(h_init, h_degree, deriv_eps, lambda0_degree, l_degrees, t_degree,
             s_degrees, ellipsoids, x_anchor, search_options, &h_sol,
             &lambda0_sol, &l_sol, &t_sol, &s_sol);
  std::cout << "h_sol: " << h_sol << "\n";
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
