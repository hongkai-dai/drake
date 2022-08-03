#include <math.h>

#include <iostream>
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
  Quadrotor2dTrigPlant<double> quadrotor;

  // Synthesize an LQR controller.
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  const Eigen::Vector2d u_des(thrust_equilibrium, thrust_equilibrium);
  auto context = quadrotor.CreateDefaultContext();
  context->SetContinuousState(Vector6d::Zero());
  quadrotor.get_input_port().FixValue(context.get(), u_des);
  auto linearized_quadrotor = Linearize(quadrotor, *context);
  Vector6d lqr_Q_diag;
  lqr_Q_diag << 1, 1, 1, 10, 10, 10;
  const Eigen::Matrix<double, 6, 6> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      linearized_quadrotor->A(), linearized_quadrotor->B(), lqr_Q,
      Eigen::Matrix2d::Identity() * 10);

  Vector6<symbolic::Variable> x;
  for (int i = 0; i < 6; ++i) {
    x(i) = symbolic::Variable(fmt::format("x{}", i));
  }

  Vector6<symbolic::Polynomial> f;
  Eigen::Matrix<symbolic::Polynomial, 6, 2> G;
  PolynomialControlAffineDynamics(quadrotor, x, &f, &G);
  const std::optional<symbolic::Polynomial> dynamics_numerator = std::nullopt;
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
  const double thrust_max = 3 * thrust_equilibrium;
  // clang-format off
  u_vertices << 0, 0, thrust_max, thrust_max,
                0, thrust_max, 0, thrust_max;
  // clang-format on
  const VectorX<symbolic::Polynomial> state_constraints(0);

  const double beta_minus = -1;
  const std::optional<double> beta_plus = std::nullopt;
  const ControlBarrier dut(f, G, dynamics_numerator, x, beta_minus, beta_plus,
                           unsafe_regions, u_vertices, state_constraints);

  // Construct h_init;
  const symbolic::Polynomial h_init(
      0.1 - x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  const int h_degree = 2;
  const double deriv_eps = 0.1;
  const int lambda0_degree = 4;
  const std::optional<int> lambda1_degree = std::nullopt;
  const std::vector<int> l_degrees = {2, 2, 2, 2};
  const std::vector<int> hdot_state_constraints_lagrangian_degrees{};
  const std::vector<int> t_degree = {0, 0};
  const std::vector<std::vector<int>> s_degrees = {{0}, {0}};
  const std::vector<std::vector<int>>
      unsafe_state_constraints_lagrangian_degrees{{}, {}};
  std::vector<ControlBarrier::Ellipsoid> ellipsoids;
  std::vector<ControlBarrier::EllipsoidBisectionOption>
      ellipsoid_bisection_options;
  ellipsoids.emplace_back(
      Vector6d::Zero(), Matrix6<double>::Identity(), 0., 0,
      std::vector<int>() /* state_constraints_lagrangian_degree */);
  ellipsoid_bisection_options.emplace_back(0, 2, 0.001);
  const Vector6d x_anchor = Vector6d::Zero();
  ControlBarrier::SearchOptions search_options;
  search_options.hsol_tiny_coeff_tol = 1E-8;
  search_options.lsol_tiny_coeff_tol = 1E-8;
  search_options.lagrangian_tiny_coeff_tol = 1E-8;
  search_options.barrier_tiny_coeff_tol = 1E-8;
  symbolic::Polynomial h_sol;
  symbolic::Polynomial lambda0_sol;
  VectorX<symbolic::Polynomial> l_sol;
  VectorX<symbolic::Polynomial> hdot_state_constraints_lagrangian_sol;
  std::vector<symbolic::Polynomial> t_sol;
  std::vector<VectorX<symbolic::Polynomial>> s_sol;
  std::vector<VectorX<symbolic::Polynomial>>
      unsafe_state_constraints_lagrangian_sol;
  const auto search_ret = dut.Search(
      h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
      hdot_state_constraints_lagrangian_degrees, t_degree, s_degrees,
      unsafe_state_constraints_lagrangian_degrees, x_anchor, search_options,
      &ellipsoids, &ellipsoid_bisection_options);
  std::cout << "h_sol: " << search_ret.h << "\n";
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
