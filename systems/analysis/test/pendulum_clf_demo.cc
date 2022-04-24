#include <math.h>

#include <limits>

#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

struct Pendulum {
  double compute_thetaddot(double theta, double theta_dot, double u) const {
    const double theta_ddot =
        (u - mass * gravity * length * std::sin(theta) - damping * theta_dot) /
        (mass * length * length);
    return theta_ddot;
  }

  void control_affine_dynamics(const Vector2<symbolic::Variable>& x,
                               double theta_des, double u_bound,
                               Vector2<symbolic::Polynomial>* f,
                               Vector2<symbolic::Polynomial>* G) const {
    (*G)(0) = symbolic::Polynomial();
    (*G)(1) = symbolic::Polynomial(u_bound / (mass * length * length));
    (*f)(0) = symbolic::Polynomial(x(1));
    (*f)(1) = symbolic::Polynomial(
        (-mass * gravity * length *
             (std::sin(theta_des) + std::cos(theta_des) * x(0) -
              std::sin(theta_des) / 2 * pow(x(0), 2) -
              std::cos(theta_des) / 6 * pow(x(0), 3)) -
         damping * x(1)) /
        (mass * length * length));
  }

  void dynamics_gradient(double theta, double u_bound, Eigen::Matrix2d* A,
                         Eigen::Vector2d* B) const {
    *A << 0, 1,
        -mass * gravity * length * std::cos(theta) / (mass * length * length),
        -damping / (mass * length * length);
    *B << 0, u_bound / (mass * length * length);
  }

  double mass{1};
  double gravity{9.81};
  double length{1};
  double damping{0.1};
};

int DoMain() {
  const Vector2<symbolic::Variable> x(symbolic::Variable("x0"),
                                      symbolic::Variable("x1"));

  const Pendulum pendulum;
  Vector2<symbolic::Polynomial> f;
  Vector2<symbolic::Polynomial> G;
  const double u_bound = 25;
  Eigen::Matrix2d A;
  Eigen::Vector2d B;
  pendulum.dynamics_gradient(M_PI, u_bound, &A, &B);
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      A, B, Eigen::Matrix2d::Identity(), Vector1<double>::Identity());
  pendulum.control_affine_dynamics(x, M_PI, u_bound, &f, &G);
  for (int i = 0; i < 2; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    G(i, 0) = G(i, 0).RemoveTermsWithSmallCoefficients(1E-6);
  }
  symbolic::Polynomial V(x.dot(lqr_result.S * x));

  const bool symmetric_dynamics = internal::IsDynamicsSymmetric(f, G);
  const int num_vdot_sos = symmetric_dynamics ? 1 : 2;
  std::vector<std::vector<symbolic::Polynomial>> l_given(1);
  l_given[0].resize(num_vdot_sos);
  for (int i = 0; i < num_vdot_sos; ++i) {
    l_given[0][i] = symbolic::Polynomial();
  }
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees(1);
  lagrangian_degrees[0].resize(num_vdot_sos);
  for (int i = 0; i < num_vdot_sos; ++i) {
    lagrangian_degrees[0][i] = {0, 4, 4};
  }
  std::vector<int> b_degrees(1);
  b_degrees[0] = 4;
  SearchLagrangianAndBGivenVBoxInputBound dut(
      V, f, G, symmetric_dynamics, l_given, lagrangian_degrees, b_degrees, x);
  dut.get_mutable_prog()->AddBoundingBoxConstraint(0.01, kInf, dut.deriv_eps());
  solvers::MosekSolver mosek_solver;
  solvers::SolverOptions solver_options;
  solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  const auto result =
      mosek_solver.Solve(dut.prog(), std::nullopt, solver_options);

  const double positivity_eps = 0.;
  ControlLyapunovBoxInputBound searcher(f, G, x, positivity_eps);
  const Eigen::Vector2d x_star(0, 0);
  Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
  int s_degree = 0;
  symbolic::Polynomial t_given{0};
  const int V_degree = 2;
  const double deriv_eps_lower = 0.01;
  const double deriv_eps_upper = kInf;
  ControlLyapunovBoxInputBound::SearchOptions search_options;
  searcher.Search(V, l_given, lagrangian_degrees, b_degrees, x_star, S,
                  s_degree, t_given, V_degree, deriv_eps_lower, deriv_eps_upper,
                  search_options);
  return 0;
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() { return drake::systems::analysis::DoMain(); }
