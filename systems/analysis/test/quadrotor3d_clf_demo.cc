#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/math/gray_code.h"
#include "drake/math/rigid_transform.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_config_functions.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

/**
 * The state is [x, y, z, roll, pitch, yaw, xdot, ydot, zdot, angular_x,
 * angular_y, angularz_z]
 * The control is the thrust generated in each propeller.
 */
struct Quadrotor {
  // Dynamics with un-normalized u.
  template <typename T>
  [[nodiscard]] Eigen::Matrix<double, 12, 1> CalcDynamics(
      const Eigen::Matrix<T, 12, 1>& x, const Vector4<T>& u) const {
    Vector4<T> plant_input;
    plant_input << u.sum(), arm_length * u(1) - arm_length * u(3),
        -arm_length * u(0) + arm_length * u(2),
        z_torque_to_force_factor * (u(0) - u(1) + u(2) - u(3));
    const math::RotationMatrix<T> R_WB(
        math::RollPitchYaw<T>(x.template segment<3>(3)));
    const Vector3<T> pos_ddot = Vector3<T>(0, 0, -gravity) +
                                R_WB.matrix().col(2) * plant_input(0) / mass;

    const auto& omega = x.template tail<3>();
    const Vector3<T> omegadot_WB =
        ((-omega.cross((inertia.array() * omega.array()).matrix()) +
          plant_input.template tail<3>())
             .array() /
         inertia.array())
            .matrix();
    Matrix3<T> omega_to_rpydot;
    const auto& rpy = x.template segment<3>(3);
    const Vector3<T> rpy_dot = CalcRpydot<T>(rpy, omega);
    Eigen::Matrix<T, 12, 1> xdot;
    xdot.template head<3>() = x.template segment<3>(6);
    xdot.template segment<3>(3) = rpy_dot;
    xdot.template segment<3>(6) = pos_ddot;
    xdot.template tail<3>() = omegadot_WB;
    return xdot;
  };

  template <typename T>
  Vector3<T> CalcRpydot(const Eigen::Ref<const Vector3<T>>& rpy,
                        const Eigen::Ref<const Vector3<T>>& omega) const {
    using std::cos;
    using std::sin;
    using std::tan;
    const T sin_roll = sin(rpy(0));
    const T cos_roll = cos(rpy(0));
    const T tan_pitch = tan(rpy(1));
    const T cos_pitch = cos(rpy(1));
    Matrix3<T> omega_to_rpydot;
    // clang-format off
    omega_to_rpydot << 1, sin_roll * tan_pitch, cos_roll * tan_pitch,
                    0, cos_roll, -sin_roll,
                    0, sin_roll / cos_pitch, cos_roll / cos_pitch;
    // clang-format on
    const Vector3<T> rpy_dot = omega_to_rpydot * omega;
    return rpy_dot;
  }

  template <typename T>
  Vector4<T> NormalizeU(const Eigen::Ref<const Vector4<T>>& u) const {
    const Vector4<T> u_mid = thrust_max / 2 * Vector4<T>::Ones();
    return (u - u_mid) / (thrust_max / 2);
  }

  template <typename T>
  Vector4<T> UnnormalizeU(const Vector4<T>& u_normalized) const {
    const Vector4<T> u_mid = thrust_max / 2 * Vector4<T>::Ones();
    return u_normalized * thrust_max / 2 + u_mid;
  }

  // This assumes normalized u, namely -1 <= u <= 1
  template <typename T>
  void ControlAffineDynamics(const Eigen::Matrix<T, 12, 1>& x,
                             Eigen::Matrix<T, 12, 1>* f,
                             Eigen::Matrix<T, 12, 4>* G) const {
    const math::RotationMatrix<T> R_WB(
        math::RollPitchYaw<T>(x.template segment<3>(3)));
    f->template head<3>() = x.template segment<3>(6);
    const auto& rpy = x.template segment<3>(3);
    const auto& omega = x.template tail<3>();
    f->template segment<3>(3) = CalcRpydot<T>(rpy, omega);
    f->template segment<3>(6) =
        Vector3<T>(0, 0, -gravity) + R_WB.col(2) * 2 * thrust_max / mass;
    f->template tail<3>() =
        -omega.cross((inertia.array() * omega.array()).matrix());

    G->template topRows<6>().setZero();
    G->template block<3, 4>(6, 0) =
        R_WB.col(2) * Eigen::RowVector4d::Ones() * (thrust_max / 2) / mass;
    // clang-format off
    G->template bottomRows<3>() << 0, arm_length, 0, -arm_length,
      -arm_length, 0, arm_length, 0,
      z_torque_to_force_factor, -z_torque_to_force_factor, z_torque_to_force_factor, -z_torque_to_force_factor;
    // clang-format on
    for (int i = 0; i < 3; ++i) {
      (*f)(9 + i) /= inertia(i);
      G->row(9 + i) *= thrust_max / (2 * inertia(i));
    }
  }

  // This assumes normalized u.
  void PolynomialControlAffineDynamics(
      const Eigen::Matrix<symbolic::Variable, 12, 1>& x,
      Eigen::Matrix<symbolic::Polynomial, 12, 1>* f,
      Eigen::Matrix<symbolic::Polynomial, 12, 4>* G) const {
    Eigen::Matrix<symbolic::Expression, 12, 1> f_expr;
    Eigen::Matrix<symbolic::Expression, 12, 4> G_expr;
    ControlAffineDynamics<symbolic::Expression>(x.cast<symbolic::Expression>(),
                                                &f_expr, &G_expr);
    const int f_order = 3;
    const int G_order = 3;
    symbolic::Environment env;
    env.insert(x, Eigen::Matrix<double, 12, 1>::Zero());
    for (int i = 0; i < 12; ++i) {
      (*f)(i) =
          symbolic::Polynomial(symbolic::TaylorExpand(f_expr(i), env, f_order));
      for (int j = 0; j < 4; ++j) {
        (*G)(i, j) = symbolic::Polynomial(
            symbolic::TaylorExpand(G_expr(i, j), env, G_order));
      }
    }
  }

  double mass = 0.468;
  double gravity = 9.81;
  double arm_length = 0.225;
  Eigen::Vector3d inertia = Eigen::Vector3d(4.9E-3, 4.9E-3, 8.8E-3);
  // The ratio between the torque along the z axis versus the force.
  double z_torque_to_force_factor = 1.1 / 29;
  double thrust_max = mass * gravity / 4 * 2.5;
};

[[maybe_unused]] void search(
    const Eigen::Matrix<symbolic::Variable, 12, 1>& x,
    const symbolic::Polynomial& V_init,
    const Eigen::Matrix<symbolic::Polynomial, 12, 1>& f,
    const Eigen::Matrix<symbolic::Polynomial, 12, 4>& G,
    symbolic::Polynomial* V_sol, double* deriv_eps_sol) {
  const Eigen::Matrix<double, 4, 16> u_vertices =
      math::CalculateReflectedGrayCodes<4>().cast<double>().transpose() * 2 -
      Eigen::Matrix<double, 4, 16>::Ones();

  VectorX<symbolic::Polynomial> state_constraints(0);

  ControlLyapunov dut(x, f, G, std::nullopt /* dynamics denominator */,
                      u_vertices, state_constraints);
  const double deriv_eps = 0.2;
  *deriv_eps_sol = deriv_eps;
  const int lambda0_degree = 2;
  const std::vector<int> l_degrees(16, 2);
  const std::vector<int> p_degrees = {};
  const std::vector<int> ellipsoid_c_lagrangian_degrees = {};
  const int V_degree = 2;
  const double positivity_eps = 0.0001;
  const int positivity_d = V_degree / 2;
  const std::vector<int> positivity_eq_lagrangian_degrees{};
  const Eigen::Matrix<double, 12, 1> x_star =
      Eigen::Matrix<double, 12, 1>::Zero();
  const Eigen::Matrix<double, 12, 12> S =
      Eigen::Matrix<double, 12, 12>::Identity();
  ControlLyapunov::SearchOptions search_options;
  search_options.bilinear_iterations = 20;
  search_options.backoff_scale = 0.02;
  search_options.lyap_tiny_coeff_tol = 2E-7;
  search_options.Vsol_tiny_coeff_tol = 1E-7;
  search_options.lsol_tiny_coeff_tol = 1E-7;
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  // search_options.lyap_step_solver_options->SetOption(
  //    solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  // search_options.lagrangian_step_solver_options->SetOption(
  //    solvers::CommonSolverOption::kPrintToConsole, 1);
  ControlLyapunov::RhoBisectionOption rho_bisection_option(0.01, 3, 0.01);
  symbolic::Polynomial lambda0;
  VectorX<symbolic::Polynomial> l;
  VectorX<symbolic::Polynomial> p_sol;
  symbolic::Polynomial r;
  VectorX<symbolic::Polynomial> positivity_eq_lagrangian_sol;
  double rho_sol;
  VectorX<symbolic::Polynomial> ellipsoid_c_lagrangian_sol;

  dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
             positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
             ellipsoid_c_lagrangian_degrees, deriv_eps, x_star, S, V_degree - 2,
             search_options, rho_bisection_option, V_sol, &lambda0, &l, &r,
             &p_sol, &positivity_eq_lagrangian_sol, &rho_sol,
             &ellipsoid_c_lagrangian_sol);
}

[[maybe_unused]] void search_w_box_bounds(
    const Eigen::Matrix<symbolic::Variable, 12, 1>& x,
    const symbolic::Polynomial& V_init,
    const Eigen::Matrix<symbolic::Polynomial, 12, 1>& f,
    const Eigen::Matrix<symbolic::Polynomial, 12, 4>& G,
    symbolic::Polynomial* V_sol, double* deriv_eps_sol) {
  const int V_degree = 2;
  const double positivity_eps{0};
  const ControlLyapunovBoxInputBound dut(f, G, x, positivity_eps);
  ControlLyapunovBoxInputBound::SearchOptions search_options;
  search_options.bilinear_iterations = 10;
  search_options.backoff_scale = 0.02;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.lyap_step_solver_options = solvers::SolverOptions();
  search_options.lyap_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  search_options.lyap_tiny_coeff_tol = 2E-7;
  search_options.lsol_tiny_coeff_tol = 1E-5;
  ControlLyapunovBoxInputBound::RhoBisectionOption rho_bisection_option(0.01, 3,
                                                                        0.01);

  const int nu = 4;
  const int num_vdot_sos = 2;
  std::vector<std::vector<symbolic::Polynomial>> l_given(nu);
  for (int i = 0; i < nu; ++i) {
    l_given[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      l_given[i][j] = symbolic::Polynomial();
    }
  }
  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees(nu);
  for (int i = 0; i < nu; ++i) {
    lagrangian_degrees[i].resize(num_vdot_sos);
    for (int j = 0; j < num_vdot_sos; ++j) {
      lagrangian_degrees[i][j] = {0, 2, 4};
    }
  }
  std::vector<int> b_degrees(nu);
  for (int i = 0; i < nu; ++i) {
    b_degrees[i] = 4;
  }

  const Eigen::Matrix<double, 12, 1> x_star =
      Eigen::Matrix<double, 12, 1>::Zero();
  const Eigen::Matrix<double, 12, 12> S =
      Eigen::Matrix<double, 12, 12>::Identity();
  const int r_degree = V_degree - 2;
  const double deriv_eps_lower = 0.2;
  const double deriv_eps_upper = kInf;
  const auto search_result =
      dut.Search(V_init, l_given, lagrangian_degrees, b_degrees, x_star, S,
                 r_degree, V_degree, deriv_eps_lower, deriv_eps_upper,
                 search_options, rho_bisection_option);
  *V_sol = search_result.V;
  *deriv_eps_sol = search_result.deriv_eps;
}

int DoMain() {
  Quadrotor quadrotor;
  const Eigen::Matrix<double, 12, 1> x_des =
      Eigen::Matrix<double, 12, 1>::Zero();
  const double thrust_equilibrium = quadrotor.mass * quadrotor.gravity / 4;
  const Eigen::Vector4d u_des = quadrotor.NormalizeU<double>(
      thrust_equilibrium * Eigen::Vector4d::Ones());
  const auto x_des_ad = math::InitializeAutoDiff(x_des);
  Eigen::Matrix<AutoDiffXd, 12, 1> f_des;
  Eigen::Matrix<AutoDiffXd, 12, 4> G_des;
  quadrotor.ControlAffineDynamics<AutoDiffXd>(x_des_ad, &f_des, &G_des);
  const auto xdot_des = f_des + G_des * u_des;
  Eigen::Matrix<double, 12, 1> lqr_Q_diag;
  lqr_Q_diag.topRows<6>() = Vector6d::Ones();
  lqr_Q_diag.bottomRows<6>() = Vector6d::Ones() * 10;
  const Eigen::Matrix<double, 12, 12> lqr_Q = lqr_Q_diag.asDiagonal();
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      math::ExtractGradient(xdot_des), math::ExtractValue(G_des), lqr_Q,
      Eigen::Matrix4d::Identity() * 10);

  Eigen::Matrix<symbolic::Variable, 12, 1> x;
  for (int i = 0; i < 12; ++i) {
    x(i) = symbolic::Variable(fmt::format("x{}", i));
  }

  Eigen::Matrix<symbolic::Polynomial, 12, 1> f;
  Eigen::Matrix<symbolic::Polynomial, 12, 4> G;
  quadrotor.PolynomialControlAffineDynamics(x, &f, &G);
  for (int i = 0; i < 12; ++i) {
    f(i) = f(i).RemoveTermsWithSmallCoefficients(1E-6);
    for (int j = 0; j < 4; ++j) {
      G(i, j) = G(i, j).RemoveTermsWithSmallCoefficients(1E-6);
    }
  }

  symbolic::Polynomial V_init(
      x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  V_init = V_init.RemoveTermsWithSmallCoefficients(1E-8);
  std::cout << "V_init: " << V_init << "\n";

  symbolic::Polynomial V_sol;
  double deriv_eps_sol;
  search(x, V_init, f, G, &V_sol, &deriv_eps_sol);
  // search_w_box_bounds(x, V_init, f, G, &V_sol, &deriv_eps_sol);

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
