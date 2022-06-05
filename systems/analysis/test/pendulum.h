#pragma once

#include "drake/common/symbolic.h"
#include "drake/examples/pendulum/pendulum_plant.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {

/**
 * Get the pendulum control affine dynamics by doing taylor approximation around
 * theta = theta_des and theta_dot = 0.
 */
void ControlAffineDynamics(
    const examples::pendulum::PendulumPlant<double>& pendulum,
    const Vector2<symbolic::Variable>& x, double theta_des,
    Vector2<symbolic::Polynomial>* f, Vector2<symbolic::Polynomial>* G);

/**
 * The state is x = (sin(theta) - sin(theta_des), cos(theta) - cos(theta_des),
 * thetadot)
 */
template <typename T>
Vector3<T> TrigDynamics(
    const examples::pendulum::PendulumPlant<double>& pendulum,
    const Eigen::Ref<const Vector3<T>>& x, double theta_des,
    const Eigen::Ref<const Vector1<T>>& u) {
  using std::cos;
  using std::sin;
  const double sin_theta_des = sin(theta_des);
  const double cos_theta_des = cos(theta_des);
  Vector3<T> xdot;
  const T s = x(0) + sin_theta_des;
  const T c = x(1) + cos_theta_des;
  xdot(0) = c * x(2);
  xdot(1) = -s * x(2);
  auto context = pendulum.CreateDefaultContext();
  const auto& p = pendulum.get_parameters(*context);
  xdot(2) =
      (u(0) - p.mass() * p.gravity() * p.length() * s - p.damping() * x(2)) /
      (p.mass() * p.length() * p.length());
  return xdot;
}

/**
 * The state is x = (sin(theta) - sin(theta_des), cos(theta) - cos(theta_des),
 * thetadot)
 */
void TrigPolyDynamics(const examples::pendulum::PendulumPlant<double>& pendulum,
                      const Vector3<symbolic::Variable>& x, double theta_des,
                      Vector3<symbolic::Polynomial>* f,
                      Vector3<symbolic::Polynomial>* G);

controllers::LinearQuadraticRegulatorResult TrigDynamicsLQR(
    const examples::pendulum::PendulumPlant<double>& pendulum, double theta_des,
    const Eigen::Ref<const Eigen::Matrix3d>& Q,
    const Eigen::Ref<const Vector1d>& R, Eigen::Matrix3d* A = nullptr,
    Eigen::Vector3d* B = nullptr);

double EquilibriumTorque(
    const examples::pendulum::PendulumPlant<double>& pendulum,
    double theta_des);

class Pendulum {
 public:
  Pendulum() {}

  double ComputeThetaddot(double theta, double theta_dot, double u) const;

  // This function normalizes the control input to be within [-1, 1].
  void ControlAffineDynamics(const Vector2<symbolic::Variable>& x,
                             double theta_des, double u_bound,
                             Vector2<symbolic::Polynomial>* f,
                             Vector2<symbolic::Polynomial>* G) const;

  // This assumes normalized control input -1 <= u <= 1
  void DynamicsGradient(double theta, double u_bound, Eigen::Matrix2d* A,
                        Eigen::Vector2d* B) const;

  double mass() const { return mass_; }

  double gravity() const { return gravity_; }

  double length() const { return length_; }

  double damping() const { return damping_; }

 private:
  double mass_{1};
  double gravity_{9.81};
  double length_{1};
  double damping_{0.1};
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
