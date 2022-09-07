#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
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

template <typename T>
Vector3<T> ToPendulumTrigState(const T& theta, const T& thetadot,
                               double theta_des) {
  const double sin_theta_des = std::sin(theta_des);
  const double cos_theta_des = std::cos(theta_des);
  using std::cos;
  using std::sin;
  return Vector3<T>(sin(theta) - sin_theta_des, cos(theta) - cos_theta_des,
                    thetadot);
}

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

/**
 * Convert (theta, thetadot) to (sin(theta)-sin(theta_des),
 * cos(theta)-cos(theta_des), thetadot)
 */
class PendulumTrigStateConverter : public LeafSystem<double> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(PendulumTrigStateConverter)

  PendulumTrigStateConverter(double theta_des);

  void Convert(const Context<double>& context, BasicVector<double>* x) const;

 private:
  double sin_theta_des;
  double cos_theta_des;
};
}  // namespace analysis
}  // namespace systems
}  // namespace drake
