#pragma once

#include <math.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {

template <typename T>
class QuadrotorPlant : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuadrotorPlant);

  QuadrotorPlant();

  /// Scalar-converting copy constructor.  See @ref system_scalar_conversion.
  template <typename U>
  explicit QuadrotorPlant(const QuadrotorPlant<U>&);

  ~QuadrotorPlant(){};

  double mass() const { return mass_; }

  double length() const { return length_; }

  double inertia() const { return inertia_; }

  double gravity() const { return gravity_; }

  const OutputPort<T>& get_state_output_port() const {
    return this->get_output_port(state_output_port_index_);
  }

  const InputPort<T>& get_actuation_input_port() const {
    return this->get_input_port();
  }

 private:
  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;

  double length_{0.25};
  double mass_{0.486};
  double inertia_{0.00383};
  double gravity_{9.81};
  OutputPortIndex state_output_port_index_;
};

template <typename T>
Eigen::Matrix<T, 7, 1> ToTrigState(
    const Eigen::Ref<const Vector6<T>>& x_original) {
  Eigen::Matrix<T, 7, 1> x_trig;
  x_trig(0) = x_original(0);
  x_trig(1) = x_original(1);
  using std::cos;
  using std::sin;
  x_trig(2) = sin(x_original(2));
  x_trig(3) = cos(x_original(2)) - 1;
  x_trig.template tail<3>() = x_original.template tail<3>();
  return x_trig;
}

/**
 * The state for the polynomial dynamics is
 * [pos_x, pos_y, sin(theta), cos(theta) - 1, vel_x, vel_y, thetadot]
 */
template <typename T>
Eigen::Matrix<T, 7, 1> TrigDynamics(
    const QuadrotorPlant<double>& quadrotor,
    const Eigen::Ref<const Eigen::Matrix<T, 7, 1>>& x,
    const Eigen::Ref<const Vector2<T>>& u) {
  Eigen::Matrix<T, 7, 1> xdot;
  xdot(0) = x(4);
  xdot(1) = x(5);
  const T c_theta = x(3) + 1;
  xdot(2) = c_theta * x(6);
  xdot(3) = -x(2) * x(6);
  xdot(4) = -x(2) / quadrotor.mass() * (u(0) + u(1));
  xdot(5) = c_theta / quadrotor.mass() * (u(0) + u(1)) - quadrotor.gravity();
  xdot(6) = quadrotor.length() / quadrotor.inertia() * (u(0) - u(1));
  return xdot;
}

/**
 * The state for the polynomial dynamics is
 * [pos_x, pos_y, sin(theta), cos(theta)-1, vel_x, vel_y, thetadot]
 */
void TrigPolyDynamics(
    const QuadrotorPlant<double>& quadrotor,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 7, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 7, 2>* G);

double EquilibriumThrust(const QuadrotorPlant<double>& quadrotor);

void PolynomialControlAffineDynamics(
    const QuadrotorPlant<double>& quadrotor,
    const Vector6<symbolic::Variable>& x, Vector6<symbolic::Polynomial>* f,
    Eigen::Matrix<symbolic::Polynomial, 6, 2>* G);

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>& x);

controllers::LinearQuadraticRegulatorResult SynthesizeTrigLqr(
    const Eigen::Ref<const Eigen::Matrix<double, 7, 7>>& Q,
    const Eigen::Ref<const Eigen::Matrix2d>& R);

template <typename T>
class ToTrigStateConverter : public LeafSystem<T> {
 public:
  ToTrigStateConverter();

  template <typename U>
  explicit ToTrigStateConverter(const ToTrigStateConverter<U>&)
      : ToTrigStateConverter<T>() {}

  ~ToTrigStateConverter(){};

 private:
  void CalcTrigState(const Context<T>& context, BasicVector<T>* x_trig) const;
};

}  // namespace analysis
}  // namespace systems
}  // namespace drake
