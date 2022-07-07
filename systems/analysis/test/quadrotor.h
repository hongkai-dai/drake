#pragma once

#include <math.h>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * The trigonometric state is (qw-1, qx, qy, qz, p_WB, v_WB, omega_WB_B)
 */
template <typename T>
class QuadrotorTrigPlant : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(QuadrotorTrigPlant);

  QuadrotorTrigPlant();

  /**
   * Scalar-converting copy constructor.
   */
  template <typename U>
  explicit QuadrotorTrigPlant(const QuadrotorTrigPlant<U>&);

  ~QuadrotorTrigPlant(){};

  double mass() const { return mass_; }

  double length() const { return length_; }

  const Eigen::Matrix3d& inertia() const { return inertia_; }

  double gravity() const { return gravity_; }

  double kF() const { return kF_; }

  double kM() const { return kM_; }

  const OutputPort<T>& get_state_output_port() const {
    return this->get_output_port(state_output_port_index_);
  }

 private:
  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;

  double length_;
  double mass_;
  Eigen::Matrix3d inertia_;
  double kF_;
  double kM_;
  double gravity_;
  OutputPortIndex state_output_port_index_;
};

template <typename T>
double EquilibriumThrust(const QuadrotorTrigPlant<T>& quadrotor) {
  return quadrotor.mass() * quadrotor.gravity() / (4 * quadrotor.kF());
}

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 13, 1>>& x);

/**
 * Convert from [p_WB, rpy, v_WB, rpyDt] to [qw-1, qx, qy, qz, p_WB, v_WB,
 * omega_WB_B]
 */
template <typename T>
Eigen::Matrix<T, 13, 1> ToTrigState(
    const Eigen::Ref<const Eigen::Matrix<T, 12, 1>>& x_original) {
  Eigen::Matrix<T, 13, 1> x_trig;
  const math::RollPitchYaw<T> rpy(x_original.template segment<3>(3));
  const Eigen::Quaternion<T> quaternion = rpy.ToQuaternion();
  x_trig(0) = quaternion.w() - T(1);
  x_trig(1) = quaternion.x();
  x_trig(2) = quaternion.y();
  x_trig(3) = quaternion.z();
  x_trig.template segment<3>(4) = x_original.template head<3>();
  x_trig.template segment<3>(7) = x_original.template segment<3>(6);
  x_trig.template tail<3>() =
      rpy.CalcAngularVelocityInChildFromRpyDt(x_original.template tail<3>());
  return x_trig;
}

void TrigPolyDynamics(
    const QuadrotorTrigPlant<double>& plant,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 13, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 13, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 13, 4>* G);
}  // namespace analysis
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::analysis::QuadrotorTrigPlant)
