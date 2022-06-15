#pragma once

#include <math.h>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/common/symbolic.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * The trigonometric state is (qw-1, qx, qy, qz, p_WB, v_WB, w_WB_B)
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
}  // namespace analysis
}  // namespace systems
}  // namespace drake
