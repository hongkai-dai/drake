#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * The state of this sytem is [pos_x, pos_y, sinθ, cosθ-1], the control is
 * [speed, θdot]
 */
template <typename T>
class DubinsCar : public LeafSystem<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DubinsCar);

  DubinsCar();

  /**
   * Scalar-converting copy constructor.
   */
  template <typename U>
  explicit DubinsCar(const DubinsCar<U>&);

  ~DubinsCar(){};

  const OutputPort<T>& get_state_output_port() const {
    return this->get_output_port(state_output_port_index_);
  }

 private:
  void DoCalcTimeDerivatives(
      const systems::Context<T>& context,
      systems::ContinuousState<T>* derivatives) const override;
  OutputPortIndex state_output_port_index_;
};

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Vector4<symbolic::Variable>>& x);
}  // namespace analysis
}  // namespace systems
}  // namespace drake
