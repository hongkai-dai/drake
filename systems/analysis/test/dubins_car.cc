#include "drake/systems/analysis/test/dubins_car.h"

namespace drake {
namespace systems {
namespace analysis {
template <typename T>
DubinsCar<T>::DubinsCar() : LeafSystem<T>(systems::SystemTypeTag<DubinsCar>{}) {
  this->DeclareVectorInputPort("input", 2);

  auto state_index = this->DeclareContinuousState(4);
  state_output_port_index_ =
      this->DeclareStateOutputPort("x", state_index).get_index();
}

template <typename T>
template <typename U>
DubinsCar<T>::DubinsCar(const DubinsCar<U>&) : DubinsCar() {}

template <typename T>
void DubinsCar<T>::DoCalcTimeDerivatives(
    const systems::Context<T>& context,
    systems::ContinuousState<T>* derivatives) const {
  const auto x = context.get_continuous_state_vector().CopyToVector();
  const auto u = this->EvalVectorInput(context, 0)->CopyToVector();
  Vector4<T> xdot;
  const T s = x(2);
  const T c = x(3) + 1;
  xdot(0) = u(0) * c;
  xdot(1) = u(0) * s;
  xdot(2) = c * u(1);
  xdot(3) = -s * u(1);
  derivatives->SetFromVector(xdot);
}

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Vector4<symbolic::Variable>>& x) {
  return symbolic::Polynomial((x(2) * x(2) + x(3) * x(3) + 2 * x(3)));
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::analysis::DubinsCar)
