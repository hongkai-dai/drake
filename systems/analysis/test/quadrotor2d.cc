#include "drake/systems/analysis/test/quadrotor2d.h"

#include "drake/math/autodiff_gradient.h"

namespace drake {
namespace systems {
namespace analysis {
template <typename T>
Quadrotor2dTrigPlant<T>::Quadrotor2dTrigPlant()
    : LeafSystem<T>(systems::SystemTypeTag<Quadrotor2dTrigPlant>{}) {
  this->DeclareVectorInputPort("u", 2);

  auto state_index = this->DeclareContinuousState(6);

  state_output_port_index_ =
      this->DeclareStateOutputPort("x", state_index).get_index();
}

template <typename T>
template <typename U>
Quadrotor2dTrigPlant<T>::Quadrotor2dTrigPlant(const Quadrotor2dTrigPlant<U>&)
    : Quadrotor2dTrigPlant() {}

template <typename T>
void Quadrotor2dTrigPlant<T>::DoCalcTimeDerivatives(
    const systems::Context<T>& context,
    systems::ContinuousState<T>* derivatives) const {
  const auto x = context.get_continuous_state_vector().CopyToVector();
  const auto u = this->EvalVectorInput(context, 0)->CopyToVector();
  const T theta = x(2);
  Vector6<T> xdot;
  xdot.template head<3>() = x.template tail<3>();
  using std::cos;
  using std::sin;
  xdot(3) = -sin(theta) / mass_ * (u(0) + u(1));
  xdot(4) = cos(theta) / mass_ * (u(0) + u(1)) - gravity_;
  xdot(5) = length_ / inertia_ * (u(0) - u(1));
  derivatives->SetFromVector(xdot);
}

void TrigPolyDynamics(
    const Quadrotor2dTrigPlant<double>& quadrotor,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 7, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 7, 2>* G) {
  (*f)(0) = symbolic::Polynomial(x(4));
  (*f)(1) = symbolic::Polynomial(x(5));
  (*f)(2) = symbolic::Polynomial((x(3) + 1) * x(6));
  (*f)(3) = symbolic::Polynomial(-x(2) * x(6));
  (*f)(4) = symbolic::Polynomial();
  (*f)(5) = symbolic::Polynomial(-quadrotor.gravity());
  (*f)(6) = symbolic::Polynomial();
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      (*G)(i, j) = symbolic::Polynomial();
    }
  }
  (*G)(4, 0) = symbolic::Polynomial(-x(2) / quadrotor.mass());
  (*G)(4, 1) = (*G)(4, 0);
  (*G)(5, 0) = symbolic::Polynomial((x(3) + 1) / quadrotor.mass());
  (*G)(5, 1) = (*G)(5, 0);
  (*G)(6, 0) = symbolic::Polynomial(quadrotor.length() / quadrotor.inertia());
  (*G)(6, 1) = -(*G)(6, 0);
}

void TrigPolyDynamicsTwinQuadrotor(
    const Quadrotor2dTrigPlant<double>& quadrotor,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 12, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 12, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 12, 4>* G) {
  (*f)(0) = symbolic::Polynomial((x(1) + 1) * x(4));
  (*f)(1) = symbolic::Polynomial(-x(0) * x(4));
  (*f)(2) = symbolic::Polynomial();
  (*f)(3) = symbolic::Polynomial(-quadrotor.gravity());
  (*f)(4) = symbolic::Polynomial();
  (*f)(5) = symbolic::Polynomial(x(9) - x(2));
  (*f)(6) = symbolic::Polynomial(x(10) - x(3));
  (*f)(7) = symbolic::Polynomial((x(8) + 1) * x(11));
  (*f)(8) = symbolic::Polynomial(-x(7) * x(11));
  (*f)(9) = symbolic::Polynomial();
  (*f)(10) = symbolic::Polynomial(-quadrotor.gravity());
  (*f)(11) = symbolic::Polynomial();
  for (int i = 0; i < 12; ++i) {
    for (int j = 0; j < 4; ++j) {
      (*G)(i, j) = symbolic::Polynomial();
    }
  }
  (*G)(2, 0) = symbolic::Polynomial(-x(0) / quadrotor.mass());
  (*G)(2, 1) = (*G)(2, 0);
  (*G)(3, 0) = symbolic::Polynomial((x(1) + 1) / quadrotor.mass());
  (*G)(3, 1) = (*G)(3, 0);
  (*G)(4, 0) = symbolic::Polynomial(quadrotor.length() / quadrotor.inertia());
  (*G)(4, 1) = -(*G)(4, 0);
  (*G)(9, 2) = symbolic::Polynomial(-x(7) / quadrotor.mass());
  (*G)(9, 3) = (*G)(9, 2);
  (*G)(10, 2) = symbolic::Polynomial((x(8) + 1) / quadrotor.mass());
  (*G)(10, 3) = (*G)(10, 2);
  (*G)(11, 2) = symbolic::Polynomial(quadrotor.length() / quadrotor.inertia());
  (*G)(11, 3) = -(*G)(11, 2);
}

double EquilibriumThrust(const Quadrotor2dTrigPlant<double>& quadrotor) {
  return quadrotor.mass() * quadrotor.gravity() / 2;
}

void PolynomialControlAffineDynamics(
    const Quadrotor2dTrigPlant<double>& quadrotor,
    const Vector6<symbolic::Variable>& x, Vector6<symbolic::Polynomial>* f,
    Eigen::Matrix<symbolic::Polynomial, 6, 2>* G) {
  for (int i = 0; i < 3; ++i) {
    (*f)(i) = symbolic::Polynomial(symbolic::Monomial(x(i + 3)));
    (*G)(i, 0) = symbolic::Polynomial();
    (*G)(i, 1) = symbolic::Polynomial();
  }
  // Use taylor expansion for sin and cos around theta=0.
  const symbolic::Polynomial s2{
      {{symbolic::Monomial(x(2)), 1}, {symbolic::Monomial(x(2), 3), -1. / 6}}};
  const symbolic::Polynomial c2{
      {{symbolic::Monomial(), 1}, {symbolic::Monomial(x(2), 2), -1. / 2}}};
  (*f)(3) = symbolic::Polynomial();
  (*f)(4) = symbolic::Polynomial(-quadrotor.gravity());
  (*f)(5) = symbolic::Polynomial();
  (*G)(3, 0) = -s2 / quadrotor.mass();
  (*G)(3, 1) = (*G)(3, 0);
  (*G)(4, 0) = c2 / quadrotor.mass();
  (*G)(4, 1) = (*G)(4, 0);
  (*G)(5, 0) = symbolic::Polynomial(quadrotor.length() / quadrotor.inertia());
  (*G)(5, 1) = -(*G)(5, 0);
}

symbolic::Polynomial Quadrotor2dStateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 7, 1>>& x) {
  return symbolic::Polynomial(x(2) * x(2) + x(3) * x(3) + 2 * x(3));
}

Vector2<symbolic::Polynomial> TwinQuadrotor2dStateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 12, 1>>& x) {
  return Vector2<symbolic::Polynomial>(
      symbolic::Polynomial(x(0) * x(0) + x(1) * x(1) + 2 * x(1)),
      symbolic::Polynomial(x(7) * x(7) + x(8) * x(8) + 2 * x(8)));
}

controllers::LinearQuadraticRegulatorResult SynthesizeQuadrotor2dTrigLqr(
    const Eigen::Ref<const Eigen::Matrix<double, 7, 7>>& Q,
    const Eigen::Ref<const Eigen::Matrix2d>& R) {
  Quadrotor2dTrigPlant<double> quadrotor;
  const double thrust_equilibrium = EquilibriumThrust(quadrotor);
  Eigen::VectorXd xu_des = Eigen::VectorXd::Zero(9);
  xu_des.tail<2>() = Eigen::Vector2d(thrust_equilibrium, thrust_equilibrium);
  const auto xu_des_ad = math::InitializeAutoDiff(xu_des);
  const auto xdot_des_ad = TrigDynamics<AutoDiffXd>(
      quadrotor, xu_des_ad.head<7>(), xu_des_ad.tail<2>());
  const auto xdot_des_grad = math::ExtractGradient(xdot_des_ad);
  // The constraint is x(2) * x(2) + (x(3) + 1) * (x(3) + 1) = 1.
  Eigen::RowVectorXd F = Eigen::RowVectorXd::Zero(7);
  F(3) = 1;
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      xdot_des_grad.leftCols<7>(), xdot_des_grad.rightCols<2>(), Q, R,
      Eigen::MatrixXd(0, 2), F);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(lqr_result.S);
  return lqr_result;
}

template <typename T>
Quadrotor2dTrigStateConverter<T>::Quadrotor2dTrigStateConverter()
    : LeafSystem<T>(SystemTypeTag<Quadrotor2dTrigStateConverter>{}) {
  this->DeclareVectorInputPort("state", 6);
  this->DeclareVectorOutputPort(
      "x_trig", 7, &Quadrotor2dTrigStateConverter<T>::CalcTrigState);
}

template <typename T>
void Quadrotor2dTrigStateConverter<T>::CalcTrigState(
    const Context<T>& context, BasicVector<T>* x_trig) const {
  const Vector6<T> x_orig = this->get_input_port().Eval(context);
  x_trig->get_mutable_value() = ToQuadrotor2dTrigState<T>(x_orig);
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::analysis::Quadrotor2dTrigPlant)
DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_SCALARS(
    class ::drake::systems::analysis::Quadrotor2dTrigStateConverter)
