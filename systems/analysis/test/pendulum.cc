#include "drake/systems/analysis/test/pendulum.h"

#include "drake/math/autodiff_gradient.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace systems {
namespace analysis {
void ControlAffineDynamics(
    const examples::pendulum::PendulumPlant<double>& pendulum,
    const Vector2<symbolic::Variable>& x, double theta_des,
    Vector2<symbolic::Polynomial>* f, Vector2<symbolic::Polynomial>* G) {
  auto context = pendulum.CreateDefaultContext();
  const auto& p = pendulum.get_parameters(*context);

  (*G)(0) = symbolic::Polynomial();
  (*G)(1) = symbolic::Polynomial(1 / (p.mass() * p.length() * p.length()));
  (*f)(0) = symbolic::Polynomial(x(1));
  (*f)(1) = symbolic::Polynomial(
      (-p.mass() * p.gravity() * p.length() *
           (std::sin(theta_des) + std::cos(theta_des) * x(0) -
            std::sin(theta_des) / 2 * pow(x(0), 2) -
            std::cos(theta_des) / 6 * pow(x(0), 3)) -
       p.damping() * x(1)) /
      (p.mass() * p.length() * p.length()));
}

void TrigPolyDynamics(const examples::pendulum::PendulumPlant<double>& pendulum,
                      const Vector3<symbolic::Variable>& x, double theta_des,
                      Vector3<symbolic::Polynomial>* f,
                      Vector3<symbolic::Polynomial>* G) {
  const double sin_theta_des = std::sin(theta_des);
  const double cos_theta_des = std::cos(theta_des);
  auto context = pendulum.CreateDefaultContext();
  const auto& p = pendulum.get_parameters(*context);
  (*f)(0) = symbolic::Polynomial((x(1) + cos_theta_des) * x(2));
  (*f)(1) = symbolic::Polynomial((-x(0) - sin_theta_des) * x(2));
  (*f)(2) = symbolic::Polynomial(
      -p.gravity() / p.length() * (x(0) + sin_theta_des) -
      p.damping() * x(2) / (p.mass() * p.length() * p.length()));
  (*G)(0) = symbolic::Polynomial();
  (*G)(1) = symbolic::Polynomial();
  (*G)(2) = symbolic::Polynomial(1 / (p.mass() * p.length() * p.length()));
}

controllers::LinearQuadraticRegulatorResult TrigDynamicsLQR(
    const examples::pendulum::PendulumPlant<double>& pendulum, double theta_des,
    const Eigen::Ref<const Eigen::Matrix3d>& Q,
    const Eigen::Ref<const Vector1d>& R, Eigen::Matrix3d* A,
    Eigen::Vector3d* B) {
  // First get the linearized dynamics.
  const double u_des = EquilibriumTorque(pendulum, theta_des);
  const Eigen::Vector4d xu_des(0, 0, 0, u_des);
  const auto xu_des_ad = math::InitializeAutoDiff(xu_des);
  const auto xdot_ad = TrigDynamics<AutoDiffXd>(pendulum, xu_des_ad.head<3>(),
                                                theta_des, xu_des_ad.tail<1>());
  const auto xdot_grad = math::ExtractGradient(xdot_ad);
  // The constraint is (x(0) + sin(theta_des))² + (x(1) + cos(theta_des))² = 1
  // The linearized constraint is sin(theta_des) * x(0) + cos(theta_des) * x(1)
  // = 0
  const Eigen::RowVector3d F(std::sin(theta_des), std::cos(theta_des), 0);
  if (A != nullptr) {
    *A = xdot_grad.leftCols<3>();
  }
  if (B != nullptr) {
    *B = xdot_grad.rightCols<1>();
  }
  return controllers::LinearQuadraticRegulator(xdot_grad.leftCols<3>(),
                                               xdot_grad.rightCols<1>(), Q, R,
                                               Eigen::MatrixXd(0, 0), F);
}

double EquilibriumTorque(
    const examples::pendulum::PendulumPlant<double>& pendulum,
    double theta_des) {
  auto context = pendulum.CreateDefaultContext();
  const auto& p = pendulum.get_parameters(*context);
  const double u_des =
      p.mass() * p.gravity() * p.length() * std::sin(theta_des);
  return u_des;
}

PendulumTrigStateConverter::PendulumTrigStateConverter(double theta_des)
    : LeafSystem<double>(),
      sin_theta_des(std::sin(theta_des)),
      cos_theta_des(std::cos(theta_des)) {
  this->DeclareVectorInputPort("theta_thetadot", 2);
  this->DeclareVectorOutputPort("x", 3, &PendulumTrigStateConverter::Convert);
}

void PendulumTrigStateConverter::Convert(const Context<double>& context,
                                         BasicVector<double>* x) const {
  const Eigen::Vector2d theta_thetadot = this->get_input_port().Eval(context);
  const double theta = theta_thetadot(0);
  x->get_mutable_value() =
      Eigen::Vector3d(std::sin(theta) - sin_theta_des,
                      std::cos(theta) - cos_theta_des, theta_thetadot(1));
}

}  // namespace analysis
}  // namespace systems
}  // namespace drake
