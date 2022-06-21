#include "drake/systems/analysis/test/quadrotor.h"

namespace drake {
namespace systems {
namespace analysis {
Eigen::Matrix3d default_inertia() {
  // clang-format off
  return (Eigen::Matrix3d() <<  0.0015, 0, 0,          
                                0, 0.0025, 0,  
                                0, 0, 0.0035)
                            .finished();
  // clang-format on
}

template <typename T>
QuadrotorTrigPlant<T>::QuadrotorTrigPlant()
    : LeafSystem<T>(systems::SystemTypeTag<QuadrotorTrigPlant>{}),
      length_{0.15},
      mass_{0.775},
      inertia_{default_inertia()},
      kF_{1.},
      kM_{0.0245},
      gravity_{9.81} {
  this->DeclareVectorInputPort("propeller_force", 4);

  auto state_index = this->DeclareContinuousState(13);
  state_output_port_index_ =
      this->DeclareStateOutputPort("x", state_index).get_index();
}

template <typename T>
template <typename U>
QuadrotorTrigPlant<T>::QuadrotorTrigPlant(const QuadrotorTrigPlant<U>&)
    : QuadrotorTrigPlant() {}

template <typename T>
void QuadrotorTrigPlant<T>::DoCalcTimeDerivatives(
    const systems::Context<T>& context,
    systems::ContinuousState<T>* derivatives) const {
  const auto x = context.get_continuous_state_vector().CopyToVector();
  const auto u = this->EvalVectorInput(context, 0)->CopyToVector();

  const Vector4<T> uF_Bz = kF_ * u;

  const Vector3<T> Faero_B(0, 0, uF_Bz.sum());
  const T Mx = length_ * (uF_Bz(1) - uF_Bz(3));
  const T My = length_ * (uF_Bz(2) - uF_Bz(0));
  const Vector4<T> uTau_Bz = kM_ * u;
  const T Mz = uTau_Bz(0) - uTau_Bz(1) + uTau_Bz(2) - uTau_Bz(3);

  const Vector3<T> Tau_B(Mx, My, Mz);
  const Vector3<T> Fgravity_N(0, 0, -mass_ * gravity_);

  const Vector4<T> quat(x(0) + 1, x(1), x(2), x(3));
  const auto w_NB_B = x.template tail<3>();
  Vector4<T> quat_dot;
  quat_dot(0) =
      0.5 * (-w_NB_B(0) * quat(1) - w_NB_B(1) * quat(2) - w_NB_B(2) * quat(3));
  quat_dot(1) =
      0.5 * (w_NB_B(0) * quat(0) + w_NB_B(2) * quat(2) - w_NB_B(1) * quat(3));
  quat_dot(2) =
      0.5 * (w_NB_B(1) * quat(0) - w_NB_B(2) * quat(1) + w_NB_B(0) * quat(3));
  quat_dot(3) =
      0.5 * (w_NB_B(2) * quat(0) + w_NB_B(1) * quat(1) - w_NB_B(0) * quat(2));

  Matrix3<T> R_NB;
  R_NB(0, 0) = 1 - 2 * quat(2) * quat(2) - 2 * quat(3) * quat(3);
  R_NB(0, 1) = 2 * quat(1) * quat(2) - 2 * quat(0) * quat(3);
  R_NB(0, 2) = 2 * quat(1) * quat(3) + 2 * quat(0) * quat(2);
  R_NB(1, 0) = 2 * quat(1) * quat(2) + 2 * quat(0) * quat(3);
  R_NB(1, 1) = 1 - 2 * quat(1) * quat(1) - 2 * quat(3) * quat(3);
  R_NB(1, 2) = 2 * quat(2) * quat(3) - 2 * quat(0) * quat(1);
  R_NB(2, 0) = 2 * quat(1) * quat(3) - 2 * quat(0) * quat(2);
  R_NB(2, 1) = 2 * quat(2) * quat(3) + 2 * quat(0) * quat(1);
  R_NB(2, 2) = 1 - 2 * quat(1) * quat(1) - 2 * quat(2) * quat(2);

  const Vector3<T> xyzDDt = (Fgravity_N + R_NB * Faero_B) / mass_;

  const Vector3<T> wIw = w_NB_B.cross(inertia_ * w_NB_B);
  const Vector3<T> alpha_NB_B((Tau_B(0) - wIw(0)) / inertia_(0, 0),
                              (Tau_B(1) - wIw(1)) / inertia_(1, 1),
                              (Tau_B(2) - wIw(2)) / inertia_(2, 2));
  Eigen::Matrix<T, 13, 1> xDt;
  xDt << quat_dot, x.template segment<3>(7), xyzDDt, alpha_NB_B;
  derivatives->SetFromVector(xDt);
}

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 13, 1>>& x) {
  return symbolic::Polynomial(x(0) * x(0) + 2 * x(0) + x(1) * x(1) +
                              x(2) * x(2) + x(3) * x(3));
}

void TrigPolyDynamics(
    const QuadrotorTrigPlant<double>& quadrotor,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 13, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 13, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 13, 4>* G) {
  const Vector4<symbolic::Expression> quat(x(0) + 1, x(1), x(2), x(3));
  const auto w_NB_B = x.tail<3>();
  // quatDt
  (*f)(0) = symbolic::Polynomial(
      0.5 * (-w_NB_B(0) * quat(1) - w_NB_B(1) * quat(2) - w_NB_B(2) * quat(3)));
  (*f)(1) = symbolic::Polynomial(
      0.5 * (w_NB_B(0) * quat(0) + w_NB_B(2) * quat(2) - w_NB_B(1) * quat(3)));
  (*f)(2) = symbolic::Polynomial(
      0.5 * (w_NB_B(1) * quat(0) - w_NB_B(2) * quat(1) + w_NB_B(0) * quat(3)));
  (*f)(3) = symbolic::Polynomial(
      0.5 * (w_NB_B(2) * quat(0) + w_NB_B(1) * quat(1) - w_NB_B(0) * quat(2)));
  // v_WB
  (*f)(4) = symbolic::Polynomial(x(7));
  (*f)(5) = symbolic::Polynomial(x(8));
  (*f)(6) = symbolic::Polynomial(x(9));
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 4; ++j) {
      (*G)(i, j) = symbolic::Polynomial();
    }
  }

  // a_WB
  (*f)(7) = symbolic::Polynomial{};
  (*f)(8) = symbolic::Polynomial{};
  (*f)(9) = symbolic::Polynomial(-quadrotor.gravity());

  Matrix3<symbolic::Expression> R_NB;
  R_NB(0, 0) = 1 - 2 * quat(2) * quat(2) - 2 * quat(3) * quat(3);
  R_NB(0, 1) = 2 * quat(1) * quat(2) - 2 * quat(0) * quat(3);
  R_NB(0, 2) = 2 * quat(1) * quat(3) + 2 * quat(0) * quat(2);
  R_NB(1, 0) = 2 * quat(1) * quat(2) + 2 * quat(0) * quat(3);
  R_NB(1, 1) = 1 - 2 * quat(1) * quat(1) - 2 * quat(3) * quat(3);
  R_NB(1, 2) = 2 * quat(2) * quat(3) - 2 * quat(0) * quat(1);
  R_NB(2, 0) = 2 * quat(1) * quat(3) - 2 * quat(0) * quat(2);
  R_NB(2, 1) = 2 * quat(2) * quat(3) + 2 * quat(0) * quat(1);
  R_NB(2, 2) = 1 - 2 * quat(1) * quat(1) - 2 * quat(2) * quat(2);
  for (int i = 0; i < 3; ++i) {
    (*G)(i + 7, 0) =
        symbolic::Polynomial(R_NB(i, 2) * quadrotor.kF() / quadrotor.mass());
    for (int j = 1; j < 4; ++j) {
      (*G)(i + 7, j) = (*G)(i + 7, 0);
    }
  }

  // alpha_NB_B
  const Vector3<symbolic::Expression> wIw =
      w_NB_B.cross(quadrotor.inertia() * w_NB_B);
  (*f)(10) = symbolic::Polynomial(-wIw(0) / quadrotor.inertia()(0, 0));
  (*f)(11) = symbolic::Polynomial(-wIw(1) / quadrotor.inertia()(1, 1));
  (*f)(12) = symbolic::Polynomial(-wIw(2) / quadrotor.inertia()(2, 2));

  (*G)(10, 0) = symbolic::Polynomial{};
  (*G)(10, 1) = symbolic::Polynomial{quadrotor.kF() * quadrotor.length() /
                                     quadrotor.inertia()(0, 0)};
  (*G)(10, 2) = symbolic::Polynomial{};
  (*G)(10, 3) = -(*G)(10, 1);
  (*G)(11, 0) = symbolic::Polynomial(-quadrotor.kF() * quadrotor.length() /
                                     quadrotor.inertia()(1, 1));
  (*G)(11, 1) = symbolic::Polynomial{};
  (*G)(11, 2) = -(*G)(11, 0);
  (*G)(11, 3) = symbolic::Polynomial{};
  (*G)(12, 0) = symbolic::Polynomial(quadrotor.kF() * quadrotor.kM() /
                                     quadrotor.inertia()(2, 2));
  (*G)(12, 1) = -(*G)(12, 0);
  (*G)(12, 2) = (*G)(12, 0);
  (*G)(12, 3) = -(*G)(12, 0);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
