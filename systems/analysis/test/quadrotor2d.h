#pragma once

#include <math.h>

#include "drake/common/eigen_types.h"
#include "drake/common/symbolic.h"

namespace drake {
namespace systems {
namespace analysis {
struct Quadrotor {
  // This is the dynamics with un-normalized u.
  template <typename T>
  Vector6<T> CalcDynamics(const Vector6<T>& x, const Vector2<T>& u) const {
    Vector6<T> xdot;
    xdot.template head<3>() = x.template tail<3>();
    using std::cos;
    using std::sin;
    xdot.template tail<3>() << -sin(x(2)) / mass * (u(0) + u(1)),
        cos(x(2)) / mass * (u(0) + u(1)) - gravity,
        length / inertia * (u(0) - u(1));
    return xdot;
  }

  template <typename T>
  Vector2<T> NormalizeU(const Vector2<T>& u) const {
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    return (u - u_mid) / (u_max / 2);
  }

  template <typename T>
  Vector2<T> UnnormalizeU(const Vector2<T>& u_normalize) const {
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    return u_normalize * (u_max / 2) + u_mid;
  }

  // This assumes normalized u, namely -1 <= u <= 1
  template <typename T>
  void ControlAffineDynamics(const Vector6<T>& x, Vector6<T>* f,
                             Eigen::Matrix<T, 6, 2>* G) const {
    f->template head<3>() = x.template tail<3>();
    G->template topRows<3>().setZero();
    using std::cos;
    using std::sin;
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    const T s2 = sin(x(2));
    const T c2 = cos(x(2));
    (*f)(3) = -s2 / mass * (u_mid(0) + u_mid(1));
    (*f)(4) = -gravity + c2 / mass * (u_mid(0) + u_mid(1));
    (*f)(5) = length / inertia * (u_mid(0) - u_mid(1));
    (*G)(3, 0) = -s2 / mass * u_max / 2;
    (*G)(3, 1) = (*G)(3, 0);
    (*G)(4, 0) = c2 / mass * u_max / 2;
    (*G)(4, 1) = (*G)(4, 0);
    (*G)(5, 0) = length / inertia * u_max / 2;
    (*G)(5, 1) = -(*G)(5, 0);
  }

  // This assumes normalized u.
  void PolynomialControlAffineDynamics(
      const Vector6<symbolic::Variable>& x, Vector6<symbolic::Polynomial>* f,
      Eigen::Matrix<symbolic::Polynomial, 6, 2>* G) const {
    for (int i = 0; i < 3; ++i) {
      (*f)(i) = symbolic::Polynomial(symbolic::Monomial(x(i + 3)));
      (*G)(i, 0) = symbolic::Polynomial();
      (*G)(i, 1) = symbolic::Polynomial();
    }
    // Use taylor expansion for sin and cos around theta=0.
    const symbolic::Polynomial s2{{{symbolic::Monomial(x(2)), 1},
                                   {symbolic::Monomial(x(2), 3), -1. / 6}}};
    const symbolic::Polynomial c2{
        {{symbolic::Monomial(), 1}, {symbolic::Monomial(x(2), 2), -1. / 2}}};
    const Eigen::Vector2d u_mid(u_max / 2, u_max / 2);
    (*f)(3) = -s2 / mass * (u_mid(0) + u_mid(1));
    (*f)(4) = -gravity + c2 / mass * (u_mid(0) + u_mid(1));
    (*f)(5) = symbolic::Polynomial();
    (*G)(3, 0) = -s2 / mass * u_max / 2;
    (*G)(3, 1) = (*G)(3, 0);
    (*G)(4, 0) = c2 / mass * u_max / 2;
    (*G)(4, 1) = (*G)(4, 0);
    (*G)(5, 0) = symbolic::Polynomial(
        {{symbolic::Monomial(), length / inertia * u_max / 2}});
    (*G)(5, 1) = -(*G)(5, 0);
  }

  double length{0.25};
  double mass{0.486};
  double inertia{0.00383};
  double gravity{9.81};
  double u_max{mass * gravity * 2};
};

}  // namespace analysis
}  // namespace systems
}  // namespace drake
