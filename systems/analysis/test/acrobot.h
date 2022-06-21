#pragma once
/**
 * Define the dynamics on the trigonometric states of the acrobot.
 * The states are (sinθ₁, cosθ₁−1, sinθ₂, cosθ₂−1, θ₁_dot, θ₂_dot)
 */

#include "examples/acrobot/_virtual_includes/acrobot_input/drake/examples/acrobot/gen/acrobot_input.h"

#include "drake/examples/acrobot/acrobot_plant.h"

namespace drake {
namespace systems {
namespace analysis {

template <typename T>
Vector6<T> ToTrigState(const Eigen::Ref<const Vector4<T>>& x_orig) {
  Vector6<T> x_trig;
  using std::cos;
  using std::sin;
  x_trig(0) = sin(x_orig(0));
  x_trig(1) = cos(x_orig(0)) - 1;
  x_trig(2) = sin(x_orig(1));
  x_trig(3) = cos(x_orig(1)) - 1;
  x_trig.template tail<2>() = x_orig.template tail<2>();
  return x_trig;
}

template <typename T>
Matrix2<T> MassMatrix(const examples::acrobot::AcrobotParams<double>& p,
                      const Eigen::Ref<const Vector6<T>>& x) {
  const T c2 = x(3) + 1;
  const T I1 = p.Ic1() + p.m1() * p.lc1() * p.lc1();
  const T I2 = p.Ic2() + p.m2() * p.lc2() * p.lc2();
  const T m2l1lc2 = p.m2() * p.l1() * p.lc2();
  const T m12 = I2 + m2l1lc2 * c2;
  Matrix2<T> M;
  M << I1 + I2 + p.m2() * p.l1() * p.l1() + 2 * m2l1lc2 * c2, m12, m12, I2;
  return M;
}

template <typename T>
Vector2<T> DynamicsBiasTerm(const examples::acrobot::AcrobotParams<double>& p,
                            const Eigen::Ref<const Vector6<T>>& x) {
  const T& s1 = x(0);
  const T& s2 = x(2);
  const T c1 = x(1) + 1;
  const T c2 = x(3) + 1;
  const T s12 = s1 * c2 + s2 * c1;
  const T theta1dot = x(4);
  const T theta2dot = x(5);
  const T m2l1lc2 = p.m2() * p.l1() * p.lc2();

  Vector2<T> bias;
  // C(q,v)*v terms.
  bias << -2 * m2l1lc2 * s2 * theta2dot * theta1dot +
              -m2l1lc2 * s2 * theta2dot * theta2dot,
      m2l1lc2 * s2 * theta1dot * theta1dot;

  // -τ_g(q) terms.
  bias(0) += p.gravity() * p.m1() * p.lc1() * s1 +
             p.gravity() * p.m2() * (p.l1() * s1 + p.lc2() * s12);
  bias(1) += p.gravity() * p.m2() * p.lc2() * s12;

  // Damping terms.
  bias(0) += p.b1() * theta1dot;
  bias(1) += p.b2() * theta2dot;

  return bias;
}

/**
 * Compute the dynamics as xdot = n(x, u) / d(x)
 * where x is the trigonometric state
 * (sinθ₁, cosθ₁−1, sinθ₂, cosθ₂−1, θ₁_dot, θ₂_dot)
 */
template <typename T>
void TrigDynamics(const examples::acrobot::AcrobotParams<double>& p,
                  const Eigen::Ref<const Vector6<T>>& x, const T& u,
                  Vector6<T>* n, T* d) {
  const Matrix2<T> M = MassMatrix<T>(p, x);
  const Vector2<T> bias = DynamicsBiasTerm<T>(p, x);
  const T det_M = M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1);
  const T& s1 = x(0);
  const T& s2 = x(2);
  const T c1 = x(1) + 1;
  const T c2 = x(3) + 1;
  const T& theta1dot = x(4);
  const T& theta2dot = x(5);
  (*n)(0) = c1 * theta1dot * det_M;
  (*n)(1) = -s1 * theta1dot * det_M;
  (*n)(2) = c2 * theta2dot * det_M;
  (*n)(3) = -s2 * theta2dot * det_M;

  Matrix2<T> M_adj;
  M_adj << M(1, 1), -M(1, 0), -M(0, 1), M(0, 0);
  n->template tail<2>() = M_adj * (Vector2<T>(0, u) - bias);
  *d = det_M;
}

/**
 * Write the dynamics in the affine form
 * xdot = f(x) / d(x) + G(x) / d(x) * u
 */
void TrigPolyDynamics(const examples::acrobot::AcrobotParams<double>& p,
                      const Eigen::Ref<const Vector6<symbolic::Variable>>& x,
                      Vector6<symbolic::Polynomial>* f,
                      Vector6<symbolic::Polynomial>* G,
                      symbolic::Polynomial* d);

Vector2<symbolic::Polynomial> StateEqConstraints(
    const Eigen::Ref<const Vector6<symbolic::Variable>>& x);
}  // namespace analysis
}  // namespace systems
}  // namespace drake
