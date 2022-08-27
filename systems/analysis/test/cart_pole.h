#pragma once
/**
 * The trigonometric state is [pos_x, sinθ, cosθ+1, vel_x, θdot]
 * Note that at pos_x = 0 and θ=π, vel_x = 0, θdot=0, the trigonometric state is
 * 0.
 */

#include <Eigen/Core>

#include "drake/common/eigen_types.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {

struct CartPoleParams {
  double mc{10};
  double mp{1};
  double l{0.5};
  double gravity{9.81};
};

template <typename T>
Eigen::Matrix<T, 5, 1> ToCartpoleTrigState(
    const Eigen::Ref<const Eigen::Matrix<T, 4, 1>>& x_orig) {
  Eigen::Matrix<T, 5, 1> x_trig;
  x_trig(0) = x_orig(0);
  using std::cos;
  using std::sin;
  x_trig(1) = sin(x_orig(1));
  x_trig(2) = cos(x_orig(1)) + 1;
  x_trig(3) = x_orig(2);
  x_trig(4) = x_orig(3);
  return x_trig;
}

template <typename T>
Matrix2<T> CartpoleMassMatrix(const CartPoleParams& params,
                      const Eigen::Ref<const Eigen::Matrix<T, 5, 1>>& x) {
  const T c = x(2) - 1;
  Matrix2<T> M;
  M << params.mc + params.mp, params.mp * params.l * c,
      params.mp * params.l * c, params.mp * params.l * params.l;
  return M;
}

template <typename T>
Vector2<T> CalcCartpoleBiasTerm(const CartPoleParams& params,
                        const Eigen::Ref<const Eigen::Matrix<T, 5, 1>>& x) {
  const T s = x(1);
  // C*v
  const Vector2<T> bias(-params.mp * params.l * s * x(4) * x(4), 0);
  return bias;
}

template <typename T>
Vector2<T> CalcCartpoleGravityVector(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<T, 5, 1>>& x) {
  const T& s = x(1);
  return Vector2<T>(0, -params.mp * params.gravity * params.l * s);
}

template <typename T>
void CartpoleTrigDynamics(const CartPoleParams& params,
                  const Eigen::Ref<const Eigen::Matrix<T, 5, 1>>& x, const T& u,
                  Eigen::Matrix<T, 5, 1>* n, T* d) {
  const Matrix2<T> M = CartpoleMassMatrix<T>(params, x);
  *d = M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1);
  const T s = x(1);
  const T c = x(2) - 1;
  (*n)(0) = x(3) * (*d);
  (*n)(1) = c * x(4) * (*d);
  (*n)(2) = -s * x(4) * (*d);
  Matrix2<T> M_adj;
  M_adj << M(1, 1), -M(1, 0), -M(0, 1), M(0, 0);
  n->template tail<2>() =
      M_adj * (Vector2<T>(u, 0) + CalcCartpoleGravityVector<T>(params, x) -
               CalcCartpoleBiasTerm<T>(params, x));
}

void TrigPolyDynamics(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* G, symbolic::Polynomial* d);

symbolic::Polynomial CartpoleStateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x);

controllers::LinearQuadraticRegulatorResult SynthesizeCartpoleTrigLqr(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<double, 5, 5>>& Q, double R);

template <typename T>
Vector3<T> CalcQdot(const Eigen::Ref<const Eigen::Matrix<T, 5, 1>>& x) {
  const T s = x(1);
  const T c = x(2) - 1;
  return Vector3<T>(x(3), c * x(4), -s * x(4));
}

template <typename T>
class CartpoleTrigStateConverter : public LeafSystem<T> {
 public:
  CartpoleTrigStateConverter();

  template <typename U>
  explicit CartpoleTrigStateConverter(const CartpoleTrigStateConverter<U>&)
      : CartpoleTrigStateConverter<T>() {}

  ~CartpoleTrigStateConverter(){};

 private:
  void CalcTrigState(const Context<T>& context, BasicVector<T>* x_trig) const;
};

class CartpoleClfController : public ClfController {
 public:
  CartpoleClfController(
      const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x,
      const Eigen::Ref<const Eigen::Matrix<symbolic::Polynomial, 5, 1>>& f,
      const Eigen::Ref<const Eigen::Matrix<symbolic::Polynomial, 5, 1>>& G,
      const symbolic::Polynomial& dynamics_denominator, symbolic::Polynomial V,
      double deriv_eps, double u_max);

 private:
  virtual void DoCalcControl(const Context<double>& context,
                             BasicVector<double>* output) const override;

  double u_max_;
};

void Simulate(const CartPoleParams& parameters,
              const Eigen::Matrix<symbolic::Variable, 5, 1>& x,
              const symbolic::Polynomial& clf, double u_bound, double deriv_eps,
              const Eigen::Vector4d& initial_state, double duration);
}  // namespace analysis
}  // namespace systems
}  // namespace drake
