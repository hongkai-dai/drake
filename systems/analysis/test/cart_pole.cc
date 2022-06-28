#include "drake/systems/analysis/test/cart_pole.h"

#include "drake/math/autodiff_gradient.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"

namespace drake {
namespace systems {
namespace analysis {
void TrigPolyDynamics(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* f,
    Eigen::Matrix<symbolic::Polynomial, 5, 1>* G, symbolic::Polynomial* d) {
  const Eigen::Matrix<symbolic::Expression, 5, 1> x_expr =
      x.cast<symbolic::Expression>();
  const Matrix2<symbolic::Expression> M_expr =
      MassMatrix<symbolic::Expression>(params, x_expr);
  *d = symbolic::Polynomial(M_expr(0, 0) * M_expr(1, 1) -
                            M_expr(1, 0) * M_expr(0, 1));
  const symbolic::Polynomial s(x(1));
  const symbolic::Polynomial c(x(2) - 1);
  (*f)(0) = x(3) * (*d);
  (*f)(1) = c * x(4) * (*d);
  (*f)(2) = -s * x(4) * (*d);

  Matrix2<symbolic::Expression> M_adj_expr;
  M_adj_expr << M_expr(1, 1), -M_expr(1, 0), -M_expr(0, 1), M_expr(0, 0);
  const Vector2<symbolic::Expression> f_tail_expr =
      M_adj_expr * (CalcGravityVector<symbolic::Expression>(params, x_expr) -
                    CalcBiasTerm<symbolic::Expression>(params, x_expr));
  (*f)(3) = symbolic::Polynomial(f_tail_expr(0));
  (*f)(4) = symbolic::Polynomial(f_tail_expr(1));
  (*G)(0) = symbolic::Polynomial();
  (*G)(1) = symbolic::Polynomial();
  (*G)(2) = symbolic::Polynomial();
  (*G)(3) = symbolic::Polynomial(M_adj_expr(0, 0));
  (*G)(4) = symbolic::Polynomial(M_adj_expr(1, 0));
}

symbolic::Polynomial StateEqConstraint(
    const Eigen::Ref<const Eigen::Matrix<symbolic::Variable, 5, 1>>& x) {
  return symbolic::Polynomial(x(1) * x(1) + x(2) * x(2) - 2 * x(2));
}

controllers::LinearQuadraticRegulatorResult SynthesizeTrigLqr(
    const CartPoleParams& params,
    const Eigen::Ref<const Eigen::Matrix<double, 5, 5>>& Q, double R) {
  const Eigen::Matrix<double, 6, 1> xu_des = Vector6d::Zero();
  const auto xu_des_ad = math::InitializeAutoDiff(xu_des);
  Eigen::Matrix<AutoDiffXd, 5, 1> n;
  AutoDiffXd d;
  TrigDynamics<AutoDiffXd>(params, xu_des_ad.head<5>(), xu_des_ad(5), &n, &d);
  const Eigen::Matrix<AutoDiffXd, 5, 1> xdot_ad = n / d;
  const auto xdot_grad = math::ExtractGradient(xdot_ad);
  Eigen::Matrix<double, 1, 5> F;
  F << 0, 0, 1, 0, 0;
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      xdot_grad.leftCols<5>(), xdot_grad.col(5), Q, Vector1d(R),
      Eigen::MatrixXd(0, 1), F);
  return lqr_result;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
