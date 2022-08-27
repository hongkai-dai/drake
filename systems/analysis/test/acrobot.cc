#include "drake/systems/analysis/test/acrobot.h"

#include "drake/systems/framework/framework_common.h"

namespace drake {
namespace systems {
namespace analysis {
void TrigPolyDynamics(const examples::acrobot::AcrobotParams<double>& p,
                      const Eigen::Ref<const Vector6<symbolic::Variable>>& x,
                      Vector6<symbolic::Polynomial>* f,
                      Vector6<symbolic::Polynomial>* G,
                      symbolic::Polynomial* d) {
  const Vector6<symbolic::Expression> x_expr = x.cast<symbolic::Expression>();
  const Matrix2<symbolic::Expression> M =
      AcrobotMassMatrix<symbolic::Expression>(p, x_expr);
  const Vector2<symbolic::Expression> bias =
      AcrobotDynamicsBiasTerm<symbolic::Expression>(p, x_expr);
  *d = symbolic::Polynomial(M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1));
  const symbolic::Expression& s1 = x_expr(0);
  const symbolic::Expression& s2 = x_expr(2);
  const symbolic::Expression c1 = x(1) - 1;
  const symbolic::Expression c2 = x(3) + 1;
  const symbolic::Expression& theta1dot = x(4);
  const symbolic::Expression& theta2dot = x(5);
  (*f)(0) = symbolic::Polynomial(c1 * theta1dot) * (*d);
  (*f)(1) = symbolic::Polynomial(-s1 * theta1dot) * (*d);
  (*f)(2) = symbolic::Polynomial(c2 * theta2dot) * (*d);
  (*f)(3) = symbolic::Polynomial(-s2 * theta2dot) * (*d);

  Matrix2<symbolic::Expression> M_adj;
  M_adj << M(1, 1), -M(1, 0), -M(0, 1), M(0, 0);
  const Vector2<symbolic::Expression> f_tail_expr = M_adj * -bias;
  (*f)(4) = symbolic::Polynomial(f_tail_expr(0));
  (*f)(5) = symbolic::Polynomial(f_tail_expr(1));
  for (int i = 0; i < 4; ++i) {
    (*G)(i) = symbolic::Polynomial();
  }
  (*G)(4) = symbolic::Polynomial(M_adj(0, 1));
  (*G)(5) = symbolic::Polynomial(M_adj(1, 1));
}

Vector2<symbolic::Polynomial> AcrobotStateEqConstraints(
    const Eigen::Ref<const Vector6<symbolic::Variable>>& x) {
  // The constraints are x(0) * x(0) + (x(1) - 1) * (x(1) - 1) = 1
  // and x(2) * x(2) + (x(3) + 1) * (x(3) + 1) = 1
  return Vector2<symbolic::Polynomial>(
      symbolic::Polynomial(x(0) * x(0) + x(1) * x(1) - 2 * x(1)),
      symbolic::Polynomial(x(2) * x(2) + x(3) * x(3) + 2 * x(3)));
}

template <typename T>
AcrobotTrigStateConverter<T>::AcrobotTrigStateConverter()
    : LeafSystem<T>(SystemTypeTag<AcrobotTrigStateConverter>{}) {
  this->DeclareInputPort("state", systems::PortDataType::kVectorValued, 4);
  this->DeclareVectorOutputPort("x_trig", 6,
                                &AcrobotTrigStateConverter<T>::CalcTrigState);
}

template <typename T>
void AcrobotTrigStateConverter<T>::CalcTrigState(const Context<T>& context,
                                                 BasicVector<T>* x_trig) const {
  const Vector4<T> x_orig = this->get_input_port().Eval(context);
  x_trig->get_mutable_value() = ToAcrobotTrigState<T>(x_orig);
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
