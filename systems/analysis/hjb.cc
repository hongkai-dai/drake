#include "drake/systems/analysis/hjb.h"

#include <iostream>

#include "drake/solvers/mathematical_program.h"
#include "drake/systems/analysis/clf_cbf_utils.h"

namespace drake {
namespace systems {
namespace analysis {
HjbUpper::HjbUpper(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    symbolic::Polynomial l, const Eigen::Ref<const Eigen::MatrixXd>& R,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& f,
    const Eigen::Ref<const MatrixX<symbolic::Polynomial>>& G,
    const std::optional<symbolic::Polynomial>& dynamics_denominator,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& state_eq_constraints)
    : x_{x},
      x_set_{x_},
      l_{std::move(l)},
      R_{R},
      llt_R_{R_},
      f_{f},
      G_{G},
      dynamics_denominator_{dynamics_denominator},
      state_eq_constraints_{state_eq_constraints} {
  R_inv_ = llt_R_.solve(Eigen::MatrixXd::Ones(G_.cols(), G_.cols()));
  const int nx = x_.rows();
  DRAKE_DEMAND(f_.rows() == nx);
  DRAKE_DEMAND(G_.rows() == nx);
  // l(0) = 0
  DRAKE_DEMAND(l_.monomial_to_coefficient_map().count(symbolic::Monomial()) ==
               0);
}

std::unique_ptr<solvers::MathematicalProgram> HjbUpper::ConstructJupperProgram(
    int J_degree, const Eigen::Ref<const Eigen::MatrixXd>& x_samples,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& policy_numerator,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& cin,
    const std::vector<int>& r_degrees,
    const std::vector<int>& state_constraints_lagrangian_degrees,
    symbolic::Polynomial* J, VectorX<symbolic::Polynomial>* r,
    VectorX<symbolic::Polynomial>* state_constraints_lagrangian) const {
  DRAKE_DEMAND(policy_numerator.rows() == G_.cols());
  auto prog = std::make_unique<solvers::MathematicalProgram>();
  prog->AddIndeterminates(x_);
  VectorX<symbolic::Monomial> J_monomial_basis;
  MatrixX<symbolic::Expression> J_gram;
  NewSosPolynomialPassOrigin(prog.get(), x_set_, J_degree,
                             symbolic::internal::DegreeType::kAny, J,
                             &J_monomial_basis, &J_gram);

  // Evaluate J on x_samples;
  Eigen::MatrixXd J_samples_A;
  Eigen::VectorXd J_samples_b;
  VectorX<symbolic::Variable> J_samples_vars;
  J->EvaluateWithAffineCoefficients(x_, x_samples, &J_samples_A,
                                    &J_samples_vars, &J_samples_b);
  prog->AddLinearCost(J_samples_A.colwise().sum().transpose(),
                      J_samples_b.sum(), J_samples_vars);

  const symbolic::Polynomial d_poly =
      dynamics_denominator_.value_or(symbolic::Polynomial(1));
  symbolic::Polynomial sos_condition = -(
      l_ * d_poly * d_poly + 0.5 * policy_numerator.dot(R_ * policy_numerator));
  sos_condition -= J->Jacobian(x_).dot(f_ * d_poly + G_ * policy_numerator);
  r->resize(cin.rows());
  DRAKE_DEMAND(cin.rows() == static_cast<int>(r_degrees.size()));
  for (int i = 0; i < cin.rows(); ++i) {
    std::tie((*r)(i), std::ignore) = prog->NewSosPolynomial(
        x_set_, r_degrees[i],
        solvers::MathematicalProgram::NonnegativePolynomial::kSos,
        "r" + std::to_string(i));
  }
  sos_condition += r->dot(cin);
  state_constraints_lagrangian->resize(state_eq_constraints_.rows());
  DRAKE_DEMAND(static_cast<int>(state_constraints_lagrangian_degrees.size()) ==
               state_eq_constraints_.rows());
  for (int i = 0; i < state_eq_constraints_.rows(); ++i) {
    (*state_constraints_lagrangian)(i) = prog->NewFreePolynomial(
        x_set_, state_constraints_lagrangian_degrees[i]);
  }
  sos_condition -= state_constraints_lagrangian->dot(state_eq_constraints_);
  prog->AddSosConstraint(sos_condition);
  return prog;
}

VectorX<symbolic::Polynomial> HjbUpper::ComputePolicyNumerator(
    const symbolic::Polynomial& J) const {
  return -(R_inv_ * J.Jacobian(x_) * G_).transpose();
}

HjbController::HjbController(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& policy_numerator,
    symbolic::Polynomial policy_denominator)
    : LeafSystem<double>(),
      x_{x},
      policy_numerator_{policy_numerator},
      policy_denominator_{std::move(policy_denominator)} {
  const int nx = x_.rows();
  x_input_index_ = this->DeclareVectorInputPort("x", nx).get_index();
  const int nu = policy_numerator_.rows();
  control_output_index_ =
      this->DeclareVectorOutputPort("control", nu, &HjbController::CalcControl)
          .get_index();
}

void HjbController::CalcControl(const Context<double>& context,
                                BasicVector<double>* output) const {
  const Eigen::VectorXd x_val = this->x_input_port().Eval(context);
  symbolic::Environment env;
  env.insert(x_, x_val);
  const double policy_denominator_val = policy_denominator_.Evaluate(env);
  for (int i = 0; i < policy_numerator_.rows(); ++i) {
    output->get_mutable_value()(i) =
        policy_numerator_(i).Evaluate(env) / policy_denominator_val;
  }
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake
