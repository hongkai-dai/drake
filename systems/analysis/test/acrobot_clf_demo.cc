#include <math.h>

#include <limits>

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/solvers/common_solver_option.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"
#include "drake/solvers/solve.h"
#include "drake/systems/analysis/control_lyapunov.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/vector_log_sink.h"

namespace drake {
namespace systems {
namespace analysis {
const double kInf = std::numeric_limits<double>::infinity();

// Given that the polynomial p(x) such that polynomial_values(i) is the value of
// the polynomial evaluated at indeterminates_values.col(i), fit the polynomial.
symbolic::Polynomial FitPolynomial(
    const VectorX<symbolic::Monomial>& monomials,
    const VectorX<symbolic::Variable>& indeterminates,
    const Eigen::MatrixXd& indeterminates_values,
    const Eigen::VectorXd& polynomial_values) {
  solvers::MathematicalProgram prog;
  const auto coeff_vars = prog.NewContinuousVariables(monomials.rows());
  const int num_samples = indeterminates_values.cols();
  Eigen::MatrixXd monomial_values(num_samples, coeff_vars.rows());
  for (int i = 0; i < coeff_vars.rows(); ++i) {
    monomial_values.col(i) =
        monomials(i).Evaluate(indeterminates, indeterminates_values);
  }
  prog.Add2NormSquaredCost(monomial_values, polynomial_values, coeff_vars);
  solvers::SolverOptions solver_options;
  // solver_options.SetOption(solvers::CommonSolverOption::kPrintToConsole, 1);
  solvers::MosekSolver mosek_solver;
  // const auto result = solvers::Solve(prog, std::nullopt, solver_options);
  solvers::MathematicalProgramResult result;
  mosek_solver.Solve(prog, std::nullopt, solver_options, &result);
  DRAKE_DEMAND(result.is_success());
  const auto coeff_sol = result.GetSolution(coeff_vars);
  symbolic::Polynomial::MapType poly_map;
  for (int i = 0; i < monomials.rows(); ++i) {
    poly_map.emplace(monomials(i), coeff_sol(i));
  }
  return symbolic::Polynomial(poly_map);
}

struct Acrobot {
  template <typename T>
  void ComputeDynamics(const Vector4<T>& x, Matrix2<T>* M, Matrix2<T>* C,
                       Vector2<T>* tau_g, Vector2<T>* B) const {
    using std::cos;
    using std::sin;
    const T s1 = sin(x(0));
    const T s2 = sin(x(1));
    const T c2 = cos(x(1));
    const T s12 = sin(x(0) + x(1));

    (*M)(0, 0) = I1 + I2 + m2 * l1 * l1 + 2 * m2 * l1 * lc2 * c2;
    (*M)(0, 1) = I2 + m2 * l1 * lc2 * c2;
    (*M)(1, 0) = (*M)(0, 1);
    (*M)(1, 1) = I2;

    (*C)(0, 0) = -2 * m2 * l1 * lc2 * s2 * x(3);
    (*C)(0, 1) = -m2 * l1 * lc2 * s2 * x(3);
    (*C)(1, 0) = m2 * l1 * lc2 * s2 * x(2);
    (*C)(1, 1) = 0;

    (*tau_g)(0) = -m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12);
    (*tau_g)(1) = -m2 * g * lc2 * s12;

    (*B) << 0, 1;
  }

  template <typename T>
  void ControlAffineDynamics(const Vector4<T>& x, Vector4<T>* f,
                             Vector4<T>* G) const {
    Matrix2<T> M;
    Matrix2<T> C;
    Vector2<T> tau_g;
    Vector2<T> B;
    ComputeDynamics(x, &M, &C, &tau_g, &B);
    (*f)(0) = x(2);
    (*f)(1) = x(3);
    f->template tail<2>() = M.inverse() * (tau_g - C * x.template tail<2>());
    (*G)(0) = 0;
    (*G)(1) = 0;
    G->template tail<2>() = M.inverse() * B * u_bound;
  }

  // @param f_q_order The order of q in f.
  // @param G_q_order The order of q in G.
  void FitControlAffineDynamics(int f_q_order, int G_q_order,
                                Vector4<symbolic::Variable>* x,
                                Vector4<symbolic::Polynomial>* f,
                                Vector4<symbolic::Polynomial>* G) const {
    for (int i = 0; i < 4; ++i) {
      (*x)(i) = symbolic::Variable(fmt::format("x{}", i));
    }
    (*f)(0) = symbolic::Polynomial((*x)(2));
    (*f)(1) = symbolic::Polynomial((*x)(3));
    (*G)(0) = symbolic::Polynomial();
    (*G)(1) = symbolic::Polynomial();

    const VectorX<symbolic::Monomial> f_q_monomial_basis =
        symbolic::OddDegreeMonomialBasis(symbolic::Variables(x->head<2>()),
                                         f_q_order);
    // f_monomial_basis include all f_q_monomial_basis, and the quadratic
    // monomial of qdot times monomials in f_q_monomial_basis that only contains
    // x(1).
    std::vector<symbolic::Monomial> f_monomial_basis_vec;
    for (int i = 0; i < f_q_monomial_basis.rows(); ++i) {
      f_monomial_basis_vec.push_back(f_q_monomial_basis(i));
      if (f_q_monomial_basis(i).degree((*x)(0)) == 0) {
        f_monomial_basis_vec.push_back(f_q_monomial_basis(i) *
                                       symbolic::Monomial((*x)(2), 2));
        f_monomial_basis_vec.push_back(f_q_monomial_basis(i) *
                                       symbolic::Monomial((*x)(3), 2));
        f_monomial_basis_vec.push_back(
            f_q_monomial_basis(i) *
            symbolic::Monomial({{(*x)(2), 1}, {(*x)(3), 1}}));
      }
    }
    const VectorX<symbolic::Monomial> f_monomial_basis =
        Eigen::Map<VectorX<symbolic::Monomial>>(f_monomial_basis_vec.data(),
                                                f_monomial_basis_vec.size());

    const VectorX<symbolic::Monomial> G_monomial_basis =
        symbolic::EvenDegreeMonomialBasis(symbolic::Variables({(*x)(1)}),
                                          G_q_order);

    // Now evaluate the dynamics at many sampled states.
    const int num_samples_per_dim = 20;
    std::array<Eigen::VectorXd, 4> xbar_samples_per_dim;
    xbar_samples_per_dim[0] = Eigen::VectorXd::LinSpaced(
        num_samples_per_dim, -0.3 * M_PI, 0.3 * M_PI);
    xbar_samples_per_dim[1] = Eigen::VectorXd::LinSpaced(
        num_samples_per_dim, -0.2 * M_PI, 0.2 * M_PI);
    xbar_samples_per_dim[2] =
        Eigen::VectorXd::LinSpaced(num_samples_per_dim, -5, 5);
    xbar_samples_per_dim[3] =
        Eigen::VectorXd::LinSpaced(num_samples_per_dim, -5, 5);
    Eigen::Matrix4Xd xbar_samples(
        4, static_cast<long>(std::pow(num_samples_per_dim, 4)));
    Eigen::Matrix2Xd f_tail(2, xbar_samples.cols());
    Eigen::Matrix2Xd G_tail(2, xbar_samples.cols());
    int sample_count = 0;
    for (int i = 0; i < num_samples_per_dim; ++i) {
      for (int j = 0; j < num_samples_per_dim; ++j) {
        for (int k = 0; k < num_samples_per_dim; ++k) {
          for (int l = 0; l < num_samples_per_dim; ++l) {
            xbar_samples.col(sample_count) << xbar_samples_per_dim[0](i),
                xbar_samples_per_dim[1](j), xbar_samples_per_dim[2](k),
                xbar_samples_per_dim[3](l);
            Eigen::Vector4d f_val;
            Eigen::Vector4d G_val;
            ControlAffineDynamics<double>(
                xbar_samples.col(sample_count) + Eigen::Vector4d(M_PI, 0, 0, 0),
                &f_val, &G_val);
            f_tail.col(sample_count) = f_val.tail<2>();
            G_tail.col(sample_count) = G_val.tail<2>();
            sample_count++;
          }
        }
      }
    }
    // Now fit the data.
    for (int i = 0; i < 2; ++i) {
      (*f)(2 + i) = FitPolynomial(f_monomial_basis, *x, xbar_samples,
                                  f_tail.row(i).transpose());
      (*G)(2 + i) = FitPolynomial(G_monomial_basis, *x, xbar_samples,
                                  G_tail.row(i).transpose());
    }
  }

  double m1{1};
  double l1{1.1};
  double lc1{0.55};
  double I1{0.083};
  double m2{1};
  double l2{2.1};
  double lc2{1.05};
  double I2{0.33};
  double g{9.81};
  double u_bound{60};
};

int DoMain() {
  Acrobot acrobot;
  Vector4<symbolic::Variable> x;
  Vector4<symbolic::Polynomial> f;
  Vector4<symbolic::Polynomial> G;
  const int f_q_order = 3;
  const int G_q_order = 2;
  acrobot.FitControlAffineDynamics(f_q_order, G_q_order, &x, &f, &G);

  // First compute the LQR controller.
  Eigen::Vector4d x_des(M_PI, 0, 0, 0);
  const auto x_des_ad = math::InitializeAutoDiff(x_des);
  Vector4<AutoDiffXd> f_des_ad;
  Vector4<AutoDiffXd> G_des_ad;
  acrobot.ControlAffineDynamics<AutoDiffXd>(x_des_ad, &f_des_ad, &G_des_ad);
  const auto lqr_result = controllers::LinearQuadraticRegulator(
      math::ExtractGradient(f_des_ad), math::ExtractValue(G_des_ad),
      Eigen::Matrix4d::Identity(), Eigen::Matrix<double, 1, 1>::Constant(10));

  // The dynamics is symmetric.
  const int num_vdot_sos = 1;
  const int V_degree = 2;
  const symbolic::Polynomial V_init(
      100 * x.cast<symbolic::Expression>().dot(lqr_result.S * x));
  std::vector<std::vector<symbolic::Polynomial>> l_given(1);
  l_given[0].push_back(
      symbolic::Polynomial(x.cast<symbolic::Expression>().dot(x)));
  std::vector<int> b_degrees(1);
  b_degrees[0] = V_degree - 1 + f_q_order + 2;

  std::vector<std::vector<std::array<int, 3>>> lagrangian_degrees(1);
  lagrangian_degrees[0].resize(num_vdot_sos);
  for (int i = 0; i < num_vdot_sos; ++i) {
    lagrangian_degrees[0][i] = {2, 6, 6};
  }

  const double positivity_eps = 0.;
  ControlLyapunovBoxInputBound dut(f, G, x, positivity_eps);

  ControlLyapunovBoxInputBound::SearchOptions search_options;
  search_options.lagrangian_step_solver_options = solvers::SolverOptions();
  search_options.lagrangian_step_solver_options->SetOption(
      solvers::CommonSolverOption::kPrintToConsole, 1);
  ControlLyapunovBoxInputBound::EllipsoidBisectionOption
      ellipsoid_bisection_option(0.01, 3, 0.01);

  const Eigen::Vector4d x_star(0, 0, 0, 0);
  const Eigen::Matrix4d S = Eigen::Matrix4d::Identity();
  const int r_degree = 0;
  const double deriv_eps_lower = 0.01;
  const double deriv_eps_upper = kInf;

  const auto search_result =
      dut.Search(V_init, l_given, lagrangian_degrees, b_degrees, x_star, S,
                 r_degree, V_degree, deriv_eps_lower, deriv_eps_upper,
                 search_options, ellipsoid_bisection_option);
  return 0;
}
}  // namespace analysis
}  // namespace systems
}  // namespace drake

int main() {
  // Ensure that we have the MOSEK license for the entire duration of this test,
  // so that we do not have to release and re-acquire the license for every
  // test.
  auto mosek_license = drake::solvers::MosekSolver::AcquireLicense();
  return drake::systems::analysis::DoMain();
}
