#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/polynomial.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/mathematical_program_result.h"

namespace drake {
namespace systems {
namespace analysis {
/**
 * For a polynomial whose coefficients are just variables, evaluate the
 * polynomial on a batch of indeterminate values.
 * return (coeff_mat, v) such that coeff_mat.row(i) * v is evaluating p with
 * indeterminates x = x_vals.col(i)
 */
void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        Eigen::MatrixXd* coeff_mat,
                        VectorX<symbolic::Variable>* v);

void EvaluatePolynomial(const symbolic::Polynomial& p,
                        const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
                        const Eigen::Ref<const Eigen::MatrixXd>& x_vals,
                        VectorX<symbolic::Expression>* p_vals);

/**
 * Evaluate h(x) at x = candidate_safe_states, split candidate_safe_states to
 * positive_states and negative_states, such that
 * h(positive_states) >= 0 and h(negative_states) < 0
 */
void SplitCandidateStates(
    const symbolic::Polynomial& h,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& candidate_safe_states,
    Eigen::MatrixXd* positive_states, Eigen::MatrixXd* negative_states);

/**
 * Remove coefficient with absolute value <= zero_tol.
 */
void RemoveTinyCoeff(solvers::MathematicalProgram* prog, double zero_tol);

solvers::MathematicalProgramResult SearchWithBackoff(
    solvers::MathematicalProgram* prog, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale);

/**
 * Find the largest inscribed ellipsoid {x | (x-x*)ᵀS(x-x*) <= d} in the
 * sub-level set {x | f(x)<= 0}, and also subject to the equality constraint
 * eq(x) = 0 Namely f(x)>= 0 and eq(x) = 0 implies (x-x*)ᵀS(x-x*) >= d.
 * Solve the following problem on the variable s(x), d
 * max d
 * s.t (1+t(x))((x-x*)ᵀS(x-x*)-d) - s(x)*f(x) - eq_l(x) * eq(x) is sos
 *     s(x) is sos
 *
 * t(x) is a given sos polynomial.
 */
bool MaximizeInnerEllipsoidSize(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    const symbolic::Polynomial& t, int s_degree,
    const std::optional<VectorX<symbolic::Polynomial>>& eq_constraints,
    const std::vector<int>& eq_lagrangian_degrees,
    const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double backoff_scale, double* d_sol, symbolic::Polynomial* s_sol,
    VectorX<symbolic::Polynomial>* eq_lagrangian_sol);

/**
 * Find the largest inscribed ellipsoid {x | (x-x*)ᵀS(x-x*) <= d} in the
 * sub-level set {x | f(x) <= 0}. Solve the following problem on the variable
 * r(x) through bisecting d
 *
 *     max d
 *     s.t -f(x) - r(x)*(d-(x-x*)ᵀS(x-x*)) is sos
 *         r(x) is sos.
 *
 * Optionally we can specify that we only care about the containment in the
 * algebraic set {x | c(x) = 0}. Namely the intersection of the ellipsoid and
 * the algebraic set {x | c(x) = 0} is contained inside the sub-level set {x |
 * f(x) <= 0}., and we solve the following problem on the variable r(x), t(x)
 * through bisecting d
 *
 *     max d
 *     s.t -f(x) - r(x)*(d-(x-x*)ᵀS(x-x*)) -t(x)ᵀ*c(x) is sos
 *         r(x) is sos.
 *
 * @return Whether we find an ellipsoid or not.
 */
bool MaximizeInnerEllipsoidSize(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const symbolic::Polynomial& f,
    const std::optional<VectorX<symbolic::Polynomial>>& c, int r_degree,
    const std::optional<std::vector<int>>& eq_lagrangian_degrees,
    double size_max, double size_min, const solvers::SolverId& solver_id,
    const std::optional<solvers::SolverOptions>& solver_options,
    double size_tol, double* d_sol, symbolic::Polynomial* r_sol,
    VectorX<symbolic::Polynomial>* eq_lagrangian_sol);

void GetPolynomialSolutions(const solvers::MathematicalProgramResult& result,
                            const VectorX<symbolic::Polynomial>& p,
                            double zero_coeff_tol,
                            VectorX<symbolic::Polynomial>* p_sol);

/**
 * Compute the monomial basis for a given set of variables up to a certain
 * degree, but remove the constant term 1 from the basis.
 */
VectorX<symbolic::Monomial> ComputeMonomialBasisNoConstant(
    const symbolic::Variables& vars, int degree,
    symbolic::internal::DegreeType degree_type);

/**
 * Creates a new sos polynomial p(x) which satisfies p(0) = 0
 */
void NewSosPolynomialPassOrigin(solvers::MathematicalProgram* prog,
                                const symbolic::Variables& indeterminates,
                                int degree,
                                symbolic::internal::DegreeType degree_type,
                                symbolic::Polynomial* p,
                                VectorX<symbolic::Monomial>* monomial_basis,
                                MatrixX<symbolic::Expression>* gram);

/**
 * Creates a new free polynomial passing the origin. Namely its constant term is
 * 0. The new free polynomial doesn't have the linear term with variables in @p
 * no_linear_term_variables either.
 */
symbolic::Polynomial NewFreePolynomialPassOrigin(
    solvers::MathematicalProgram* prog,
    const symbolic::Variables& indeterminates, int degree,
    const std::string& coeff_name, symbolic::internal::DegreeType degree_type,
    const symbolic::Variables& no_linear_term_variables);

/**
 * Return the variables x in @p indeterminates, such that p doesn't have the
 * linear term x.
 */
symbolic::Variables FindNoLinearTermVariables(
    const symbolic::Variables& indeterminates,
    const VectorX<symbolic::Polynomial>& p);

/**
 * Constructs a program to find Lyapunov candidate V that satisfy the following
 * constraints
 * <pre>
 * min ∑ᵢVdot(xⁱ)
 * s.t V - ε*(xᵀx)ᵈ >= 0 when c(x) = 0
 *     V(0) = 0
 *     Vdot(xⁱ) <= 0
 *     V(xⁱ) <= 1
 * </pre>
 * @param x The indeterminates
 * @param V_degree The total degree of V
 * @param x_vals The sampled value of x, x_vals.col(i) is xⁱ
 * @param xdot_vals xdot_vals.col(i) is the dynamics derivative at xⁱ
 */
std::unique_ptr<solvers::MathematicalProgram> FindCandidateLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x, int V_degree,
    double positivity_eps, int d,
    const VectorX<symbolic::Polynomial>& state_constraints,
    const std::vector<int>& eq_lagrangian_degrees,
    const Eigen::Ref<const Eigen::MatrixXd>& x_val,
    const Eigen::Ref<const Eigen::MatrixXd>& xdot_val, symbolic::Polynomial* V,
    VectorX<symbolic::Polynomial>* eq_lagrangian);

struct FindCandidateRegionalLyapunovReturn {
  FindCandidateRegionalLyapunovReturn()
      : prog{std::make_unique<solvers::MathematicalProgram>()} {}
  FindCandidateRegionalLyapunovReturn(FindCandidateRegionalLyapunovReturn&&) =
      default;
  FindCandidateRegionalLyapunovReturn& operator=(
      FindCandidateRegionalLyapunovReturn&&) = default;

  std::unique_ptr<solvers::MathematicalProgram> prog;
  symbolic::Polynomial V;
  VectorX<symbolic::Polynomial> positivity_cin_lagrangian;
  VectorX<symbolic::Polynomial> positivity_ceq_lagrangian;
  VectorX<symbolic::Polynomial> derivative_cin_lagrangian;
  VectorX<symbolic::Polynomial> derivative_ceq_lagrangian;
  symbolic::Polynomial positivity_sos_condition;
  symbolic::Polynomial derivative_sos_condition;
};

/**
 * Constructs a program to find Lyapunov candidate V that satisfy the Lyapunov
 * condition within a region cin(x) <= 0
 * <pre> Find V(x), p1(x), p2(x), q1(1), q2(x)
 * s.t V - ε1*(xᵀx)ᵈ + p1(x) * cin(x) - p2(x) * ceq(x) is sos  (1)
 *     -Vdot - ε2 * V + q1(x) * cin(x) - q2(x) * ceq(x) is sos  (2)
 *     p1(x) is sos, q1(x) is sos.
 * </pre>
 * @param positivity_eps ε1 in the documentation above.
 * @param deriv_eps ε2 in the documentation above.
 * @param[out] positivity_sos_condition The sos condition (1)
 * @param[out] derivative_sos_condition The sos condition (2)
 */
FindCandidateRegionalLyapunovReturn FindCandidateRegionalLyapunov(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const VectorX<symbolic::Polynomial>& dynamics,
    const std::optional<symbolic::Polynomial>& dynamics_denominator,
    int V_degree, double positivity_eps, int d, double deriv_eps,
    const VectorX<symbolic::Polynomial>& state_eq_constraints,
    const std::vector<int>& positivity_ceq_lagrangian_degrees,
    const std::vector<int>& derivative_ceq_lagrangian_degrees,
    const VectorX<symbolic::Polynomial>& state_ineq_constraints,
    const std::vector<int>& positivity_cin_lagrangian_degrees,
    const std::vector<int>& derivative_cin_lagrangian_degrees);

/**
 * Each x[i] contains the coordinate along one dimension, returns the matrix
 * such that each column of the matrix corresponds to one point in the meshgrid.
 */
Eigen::MatrixXd Meshgrid(const std::vector<Eigen::VectorXd>& x);

void Save(const symbolic::Polynomial& p, const std::string& file_name);

[[nodiscard]] symbolic::Polynomial Load(
    const symbolic::Variables& indeterminates, const std::string& file_name);

/**
 * Construct a nonlinear optimization problem to search for x with the maximal
 * Vdot max s s.t ∂V/∂x*(f(x)+G(x)uⁱ)≥ s where uⁱ is the i'th vertex in
 * u_vertices.
 */
std::unique_ptr<solvers::MathematicalProgram> ConstructMaxVdotProgram(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const symbolic::Polynomial& V, const VectorX<symbolic::Polynomial>& f,
    const MatrixX<symbolic::Polynomial>& G,
    const Eigen::Ref<const Eigen::MatrixXd>& u_vertices,
    symbolic::Variable* max_Vdot);

void CheckPolynomialsPassOrigin(const VectorX<symbolic::Polynomial>& p);

double SmallestCoeff(const solvers::MathematicalProgram& prog);

double LargestCoeff(const solvers::MathematicalProgram& prog);

enum class OptimizePolynomialMode {
  kMinimizeMaximal,
  kMaximizeMinimal,
  kMinimizeAverage,
  kMaximizeAverage,
};

/**
 * For a polynomial p(x) whose coefficients are all linear expressions of
 * decision variables in prog, optimize the value of p(x) evaluated at
 * x_samples. Depending on the optimize_polynomial_mode, we choose diffent cost
 * form.
 */
solvers::Binding<solvers::LinearCost> OptimizePolynomialAtSamples(
    solvers::MathematicalProgram* prog, const symbolic::Polynomial& p,
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::MatrixXd>& x_samples,
    OptimizePolynomialMode optimize_polynomial_mode);

enum BilinearIterationStatus {
  kIterationLimit,
  kFailLagrangian,
  kFailLyapunov,
  kInsufficientDecrease,
  kUnknown,
};

struct SearchResultDetails {
  int num_bilinear_iterations = 0;
  BilinearIterationStatus bilinear_iteration_status{
      BilinearIterationStatus::kUnknown};
};

/**
 * When minimizing the positive polynomial a(x) (and wishing to minimize it to
 * 0-polynomial), we can set the type of a(x).
 */
enum class SlackPolynomialType {
  kSos,       ///< a(x) is a sos polynomial.
  kSquare,    ///< a(x) = ε*m(x)ᵀm(x) where ε is a constant and m(x) is the
              ///< monomial basis
  kDiagonal,  ///< a(x) = m(x)ᵀ*diag(s)*m(x) where s is a vector with
              ///< non-negative entries and m(x) is the monomial basis.
};

struct SlackPolynomialInfo {
  SlackPolynomialInfo(int m_degree,
                      SlackPolynomialType m_type = SlackPolynomialType::kSos)
      : degree{m_degree}, type{m_type} {}

  int degree;
  SlackPolynomialType type;

  void AddToProgram(solvers::MathematicalProgram* prog,
                    const symbolic::Variables& x, const std::string& gram_name,
                    symbolic::Polynomial* a,
                    MatrixX<symbolic::Expression>* a_gram) const;
};

Eigen::MatrixXd GetGramSolution(
    const solvers::MathematicalProgramResult& result,
    const Eigen::Ref<const MatrixX<symbolic::Expression>>& gram);

namespace internal {
/** The ellipsoid polynomial (x−x*)ᵀS(x−x*)−ρ
 */
template <typename RhoType>
symbolic::Polynomial EllipsoidPolynomial(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& x,
    const Eigen::Ref<const Eigen::VectorXd>& x_star,
    const Eigen::Ref<const Eigen::MatrixXd>& S, const RhoType& rho);
}  // namespace internal
}  // namespace analysis
}  // namespace systems
}  // namespace drake
