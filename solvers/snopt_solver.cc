#include "drake/solvers/snopt_solver.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "drake/common/text_logging.h"
#include "drake/math/autodiff.h"
#include "drake/solvers/mathematical_program.h"

extern "C" {
// NOLINTNEXTLINE(build/include)
#include "snopt_cwrap.h"
}

// TODO(jwnimmer-tri) Eventually resolve these warnings.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

// todo(sammy-tri) :  return more information that just the solution (INFO,
// infeasible constraints, ...)
// todo(sammy-tri) :  avoid all dynamic allocation

namespace drake {
namespace solvers {
namespace {
// This struct is used for passing additional info to the snopt_userfun, which
// evaluates the value and gradient of the cost and constraints. Apart from the
// standard information such as decision variable values, snopt_userfun could
// rely on additional information such as the cost gradient sparsity pattern.
struct SnoptUserFunInfo {
  const MathematicalProgram* prog{nullptr};
  const std::set<int>* nonlinear_cost_gradient_indices{nullptr};
};

// Return the number of rows in the nonlinear constraint.
template <typename C>
int SingleNonlinearConstraintSize(const C& constraint) {
  return constraint.num_constraints();
}

template <>
int SingleNonlinearConstraintSize<LinearComplementarityConstraint>(
    const LinearComplementarityConstraint& constraint) {
  return 1;
}

// Evaluate a single nonlinear constraints. For generic Constraint,
// LorentzConeConstraint, RotatedLorentzConeConstraint, we call Eval function
// of the constraint directly. For some other constraint, such as
// LinearComplementaryConstraint, we will evaluate its nonlinear constraint
// differently, than its Eval function.
template <typename C>
void EvaluateSingleNonlinearConstraint(const C& constraint,
                                       const Eigen::VectorXd& this_x,
                                       AutoDiffVecXd* ty) {
  ty->resize(SingleNonlinearConstraintSize(constraint));
  constraint.Eval(math::initializeAutoDiff(this_x), ty);
}

template <>
void EvaluateSingleNonlinearConstraint<LinearComplementarityConstraint>(
    const LinearComplementarityConstraint& constraint,
    const Eigen::VectorXd& this_x, AutoDiffVecXd* ty) {
  ty->resize(1);
  auto tx = math::initializeAutoDiff(this_x);
  (*ty)(0) = tx.dot(constraint.M().cast<AutoDiffXd>() * tx +
                    constraint.q().cast<AutoDiffXd>());
}

/*
 * Evaluate the value and gradients of nonlinear constraints.
 * The template type Binding is supposed to be a
 * MathematicalProgram::Binding<Constraint> type.
 * @param constraint_list A list of Binding<Constraint>
 * @param F The value of the constraints
 * @param G The value of the non-zero entries in the gradient
 * @param constraint_index The starting index of the constraint_list(0) in the
 * optimization problem.
 * @param grad_index The starting index of the gradient of constraint_list(0)
 * in the optimization problem.
 * @param tx the AutoDiffMatrixType that stores the value of the decision
 * variable.
 */
template <typename C>
void EvaluateNonlinearConstraints(
    const MathematicalProgram& prog,
    const std::vector<Binding<C>>& constraint_list, double F[], double G[],
    size_t* constraint_index, size_t* grad_index, const Eigen::VectorXd& xvec) {
  Eigen::VectorXd this_x;
  for (const auto& binding : constraint_list) {
    const auto& c = binding.evaluator();
    int num_constraints = SingleNonlinearConstraintSize(*c);

    int num_v_variables = binding.GetNumElements();
    this_x.resize(num_v_variables);
    for (int i = 0; i < num_v_variables; ++i) {
      this_x(i) = xvec(prog.FindDecisionVariableIndex(binding.variables()(i)));
    }

    AutoDiffVecXd ty;
    ty.resize(num_constraints);
    EvaluateSingleNonlinearConstraint(*c, this_x, &ty);

    for (int i = 0; i < num_constraints; i++) {
      F[(*constraint_index)++] = ty(i).value();
    }

    for (int i = 0; i < num_constraints; i++) {
      for (int j = 0; j < num_v_variables; ++j) {
        G[(*grad_index)++] = ty(i).derivatives()(j);
      }
    }
  }
}

// Find the variables with non-zero gradient in @p costs, and add the indices of
// these variable to cost_gradient_indices.
template <typename C>
void GetNonlinearCostNonzeroGradientIndices(
    const MathematicalProgram& prog, const std::vector<Binding<C>>& costs,
    std::set<int>* cost_gradient_indices) {
  for (const auto& cost : costs) {
    for (int i = 0; i < static_cast<int>(cost.GetNumElements()); ++i) {
      cost_gradient_indices->insert(
          prog.FindDecisionVariableIndex(cost.variables()(i)));
    }
  }
}

/*
 * Loops through each nonlinear cost stored in MathematicalProgram, and obtains
 * the indices of the non-zero gradient, in the summed cost.
 */
std::set<int> GetAllNonlinearCostNonzeroGradientIndices(
    const MathematicalProgram& prog) {
  // The nonlinear costs include the quadratic and generic costs. Linear costs
  // are not included. In the future if we support more costs (like polynomial
  // costs), we should add it here.
  std::set<int> cost_gradient_indices;
  GetNonlinearCostNonzeroGradientIndices(prog, prog.quadratic_costs(),
                                         &cost_gradient_indices);
  GetNonlinearCostNonzeroGradientIndices(prog, prog.generic_costs(),
                                         &cost_gradient_indices);
  return cost_gradient_indices;
}

/*
 * Evaluates all the nonlinear costs, adds the value of the costs to
 * @p total_cost, and also adds the gradients to @p nonlinear_cost_gradients.
 */
template <typename C>
void EvaluateAndAddNonlinearCosts(
    const MathematicalProgram& prog,
    const std::vector<Binding<C>>& nonlinear_costs, const Eigen::VectorXd& x,
    double* total_cost, std::vector<double>* nonlinear_cost_gradients) {
  for (const auto& binding : nonlinear_costs) {
    const auto& obj = binding.evaluator();
    const int num_v_variables = binding.GetNumElements();

    Eigen::VectorXd this_x(num_v_variables);
    // binding_var_indices[i] is the index of binding.variables()(i) in prog's
    // decision variables.
    std::vector<int> binding_var_indices(num_v_variables);
    for (int i = 0; i < num_v_variables; ++i) {
      binding_var_indices[i] =
          prog.FindDecisionVariableIndex(binding.variables()(i));
      this_x(i) = x(binding_var_indices[i]);
    }
    AutoDiffVecXd ty(1);
    obj->Eval(math::initializeAutoDiff(this_x), &ty);

    *total_cost += ty(0).value();
    for (int i = 0; i < num_v_variables; ++i) {
      (*nonlinear_cost_gradients)[binding_var_indices[i]] +=
          ty(0).derivatives()(i);
    }
  }
}

// Evaluates all nonlinear costs, including the quadratic and generic costs.
// SNOPT stores the value of the total cost in F[0], and the nonzero gradient
// in array G. After calling this function, G[0], G[1], ..., G[grad_index-1]
// will store the nonzero gradient of the cost.
void EvaluateAllNonlinearCosts(
    const MathematicalProgram& prog, const Eigen::VectorXd& xvec,
    const std::set<int>& nonlinear_cost_gradient_indices, double F[],
    double G[], size_t* grad_index) {
  std::vector<double> cost_gradients(prog.num_vars(), 0);
  // Quadratic costs.
  EvaluateAndAddNonlinearCosts(prog, prog.quadratic_costs(), xvec, &(F[0]),
                               &cost_gradients);
  // Generic costs.
  EvaluateAndAddNonlinearCosts(prog, prog.generic_costs(), xvec, &(F[0]),
                               &cost_gradients);

  for (const int cost_gradient_index : nonlinear_cost_gradient_indices) {
    G[*grad_index] = cost_gradients[cost_gradient_index];
    ++(*grad_index);
  }
}

void PackSnoptUserFunInfoIntoWorkspace(
    snProblem* snopt_problem, const SnoptUserFunInfo* snopt_userfun_info) {
  // snopt-interface does not allow using the char workspace, as explained in
  // http://ccom.ucsd.edu/~optimizers/usage/c/, that "due to the incompatibility
  // of strings in Fortran/C, all character arrays are disabled."
  //
  // This is a tricky part, we use the integer user workspace iu[0] to
  // store snopt_user_info. In snopt_userfun, we will need to read
  // snopt_user_info from the integer workspace.
  // Set the length of the integer user workspace to be 100. This should be
  // large enough to accomodate snopt_userfun_info. If not, increase this
  // number, and also change the static assertion below.
  snopt_problem->leniu = 100;
  static_assert(100 * sizeof(int) > sizeof(snopt_userfun_info),
                "snopt_problem->leniu is not large enough.");
  snopt_problem->iu =
      static_cast<int*>(malloc(sizeof(int) * snopt_problem->leniu));

  const int* const p_snopt_userfun_info =
      reinterpret_cast<int*>(&snopt_userfun_info);
  int* const iu_snopt_userfun_info = &snopt_problem->iu[0];
  // copy_size is the round up of the division sizeof(snopt_userfun_info) /
  // sizeof(int).
  const int copy_size =
      (sizeof(snopt_userfun_info) + sizeof(int) - 1) / sizeof(int);
  std::copy(p_snopt_userfun_info, p_snopt_userfun_info + copy_size,
            iu_snopt_userfun_info);
}

SnoptUserFunInfo* UnpackSnoptUserFunInfoFromWorkspace(int iu[]) {
  SnoptUserFunInfo* snopt_userfun_info;
  int* const p_snopt_userfun_info = reinterpret_cast<int*>(&snopt_userfun_info);
  const int* const iu_snopt_userfun_info = iu;
  std::copy(iu_snopt_userfun_info,
            iu_snopt_userfun_info + sizeof(snopt_userfun_info),
            p_snopt_userfun_info);
  return snopt_userfun_info;
}

void snopt_userfun(int* Status, int* n, double x[], int* needF, int* neF,
                   double F[], int* needG, int* neG, double G[], char* cu,
                   int* lencu, int iu[], int* leniu, double ru[], int* lenru) {
  const SnoptUserFunInfo* snopt_userfun_info =
      UnpackSnoptUserFunInfoFromWorkspace(iu);

  const MathematicalProgram* current_problem = snopt_userfun_info->prog;
  const std::set<int>* nonlinear_cost_gradient_indices =
      snopt_userfun_info->nonlinear_cost_gradient_indices;

  Eigen::VectorXd xvec(*n);
  for (int i = 0; i < *n; i++) {
    xvec(i) = x[i];
  }

  F[0] = 0.0;
  memset(G, 0, (*n) * sizeof(double));

  size_t grad_index = 0;

  current_problem->EvalVisualizationCallbacks(xvec);

  EvaluateAllNonlinearCosts(*current_problem, xvec,
                            *nonlinear_cost_gradient_indices, F, G,
                            &grad_index);

  // The constraint index starts at 1 because the cost is the
  // first row.
  size_t constraint_index = 1;
  // The gradient_index also starts after the cost.
  EvaluateNonlinearConstraints(*current_problem,
                               current_problem->generic_constraints(), F, G,
                               &constraint_index, &grad_index, xvec);
  EvaluateNonlinearConstraints(*current_problem,
                               current_problem->lorentz_cone_constraints(), F,
                               G, &constraint_index, &grad_index, xvec);
  EvaluateNonlinearConstraints(
      *current_problem, current_problem->rotated_lorentz_cone_constraints(), F,
      G, &constraint_index, &grad_index, xvec);
  EvaluateNonlinearConstraints(
      *current_problem, current_problem->linear_complementarity_constraints(),
      F, G, &constraint_index, &grad_index, xvec);
}

/*
 * Updates the number of nonlinear constraints and the number of gradients by
 * looping through the constraint list
 * @tparam C A Constraint type. Note that some derived classes of Constraint
 * is regarded as generic constraint by SNOPT solver, such as
 * LorentzConeConstraint and RotatedLorentzConeConstraint, so @tparam C can also
 * be these derived classes.
 */
template <typename C>
void UpdateNumNonlinearConstraintsAndGradients(
    const std::vector<Binding<C>>& constraint_list,
    int* num_nonlinear_constraints, int* max_num_gradients) {
  for (auto const& binding : constraint_list) {
    auto const& c = binding.evaluator();
    int n = c->num_constraints();
    *max_num_gradients += n * binding.GetNumElements();
    *num_nonlinear_constraints += n;
  }
}

// For linear complementary condition
// 0 <= x ⊥ Mx + q >= 0
// we add the nonlinear constraint xᵀ(Mx+q) = 0
// So we only add one row of nonlinear constraint, and update the gradient of
// this nonlinear constraint accordingly.
template <>
void UpdateNumNonlinearConstraintsAndGradients<LinearComplementarityConstraint>(
    const std::vector<Binding<LinearComplementarityConstraint>>&
        constraint_list,
    int* num_nonlinear_constraints, int* max_num_gradients) {
  *num_nonlinear_constraints += constraint_list.size();
  for (const auto& binding : constraint_list) {
    *max_num_gradients += binding.evaluator()->M().rows();
  }
}

template <typename C>
void UpdateConstraintBoundsAndGradients(
    const MathematicalProgram& prog,
    const std::vector<Binding<C>>& constraint_list, std::vector<double>* Flow,
    std::vector<double>* Fupp, std::vector<int>* iGfun, std::vector<int>* jGvar,
    size_t* constraint_index, size_t* grad_index) {
  for (auto const& binding : constraint_list) {
    auto const& c = binding.evaluator();
    int n = c->num_constraints();

    auto const lb = c->lower_bound(), ub = c->upper_bound();
    for (int i = 0; i < n; i++) {
      const int constraint_index_i = *constraint_index + i;
      (*Flow)[constraint_index_i] = lb(i);
      (*Fupp)[constraint_index_i] = ub(i);
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < static_cast<int>(binding.GetNumElements()); ++j) {
        (*iGfun)[*grad_index] = *constraint_index + i;  // row order
        (*jGvar)[*grad_index] =
            prog.FindDecisionVariableIndex(binding.variables()(j));
        (*grad_index)++;
      }
    }

    (*constraint_index) += n;
  }
}

// For linear complementary condition
// 0 <= x ⊥ Mx + q >= 0
// we add the nonlinear constraint xᵀ(Mx + q) = 0
// The bound of this constraint is 0. The indices of the non-zero gradient
// of this constraint is updated accordingly.
template <>
void UpdateConstraintBoundsAndGradients<LinearComplementarityConstraint>(
    const MathematicalProgram& prog,
    const std::vector<Binding<LinearComplementarityConstraint>>&
        constraint_list,
    std::vector<double>* Flow, std::vector<double>* Fupp,
    std::vector<int>* iGfun, std::vector<int>* jGvar, size_t* constraint_index,
    size_t* grad_index) {
  for (const auto& binding : constraint_list) {
    (*Flow)[*constraint_index] = 0;
    (*Fupp)[*constraint_index] = 0;
    for (int j = 0; j < binding.evaluator()->M().rows(); ++j) {
      (*iGfun)[*grad_index] = *constraint_index;
      (*jGvar)[*grad_index] =
          prog.FindDecisionVariableIndex(binding.variables()(j));
      (*grad_index)++;
    }
    ++(*constraint_index);
  }
}

template <typename C>
Eigen::SparseMatrix<double> LinearEvaluatorA(const C& evaluator) {
  return evaluator.GetSparseMatrix();
}

// Return the number of rows in the linear constraint
template <typename C>
int LinearConstraintSize(const C& constraint) {
  return constraint.num_constraints();
}

// For linear complementary condition
// 0 <= x ⊥ Mx + q >= 0
// The linear constraint we add to the program is Mx >= -q
// This linear constraint has the same number of rows, as matrix M.
template <>
int LinearConstraintSize<LinearComplementarityConstraint>(
    const LinearComplementarityConstraint& constraint) {
  return constraint.M().rows();
}

template <>
Eigen::SparseMatrix<double> LinearEvaluatorA<LinearComplementarityConstraint>(
    const LinearComplementarityConstraint& constraint) {
  return constraint.M().sparseView();
}

template <typename C>
std::pair<Eigen::VectorXd, Eigen::VectorXd> LinearConstraintBounds(
    const C& constraint) {
  return std::make_pair(constraint.lower_bound(), constraint.upper_bound());
}

// For linear complementary condition
// 0 <= x ⊥ Mx + q >= 0
// we add the constraint Mx >= -q
template <>
std::pair<Eigen::VectorXd, Eigen::VectorXd>
LinearConstraintBounds<LinearComplementarityConstraint>(
    const LinearComplementarityConstraint& constraint) {
  return std::make_pair(
      -constraint.q(),
      Eigen::VectorXd::Constant(constraint.q().rows(),
                                std::numeric_limits<double>::infinity()));
}

void UpdateLinearCost(
    const MathematicalProgram& prog,
    std::unordered_map<int, double>* variable_to_coefficient_map,
    double* linear_cost_constant_term) {
  for (const auto& binding : prog.linear_costs()) {
    *linear_cost_constant_term += binding.evaluator()->b();
    for (int k = 0; k < static_cast<int>(binding.GetNumElements()); ++k) {
      const int variable_index =
          prog.FindDecisionVariableIndex(binding.variables()(k));
      (*variable_to_coefficient_map)[variable_index] +=
          binding.evaluator()->a()(k);
    }
  }
}

template <typename C>
void UpdateLinearConstraint(const MathematicalProgram& prog,
                            const std::vector<Binding<C>>& linear_constraints,
                            std::vector<Eigen::Triplet<double>>* tripletList,
                            std::vector<double>* Flow,
                            std::vector<double>* Fupp, size_t* constraint_index,
                            size_t* linear_constraint_index) {
  for (auto const& binding : linear_constraints) {
    auto const& c = binding.evaluator();
    int n = LinearConstraintSize(*c);

    const Eigen::SparseMatrix<double> A_constraint = LinearEvaluatorA(*c);

    for (int k = 0; k < static_cast<int>(binding.GetNumElements()); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(A_constraint, k); it;
           ++it) {
        tripletList->emplace_back(
            *linear_constraint_index + it.row(),
            prog.FindDecisionVariableIndex(binding.variables()(k)), it.value());
      }
    }

    const auto bounds = LinearConstraintBounds(*c);
    for (int i = 0; i < n; i++) {
      const int constraint_index_i = *constraint_index + i;
      (*Flow)[constraint_index_i] = bounds.first(i);
      (*Fupp)[constraint_index_i] = bounds.second(i);
    }
    *constraint_index += n;
    *linear_constraint_index += n;
  }
}

void SolveWithGivenOptions(
    const MathematicalProgram& prog,
    const Eigen::Ref<const Eigen::VectorXd>& x_init,
    const std::map<std::string, std::string>& snopt_options_string,
    const std::map<std::string, int>& snopt_options_int,
    const std::map<std::string, double>& snopt_options_double,
    int* snopt_status, double* objective, EigenPtr<Eigen::VectorXd> x_val) {
  DRAKE_ASSERT(x_val->rows() == prog.num_vars());
  snProblem drake_problem{};
  char problem_name[] = "drake_problem";
  char no_print_file[] = "";
  snInit(&drake_problem, problem_name, no_print_file, 0 /* no summary */);

  SnoptUserFunInfo snopt_userfun_info;
  snopt_userfun_info.prog = &prog;
  const std::set<int> nonlinear_cost_gradient_indices =
      GetAllNonlinearCostNonzeroGradientIndices(prog);
  snopt_userfun_info.nonlinear_cost_gradient_indices =
      &nonlinear_cost_gradient_indices;
  PackSnoptUserFunInfoIntoWorkspace(&drake_problem, &snopt_userfun_info);

  const int nx = prog.num_vars();
  std::vector<double> x(nx, 0.0);
  std::vector<double> xlow(nx, -std::numeric_limits<double>::infinity());
  std::vector<double> xupp(nx, std::numeric_limits<double>::infinity());
  std::vector<double> xmul(nx, 0.0);
  std::vector<int> xstate(nx, 0);

  // Initialize the guess for x.
  for (int i = 0; i < nx; ++i) {
    if (!std::isnan(x_init(i))) {
      x[i] = x_init(i);
    } else {
      x[i] = 0.0;
    }
  }

  // Set up the lower and upper bounds.
  for (auto const& binding : prog.bounding_box_constraints()) {
    const auto& c = binding.evaluator();
    const auto& lb = c->lower_bound();
    const auto& ub = c->upper_bound();

    for (int k = 0; k < static_cast<int>(binding.GetNumElements()); ++k) {
      const size_t vk_index =
          prog.FindDecisionVariableIndex(binding.variables()(k));
      xlow[vk_index] = std::max(lb(k), xlow[vk_index]);
      xupp[vk_index] = std::min(ub(k), xupp[vk_index]);
    }
  }

  // For linear complementary condition
  // 0 <= x ⊥ Mx + q >= 0
  // we add the bounding box constraint x >= 0
  for (const auto& binding : prog.linear_complementarity_constraints()) {
    for (int k = 0; k < static_cast<int>(binding.GetNumElements()); ++k) {
      const size_t vk_index =
          prog.FindDecisionVariableIndex(binding.variables()(k));
      xlow[vk_index] = std::max(xlow[vk_index], 0.0);
    }
  }

  // Update nonlinear constraints.
  int num_nonlinear_constraints = 0;
  int max_num_gradients = nonlinear_cost_gradient_indices.size();
  UpdateNumNonlinearConstraintsAndGradients(prog.generic_constraints(),
                                            &num_nonlinear_constraints,
                                            &max_num_gradients);
  UpdateNumNonlinearConstraintsAndGradients(prog.lorentz_cone_constraints(),
                                            &num_nonlinear_constraints,
                                            &max_num_gradients);
  UpdateNumNonlinearConstraintsAndGradients(
      prog.rotated_lorentz_cone_constraints(), &num_nonlinear_constraints,
      &max_num_gradients);
  UpdateNumNonlinearConstraintsAndGradients(
      prog.linear_complementarity_constraints(), &num_nonlinear_constraints,
      &max_num_gradients);

  // Update linear constraints.
  int num_linear_constraints = 0;
  const auto linear_constraints = prog.GetAllLinearConstraints();
  for (auto const& binding : linear_constraints) {
    num_linear_constraints += binding.evaluator()->num_constraints();
  }

  // For linear complementary condition
  // 0 <= x ⊥ Mx + q >= 0
  // The linear constraint we add is Mx + q >= 0, so we will append
  // M.rows() rows to the linear constraints.
  for (const auto& binding : prog.linear_complementarity_constraints()) {
    num_linear_constraints += binding.evaluator()->M().rows();
  }

  // Update the bound of the constraint.
  const int nF = 1 + num_nonlinear_constraints + num_linear_constraints;
  std::vector<double> F(nF, 0.0);
  std::vector<double> Flow(nF, -std::numeric_limits<double>::infinity());
  std::vector<double> Fupp(nF, std::numeric_limits<double>::infinity());
  std::vector<double> Fmul(nF, 0.0);
  std::vector<int> Fstate(nF, 0);

  // Set up the gradient sparsity pattern.
  const int lenG = max_num_gradients;
  std::vector<int> iGfun(lenG, 0);
  std::vector<int> jGvar(lenG, 0);
  size_t grad_index = 0;
  for (const auto cost_gradient_index : nonlinear_cost_gradient_indices) {
    iGfun[grad_index] = 0;
    jGvar[grad_index] = cost_gradient_index;
    ++grad_index;
  }

  // constraint_index starts at 1 because row 0 is the cost.
  size_t constraint_index = 1;
  UpdateConstraintBoundsAndGradients(prog, prog.generic_constraints(), &Flow,
                                     &Fupp, &iGfun, &jGvar, &constraint_index,
                                     &grad_index);
  UpdateConstraintBoundsAndGradients(prog, prog.lorentz_cone_constraints(),
                                     &Flow, &Fupp, &iGfun, &jGvar,
                                     &constraint_index, &grad_index);
  UpdateConstraintBoundsAndGradients(
      prog, prog.rotated_lorentz_cone_constraints(), &Flow, &Fupp, &iGfun,
      &jGvar, &constraint_index, &grad_index);
  UpdateConstraintBoundsAndGradients(
      prog, prog.linear_complementarity_constraints(), &Flow, &Fupp, &iGfun,
      &jGvar, &constraint_index, &grad_index);

  // Now find the sparsity pattern of the linear constraints/costs, and also
  // update Flow and Fupp corresponding to the linear constraints.
  // We use Eigen::Triplet to store the non-zero entries in the linear
  // constraint
  // http://eigen.tuxfamily.org/dox/group__TutorialSparse.html
  typedef Eigen::Triplet<double> T;
  double linear_cost_constant_term = 0;
  std::unordered_map<int, double> variable_to_linear_cost_coefficient;
  UpdateLinearCost(prog, &variable_to_linear_cost_coefficient,
                   &linear_cost_constant_term);

  std::vector<T> linear_constraints_triplets;
  linear_constraints_triplets.reserve(num_linear_constraints * prog.num_vars());
  size_t linear_constraint_index = 0;
  UpdateLinearConstraint(prog, linear_constraints, &linear_constraints_triplets,
                         &Flow, &Fupp, &constraint_index,
                         &linear_constraint_index);
  UpdateLinearConstraint(prog, prog.linear_complementarity_constraints(),
                         &linear_constraints_triplets, &Flow, &Fupp,
                         &constraint_index, &linear_constraint_index);

  const int lenA = variable_to_linear_cost_coefficient.size() +
                   linear_constraints_triplets.size();
  std::vector<double> A(lenA, 0.0);
  std::vector<int> iAfun(lenA, 0);
  std::vector<int> jAvar(lenA, 0);
  size_t A_index = 0;
  for (const auto& it : variable_to_linear_cost_coefficient) {
    A[A_index] = it.second;
    iAfun[A_index] = 0;
    jAvar[A_index] = it.first;
    A_index++;
  }
  for (const auto& it : linear_constraints_triplets) {
    A[A_index] = it.value();
    iAfun[A_index] = 1 + num_nonlinear_constraints + it.row();
    jAvar[A_index] = it.col();
    A_index++;
  }

  // Determines if we should print out snopt debugging info.
  const auto print_file_it = snopt_options_string.find("Print file");
  if (print_file_it != snopt_options_string.end()) {
    setPrintfile(&drake_problem,
                 const_cast<char*>(print_file_it->second.c_str()));
    char major_print_level[] = "Major print level";
    // According to SNOPT user manual, set major print level to 11 gives
    // additional details of the Jacobian factorization that commences each
    // major iteration.
    setIntParameter(&drake_problem, major_print_level, 11);
  }

  for (const auto& it : snopt_options_double) {
    setRealParameter(&drake_problem, const_cast<char*>(it.first.c_str()),
                     it.second);
  }

  for (const auto& it : snopt_options_int) {
    setIntParameter(&drake_problem, const_cast<char*>(it.first.c_str()),
                    it.second);
  }

  const int Cold = 0;
  const double ObjAdd = linear_cost_constant_term;
  const int ObjRow = 0;
  int nS = 0;
  int nInf{0};
  double sInf{0.0};

  *snopt_status =
      solveA(&drake_problem, Cold, nF, nx, ObjAdd, ObjRow, snopt_userfun, lenA,
             iAfun.data(), jAvar.data(), A.data(), lenG, iGfun.data(),
             jGvar.data(), xlow.data(), xupp.data(), Flow.data(), Fupp.data(),
             x.data(), xstate.data(), xmul.data(), F.data(), Fstate.data(),
             Fmul.data(), &nS, &nInf, &sInf);
  *x_val = Eigen::Map<Eigen::VectorXd>(x.data(), nx);
  *objective = F[0];

  free(drake_problem.iu);
  deleteSNOPT(&drake_problem);
}

SolutionResult MapSnoptInfoToSolutionResult(int snopt_info) {
  SolutionResult solution_result{SolutionResult::kUnknownError};
  if (snopt_info >= 1 && snopt_info <= 6) {
    solution_result = SolutionResult::kSolutionFound;
  } else {
    log()->debug("Snopt returns code {}\n", snopt_info);
    if (snopt_info >= 11 && snopt_info <= 16) {
      solution_result = SolutionResult::kInfeasibleConstraints;
    } else if (snopt_info >= 20 && snopt_info <= 22) {
      solution_result = SolutionResult::kUnbounded;
    } else if (snopt_info >= 30 && snopt_info <= 32) {
      solution_result = SolutionResult::kIterationLimit;
    } else if (snopt_info == 91) {
      solution_result = SolutionResult::kInvalidInput;
    }
  }
  return solution_result;
}

}  // anon namespace

bool SnoptSolver::available() const { return true; }

SolutionResult SnoptSolver::Solve(MathematicalProgram& prog) const {
  int snopt_status{0};
  double objective{0};
  Eigen::VectorXd x_val(prog.num_vars());
  SolveWithGivenOptions(
      prog, prog.initial_guess(), prog.GetSolverOptionsStr(id()),
      prog.GetSolverOptionsInt(id()), prog.GetSolverOptionsDouble(id()),
      &snopt_status, &objective, &x_val);
  // TODO(hongkai.dai) add other useful quantities to a struct specific to
  // SNOPT, to include information on constraint values, xstate, Fstate, xmul,
  // Fmul, etc.
  SolverResult solver_result(id());
  solver_result.set_decision_variable_values(x_val);
  solver_result.set_optimal_cost(objective);

  const SolutionResult solution_result =
      MapSnoptInfoToSolutionResult(snopt_status);
  if (solution_result == SolutionResult::kUnbounded) {
    solver_result.set_optimal_cost(MathematicalProgram::kUnboundedCost);
  }
  prog.SetSolverResult(solver_result);
  return solution_result;
}

}  // namespace solvers
}  // namespace drake

#pragma GCC diagnostic pop  // "-Wunused-parameter"
