#pragma once

#include <list>
#include <memory>

// clang-format off
// scs.h should be included before amatrix.h, since amatrix.h uses deta types
// defined in scs.h
#include <scs.h>
#include "linsys/amatrix.h"
// clang-format on

#include "drake/common/symbolic.h"
namespace drake {
namespace solvers {
/**
 * Inside each node, we solve an optimization problem in SCS form
 * <pre>
 * min cᵀx
 * s.t Ax + s = b
 *     s in K
 * </pre>
 * This node is created from its parent node, by fixing a binary variable y to
 * either 0 or 1. The parent node solves the problem
 * <pre>
 * min c_primeᵀ * x_prime
 * s.t A_prime * x_prime + s_prime = b_prime
 *     s_prime in K_prime
 * </pre>
 * where x is obtained by removing the binary variable y from x_prime.
 * Notice that the matrix A, b, c will change from node to node.
 */
class ScsNode {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ScsNode)

  /**
   * Each node solves this optimization program
   * <pre>
   *   min cᵀx + d
   *   s.t Ax + s = b
   *       s in K
   * </pre>
   * Construct a node used in the tree for branch-and-bound. Pre-allocate the
   * memory for the matrix A, b, and c.
   * @param num_A_rows The number of rows in the matrix A.
   * @param num_A_cols The number of columns in the matrix A.
   */
  ScsNode(int num_A_rows, int num_A_cols);

  /**
   * We want to solve this mixed-integer optimization program
   * <pre>
   *   min cᵀx + d
   *   s.t Ax + s = b
   *       s in K
   *       y are binary
   * </pre>
   * where y is a subset of decision variables x. To solve this problem using
   * branch and bound algorithm, we create the root node from the SCS problem
   * data. The constraint Ax + s = b does NOT include the integral constraint on
   * the binary variables. Neither does it include the relaxed constraint on the
   * binary variable y (i.e, 0 ≤ y ≤ 1)
   * @param A The left-hand side of the constraint.
   * @param b The right-hand side of the constraint.
   * @param c The coefficients of the linear cost.
   * @param cone The cone K in the documentation above. Note that cone does not
   * include the relaxation on the binary variables 0 ≤ y ≤ 1. Also note that
   * this is aliased for the lifetime of the ScsNode object.
   * tree for the branch and bound.
   * @param binary_var_indices The indices of the binary variables, y, in the
   * vector of decision variables, x.
   * @param cost_constant The constant term in the cost.
   * @pre 1. binary_var_indices is within the range of [0, A.n).
   * @pre 2. binary_var_indices does not contain duplicate entries.
   * @throws std::runtime_error if the preconditions are not met.
   */
  static std::unique_ptr<ScsNode> ConstructRootNode(
      const AMatrix& A, const scs_float* const b, const scs_float* const c,
      const SCS_CONE& cone, const std::list<int>& binary_var_indices,
      double cost_constant);

  ~ScsNode();

  /**
   * Branches on one binary variable, and creates two child nodes. In each child
   * node, that binary variable is fixed to either 0 or 1.
   * @param binary_var_index The binary variable with this index will be fixed
   * to either 0 or 1 in the child node, and being removed from the decision
   * variables.
   * @pre The variable with the index binary_var_index is a binary variable,
   * namely it is contained in binary_var_indices_.
   * @throw std::runtime_error if the preconditions are not met.
   */
  void Branch(int binary_var_index);

  /**
   * Solve the optimization problem in this node.
   * <pre>
   *  min c_ᵀx + d
   *  s.t A_*x + s = b_
   *      s_ in cones_
   * </pre>
   * There are several possible outcomes by solving the optimization program
   * in this node.
   * 1. The problem is infeasible. Then we do not need to branch on this node.
   * 2. The problem is feasible, we can then update the lower bound, as the
   *    minimal among all the costs in the leaf nodes.
   * 3. The problem is feasible, and we find a solution that satisfies the
   *    integral constraints. The cost of this solution is an upper bound of
   *    the original mixed-integer problem. If the cost is smaller than the best
   *    upper bound, then we update the best upper bound to this cost.
   * 4. The problem is feasible, but the optimal cost is larger than the
   *    best upper bound. Then there is no need to branch on the node.
   * @param scs_settings. The settings (parameters) for solving the SCS problem.
   */
  scs_int Solve(const SCS_SETTINGS& scs_settings);

  /// Getter for A matrix.
  const AMatrix* A() const { return A_.get(); }

  /// Getter for b vector.
  const scs_float* b() const { return b_.get(); }

  /// Getter for c vector, the linear coefficient of the cost.
  const scs_float* c() const { return c_.get(); }

  /// Getter for the cones.
  const SCS_CONE* cone() const { return cone_.get(); }

  /**
   * True if the optimal solution in this node satisfies all integral
   * constraints.
   */
  bool found_integral_sol() const { return found_integral_sol_; }

  /// Getter for the indices of all binary variables in this node.
  const std::list<int>& binary_var_indices() const {
    return binary_var_indices_;
  }

  /**
   * This node was created from its parent node, by branching on a binary
   * variable. Return the index of the branching variable in the parent node.
   */
  int y_index() const { return y_index_; }

  /**
   * This node was created from its parent node, by fixing a binary variable to
   * either 0 or 1. Returns the value of the branching binary variable.
   */
  int y_val() const { return y_val_; }

  /// Getter for the constant term in the cost.
  double cost_constant() const { return cost_constant_; }

  ScsNode* left_child() const { return left_child_; }

  ScsNode* right_child() const { return right_child_; }

  ScsNode* parent() const { return parent_; }

  /// Getter for the optimal cost of the optimization program in this node.
  double cost() const { return cost_; }

  const SCS_SOL_VARS* scs_sol() const { return scs_sol_.get(); }

  SCS_INFO scs_info() const { return scs_info_; }

 private:
  // We will solve the problem
  // min c_ᵀx
  // s.t A_ * x + s = b_
  //     s in K
  // in this node.
  // We will put the constraint 0 ≤ y ≤ 1 in the first rows of the "linear
  // cones" in A. Namely starting from cone_->f'th row in A, to cone_->f + 2N
  // row in A, are of the form
  // -y + s = 0
  // y + s = 1
  // s in positive cone
  // where N is the length of the binary_var_indices_;
  std::unique_ptr<AMatrix, void (*)(AMatrix*)> A_;
  std::unique_ptr<scs_float, void (*)(void*)> b_;
  std::unique_ptr<scs_float, void (*)(void*)> c_;
  // ScsNode does not own cone_->q, cone_->s, cone_->p. Notice that only
  // cone_->l changes between each node, the length of other constraints, such
  // as second order cone, semi-definite cone, etc, do not change.
  std::unique_ptr<SCS_CONE, void (*)(void*)> cone_;
  // This node is created from its parent node, by fixing a variable y to a
  // binary value. That variable y has index y_index_ in the parent node.
  int y_index_;
  int y_val_;
  // The optimization program can add a constant term to the cost.
  double cost_constant_;
  std::unique_ptr<SCS_SOL_VARS, void (*)(SCS_SOL_VARS*)> scs_sol_;
  SCS_INFO scs_info_;
  double cost_;
  // Whether the solution of the optimization problem in this node satisfies all
  // integral constraints.
  bool found_integral_sol_;
  // binary_var_indices_ are the indices of the remaining binary variables, in
  // the vector x.
  std::list<int> binary_var_indices_;

  ScsNode* left_child_ = nullptr;
  ScsNode* right_child_ = nullptr;
  ScsNode* parent_ = nullptr;

  // If the solution is within integer_tol to an integer value, then we regard
  // the solution as taking the integer value.
  // TODO(hongkai.dai) Add a function to set the integer tolerance.
  double integer_tol_ = 1E-2;
};

/**
 * Given a mixed-integer convex optimization program in SCS format
 * <pre>
 * min cᵀx + d
 * s.t Ax + s = b
 *     s in K
 *     y are binary variables.
 * <pre>
 * where y is a subset of the variables x, and the indices of binary variable y
 * in x that should only take binary value {0, 1}, solve this mixed-integer
 * optimization problem through branch-and-bound.
 *
 * The performance of the branch-and-bound highly depends on some choices,
 * including
 * 1. Which leaf node to branch.
 * 2. Which binary variable to branch.
 * 3. How to find a feasible solution to the mixed-integer problem, from the
 *    solution to the problem in a node. This feasible solution will generate
 *    an upper bound of the mixed-integer problem.
 * In this class we provide default choices, and also virtual functions as
 * interfaces to implement the user's own choices. The user could inherit this
 * class, and implement their choices in the sub-class.
 */
class ScsBranchAndBound {
 public:
  enum class PickVariable {
    LeastAmbivalent,  // pick the variable that is closest to either 0 or 1
    MostAmbivalent    // pick the variable that is closest to 0.5
  };

  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ScsBranchAndBound)

  /**
   * Construct the root of the tree for the branch and bound. The mixed-integer
   * optimization problem is
   * <pre>
   * min cᵀx + d
   * s.t Ax + s = b
   *     s in K
   *     y are binary variables.
   * </pre>
   * @param scs_data scs_data contains the A, b, c matrices of the problem,
   * together with the settings of the problem. Notice that the data A, b, c do
   * NOT include the integral constraints on y, nor the relaxation 0 ≤ y ≤ 1.
   * scs_data should outlive the ScsBranchAndBound object constructed in this
   * function.
   * @param cone The cone `K` in the documentation above. This cone should
   * outlive the ScsBranchAndBound object constructed in this function.
   * @param cost_constant The constant term in the cost, `d` in the
   * documentation above.
   * @param binary_var_indices The indices of the binary variables y in x.
   */
  ScsBranchAndBound(const SCS_PROBLEM_DATA& scs_data, const SCS_CONE& cone,
                    double cost_constant,
                    const std::list<int>& binary_var_indices);

  virtual ~ScsBranchAndBound() {}

 protected:
  /**
   * Given a node, pick a binary variable in the node to branch.
   * The two child nodes are created in this function. The optimization problems
   * in the two child nodes are solved.
   * @param node
   */
  void PickBranchingVariableAndSolve(ScsNode* node);

 private:
  friend class ScsBranchAndBoundTest;  // Forward declaration

  /**
   * Pick one node to branch. The default is to pick the node with the smallest
   * optimal cost.
   */
  virtual ScsNode* PickBranchingNode() const;

  /**
   * Pick one variable to branch, returns the index of the branching variable.
   */
  virtual int PickBranchingVariable(ScsNode* node);

  // scs_data_ includes the data on c, A, b, and the cone K. It also contains
  // the settings of the problem, such as iteration limit, accuracy, etc.
  SCS_PROBLEM_DATA scs_data_;
  // binary_var_indices_ records the indices of all binary variables in x.
  std::list<int> binary_var_indices_;

  // The root of the tree
  std::unique_ptr<ScsNode> root_;

  // The best upper bound of the mixed-integer optimization optimal cost. An
  // upper bound is obtained by evaluating the cost at a solution satisfying
  // all the constraints in the mixed-integer problem.
  double best_upper_bound_;

  // The best lower bound of the mixed-integer optimization optimal cost. This
  // best lower bound is obtained by taking the minimal of the optimal cost in
  // each leaf node.
  double best_lower_bound_;

  // We will stop the branch and bound, and regard the best upper bound is
  // sufficiently close to the best lower bound, if
  // (best_upper_bound_ - best_lower_bound_) / abs(best_lower_bound) <
  // relative_gap_tol_
  double relative_gap_tol_;

  // The list of active leaves, i.e., the leaf nodes that have not been
  // fathomed. A leaf node is fathomed if
  // 1. The optimization problem in the node is infeasible.
  // 2. The optimal cost of the node is larger than the best upper bound.
  std::list<ScsNode*> active_leaves_;

  // Default way to pick a branching variable is "most ambivalent".
  constexpr PickVariable pick_variable = PickVariable::MostAmbivalent;
};
}  // namespace solvers
}  // namespace drake
