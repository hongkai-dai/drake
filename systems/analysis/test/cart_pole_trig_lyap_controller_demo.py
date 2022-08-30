"""
Jointly searches for the controller, the Lyapunov function and the Lagrangian multipliers.
"""
import numpy as np
import pickle

import clf_cbf_utils
import pydrake.systems.analysis as analysis
import pydrake.systems.primitives as primitives
from pydrake.solvers import MathematicalProgram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym
import cart_pole_trig_clf_demo

"""
We search for the control policy
−κ*V(x)*d(x)−∂V/∂x*(f(x)+G(x)π(x))−λ₁(x)(1−V) - l1(x)*(π(x)+u_max) - l2(x)*(u_max - π(x)) - state_eq_lagrangian1(x) * state_eq_lagrangian(x) is sos
λ₁(x) is sos, l1(x) is sos, l2(x) is sos.
−κ*V(x)*d(x)−∂V/∂x*(f(x)+G(x)u_max)−λ2(x)(1−V) - l3(x)*(π(x)-u_max) - state_eq_lagrangian2(x) * state_eq_lagrangian(x) is sos
λ2(x) is sos, l3(x) is sos
−κ*V(x)*d(x)−∂V/∂x*(f(x)-G(x)u_max)−λ3(x)(1−V) - l4(x)*(-π(x)-u_max) - state_eq_lagrangian3(x) * state_eq_lagrangian(x) is sos
λ3(x) is sos, l4(x) is sos
"""

def add_vdot_sos_constraint(prog, x, f, G, dynamics_denominator, V, u, kappa, u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4):
    x_set = sym.Variables(x)
    state_eq_lagrangian1 = prog.NewFreePolynomial(x_set, 2)
    dVdx = V.Jacobian(x)
    vdot_sos_condition1 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        - dVdx.dot(G * u) - lambda1 * (1 - V) - l1 * (u + u_max)\
        - l2 * (u_max - u) - state_eq_lagrangian1 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition1)

    state_eq_lagrangian2 = prog.NewFreePolynomial(x_set, 2)
    vdot_sos_condition2 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        - dVdx.dot(G * u_max) - lambda2 * (1 - V) - l3 * (u - u_max)\
        - state_eq_lagrangian2 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition2)

    state_eq_lagrangian3 = prog.NewFreePolynomial(x_set, 2)
    vdot_sos_condition3 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        + dVdx.dot(G * u_max) - lambda3 * (1-V) - l4*(-u - u_max)\
        - state_eq_lagrangian3 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition3)


def search_lagrangian(x, f, G, dynamics_denominator, V, u, kappa, u_max):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    lambda1, _ = prog.NewSosPolynomial(x_set, 2)
    lambda2, _ = prog.NewSosPolynomial(x_set, 2)
    lambda3, _ = prog.NewSosPolynomial(x_set, 2)
    l1, _ = prog.NewSosPolynomial(x_set, 2)
    l2, _ = prog.NewSosPolynomial(x_set, 2)
    l3, _ = prog.NewSosPolynomial(x_set, 2)
    l4, _ = prog.NewSosPolynomial(x_set, 2)
    add_vdot_sos_constraint(prog, x, f, G, dynamics_denominator, V, u, kappa, u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4)
    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    lambda1_sol = result.GetSolution(lambda1)
    lambda2_sol = result.GetSolution(lambda2)
    lambda3_sol = result.GetSolution(lambda3)
    l1_sol = result.GetSolution(l1)
    l2_sol = result.GetSolution(l2)
    l3_sol = result.GetSolution(l3)
    l4_sol = result.GetSolution(l4)
    return lambda1_sol, lambda2_sol, lambda3_sol, l1_sol, l2_sol, l3_sol, l4_sol


def search_controller(x, f, G, dynamics_denominator, V, pi_degree, kappa, u_max, l1, l2, l3, l4):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    u = prog.NewFreePolynomial(x_set, pi_degree)
    lambda1, _ = prog.NewSosPolynomial(x_set, 2)
    lambda2, _ = prog.NewSosPolynomial(x_set, 2)
    lambda3, _ = prog.NewSosPolynomial(x_set, 2)

    add_vdot_sos_constraint(prog, x, f, G, dynamics_denominator, V, u, kappa, u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    u_sol = result.GetSolution(u)
    lambda1_sol = result.GetSolution(lambda1)
    lambda2_sol = result.GetSolution(lambda2)
    lambda3_sol = result.GetSolution(lambda3)
    return u_sol, lambda1_sol, lambda2_sol, lambda3_sol

def search_V(x, f, G, dynamics_denominator, V_degree, u, kappa, u_max, lambda1, lambda2, lambda3, positivity_eps, x_samples):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    # Add constraint V>=0
    V = prog.NewFreePolynomial(x_set, V_degree)
    positivity_eq_lagrangian = prog.NewFreePolynomial(x_set, V_degree - 2)
    prog.AddSosConstraint(V - positivity_eps * sym.Polynomial(sym.pow(x.dot(x), V_degree / 2)) - positivity_eq_lagrangian * state_constraint)
    l1, _ = prog.NewSosPolynomial(x_set, 2)
    l2, _ = prog.NewSosPolynomial(x_set, 2)
    l3, _ = prog.NewSosPolynomial(x_set, 2)
    l4, _ = prog.NewSosPolynomial(x_set, 2)
    add_vdot_sos_constraint(
        prog, x, f, G, dynamics_denominator, V, u, kappa, u_max, lambda1,
        lambda2, lambda3, l1, l2, l3, l4)
    analysis.OptimizePolynomialAtSamples(prog, V, x, x_samples, analysis.OptimizePolynomialMode, kMinimizeMaximal)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = analysis.SearchWithBackoff(
        prog, MosekSolver.id(), solver_options, backoff_scale=0.01)
    assert (result.is_success())
    V_sol = result.GetSolution(V)
    l1_sol = result.GetSolution(l1)
    l2_sol = result.GetSolution(l2)
    l3_sol = result.GetSolution(l3)
    l4_sol = result.GetSolution(l4)
    return V_sol, l1_sol, l2_sol, l3_sol, l4_sol

def search():
    params = analysis.CartPoleParams()
    x = sym.MakeVectorContinuousVariable(5, "x")
    u_max = 176
    kappa = 0.01
    V_degree = 2
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)

    V_init = cart_pole_trig_clf_demo.FindClfInit(params, V_degree, x)
    lqr_Q_diag = np.array([1, 1, 1, 10, 10.])
    lqr_Q = np.diag(lqr_Q_diag)
    K, _ = analysis.SynthesizeCartpoleTrigLqr(params, lqr_Q, R=20)
    u_lqr = sym.Polynomial(-K[0, :] @ x)

    lambda1, lambda2, lambda3, l1, l2, l3, l4 = search_lagrangian(x, f, G, dynamics_denominator, V, u_lqr, kappa, u_max)


if __name__ == "__main__":
    search()


