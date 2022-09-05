"""
Jointly searches for the controller, the Lyapunov function and the Lagrangian multipliers.
"""
import numpy as np
import pickle

import clf_cbf_utils
import pydrake.systems.analysis as analysis
import pydrake.systems.primitives as primitives
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym
import cart_pole_trig_clf_demo

"""
We search for the control policy
−κ*V(x)*d(x)−∂V/∂x*(f(x)+G(x)π(x))−λ₁(x)(ρ−V) - l1(x)*(π(x)+u_max) - l2(x)*(u_max - π(x)) - state_eq_lagrangian1(x) * state_eq_lagrangian(x) is sos
λ₁(x) is sos, l1(x) is sos, l2(x) is sos.
−κ*V(x)*d(x)−∂V/∂x*(f(x)+G(x)u_max)−λ2(x)(ρ−V) - l3(x)*(π(x)-u_max) - state_eq_lagrangian2(x) * state_eq_lagrangian(x) is sos
λ2(x) is sos, l3(x) is sos
−κ*V(x)*d(x)−∂V/∂x*(f(x)-G(x)u_max)−λ3(x)(ρ−V) - l4(x)*(-π(x)-u_max) - state_eq_lagrangian3(x) * state_eq_lagrangian(x) is sos
λ3(x) is sos, l4(x) is sos
"""

def add_vdot_sos_constraint(prog, x, f, G, dynamics_denominator, state_constraint, V, rho, u, kappa, u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4, V_degree, pi_degree):
    x_set = sym.Variables(x)
    state_eq_lagrangian1 = prog.NewFreePolynomial(x_set, V_degree + pi_degree - 1)
    dVdx = V.Jacobian(x)
    vdot_sos_condition1 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        - dVdx.dot(G * u) - lambda1 * (rho - V) - l1 * (u + u_max)\
        - l2 * (u_max - u) - state_eq_lagrangian1 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition1)

    state_eq_lagrangian2 = prog.NewFreePolynomial(x_set, V_degree + pi_degree - 1)
    vdot_sos_condition2 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        - dVdx.dot(G * u_max) - lambda2 * (rho - V) - l3 * (u - u_max)\
        - state_eq_lagrangian2 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition2)

    state_eq_lagrangian3 = prog.NewFreePolynomial(x_set, V_degree + pi_degree - 1)
    vdot_sos_condition3 = -kappa * V * dynamics_denominator - dVdx.dot(f)\
        + dVdx.dot(G * u_max) - lambda3 * (rho-V) - l4*(-u - u_max)\
        - state_eq_lagrangian3 * state_constraint
    prog.AddSosConstraint(vdot_sos_condition3)


def search_lagrangian(x, f, G, dynamics_denominator, V, rho, u, kappa, u_max, V_degree, pi_degree, lagrangian_tiny_coeff_tol):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    lambda_degree = V_degree + pi_degree - 1
    lambda1, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    lambda2, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    lambda3, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    l1, _ = prog.NewSosPolynomial(x_set, V_degree)
    l2, _ = prog.NewSosPolynomial(x_set, V_degree)
    l3, _ = prog.NewSosPolynomial(x_set, V_degree)
    l4, _ = prog.NewSosPolynomial(x_set, V_degree)
    add_vdot_sos_constraint(
        prog, x, f, G, dynamics_denominator, state_constraint, V, rho, u, kappa,
        u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4, V_degree, pi_degree)
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
    if lagrangian_tiny_coeff_tol is not None:
        lambda1_sol = lambda1_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        lambda2_sol = lambda2_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        lambda3_sol = lambda3_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l1_sol = l1_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l2_sol = l2_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l3_sol = l3_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
    return lambda1_sol, lambda2_sol, lambda3_sol, l1_sol, l2_sol, l3_sol, l4_sol


def search_controller(x, f, G, dynamics_denominator, V, rho, pi_degree, kappa, u_max, l1, l2, l3, l4):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    u = prog.NewFreePolynomial(x_set, pi_degree)
    V_degree = V.TotalDegree()
    lambda_degree = V_degree + pi_degree - 1
    lambda1, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    lambda2, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    lambda3, _ = prog.NewSosPolynomial(x_set, lambda_degree)

    add_vdot_sos_constraint(prog, x, f, G, dynamics_denominator, state_constraint, V, rho, u, kappa, u_max, lambda1, lambda2, lambda3, l1, l2, l3, l4, V_degree, pi_degree)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    u_sol = result.GetSolution(u)
    lambda1_sol = result.GetSolution(lambda1)
    lambda2_sol = result.GetSolution(lambda2)
    lambda3_sol = result.GetSolution(lambda3)
    return u_sol, lambda1_sol, lambda2_sol, lambda3_sol

def search_controller2(x, f, G, dynamics_denominator, V, rho, pi_degree, kappa, u_max):
    """
    Try the alternative formulation that doesn't use piecewise polynomial controller.
     −κ V(x)d(x)-∂V/∂x*(f(x)+G(x)π(x)) - λ₁(x)(1-V) - state_eq_lagrangian(x) * state_eq_constraint(x) is sos
     λ₁(x) is sos
     Since we also need -u_max <= π(x) <= u_max when V(x)<=1, we impose
     π(x)+u_max - l1(x)(1-V) is sos
     (u_max - π(x)) - l2(x)(1-V) is sos
     l1(x) is sos, l2(x) is sos
    """
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    dVdx = V.Jacobian(x)
    lambda1_degree = pi_degree + 1
    lambda1, _ = prog.NewSosPolynomial(x_set, lambda1_degree)
    u = prog.NewFreePolynomial(x_set, pi_degree)
    state_eq_lagrangian_degree = pi_degree + 1
    state_eq_lagrangian = prog.NewFreePolynomial(x_set, state_eq_lagrangian_degree)
    Vdot_sos_condition = -kappa * V * dynamics_denominator - dVdx.dot(f) - dVdx.dot(G * u) - lambda1 * (rho - V) - state_eq_lagrangian * state_constraint
    prog.AddSosConstraint(Vdot_sos_condition)
    # -u_max <= u <= u_max
    l1, _ = prog.NewSosPolynomial(x_set, pi_degree - 1)
    u_lb_state_eq_lagrangian = prog.NewFreePolynomial(x_set, pi_degree - 1)
    prog.AddSosConstraint(u + u_max - l1 * (rho - V) - u_lb_state_eq_lagrangian * state_constraint)
    l2, _ = prog.NewSosPolynomial(x_set, pi_degree - 1)
    u_ub_state_eq_lagrangian = prog.NewFreePolynomial(x_set, pi_degree - 1)
    prog.AddSosConstraint(u_max - u - l2 * (rho - V) - u_ub_state_eq_lagrangian * state_constraint)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    u_sol = result.GetSolution(u)
    lambda1_sol = result.GetSolution(lambda1)
    l1_sol = result.GetSolution(l1)
    l2_sol = result.GetSolution(l2)
    return u_sol, lambda1_sol, l1_sol, l2_sol

def search_V(x, f, G, dynamics_denominator, V_degree, u, kappa, u_max, lambda1, lambda2, lambda3, positivity_eps, x_samples, pi_degree, lagrangian_tiny_coeff_tol):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    # Add constraint V>=0
    V = prog.NewFreePolynomial(x_set, V_degree)
    rho = prog.NewContinuousVariables(1)[0]
    positivity_eq_lagrangian = prog.NewFreePolynomial(x_set, V_degree - 2)
    prog.AddSosConstraint(V - positivity_eps * sym.Polynomial(sym.pow(x.dot(x), V_degree / 2)) - positivity_eq_lagrangian * state_constraint)
    l1, _ = prog.NewSosPolynomial(x_set, 2)
    l2, _ = prog.NewSosPolynomial(x_set, 2)
    l3, _ = prog.NewSosPolynomial(x_set, 2)
    l4, _ = prog.NewSosPolynomial(x_set, 2)
    add_vdot_sos_constraint(
        prog, x, f, G, dynamics_denominator, state_constraint, V, rho, u, kappa, u_max, lambda1,
        lambda2, lambda3, l1, l2, l3, l4, V_degree, pi_degree)
    analysis.OptimizePolynomialAtSamples(prog, V, x, x_samples, analysis.OptimizePolynomialMode.kMinimizeMaximal)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = analysis.SearchWithBackoff(
        prog, MosekSolver.id(), solver_options, backoff_scale=0.03)
    assert (result.is_success())
    V_sol = result.GetSolution(V)
    rho_sol = result.GetSolution(rho)
    l1_sol = result.GetSolution(l1)
    l2_sol = result.GetSolution(l2)
    l3_sol = result.GetSolution(l3)
    l4_sol = result.GetSolution(l4)
    if lagrangian_tiny_coeff_tol is not None:
        l1_sol = l1_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l2_sol = l2_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l3_sol = l3_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
        l4_sol = l4_sol.RemoveTermsWithSmallCoefficients(lagrangian_tiny_coeff_tol)
    return V_sol, rho_sol, l1_sol, l2_sol, l3_sol, l4_sol

def search():
    params = analysis.CartPoleParams()
    x = sym.MakeVectorContinuousVariable(5, "x")
    x_set = sym.Variables(x)
    u_max = 176
    kappa = 0.01
    V_degree = 2
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)

    load_V_init = True
    if load_V_init:
        with open("/home/hongkaidai/sos_clf_cbf_data/cart_pole/cartpole_trig_clf50.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(x_set, pickle.load(input_file)["V"])
            rho_init = 1.
            pi_degree = 3
            u_init, _, _, _ = search_controller2(x, f, G, dynamics_denominator, V_init, rho_init, pi_degree, kappa, u_max)
    else:
        V_init = cart_pole_trig_clf_demo.FindClfInit(params, V_degree, x)
        rho_init = 1.
        lqr_Q_diag = np.array([1, 1, 1, 10, 10.])
        lqr_Q = np.diag(lqr_Q_diag)
        K, _ = analysis.SynthesizeCartpoleTrigLqr(params, lqr_Q, R=20)
        u_init = sym.Polynomial(-K[0, :] @ x)
        pi_degree = 3

    lagrangian_tiny_coeff_tol = 1E-6
    lambda1_sol, lambda2_sol, lambda3_sol, l1_sol, l2_sol, l3_sol, l4_sol = \
        search_lagrangian(
            x, f, G, dynamics_denominator, V_init, rho_init, u_init, kappa, u_max,
            V_degree, pi_degree, lagrangian_tiny_coeff_tol)
    iter_count = 0
    V_sol = V_init
    rho_sol = rho_init
    u_sol = u_init
    state_swingup, control_swingup = cart_pole_trig_clf_demo.SwingUpTrajectoryOptimization()
    x_swingup = np.empty((5, state_swingup.shape[1]))
    for i in range(state_swingup.shape[1]):
        x_swingup[:, i] = analysis.ToCartpoleTrigState(state_swingup[:, i])
    x_samples = x_swingup[:, -18]
    positivity_eps = 0.0001
    while iter_count < 20:
        u_sol, lambda1_sol, lambda2_sol, lambda3_sol = search_controller(
            x, f, G, dynamics_denominator, V_sol, rho_sol, pi_degree, kappa, u_max,
            l1_sol, l2_sol, l3_sol, l4_sol)
        V_sol, rho_sol, l1_sol, l2_sol, l3_sol, l4_sol = search_V(
            x, f, G, dynamics_denominator, V_degree, u_sol, kappa, u_max,
            lambda1_sol, lambda2_sol, lambda3_sol, positivity_eps, x_samples,
            pi_degree, lagrangian_tiny_coeff_tol)
        print(f"iter={iter_count}, V(x_samples)={V_sol.EvaluateIndeterminates(x, x_samples).T}")
        iter_count += 1
    pass


if __name__ == "__main__":
    search()


