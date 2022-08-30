"""
Jointly searches for the controller, the Lyapunov function and the Lagrangian multipliers.
"""
import numpy as np
import pickle

import clf_cbf_utils
import pydrake.systems.analysis as analysis
import pydrake.systems.primitives as primitives
import pydrake.systems.controllers as controllers
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
from pydrake.solvers.gurobi import GurobiSolver
import pydrake.symbolic as sym
import pydrake.common
from pydrake.systems.framework import (
    DiagramBuilder,
    LeafSystem,
)
from pydrake.systems.primitives import LogVectorOutput
from pydrake.examples import (
    QuadrotorGeometry,
    QuadrotorPlant,
)
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    SceneGraph,
)
import quadrotor3d_trig_clf_demo

def search_controller(x, f, G, V, pi_degree, kappa, thrust_max, lambda1_degree, l1_degrees, l2_degrees):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    state_constraint = analysis.QuadrotorStateEqConstraint(x)
    dVdx = V.Jacobian(x)
    lambda1, _ = prog.NewSosPolynomial(x_set, lambda1_degree)
    u = np.empty((4,), dtype=object)
    for i in range(4):
        u[i] = prog.NewFreePolynomial(x_set, pi_degree)
    state_eq_lagrangian_degree = 4
    state_eq_lagrangian = prog.NewFreePolynomial(x_set, state_eq_lagrangian_degree)
    Vdot_sos_condition = -kappa * V - dVdx.dot(f) - dVdx.dot(G @ u) - lambda1 * (1 - V) - state_eq_lagrangian * state_constraint
    prog.AddSosConstraint(Vdot_sos_condition)
    # Now add the constraint that 0 <= u <= thrust_max
    l1 = np.empty((4,), dtype=object)
    l2 = np.empty((4,), dtype=object)
    u_lb_state_eq_lagrangian = np.empty((4,), dtype=object)
    u_ub_state_eq_lagrangian = np.empty((4,), dtype=object)
    for i in range(4):
        l1[i], _ = prog.NewSosPolynomial(x_set, l1_degrees[i])
        u_lb_state_eq_lagrangian[i] = prog.NewFreePolynomial(x_set, 2)
        prog.AddSosConstraint(
            u[i] - l1[i] * (1 - V)
            - u_lb_state_eq_lagrangian[i] * state_constraint)
        l2[i], _ = prog.NewSosPolynomial(x_set, l2_degrees[i])
        u_ub_state_eq_lagrangian[i] = prog.NewFreePolynomial(x_set, 2)
        prog.AddSosConstraint(
            thrust_max - u[i] - l2[i] * (1-V)
            - u_ub_state_eq_lagrangian[i] * state_constraint)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    lambda1_sol = result.GetSolution(lambda1)
    u_sol = np.array([result.GetSolution(u[i]) for i in range(4)])
    l1_sol = np.array([result.GetSolution(l1[i]) for i in range(4)])
    l2_sol = np.array([result.GetSolution(l2[i]) for i in range(4)])
    return u_sol, lambda1_sol, l1_sol, l2_sol


def search_V(x, f, G, u, V_degree, kappa, thrust_max, lambda1, l1, l2, positivity_eps, x_samples):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    V = prog.NewFreePolynomial(x_set, V_degree)
    state_constraint = analysis.QuadrotorStateEqConstraint(x)
    positivity_eq_lagrangian = prog.NewFreePolynomial(x_set, V_degree - 2)
    prog.AddSosConstraint(
        V - positivity_eps * sym.Polynomial(pow(x.dot(x), V_degree / 2))
        - positivity_eq_lagrangian * state_constraint)
    dVdx = V.Jacobian(x)
    state_eq_lagrangian_degree = 4
    state_eq_lagrangian = prog.NewFreePolynomial(x_set, state_eq_lagrangian_degree)
    Vdot_sos_condition = -kappa * V - dVdx.dot(f) - dVdx.dot(G @ u) - lambda1 * (1 - V) - state_eq_lagrangian * state_constraint
    prog.AddSosConstraint(Vdot_sos_condition)
    # Now add the constraint V<=1 => 0<=u<=thrust_max
    u_lb_state_eq_lagrangian = np.empty((4,), dtype=object)
    u_ub_state_eq_lagrangian = np.empty((4,), dtype=object)
    for i in range(4):
        u_lb_state_eq_lagrangian[i] = prog.NewFreePolynomial(x_set, 2)
        prog.AddSosConstraint(
            u[i] - l1[i] * (1 - V)
            - u_lb_state_eq_lagrangian[i] * state_constraint)
        u_ub_state_eq_lagrangian[i] = prog.NewFreePolynomial(x_set, 2)
        prog.AddSosConstraint(
            thrust_max - u[i] - l2[i] * (1-V)
            - u_ub_state_eq_lagrangian[i] * state_constraint)

    analysis.OptimizePolynomialAtSamples(prog, V, x, x_samples, analysis.OptimizePolynomialMode.kMinimizeMaximal)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = analysis.SearchWithBackoff(
        prog, MosekSolver.id(), solver_options, backoff_scale=0.01)
    assert (result.is_success())
    V_sol = result.GetSolution(V)
    return V_sol


def search():
    quadrotor = analysis.QuadrotorTrigPlant()
    x = sym.MakeVectorContinuousVariable(13, "x")
    x_set = sym.Variables(x)
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
    state_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])

    V_degree = 2
    pi_degree = 3
    load_V_init = True
    if load_V_init:
        with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_clf/quadrotor3d_trig_clf_init.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, pickle.load(input_file)["V"])
    else:
        V_init = quadrotor3d_trig_clf_demo.FindClfInit(V_degree, x)
    # The condition is V<=1 => ∂V/∂x*(f(x)+G(x)π(x))≤ −κ V(x)
    # where the policy π(x) is within [0, thrust_max]
    # So we impose the following sos
    # −κ V(x)-∂V/∂x*(f(x)+G(x)π(x)) - λ₁(x)(1-V) - state_eq_lagrangian(x) * state_eq_constraint(x) is sos
    # λ₁(x) is sos
    # Since we also need 0 <= π(x) <= thrust_max when V(x)<=1, we impose
    # π(x) - l1(x)(1-V) is sos
    # (thrust_max - π(x)) - l2(x)(1-V) is sos
    # l1(x) is sos, l2(x) is sos
    lambda1_degree = 4
    l1_degrees = [2, 2, 2, 2]
    l2_degrees = [2, 2, 2, 2]
    iter_count = 0
    V_sol = V_init * 1000
    state_samples = np.zeros((12, 4))
    state_samples[:, 0] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 1] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 2] = np.array(
        [1, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 3] = np.array(
        [1, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0.])
    x_samples = np.zeros((13, state_samples.shape[1]))

    positivity_eps = 0.0001
    kappa = 0.1
    for i in range(state_samples.shape[1]):
        x_samples[:, i] = analysis.ToQuadrotorTrigState(state_samples[:, i])
    while iter_count < 28:
        print(f"iter={iter_count}")
        u, lambda1, l1, l2 = search_controller(
            x, f, G, V_sol, pi_degree, kappa, thrust_max, lambda1_degree,
            l1_degrees, l2_degrees)
        V_sol = search_V(
            x, f, G, u, V_degree, kappa, thrust_max, lambda1, l1, l2,
            positivity_eps, x_samples)
        print(f"V(x_samples)={V_sol.EvaluateIndeterminates(x, x_samples).T}")
        with open(f"quadrotor3d_trig_V_cubic_controller{iter_count}.pickle", "wb") as handle:
            pickle.dump({
                "V": clf_cbf_utils.serialize_polynomial(V_sol),
                "u": [clf_cbf_utils.serialize_polynomial(u[i]) for i in range(4)],
                "kappa": kappa,
                "thrust_max": thrust_max}, handle)
        iter_count += 1
    pass

if __name__ == "__main__":
    search()
