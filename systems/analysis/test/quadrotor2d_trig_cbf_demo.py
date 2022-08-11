import numpy as np

import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym


def SearchWithSlackA(
    quadrotor: analysis.Quadrotor2dTrigPlant, x: np.array, thrust_max: float,
    deriv_eps: float, unsafe_regions: list, x_safe: np.array
) -> sym.Polynomial:
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    u_vertices = np.array([
        [0., 0, thrust_max, thrust_max],
        [0, thrust_max, 0., thrust_max]])
    state_constraints = np.array([analysis.Quadrotor2dStateEqConstraint(x)])

    dynamics_denominator = None

    beta_minus = 0.
    beta_plus = None

    dut = analysis.ControlBarrier(
        f, G, dynamics_denominator, x, beta_minus, beta_plus, unsafe_regions,
        u_vertices, state_constraints)
    h_degree = 2
    x_set = sym.Variables(x)
    h_init = sym.Polynomial(1 - x.dot(x))

    h_init_x_safe = h_init.EvaluateIndeterminates(x, x_safe)
    print(f"h_init(x_safe): {h_init_x_safe.squeeze()}")
    if np.any(h_init_x_safe < 0):
        h_init -= h_init_x_safe.min()
        h_init += 0.1

    lambda0_degree = 4
    lambda1_degree = None
    l_degrees = [2, 2, 2, 2]
    hdot_eq_lagrangian_degrees = [h_degree + lambda0_degree - 2]
    hdot_a_degree = h_degree + lambda0_degree

    t_degrees = [0, 0]
    s_degrees = [[h_degree - 2], [h_degree - 2]]
    unsafe_eq_lagrangian_degrees = [[h_degree - 2], [h_degree - 2]]
    unsafe_a_degrees = [h_degree, h_degree]
    h_x_safe_min = np.array([0.01])
    hdot_a_zero_tol = 3E-9
    unsafe_a_zero_tol = 1E-9

    search_options = analysis.ControlBarrier.SearchWithSlackAOptions(
        hdot_a_zero_tol, unsafe_a_zero_tol, use_zero_a=True,
        hdot_a_cost_weight=1., unsafe_a_cost_weight=[1., 1.])
    search_options.bilinear_iterations = 100
    search_result = dut.SearchWithSlackA(
        h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
        hdot_eq_lagrangian_degrees, hdot_a_degree, t_degrees, s_degrees,
        unsafe_eq_lagrangian_degrees, unsafe_a_degrees, x_safe,
        h_x_safe_min, search_options)
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_lagrangian_ret = dut.SearchLagrangian(
        search_result.h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
        hdot_eq_lagrangian_degrees, None, t_degrees, s_degrees,
        unsafe_eq_lagrangian_degrees, [None] * len(unsafe_regions),
        search_options, backoff_scale=None)
    return search_result.h


def DoMain():
    plant = analysis.Quadrotor2dTrigPlant()
    x = np.empty(7, dtype=object)
    for i in range(7):
        x[i] = sym.Variable(f"x{i}")

    thrust_equilibrium = analysis.EquilibriumThrust(plant)
    thrust_max = 3 * thrust_equilibrium
    deriv_eps = 0.5
    unsafe_regions = [None, None]
    unsafe_regions[0] = np.array([sym.Polynomial(x[1] + 0.3)])
    unsafe_regions[1] = np.array([sym.Polynomial(0.5 - x[1])])
    safe_states = np.empty((6, 1))
    safe_states[:, 0] = np.array([0., 0., 0., 0., 0., 0.])
    x_safe = np.empty((7, safe_states.shape[1]))
    for i in range(x_safe.shape[1]):
        x_safe[:, i] = analysis.ToQuadrotor2dTrigState(safe_states[:, i])

    h_sol = SearchWithSlackA(plant, x, thrust_max,
                             deriv_eps, unsafe_regions, x_safe)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        DoMain()
