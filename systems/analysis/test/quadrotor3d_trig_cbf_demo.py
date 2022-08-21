import numpy as np
import pickle

import clf_cbf_utils

import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym
import pydrake.common


def Search(
    quadrotor: analysis.QuadrotorTrigPlant, x: np.ndarray, thrust_max: float,
        deriv_eps: float, unsafe_regions: list, x_safe: np.ndarray) -> sym.Polynomial:
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    u_vertices = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 0, 1, 0]]).T * thrust_max

    state_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])

    dynamics_denominator = None

    beta_minus = -0.005

    beta_plus = 0.005

    dut = analysis.ControlBarrier(
        f, G, dynamics_denominator, x, beta_minus, beta_plus, unsafe_regions,
        u_vertices, state_constraints)

    h_degree = 2
    x_set = sym.Variables(x)

    #h_init = sym.Polynomial((x[4] - 0.5) ** 2 + x[5] **
    #                        2 + x[6] ** 2 + 0.01 * x[7:].dot(x[7:]) - 0.07)
    with open("quadrotor3d_trig_cbf12.pickle", "rb") as input_file:
       h_init = clf_cbf_utils.deserialize_polynomial(
           x_set, pickle.load(input_file)["h"])

    h_init_x_safe = h_init.EvaluateIndeterminates(x, x_safe)
    print(f"h_init(x_safe): {h_init_x_safe.squeeze()}")
    if np.any(h_init_x_safe < 0):
        h_init -= h_init_x_safe.min()
        h_init += 0.1

    with_slack_a = True

    lambda0_degree = 4
    lambda1_degree = 4
    l_degrees = [2] * 16
    hdot_eq_lagrangian_degrees = [h_degree + lambda0_degree - 2]

    if with_slack_a:
        hdot_a_info = analysis.SlackPolynomialInfo(
            degree=6, poly_type=analysis.SlackPolynomialType.kSos,
            cost_weight=1.)

    t_degrees = [0]
    s_degrees = [[h_degree - 2]]
    unsafe_eq_lagrangian_degrees = [[h_degree - 2]]
    if with_slack_a:
        #unsafe_a_info = [analysis.SlackPolynomialInfo(
        #    degree=h_degree, poly_type=analysis.SlackPolynomialType.kSos,
        #    cost_weight=1.)]
        unsafe_a_info = [None]
    h_x_safe_min = np.array([0.01] * x_safe.shape[1])

    if with_slack_a:
        hdot_a_zero_tol = 1E-9
        unsafe_a_zero_tol = 1E-8
        search_options = analysis.ControlBarrier.SearchWithSlackAOptions(
            hdot_a_zero_tol, unsafe_a_zero_tol, use_zero_a=True)
        search_options.bilinear_iterations = 10
        search_options.lagrangian_step_solver_options = mp.SolverOptions()
        search_options.lagrangian_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.lagrangian_step_solver_options.SetOption(
            MosekSolver().id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1E-9)
        search_options.barrier_step_solver_options = mp.SolverOptions()
        search_options.barrier_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_solver_options.SetOption(
            MosekSolver().id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1E-9)
        search_options.barrier_step_backoff_scale = 0.01
        search_options.lagrangian_step_backoff_scale = 0.01
        search_options.hsol_tiny_coeff_tol = 1E-6
        search_result = dut.SearchWithSlackA(
            h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
            hdot_eq_lagrangian_degrees, hdot_a_info, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, unsafe_a_info, x_safe,
            h_x_safe_min, search_options)
        search_lagrangian_ret = dut.SearchLagrangian(
            search_result.h, deriv_eps, lambda0_degree, lambda1_degree, l_degrees,
            hdot_eq_lagrangian_degrees, None, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, [None] * len(unsafe_regions),
            search_options, backoff_scale=None)
    else:
        ellipsoids = [analysis.ControlBarrier.Ellipsoid(
            c=x_safe[:, 0], S=np.eye(13), d=0., r_degree=0,
            eq_lagrangian_degrees=[0])]
        ellipsoid_options = [
            analysis.ControlBarrier.EllipsoidMaximizeOption(
                t=sym.Polynomial(), s_degree=0, backoff_scale=0.04)]
        search_options = analysis.ControlBarrier.SearchOptions()
        search_options.lagrangian_step_solver_options = mp.SolverOptions()
        search_options.lagrangian_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_solver_options = mp.SolverOptions()
        search_options.barrier_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.lsol_tiny_coeff_tol = 0#1E-6
        search_options.hsol_tiny_coeff_tol = 0#1E-6
        search_options.barrier_step_backoff_scale = 0.1
        x_anchor = x_safe[:, 0]
        h_x_anchor_max = h_init.EvaluateIndeterminates(x, x_anchor)[0] * 2
        search_result = dut.Search(
            h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree,
            l_degrees, hdot_eq_lagrangian_degrees, t_degrees, s_degrees,
            unsafe_eq_lagrangian_degrees, x_anchor, h_x_anchor_max,
            search_options, ellipsoids, ellipsoid_options)
    with open("quadrotor3d_trig_cbf13.pickle", "wb") as handle:
        pickle.dump({
            "h": clf_cbf_utils.serialize_polynomial(search_result.h),
            "beta_plus": beta_plus, "beta_minus": beta_minus,
            "deriv_eps": deriv_eps, "thrust_max": thrust_max,
            "x_safe": x_safe}, handle)
    return search_result.h

#def reexecute_if_unbuffered():
#    """Ensures that output is immediately flushed (e.g. for segfaults).
#    ONLY use this at your entrypoint. Otherwise, you may have code be
#    re-executed that will clutter your console."""
#    import os
#    import shlex
#    import sys
#    if os.environ.get("PYTHONUNBUFFERED") in (None, ""):
#        os.environ["PYTHONUNBUFFERED"] = "1"
#        argv = list(sys.argv)
#        if argv[0] != sys.executable:
#            argv.insert(0, sys.executable)
#        cmd = " ".join([shlex.quote(arg) for arg in argv])
#        sys.stdout.flush()
#        os.execv(argv[0], argv)
#
#
#def traced(func, ignoredirs=None):
#    """Decorates func such that its execution is traced, but filters out any
#     Python code outside of the system prefix."""
#    import functools
#    import sys
#    import trace
#    if ignoredirs is None:
#        ignoredirs = ["/usr", sys.prefix]
#    tracer = trace.Trace(trace=1, count=0, ignoredirs=ignoredirs)
#
#    @functools.wraps(func)
#    def wrapped(*args, **kwargs):
#        return tracer.runfunc(func, *args, **kwargs)
#
#    return wrapped
#
#@traced
def main():
    pydrake.common.configure_logging()
    quadrotor = analysis.QuadrotorTrigPlant()
    x = sym.MakeVectorContinuousVariable(13, "x")
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
    deriv_eps = 1.
    # Unsafe region.
    # 1. The quadrotor gets close to a sphere at (0.5, 0, 0)
    unsafe_regions = [np.array([sym.Polynomial(
        (x[4] - 0.5) ** 2 + x[5] ** 2 + x[6] ** 2- quadrotor.length() ** 2)])]
    x_safe = np.empty((13, 2))
    x_safe[:, 0] = np.zeros(13)
    x_safe[:, 1] = np.zeros(13)
    x_safe[4, 1] = 1
    h_sol = Search(quadrotor, x, thrust_max, deriv_eps, unsafe_regions, x_safe)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
