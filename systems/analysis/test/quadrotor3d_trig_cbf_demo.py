import numpy as np
import pickle

import clf_cbf_utils

import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
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

from pydrake.geometry import(
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    SceneGraph,
)

meshcat = StartMeshcat()

class QuadrotorCbfController(LeafSystem):
    def __init__(self, x, f, G, cbf, deriv_eps, thrust_max, beta_minus, beta_plus):
        LeafSystem.__init__(self)
        assert (x.shape == (13,))
        self.x = x
        assert (f.shape == (13,))
        self.f = f
        assert (G.shape == (13, 4))
        self.G = G
        self.cbf = cbf
        self.deriv_eps = deriv_eps
        self.thrust_max = thrust_max
        self.beta_minus = beta_minus
        self.beta_plus = beta_plus
        dhdx = self.cbf.Jacobian(self.x)
        self.dhdx_times_f = dhdx.dot(self.f)
        self.dhdx_times_G = dhdx @ self.G

        self.x_input_index = self.DeclareVectorInputPort("x", 13).get_index()
        self.control_output_index = self.DeclareVectorOutputPort("control", 4, self.CalcControl).get_index()
        self.cbf_output_index = self.DeclareVectorOutputPort("cbf", 1, self.CalcCbf).get_index()

    def x_input_port(self):
        return self.get_input_port(self.x_input_index)

    def control_output_port(self):
        return self.get_output_port(self.control_output_index)

    def cbf_output_port(self):
        return self.get_output_port(self.cbf_output_index)

    def CalcControl(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(13)}

        prog = mp.MathematicalProgram()
        nu = 4
        u = prog.NewContinuousVariables(nu, "u")
        prog.AddBoundingBoxConstraint(0, self.thrust_max, u)
        prog.AddQuadraticCost(np.identity(nu), np.zeros((nu,)), 0, u)
        dhdx_times_f_val = self.dhdx_times_f.Evaluate(env)
        dhdx_times_G_val = np.array([
            self.dhdx_times_G[i].Evaluate(env) for i in range(nu)])
        h_val = self.cbf.Evaluate(env)
        # dhdx * G * u + dhdx * f >= -eps * h
        if self.beta_minus <= h_val <= self.beta_plus:
            prog.AddLinearConstraint(
                dhdx_times_G_val.reshape((1, -1)),
                np.array([-self.deriv_eps * h_val - dhdx_times_f_val]),
                np.array([np.inf]), u)
        result = mp.Solve(prog)
        if not result.is_success():
            raise Exception("CBF controller cannot find u")
        output.SetFromVector(result.GetSolution(u))

    def CalcCbf(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i] : x_val[i] for i in range(13)}
        output.SetFromVector(np.array([self.cbf.Evaluate(env)]))

def simulate(x, f, G, cbf, thrust_max, deriv_eps, beta_minus, beta_plus, initial_state, duration):
    builder = DiagramBuilder()

    quadrotor = builder.AddSystem(QuadrotorPlant())

    scene_graph = builder.AddSystem(pydrake.geometry.SceneGraph())

    geom = QuadrotorGeometry.AddToBuilder(
        builder, quadrotor.get_output_port(0), scene_graph)

    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception))

    state_converter = builder.AddSystem(analysis.QuadrotorTrigStateConverter())

    builder.Connect(quadrotor.get_output_port(0), state_converter.get_input_port())

    cbf_controller = builder.AddSystem(QuadrotorCbfController(x,f, G, cbf, deriv_eps, thrust_max, beta_minus, beta_plus))

    builder.Connect(cbf_controller.control_output_port(), quadrotor.get_input_port())
    builder.Connect(state_converter.get_output_port(), cbf_controller.x_input_port())

    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)
    cbf_logger = LogVectorOutput(cbf_controller.cbf_output_port(), builder)
    control_logger = LogVectorOutput(cbf_controller.control_output_port(), builder)

    diagram = builder.Build()

    simulator = analysis.Simulator(diagram)

    analysis.ResetIntegratorFromFlags(simulator, "implicit_euler", 0.001)

    simulator.get_mutable_context().SetContinuousState(initial_state)
    simulator.AdvanceTo(duration)

    state_data = state_logger.FindLog(simulator.get_context()).data()
    cbf_data = cbf_logger.FindLog(simulator.get_context()).data()
    control_data = control_logger.FindLog(simulator.get_context()).data()
    pass

def get_u_vertices(thrust_max):
    return np.array([
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

def search_sphere_obstacle_cbf(x, f, G, beta_minus, beta_plus, thrust_max, kappa, x_safe, h_init):
    """
    Given h_init that already satisfies hdot >= -kappa*h, try to minimize h(sphere_center) while keeping h(x_safe) >= 0
    """
    state_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])
    dut = analysis.ControlBarrier(f, G, None, x, beta_minus, beta_plus, [], u_vertices, state_constraints)

    assert (np.all(h_init.EvaluateIndeterminates(x, x_safe) >= 0))
    iter_count = 0
    h_sol = h_init

    h_degree = 2
    lambda0_degree = 4
    lambda1_degree = 4
    l_degrees = [2] * 16
    hdot_eq_lagrangian_degrees = [h_degree + lambda0_degree - 2]
    t_degrees = []
    s_degrees = [[]]
    unsafe_state_constraints_lagrangian_degrees = [[]]
    unsafe_a_info = [None]
    search_options = analysis.ControlBarrier.SearchWithSlackAOptions(
        hdot_a_zero_tol, unsafe_a_zero_tol, use_zero_a=True)
    search_options.bilinear_iterations = 20
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.barrier_step_solver_options = mp.SolverOptions()
    search_options.barrier_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.barrier_step_backoff_scale = 0.02
    sphere_center = np.zeros((13,))
    sphere_center[4] = 0.5
    while iter_count <= search_options.bilinear_iterations:
        search_lagrangian_ret = dut.SearchLagrangian(
            h_sol, kappa, lambda0_degree, lambda1_degree, l_degrees,
            hdot_eq_lagrangian_degrees, None, t_degrees, s_degrees,
            unsafe_state_constraints_lagrangian_degrees, unsafe_a_info,
            search_options, backoff_scale=None)
        assert (search_lagrangian_ret.success)

        # Now construct the barrier program.
        barrier_ret = dut.ConstructBarrierProgram(
            search_lagrangian_ret.lambda0, search_lagrangian_ret.lambda1,
            search_lagrangian_ret.l, hdot_eq_lagrangian_degrees, None, [],
            [[]], h_degree, kappa, s_degrees, [[]])
        # Add constraint h(x_safe) >= 0
        A_h_safe, var_h_safe, b_h_safe = barrier_ret.h.EvaluateWithAffineCoefficients(x, x_safe)
        barrier_ret.prog().AddLinearConstraint(
            A_h_safe, -b_h_safe, np.full_like(b_h_safe, np.inf), var_h_safe)
        # Add cost to minimize h(sphere_center)
        A_sphere_center, var_sphere_center, b_sphere_center = barrier_ret.h.EvaluateWithAffineCoefficients(x, sphere_center)
        barrier_ret.prog().AddLinearCost(
            A_sphere_center[0, :], b_sphere_center[0], var_sphere_center)
        result = analysis.SearchWithBackoff(
            barrier_ret.prog(), search_options.barrier_step_solver_id,
            search_options.barrier_step_solver_options,
            search_options.barrier_step_backoff_scale)
        assert (result.is_success())
        h_sol = result.GetSolution(barrier_ret.h)
        iter_count += 1

    return h_sol








def search(
    x: np.ndarray, f, G, thrust_max: float,
        deriv_eps: float, unsafe_regions: list, x_safe: np.ndarray) -> sym.Polynomial:
    u_vertices = get_u_vertices(thrust_max)

    state_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])

    dynamics_denominator = None

    beta_minus = -0.01

    beta_plus = 0.01

    dut = analysis.ControlBarrier(
        f, G, dynamics_denominator, x, beta_minus, beta_plus, unsafe_regions,
        u_vertices, state_constraints)

    h_degree = 2
    x_set = sym.Variables(x)

    #h_init = sym.Polynomial((x[4] - 0.5) ** 2 + x[5] **
    #                        2 + x[6] ** 2 + 0.01 * x[7:].dot(x[7:]) - 0.2)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_trig_cbf17.pickle", "rb") as input_file:
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
            degree=4, poly_type=analysis.SlackPolynomialType.kSos,
            cost_weight=1.)
        #hdot_a_info = None

    t_degrees = [0]
    s_degrees = [[h_degree - 2]]
    unsafe_eq_lagrangian_degrees = [[h_degree - 2]]
    if with_slack_a:
        unsafe_a_info = [analysis.SlackPolynomialInfo(
            degree=h_degree, poly_type=analysis.SlackPolynomialType.kSos,
            cost_weight=1.)]
        #unsafe_a_info = [None]
    h_x_safe_min = np.array([0.01] * x_safe.shape[1])

    if with_slack_a:
        hdot_a_zero_tol = 1E-9
        unsafe_a_zero_tol = 1E-8
        search_options = analysis.ControlBarrier.SearchWithSlackAOptions(
            hdot_a_zero_tol, unsafe_a_zero_tol, use_zero_a=True)
        search_options.bilinear_iterations = 20
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
        search_options.barrier_step_backoff_scale = 0.02
        search_options.lagrangian_step_backoff_scale = 0.02
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
        search_options = analysis.ControlBarrier.SearchOptions()
        search_options.lagrangian_step_solver_options = mp.SolverOptions()
        search_options.lagrangian_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.barrier_step_solver_options = mp.SolverOptions()
        search_options.barrier_step_solver_options.SetOption(
            mp.CommonSolverOption.kPrintToConsole, 1)
        search_options.bilinear_iterations = 15

        search_with_ellipsoid = False
        if search_with_ellipsoid:
            ellipsoids = [analysis.ControlBarrier.Ellipsoid(
                c=x_safe[:, 0], S=np.eye(13), d=0., r_degree=0,
                eq_lagrangian_degrees=[0])]
            ellipsoid_options = [
                analysis.ControlBarrier.EllipsoidMaximizeOption(
                    t=sym.Polynomial(), s_degree=0, backoff_scale=0.04)]
            search_options.lsol_tiny_coeff_tol = 0#1E-6
            search_options.hsol_tiny_coeff_tol = 0#1E-6
            search_options.barrier_step_backoff_scale = 0.01
            x_anchor = x_safe[:, 0]
            h_x_anchor_max = h_init.EvaluateIndeterminates(x, x_anchor)[0] * 10 
            search_result = dut.Search(
                h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree,
                l_degrees, hdot_eq_lagrangian_degrees, t_degrees, s_degrees,
                unsafe_eq_lagrangian_degrees, x_anchor, h_x_anchor_max,
                search_options, ellipsoids, ellipsoid_options)
        else:
            search_options.barrier_step_backoff_scale = 0.02
            x_anchor = np.zeros((13,))
            h_x_anchor_max = h_init.EvaluateIndeterminates(
                x, x_anchor.reshape((-1, 1)))[0] * 10
            x_samples = np.zeros((13, 3))
            x_samples[4, 1] = 1
            x_samples[4, 2] = 0.5
            maximize_minimal = True
            search_result = dut.Search(
                h_init, h_degree, deriv_eps, lambda0_degree, lambda1_degree,
                l_degrees, hdot_eq_lagrangian_degrees, t_degrees, s_degrees,
                unsafe_eq_lagrangian_degrees, x_anchor, h_x_anchor_max, x_safe,
                x_samples, maximize_minimal, search_options)

    with open("quadrotor3d_trig_cbf18.pickle", "wb") as handle:
        pickle.dump({
            "h": clf_cbf_utils.serialize_polynomial(search_result.h),
            "beta_plus": beta_plus, "beta_minus": beta_minus,
            "deriv_eps": deriv_eps, "thrust_max": thrust_max,
            "x_safe": x_safe,
            "unsafe_region0": clf_cbf_utils.serialize_polynomial(unsafe_regions[0][0])}, handle)
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
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
    deriv_eps = 0.1
    unsafe_regions = [np.array([#sym.Polynomial(x[6] + 0.15)])]
        sym.Polynomial((x[4] - 0.5) ** 2 + x[5] ** 2 + x[6] ** 2- (0.8*quadrotor.length()) ** 2)])]
    x_safe = np.empty((13, 2))
    x_safe[:, 0] = np.zeros(13)
    x_safe[:, 1] = np.zeros(13)
    x_safe[4, 1] = 1
    h_sol = search(x, f, G, thrust_max, deriv_eps, unsafe_regions, x_safe)

    #with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_trig_cbf15.pickle", "rb") as input_file:
    #    input_data = pickle.load(input_file)
    #x_set = sym.Variables(x)
    #cbf = clf_cbf_utils.deserialize_polynomial(x_set, input_data["h"])
    #simulate(x, f, G, cbf, thrust_max, input_data["deriv_eps"], input_data["beta_minus"], input_data["beta_plus"], np.zeros((12,)), 1)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
