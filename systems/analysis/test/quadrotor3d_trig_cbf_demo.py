import numpy as np
import pickle
import matplotlib.pyplot as plt

import clf_cbf_utils

import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
from pydrake.solvers.gurobi import GurobiSolver 
import pydrake.symbolic as sym
import pydrake.common
from pydrake.common import RandomGenerator
from pydrake.systems.framework import (
    DiagramBuilder,
    LeafSystem,
)
import pydrake.math
from pydrake.systems.primitives import LogVectorOutput
from pydrake.examples import (
    QuadrotorGeometry,
    QuadrotorPlant,
    QuadrotorTrigGeometry,
)
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    SceneGraph,
)
import pydrake.geometry as geometry
import pydrake.math as math
from quadrotor3d_trig_clf_demo import QuadrotorClfController



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
        self.beta_plus = np.inf if beta_plus is None else beta_plus
        dhdx = self.cbf.Jacobian(self.x)
        self.dhdx_times_f = dhdx.dot(self.f)
        self.dhdx_times_G = dhdx @ self.G

        self.x_input_index = self.DeclareVectorInputPort("x", 13).get_index()
        self.control_output_index = self.DeclareVectorOutputPort(
            "control", 4, self.CalcControl).get_index()
        self.cbf_output_index = self.DeclareVectorOutputPort(
            "cbf", 1, self.CalcCbf).get_index()

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
        env = {self.x[i]: x_val[i] for i in range(13)}
        output.SetFromVector(np.array([self.cbf.Evaluate(env)]))

class QuadrotorClfCbfController(LeafSystem):
    def __init__(self, x, f, G, clf, cbf, kappa_V, kappa_h, thrust_max, beta_minus, beta_plus):
        LeafSystem.__init__(self)
        assert (x.shape == (13,))
        self.x = x
        assert (f.shape == (13,))
        self.f = f
        assert (G.shape == (13, 4))
        self.G = G
        self.clf = clf
        self.cbf = cbf
        self.kappa_V = kappa_V
        self.kappa_h = kappa_h
        self.thrust_max = thrust_max
        self.beta_minus = beta_minus
        self.beta_plus = np.inf if beta_plus is None else beta_plus
        dVdx = self.clf.Jacobian(self.x)
        self.dVdx_times_f = dVdx.dot(self.f)
        self.dVdx_times_G = dVdx @ self.G
        dhdx = self.cbf.Jacobian(self.x)
        self.dhdx_times_f = dhdx.dot(self.f)
        self.dhdx_times_G = dhdx @ self.G

        self.x_input_index = self.DeclareVectorInputPort("x", 13).get_index()
        self.control_output_index = self.DeclareVectorOutputPort(
            "control", 4, self.CalcControl).get_index()
        self.cbf_output_index = self.DeclareVectorOutputPort(
            "cbf", 1, self.CalcCbf).get_index()
        self.clf_output_index = self.DeclareVectorOutputPort(
            "clf", 1, self.CalcClf).get_index()

    def x_input_port(self):
        return self.get_input_port(self.x_input_index)

    def control_output_port(self):
        return self.get_output_port(self.control_output_index)

    def cbf_output_port(self):
        return self.get_output_port(self.cbf_output_index)

    def clf_output_port(self):
        return self.get_output_port(self.clf_output_index)

    def construct_qp(self, x_val, include_cbf=True):
        env = {self.x[i]: x_val[i] for i in range(13)}

        prog = mp.MathematicalProgram()
        nu = 4
        u = prog.NewContinuousVariables(nu, "u")
        prog.AddBoundingBoxConstraint(0, self.thrust_max, u)
        prog.AddQuadraticCost(np.identity(nu), np.zeros((nu,)), 0, u)
        dVdx_times_f_val = self.dVdx_times_f.Evaluate(env)
        dVdx_times_G_val = np.array([
            self.dVdx_times_G[i].Evaluate(env) for i in range(nu)])
        V_val = self.clf.Evaluate(env)
        # Add Vdot(x, u) <= -k * V + delta
        delta = prog.NewContinuousVariables(1)
        Vdot_A = np.empty((1, 5))
        Vdot_A[0, :4] = dVdx_times_G_val
        Vdot_A[0, 4] = -1
        Vdot_A_vars = np.empty((5,), dtype=object)
        Vdot_A_vars[:4] = u
        Vdot_A_vars[4] = delta[0]
        prog.AddLinearConstraint(
            Vdot_A, np.array([-np.inf]), np.array([-dVdx_times_f_val  - self.kappa_V * V_val]), Vdot_A_vars)
        Vdot_cost_weight = 1000000
        prog.AddLinearCost(np.array([Vdot_cost_weight]), 0, delta)
        dhdx_times_f_val = self.dhdx_times_f.Evaluate(env)
        dhdx_times_G_val = np.array([
            self.dhdx_times_G[i].Evaluate(env) for i in range(nu)])
        h_val = self.cbf.Evaluate(env)
        # dhdx * G * u + dhdx * f >= -eps * h
        if include_cbf and self.beta_minus <= h_val <= self.beta_plus:
            prog.AddLinearConstraint(
                dhdx_times_G_val.reshape((1, -1)),
                np.array([-self.kappa_h * h_val - dhdx_times_f_val]),
                np.array([np.inf]), u)
        return prog, u, delta
        

    def CalcControl(self, context, output):
        x_val = self.x_input_port().Eval(context)
        prog, u, delta = self.construct_qp(x_val)
        gurobi_solver = GurobiSolver()
        result = gurobi_solver.Solve(prog)
        if not result.is_success():
            raise Exception("CBF controller cannot find u")
        output.SetFromVector(result.GetSolution(u))

    def CalcCbf(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(13)}
        output.SetFromVector(np.array([self.cbf.Evaluate(env)]))

    def CalcClf(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(13)}
        output.SetFromVector(np.array([self.clf.Evaluate(env)]))


def simulate(x, f, G, clf, cbf, thrust_max, kappa_V, kappa_h, beta_minus, beta_plus, initial_state, duration, meshcat):
    builder = DiagramBuilder()

    quadrotor = builder.AddSystem(analysis.QuadrotorTrigPlant())

    scene_graph = builder.AddSystem(pydrake.geometry.SceneGraph())
    visual_properties = geometry.MakePhongIllustrationProperties(
      np.array([0.7, 0.2, 0.1, 1]))
    s_id = scene_graph.RegisterSource("python")
    ground_id = scene_graph.RegisterAnchoredGeometry(
      s_id,
      geometry.GeometryInstance(
          math.RigidTransform(np.array([0, 0, -0.25])), geometry.Box(10, 10, 0.01), "ground"));
    scene_graph.AssignRole(s_id, ground_id, visual_properties);

    geom = QuadrotorTrigGeometry.AddToBuilder(
        builder, quadrotor.get_output_port(0), "quadrotor", scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kIllustration))

    state_converter = builder.AddSystem(analysis.QuadrotorTrigStateConverter())

    if clf is not None and cbf is not None:
        controller = builder.AddSystem(QuadrotorClfCbfController(
            x, f, G, clf, cbf, kappa_V, kappa_h, thrust_max, beta_minus, beta_plus))
    elif clf is None and cbf is not None:
        controller = builder.AddSystem(QuadrotorCbfController(
            x, f, G, cbf, kappa_h, thrust_max, beta_minus, beta_plus))
    elif clf is not None and cbf is None:
        controller = builder.AddSystem(QuadrotorClfController(x, f, G, clf, kappa_V, thrust_max, Vdot_cost_weight=2))

    builder.Connect(controller.control_output_port(),
                    quadrotor.get_input_port())
    builder.Connect(quadrotor.get_output_port(0),
                    controller.x_input_port())

    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)
    if cbf is not None:
        cbf_logger = LogVectorOutput(controller.cbf_output_port(), builder)
    if clf is not None:
        clf_logger = LogVectorOutput(controller.clf_output_port(), builder)
    control_logger = LogVectorOutput(
        controller.control_output_port(), builder)

    diagram = builder.Build()

    simulator = analysis.Simulator(diagram)

    analysis.ResetIntegratorFromFlags(simulator, "implicit_euler", 0.01)

    x0 = analysis.ToQuadrotorTrigState(initial_state)
    simulator.get_mutable_context().SetContinuousState(x0)
    visualizer.StartRecording()
    simulator.AdvanceTo(duration)
    visualizer.StopRecording()

    state_data = state_logger.FindLog(simulator.get_context()).data()
    if cbf is not None:
        cbf_data = cbf_logger.FindLog(simulator.get_context()).data()
    else:
        cbf_data = None
    if clf is not None:
        clf_data = clf_logger.FindLog(simulator.get_context()).data()
    else:
        clf_data = None
    control_data = control_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    print(f"final state: {state_data[:, -1]}")
    return state_data, control_data, clf_data, cbf_data, time_data


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


def search_cbf_init(x, f, G, kappa: float, h_degree: int, unsafe_region: sym.Polynomial, x_safe):
    """
    Find a CBF with the condition
    maxᵤ ∂h/∂x*(f(x)+G(x)u)− r * uᵀu ≥ −κ h(x)
    h(x) <= 0 ∀ x ∈ unsafe_regions
    The first condition is that the matrix
    [∂h/∂x*f(x)+κh(x)  ∂h/∂x*G(x)] is psd
    [(∂h/∂x*G(x))ᵀ             4r]
    This is a matrix-sos constraint. We impose it on some sample x.
    """
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    h = prog.NewFreePolynomial(x_set, h_degree)
    dhdx = h.Jacobian(x)
    dhdx_times_f = dhdx.dot(f)
    dhdx_times_G = dhdx @ G
    state_eq_constraint = analysis.QuadrotorStateEqConstraint(x)
    state_eq_lagrangian = prog.NewFreePolynomial(x_set, h_degree)
    g = RandomGenerator()
    x_samples = np.empty((10000, 13))
    for i in range(x_samples.shape[0]):
        x_samples[i, :4] = pydrake.math.UniformlyRandomQuaternion(g).wxyz()
    x_samples[:, 4] = np.random.rand(x_samples.shape[0])
    x_samples[:, 5:] = np.random.rand(x_samples.shape[0], 8) * 2 - 1

    psd_top_left = (dhdx_times_f + kappa * h -
                    state_eq_lagrangian * state_eq_constraint).ToExpression()
    r = 0.01
    for i in range(x_samples.shape[0]):
        env = {x[j]: x_samples[i, j] for j in range(13)}
        psd_sample = np.empty((5, 5), dtype=object)
        psd_sample[0, 0] = psd_top_left.EvaluatePartial(env)
        for j in range(4):
            psd_sample[0, j +
                       1] = dhdx_times_G[j].ToExpression().EvaluatePartial(env)
            psd_sample[j+1, 0] = psd_sample[0, j+1]
        psd_sample[1:, 1:] = 4 * r * np.eye(4)
        prog.AddPositiveSemidefiniteConstraint(psd_sample)

    # Add the constraint that h(x) <= 0 in the unsafe region.
    unsafe_eq_lagrangian = prog.NewFreePolynomial(x_set, h_degree - 2)
    unsafe_lagrangian, _ = prog.NewSosPolynomial(x_set, h_degree - 2)
    prog.AddSosConstraint(-h + unsafe_lagrangian * unsafe_region -
                          unsafe_eq_lagrangian * state_eq_constraint)

    # Add the constraint h(x_safe) >= 0
    A_h_safe, var_h_safe, b_h_safe = h.EvaluateWithAffineCoefficients(
        x, x_safe)
    prog.AddLinearConstraint(
        A_h_safe, -b_h_safe, np.full_like(b_h_safe, np.inf), var_h_safe)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    return result.GetSolution(h)


def search_sphere_obstacle_cbf(x, f, G, beta_minus, beta_plus, thrust_max, kappa, x_safe):
    """
    Given h_init that already satisfies hdot >= -kappa*h, try to minimize h(sphere_center) while keeping h(x_safe) >= 0
    """
    x_set = sym.Variables(x)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_trig_cbf25.pickle", "rb") as input_file:
        h_init = clf_cbf_utils.deserialize_polynomial(
            x_set, pickle.load(input_file)["h"])
    u_vertices = get_u_vertices(thrust_max)
    state_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])
    dut = analysis.ControlBarrier(
        f, G, None, x, beta_minus, beta_plus, [], u_vertices, state_constraints)

    assert (np.all(h_init.EvaluateIndeterminates(x, x_safe) >= 0))
    iter_count = 0
    h_sol = h_init

    h_degree = 2
    lambda0_degree = 4
    lambda1_degree = 4
    l_degrees = [2] * 16
    hdot_eq_lagrangian_degrees = [h_degree + lambda0_degree - 2]
    t_degrees = []
    s_degrees = []
    unsafe_state_constraints_lagrangian_degrees = []
    unsafe_a_info = [None]
    search_options = analysis.ControlBarrier.SearchOptions()
    search_options.bilinear_iterations = 40
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.barrier_step_solver_options = mp.SolverOptions()
    search_options.barrier_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    #search_options.barrier_step_backoff_scale = 0.02
    sphere_center = np.zeros((13,))
    sphere_center[4] = 0.5
    while iter_count <= search_options.bilinear_iterations:
        print(f"iteration {iter_count}")
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
            [[]], h_degree, kappa, s_degrees, [])
        # Add constraint h(x_safe) >= 0
        A_h_safe, var_h_safe, b_h_safe = barrier_ret.h.EvaluateWithAffineCoefficients(
            x, x_safe)
        barrier_ret.prog().AddLinearConstraint(
            A_h_safe, np.full_like(b_h_safe, 0.01)-b_h_safe,
            np.full_like(b_h_safe, np.inf), var_h_safe)
        # Add cost to minimize h(sphere_center)
        A_sphere_center, var_sphere_center, b_sphere_center = barrier_ret.h.EvaluateWithAffineCoefficients(
            x, sphere_center)
        barrier_ret.prog().AddLinearCost(
            A_sphere_center[0, :], b_sphere_center[0], var_sphere_center)
        result = analysis.SearchWithBackoff(
            barrier_ret.prog(), search_options.barrier_step_solver,
            search_options.barrier_step_solver_options,
            search_options.barrier_step_backoff_scale)
        assert (result.is_success())
        h_sol = result.GetSolution(barrier_ret.h)
        iter_count += 1

    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_trig_cbf26.pickle", "wb") as handle:
        pickle.dump({
            "h": clf_cbf_utils.serialize_polynomial(h_sol),
            "beta_plus": beta_plus, "beta_minus": beta_minus,
            "deriv_eps": kappa, "thrust_max": thrust_max,
            "x_safe": x_safe}, handle)
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

    # h_init = sym.Polynomial((x[4] - 0.5) ** 2 + x[5] **
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
        # unsafe_a_info = [analysis.SlackPolynomialInfo(
        #    degree=h_degree, poly_type=analysis.SlackPolynomialType.kSos,
        #    cost_weight=1.)]
        unsafe_a_info = [None]
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
            search_options.lsol_tiny_coeff_tol = 0  # 1E-6
            search_options.hsol_tiny_coeff_tol = 0  # 1E-6
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

# def reexecute_if_unbuffered():
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
# def traced(func, ignoredirs=None):
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
# @traced


def main():
    pydrake.common.configure_logging()
    quadrotor = analysis.QuadrotorTrigPlant()
    x = sym.MakeVectorContinuousVariable(13, "x")
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
    deriv_eps = 0.3
    unsafe_regions = [np.array([  # sym.Polynomial(x[6] + 0.15)])]
        sym.Polynomial((x[4] - 0.5) ** 2 + x[5] ** 2 + x[6] ** 2 - (0.8*quadrotor.length()) ** 2)])]
    x_safe = np.empty((13, 2))
    x_safe[:, 0] = np.zeros(13)
    x_safe[:, 1] = np.zeros(13)
    x_safe[4, 1] = 1
    ##h_sol = search(x, f, G, thrust_max, deriv_eps, unsafe_regions, x_safe)
    #h_sol = search_sphere_obstacle_cbf(
    #    x, f, G, -0.01, 0.01, thrust_max, deriv_eps, x_safe)

    meshcat = StartMeshcat()
    x_set = sym.Variables(x)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_trig_cbf21.pickle", "rb") as input_file:
        cbf_input_data = pickle.load(input_file)
        cbf = clf_cbf_utils.deserialize_polynomial(x_set, cbf_input_data["h"])
        kappa_h = cbf_input_data["deriv_eps"]
        beta_minus = cbf_input_data["beta_minus"]
        beta_plus = cbf_input_data["beta_plus"]
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_clf/quadrotor3d_trig_clf_sol3.pickle", "rb") as input_file:
        clf_input_data = pickle.load(input_file)
        clf = clf_cbf_utils.deserialize_polynomial(x_set, clf_input_data["V"])
        kappa_V = clf_input_data["kappa"]
    x0 = np.zeros((12,))
    x0[0] = 1 
    x0[1] = 0.
    x0[2] = 0.
    x0[3] = 0.0*np.pi
    x0[6] = 0
    if clf is not None:
        print(f"CLF_init {clf.EvaluateIndeterminates(x, analysis.ToQuadrotorTrigState(x0).reshape((-1, 1)))}")
    if cbf is not None:
        print(f"CBF_init {cbf.EvaluateIndeterminates(x, analysis.ToQuadrotorTrigState(x0).reshape((-1, 1)))}")

    sim_T = 20
    state_data, control_data, clf_data, cbf_data, time_data = simulate(x, f, G, clf, cbf, thrust_max, kappa_V, kappa_h, beta_minus, beta_plus, x0, sim_T, meshcat)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_sim_clf3_cbf21_3.pickle", "wb") as handle:
        pickle.dump({
            "state_data": state_data,
            "control_data": control_data,
            "clf_data": clf_data,
            "cbf_data": cbf_data,
            "time_data": time_data}, handle)
    state_data_clf, control_data_clf, clf_data_clf, cbf_data_clf, time_data_clf = simulate(x, f, G, clf, None, thrust_max, kappa_V, kappa_h, beta_minus, beta_plus, x0, sim_T, meshcat)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_cbf/quadrotor3d_sim_clf3__2.pickle", "wb") as handle:
        pickle.dump({
            "state_data_clf": state_data_clf,
            "control_data_clf": control_data_clf,
            "clf_data_clf": clf_data_clf,
            "cbf_data_clf": cbf_data_clf,
            "time_data_clf": time_data_clf}, handle)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(time_data, state_data[6] + 0.15, 'b', label="CBF-CLF-QP controller")
    ax.plot(time_data_clf, state_data_clf[6] + 0.15, color='r', label="CLF-QP controller")
    ax.plot([0, sim_T], [0, 0], 'g--')
    ax.legend()
    ax.set_title("Quadrotor height above ground", fontsize=18)
    ax.set_xlabel("time (s)", fontsize=18)
    ax.set_ylabel("z (m)", fontsize=18)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    for fig_format in ("png", "pdf"):
        fig.savefig(f"/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/quadrotor_clf_cbf_z_3.{fig_format}", format=fig_format, bbox_inches="tight")
    return


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
