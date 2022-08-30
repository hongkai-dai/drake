import numpy as np
import pickle

import clf_cbf_utils

import pydrake.systems.analysis as analysis
import pydrake.systems.primitives as primitives
import pydrake.systems.controllers as controllers
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
import pydrake.symbolic as sym
import pydrake.common
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import DiagramBuilder
import pydrake.systems.trajectory_optimization as trajectory_optimization
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
)
from pydrake.systems.primitives import LogVectorOutput


meshcat = StartMeshcat()

def construct_builder():
    builder = DiagramBuilder()
    cart_pole, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.)
    Parser(cart_pole).AddModelFromFile(pydrake.common.FindResourceOrThrow(
        "drake/examples/multibody/cart_pole/cart_pole.sdf"))
    cart_pole.Finalize()
    visual = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception))
    return builder, cart_pole, scene_graph


def SwingUpTrajectoryOptimization():
    builder, cart_pole, scene_graph = construct_builder()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    context = cart_pole.CreateDefaultContext()
    num_time_samples = 50
    minimum_timestep = 0.04
    maximum_timestep = 0.06
    dircol = trajectory_optimization.DirectCollocation(
        cart_pole, context, num_time_samples, minimum_timestep,
        maximum_timestep, cart_pole.get_actuation_input_port().get_index())
    dircol.prog().AddBoundingBoxConstraint(
        np.zeros((4,)), np.zeros((4,)), dircol.state(0))
    dircol.prog().AddBoundingBoxConstraint(
        np.array([0, np.pi, 0, 0]), np.array([0, np.pi, 0, 0]),
        dircol.state(num_time_samples - 1))
    dircol.prog().AddBoundingBoxConstraint(
        0, 0, dircol.input(num_time_samples - 1)[0])
    dircol.AddRunningCost(dircol.input().dot(dircol.input()))
    result = mp.Solve(dircol.prog())
    assert (result.is_success())
    x_traj = dircol.GetStateSamples(result)
    u_traj = dircol.GetInputSamples(result)
    return x_traj, u_traj

def simulate(
    parameters: analysis.CartPoleParams, x, clf, kappa, u_bound,
        initial_state, duration):
    builder, cart_pole, scene_graph = construct_builder()
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(parameters, x)
    clf_controller = builder.AddSystem(analysis.CartpoleClfController(
        x, f, G, dynamics_denominator, clf, kappa, u_bound))

    state_logger = LogVectorOutput(cart_pole.get_state_output_port(), builder)
    clf_logger = LogVectorOutput(clf_controller.clf_output_port(), builder)
    control_logger = LogVectorOutput(clf_controller.control_output_port(), builder)
    trig_state_converter = builder.AddSystem(analysis.CartpoleTrigStateConverter())
    builder.Connect(cart_pole.get_state_output_port(), trig_state_converter.get_input_port())
    builder.Connect(trig_state_converter.get_output_port(), clf_controller.x_input_port())
    builder.Connect(clf_controller.control_output_port(), cart_pole.get_actuation_input_port())
    x0 = analysis.ToCartpoleTrigState(initial_state)
    print(f"V(initial_state)={clf.EvaluateIndeterminates(x, x0)}")

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()

    simulator = analysis.Simulator(diagram)
    simulator.get_mutable_context().SetContinuousState(initial_state)

    diagram.Publish(simulator.get_context())
    simulator.AdvanceTo(duration)

    states = state_logger.FindLog(simulator.get_context()).data()
    controls = control_logger.FindLog(simulator.get_context()).data()
    clfs = clf_logger.FindLog(simulator.get_context()).data()
    return states, controls, clfs


def FindHjbLower(params: analysis.CartPoleParams, J_degree, x):
    # max J(x)
    # dJ/dx*(f(x)+G(x)u)/d(x)+l(x, u)>=0
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    u = prog.NewIndeterminates(1, "u")
    x_set = sym.Variables(x)
    xu_set = sym.Variables(x)
    xu_set.insert(u[0])
    state_constraint = analysis.CartpoleStateEqConstraint(x)
    J, _ = prog.NewSosPolynomial(x_set, J_degree)
    dJdx = J.Jacobian(x)
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)
    l = sym.Polynomial(
        x.dot(np.array([1, 1, 1, 10., 10]) * x) + 10 * u[0]*u[0])
    state_eq_lagrangian = prog.NewFreePolynomial(xu_set, J_degree)
    box_lower = np.array([-1, -1, 0, -5, -5.])
    box_upper = np.array([1, 1, 2, 5, 5.])
    box_regions = np.array([sym.Polynomial(x[i] - box_lower[i])
                           for i in range(5)] + [sym.Polynomial(box_upper[i] - x[i]) for i in range(5)])
    box_region_lagrangian = [None] * 10
    for i in range(10):
        box_region_lagrangian[i], _ = prog.NewSosPolynomial(
            xu_set, J_degree + 2)
    sos_condition = dJdx @ (f + G * u) + l * dynamics_denominator - state_eq_lagrangian * \
        state_constraint - np.array(box_region_lagrangian) @ box_regions

    prog.AddSosConstraint(sos_condition)

    # Add constraint J(0) = 0
    A_J_0, var_J_0, b_J_0 = J.EvaluateWithAffineCoefficients(
        x, np.zeros((5, 1)))
    prog.AddLinearEqualityConstraint(A_J_0, -b_J_0, var_J_0)

    # Maximize integral(J)
    J_integral = J.Integrate(x[0], box_lower[0], box_upper[0]).Integrate(
        x[3], box_lower[3], box_upper[3]).Integrate(x[4], box_lower[4], box_upper[4])
    theta = np.linspace(-np.pi, np.pi, 100)
    x1_samples = np.sin(theta)
    x2_samples = np.cos(theta) + 1
    x12_samples = np.vstack(
        (x1_samples.reshape((1, -1)), x2_samples.reshape((1, -1))))
    A_J_integral, var_J_integral, b_J_integral = J_integral.EvaluateWithAffineCoefficients(
        x[1:3], x12_samples)
    prog.AddLinearCost(-A_J_integral[0, :], -b_J_integral[0], var_J_integral)
    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    assert (result.is_success())
    return result.GetSolution(J)


def FindClfInit(params: analysis.CartPoleParams, V_degree, x):
    lqr_Q_diag = np.array([1, 1, 1, 10, 10.])
    lqr_Q = np.diag(lqr_Q_diag)
    K, _ = analysis.SynthesizeCartpoleTrigLqr(params, lqr_Q, R=20)
    u_lqr = -K[0, :] @ x
    n_expr, d_expr = analysis.CartpoleTrigDynamics[sym.Expression](
        params, x, u_lqr)
    dynamics_numerator = np.array(
        [sym.Polynomial(n_expr[i]) for i in range(5)])
    dynamics_denominator = sym.Polynomial(d_expr)

    positivity_eps = 0.001
    d = int(V_degree / 2)
    kappa = 0.01
    state_eq_constraints = np.array([analysis.CartpoleStateEqConstraint(x)])
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [V_degree + 2]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 0.04)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = [V_degree]

    ret = analysis.FindCandidateRegionalLyapunov(
        x, dynamics_numerator, dynamics_denominator, V_degree, positivity_eps,
        d, kappa, state_eq_constraints, positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees, state_ineq_constraints,
        positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees)
    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(ret.prog(), None, solver_options)
    assert (result.is_success())
    V_sol = result.GetSolution(ret.V)
    return V_sol


def SearchWithSlackA(params: analysis.CartPoleParams, x, u_max, kappa):
    state_constraints = np.array([analysis.CartpoleStateEqConstraint(x)])
    u_vertices = np.array([[-u_max, u_max]])
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)
    dut = analysis.ControlLyapunov(
        x, f, G, dynamics_denominator, u_vertices, state_constraints)

    search_options = analysis.ControlLyapunov.SearchWithSlackAOptions()
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.a_zero_tol = 1E-9
    search_options.lagrangian_step_backoff_scale = 0.02
    search_options.lyap_step_backoff_scale = 0.02
    search_options.lsol_tiny_coeff_tol = 1E-6
    search_options.Vsol_tiny_coeff_tol = 1E-6
    search_options.lyap_tiny_coeff_tol = 1E-6
    search_options.bilinear_iterations = 200

    lambda0_degree = 4
    l_degrees = [4, 4]
    V_degree = 2
    p_degrees = [8]
    positivity_eps = 0.0001
    positivity_d = int(V_degree / 2)
    positivity_eq_lagrangian_degrees = [V_degree - 2]
    in_roa_samples = np.empty((5, 1))
    in_roa_samples[:, 0] = analysis.ToCartpoleTrigState(
        np.array([0, 0, 0, 0.]))

    x_set = sym.Variables(x)

    load_clf = True
    if load_clf:
        with open("/home/hongkaidai/sos_clf_cbf_data/cart_pole/cartpole_trig_clf30.pickle", "rb") as input_file:
            load_data = pickle.load(input_file)
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, load_data["V"])
            rho = load_data["rho"]
    else:
        V_init = FindClfInit(params, V_degree, x)
        V_init = V_init.RemoveTermsWithSmallCoefficients(1E-7)
    V_init_at_samples = V_init.EvaluateIndeterminates(x, in_roa_samples)
    search_options.rho = V_init_at_samples.max()
    a_info = analysis.SlackPolynomialInfo(
        degree=10, poly_type=analysis.SlackPolynomialType.kSos)

    search_result = dut.SearchWithSlackA(
        V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees, kappa,
        in_roa_samples, a_info, search_options)
    return search_result.V


def SearchWTrigDynamics(params, x, u_max, kappa):
    x_set = sym.Variables(x)
    V_degree = 2
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)

    state_constraints = np.array([analysis.CartpoleStateEqConstraint(x)])
    u_vertices = np.array([[-u_max, u_max]])

    dut = analysis.ControlLyapunov(
        x, f, G, dynamics_denominator, u_vertices, state_constraints)

    lambda0_degree = 4
    l_degrees = [4, 4]
    p_degrees = [8]

    load_clf = True
    if load_clf:
        with open("/home/hongkaidai/sos_clf_cbf_data/cart_pole/cartpole_trig_clf50.pickle", "rb") as input_file:
            load_data = pickle.load(input_file)
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, load_data["V"])
            rho = load_data["rho"]
    else:
        V_init = FindClfInit(params, V_degree, x)
        #V_init = FindHjbLower(params, V_degree, x)
        d_degree = int(lambda0_degree / 2) + 1
        lagrangian_ret = dut.ConstructLagrangianProgram(
            V_init, sym.Polynomial(), d_degree, l_degrees, p_degrees, kappa)
        solver_options = mp.SolverOptions()
        solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        result = mp.Solve(lagrangian_ret.prog(), None, solver_options)
        assert (result.is_success())
        rho_sol = result.GetSolution(lagrangian_ret.rho)
        V_init = V_init / rho_sol * 0.1
        rho = 1.

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.rho = rho

    verify_V_init = False
    if verify_V_init:
        dut.SearchLagrangian(V_init, 1., lambda0_degree, l_degrees,
                             p_degrees, kappa, search_options, None, None, None)

    state_swingup, control_swingup = SwingUpTrajectoryOptimization()
    x_swingup = np.empty((5, state_swingup.shape[1]))
    for i in range(state_swingup.shape[1]):
        x_swingup[:, i] = analysis.ToCartpoleTrigState(state_swingup[:, i])
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.bilinear_iterations = 10
    search_options.lyap_step_backoff_scale = 0.02
    positivity_eps = 0.000
    positivity_d = int(V_degree / 2)
    positivity_eq_lagrangian_degrees = [V_degree - 2]
    ellipsoid_eq_lagrangian_degrees = [V_degree - 2]
    x_star = analysis.ToCartpoleTrigState(
        0.2*state_swingup[:, -18] + 0.8 * state_swingup[:, -17])

    S = np.zeros((5, 5))
    S[0, 0] = 10
    S[1, 1] = 100
    S[2, 2] = 100
    S[3, 3] = 1
    S[4, 4] = 1
    r_degree = 0
    ellipsoid_maximize_option = \
        analysis.ControlLyapunov.EllipsoidMaximizeOption(
            t=sym.Polynomial(), s_degree=0, backoff_scale=0.01)
    search_result = dut.Search(
        V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
        ellipsoid_eq_lagrangian_degrees, kappa, x_star, S, r_degree,
        search_options, ellipsoid_maximize_option)
    with open("/home/hongkaidai/sos_clf_cbf_data/cart_pole/cartpole_trig_clf54.pickle", "wb") as handle:
        pickle.dump({
            "V": clf_cbf_utils.serialize_polynomial(search_result.V),
            "kappa": kappa, "u_max": u_max, "rho": search_options.rho,
            "x_swingup": x_swingup,
            "V_swingup": search_result.V.EvaluateIndeterminates(x, x_swingup)},
            handle)
    return search_result.V


def main():
    pydrake.common.configure_logging()
    params = analysis.CartPoleParams()
    x = sym.MakeVectorContinuousVariable(5, "x")
    u_max = 176
    kappa = 0.01
    #V_sol = SearchWithSlackA(params, x, u_max, kappa)
    SearchWTrigDynamics(params, x, u_max, kappa)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
