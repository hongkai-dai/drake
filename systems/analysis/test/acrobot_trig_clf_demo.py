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
    SceneGraph
)
import pydrake.examples.acrobot as acrobot
from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    InitializeAutoDiff,
)

meshcat = StartMeshcat()

def construct_builder():
    builder = DiagramBuilder()
    plant = builder.AddSystem(acrobot.AcrobotPlant())
    scene_graph = builder.AddSystem(SceneGraph())
    acrobot.AcrobotGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, MeshcatVisualizerParams(role=Role.kPerception))
    return builder, plant, scene_graph

def SwingUpTrajectoryOptimization():
    builder, plant, scene_graph = construct_builder()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    context = plant.CreateDefaultContext()
    num_time_samples = 30
    minimum_timestep = 0.04
    maximum_timestep = 0.06
    dircol = trajectory_optimization.DirectCollocation(
        plant, context, num_time_samples, minimum_timestep, maximum_timestep,
        plant.get_input_port().get_index())
    dircol.prog().AddBoundingBoxConstraint(np.array([0, 0, 0., 0]), np.array([0., 0, 0, 0]), dircol.state(0))
    dircol.prog().AddBoundingBoxConstraint(np.array([np.pi, 0, 0, 0]), np.array([np.pi, 0, 0, 0]), dircol.state(num_time_samples - 1))
    u_max = 30
    for i in range(num_time_samples):
        dircol.prog().AddBoundingBoxConstraint(-u_max, u_max, dircol.input(i)[0])

    dircol.AddRunningCost(dircol.input().dot(dircol.input()))

    result = mp.Solve(dircol.prog())
    assert (result.is_success())
    state_traj = dircol.GetStateSamples(result)
    control_traj = dircol.GetInputSamples(result)

    return state_traj, control_traj


def SynthesizeTrigLqr(params: acrobot.AcrobotParams):
    xu_des_ad = InitializeAutoDiff(np.zeros((7,)))
    n, d = analysis.AcrobotTrigDynamics[AutoDiffXd](params, xu_des_ad[:6], xu_des_ad[6,0])
    xdot_des_ad = n / d
    xdot_des_grad = ExtractGradient(xdot_des_ad)

    F = np.zeros((2, 6))
    F[0, 1] = 1
    F[1, 3] = 1
    lqr_Q = np.diag(np.array([1., 1., 1, 1, 10, 10]))
    K, S = controllers.LinearQuadraticRegulator(xdot_des_grad[:, :6], xdot_des_grad[:, -1], lqr_Q, np.array([[1000]]), np.zeros((0, 1)), F)
    return K, S


def FindClfInit(params: acrobot.AcrobotParams, V_degree, x):
    K, S = SynthesizeTrigLqr(params)
    u_lqr = -K[0, :].dot(x)
    n_expr, d_expr = analysis.AcrobotTrigDynamics[sym.Expression](params, x, u_lqr)
    dynamics_numerator = np.array([sym.Polynomial(n_expr[i]) for i in range(6)])
    dynamics_denominator = sym.Polynomial(d_expr)

    positivity_eps = 0.001
    d = int(V_degree / 2)
    kappa = 0.01
    state_eq_constraints = analysis.AcrobotStateEqConstraints(x)
    positivity_ceq_lagrangian_degrees = [V_degree - 2, V_degree - 2]
    derivative_ceq_lagrangian_degrees = [4, 4]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 0.01)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = [2]

    ret = analysis.FindCandidateRegionalLyapunov(
        x, dynamics_numerator, dynamics_denominator, V_degree, positivity_eps,
        d, kappa, state_eq_constraints, positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees, state_ineq_constraints,
        positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(ret.prog(), None, solver_options)
    assert (result.is_success())
    V_sol = result.GetSolution(ret.V).RemoveTermsWithSmallCoefficients(1E-8)
    return V_sol

def SearchWTrigDynamics(params, x, u_max, kappa):
    f, G, dynamics_denominator = analysis.TrigPolyDynamics(params, x)
    state_constraints = analysis.AcrobotStateEqConstraints(x)
    u_vertices = np.array([[-u_max, u_max]])
    dut = analysis.ControlLyapunov(x, f, G, dynamics_denominator, u_vertices, state_constraints)

    V_degree = 2
    lambda0_degree = 2
    l_degrees = [2, 2]
    p_degrees = [6, 6]

    x_set = sym.Variables(x)

    load_clf = True
    if load_clf:
        with open("/home/hongkaidai/sos_clf_cbf_data/acrobot/acrobot_trig_clf2.pickle", "rb") as input_file:
            load_data = pickle.load(input_file)
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, load_data["V"]) * 100
            rho = load_data["rho"] * 100
    else:
        V_init = FindClfInit(params, V_degree, x)
        d_degree = int(lambda0_degree / 2) + 1
        lagrangian_ret = dut.ConstructLagrangianProgram(V_init, sym.Polynomial(), d_degree, l_degrees, p_degrees, kappa)
        solver_options = mp.SolverOptions()
        solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        result = mp.Solve(lagrangian_ret.prog(), None, solver_options)
        assert (result.is_success)
        rho_sol = result.GetSolution(lagrangian_ret.rho)
        #V_init = V_init / rho_sol 
        rho = rho_sol

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.rho = rho

    verify_V_init = False
    if verify_V_init:
        dut.SearchLagrangian(
            V_init, search_options.rho, lambda0_degree, l_degrees, p_degrees,
            kappa, search_options, None, None, None)

    state_swingup, control_swingup = SwingUpTrajectoryOptimization()
    x_swingup = np.empty((6, state_swingup.shape[1]))
    for i in range(state_swingup.shape[1]):
        x_swingup[:, i] = analysis.ToAcrobotTrigState(state_swingup[:, i])
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.bilinear_iterations = 20
    search_options.lyap_step_backoff_scale = 0.02
    search_options.lsol_tiny_coeff_tol = 1E-7
    search_options.lyap_tiny_coeff_tol = 1E-7
    positivity_eps = 0.000
    positivity_d = int(V_degree / 2)
    positivity_eq_lagrangian_degrees = [V_degree - 2, V_degree - 2]
    ellipsoid_eq_lagrangian_degrees = [V_degree - 2, V_degree - 2]
    x_star = analysis.ToAcrobotTrigState(
        0.1*state_swingup[:, -2] + 0.9 * state_swingup[:, -1])

    S = np.zeros((6, 6))
    S[0, 0] = 100
    S[1, 1] = 100
    S[2, 2] = 100
    S[3, 3] = 100
    S[4, 4] = 1
    S[5, 5] = 1
    r_degree = 0
    ellipsoid_maximize_option = \
        analysis.ControlLyapunov.EllipsoidMaximizeOption(
            t=sym.Polynomial(), s_degree=0, backoff_scale=0.02)
    search_result = dut.Search(
        V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees,
        ellipsoid_eq_lagrangian_degrees, kappa, x_star, S, r_degree,
        search_options, ellipsoid_maximize_option)

    with open("/home/hongkaidai/sos_clf_cbf_data/acrobot/acrobot_trig_clf3.pickle", "wb") as handle:
        pickle.dump({
            "V": clf_cbf_utils.serialize_polynomial(search_result.V),
            "kappa": kappa, "u_max": u_max, "rho": search_options.rho,
            "x_swingup": x_swingup,
            "V_swingup": search_result.V.EvaluateIndeterminates(x, x_swingup)},
            handle)
    return search_result.V


def main():
    builder, plant, scene_graph = construct_builder()
    context = plant.CreateDefaultContext()
    params = plant.get_parameters(context)
    x = sym.MakeVectorContinuousVariable(6, "x")
    u_max = 50
    kappa = 0.01
    SearchWTrigDynamics(params, x, u_max, kappa)

if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
