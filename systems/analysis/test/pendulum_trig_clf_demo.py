import numpy as np
import pickle

import pdb
import matplotlib.pyplot as plt

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
    PendulumPlant,
    PendulumGeometry,
)
import pydrake.systems.trajectory_optimization as trajectory_optimization
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    SceneGraph,
)


def construct_builder():
    builder = DiagramBuilder()
    pendulum = builder.AddSystem(PendulumPlant())
    scene_graph = builder.AddSystem(SceneGraph())
    return builder, pendulum, scene_graph


def SwingUpTrajectoryOptimization(u_max):
    builder, pendulum, scene_graph = construct_builder()
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    context = pendulum.CreateDefaultContext()
    num_time_samples = 50
    minimum_timestep = 0.04
    maximum_timestep = 0.06
    dircol = trajectory_optimization.DirectCollocation(
        pendulum, context, num_time_samples, minimum_timestep,
        maximum_timestep, pendulum.get_input_port().get_index())
    for i in range(num_time_samples):
        dircol.prog().AddBoundingBoxConstraint(-u_max, u_max, dircol.input(i))
    dircol.prog().AddBoundingBoxConstraint(
        np.zeros((2,)), np.zeros((2,)), dircol.state(0))
    dircol.prog().AddBoundingBoxConstraint(
        np.array([np.pi, 0]), np.array([np.pi, 0]),
        dircol.state(num_time_samples - 1))
    dircol.prog().AddBoundingBoxConstraint(
        0, 0, dircol.input(num_time_samples - 1)[0])
    dircol.AddRunningCost(dircol.input().dot(dircol.input()))
    result = mp.Solve(dircol.prog())
    assert (result.is_success())
    x_traj = dircol.GetStateSamples(result)
    u_traj = dircol.GetInputSamples(result)
    return x_traj, u_traj


class PendulumClfController(LeafSystem):
    def __init__(self, x, f, G, clf, kappa, u_max, Vdot_cost_weight):
        LeafSystem.__init__(self)
        self.x = x
        self.f = f
        self.G = G
        self.clf = clf
        self.kappa = kappa
        self.u_max = u_max
        self.Vdot_cost_weight = Vdot_cost_weight

        dVdx = self.clf.Jacobian(self.x)

        self.dVdx_times_f = dVdx.dot(self.f)
        self.dVdx_times_G = dVdx @ self.G
        self.x_input_index = self.DeclareVectorInputPort("x", 3).get_index()
        self.control_output_index = self.DeclareVectorOutputPort(
            "control", 1, self.CalcControl).get_index()
        self.clf_output_index = self.DeclareVectorOutputPort(
            "clf", 1, self.CalcClf).get_index()

    def x_input_port(self):
        return self.get_input_port(self.x_input_index)

    def control_output_port(self):
        return self.get_output_port(self.control_output_index)

    def clf_output_port(self):
        return self.get_output_port(self.clf_output_index)

    def CalcControl(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(3)}

        prog = mp.MathematicalProgram()
        nu = 1
        u = prog.NewContinuousVariables(nu, "u")
        prog.AddBoundingBoxConstraint(-self.u_max, self.u_max, u)
        prog.AddQuadraticCost(np.identity(nu), np.zeros((nu,)), 0, u)
        dVdx_times_f_val = self.dVdx_times_f.Evaluate(env)
        dVdx_times_G_val = self.dVdx_times_G.Evaluate(env)
        V_val = self.clf.Evaluate(env)
        # dVdx * G * u + dVdx * f <= -kappa * V
        prog.AddLinearConstraint(
            np.array([[dVdx_times_G_val]]), np.array([-np.inf]),
            np.array([-self.kappa * V_val - dVdx_times_f_val]), u)

        # Add the cost of Vdot = dVdx*G*u + dVdx * f
        prog.AddLinearCost(
            np.array([self.Vdot_cost_weight * dVdx_times_G_val]),
            self.Vdot_cost_weight * dVdx_times_f_val, u)
        gurobi_solver = GurobiSolver()
        result = gurobi_solver.Solve(prog)
        if not result.is_success():
            raise Exception("CLF controller cannot find u")
        u_sol = result.GetSolution(u)
        output.SetFromVector(u_sol)

    def CalcClf(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(3)}
        output.SetFromVector(np.array([self.clf.Evaluate(env)]))


def simulate(x, f, G, clf, u_max, kappa, initial_state, duration, meshcat):
    builder = DiagramBuilder()

    pendulum = builder.AddSystem(PendulumPlant())
    scene_graph = builder.AddSystem(pydrake.geometry.SceneGraph())
    geom = PendulumGeometry.AddToBuilder(
        builder, pendulum.get_output_port(0), scene_graph)

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, meshcat, MeshcatVisualizerParams(role=Role.kPerception))

    state_converter = builder.AddSystem(
        analysis.PendulumTrigStateConverter(theta_des=np.pi))

    builder.Connect(pendulum.get_output_port(
        0), state_converter.get_input_port())

    clf_controller = builder.AddSystem(PendulumClfController(
        x, f, G, clf, kappa, u_max, Vdot_cost_weight=0))
    builder.Connect(clf_controller.control_output_port(),
                    pendulum.get_input_port())
    builder.Connect(state_converter.get_output_port(),
                    clf_controller.x_input_port())

    state_logger = LogVectorOutput(pendulum.get_output_port(), builder)
    clf_logger = LogVectorOutput(clf_controller.clf_output_port(), builder)
    control_logger = LogVectorOutput(
        clf_controller.control_output_port(), builder)

    diagram = builder.Build()

    simulator = analysis.Simulator(diagram)

    #analysis.ResetIntegratorFromFlags(simulator, "radau3", 0.01)

    simulator.get_mutable_context().SetContinuousState(initial_state)
    simulator.AdvanceTo(duration)

    state_data = state_logger.FindLog(simulator.get_context()).data()
    clf_data = clf_logger.FindLog(simulator.get_context()).data()
    control_data = control_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    return state_data, control_data, clf_data, time_data


def state_eq_constraints(x):
    return np.array([sym.Polynomial(x[0] * x[0] + x[1]*x[1] - 2*x[1])])


def find_clf_init(pendulum, V_degree, x, f, G):
    theta_des = np.pi
    Q = np.diag(np.array([1, 1, 1.]))
    R = np.diag([[1]])
    K, S = analysis.TrigDynamicsLQR(pendulum, theta_des, Q, R)
    theta_samples = np.linspace(np.pi-0.2, np.pi+0.2, 10)
    thetadot_samples = np.linspace(-0.3, 0.3, 10)
    x_val = np.empty((3, theta_samples.shape[0] * thetadot_samples.shape[0]))
    xdot_val = np.empty_like(x_val)
    x_count = 0
    for i in range(theta_samples.shape[0]):
        for j in range(thetadot_samples.shape[0]):
            x_val[:, x_count] = analysis.ToPendulumTrigState(
                theta_samples[i], thetadot_samples[j], theta_des)
            u = -K @ x_val[:, x_count]
            env = {x[i]: x_val[i, x_count] for i in range(3)}
            for k in range(3):
                xdot_val[k, x_count] = (f[k] + G[k] * u).Evaluate(env)
            x_count += 1

    positivity_eps = 0.
    d = int(V_degree / 2)
    state_constraints = np.array([])
    eq_lagrangian_degrees = []

    find_candidate_lyap_ret = analysis.FindCandidateLyapunov(
        x, V_degree, positivity_eps, d, state_constraints, eq_lagrangian_degrees, x_val, xdot_val)
    result = mp.Solve(find_candidate_lyap_ret.prog())
    assert (result.is_success())
    V_init = result.GetSolution(find_candidate_lyap_ret.V)
    return V_init


def search(u_max, kappa):
    x = sym.MakeVectorContinuousVariable(3, "x")
    pendulum = PendulumPlant()
    theta_des = np.pi
    f, G = analysis.TrigPolyDynamics(pendulum, x, theta_des)
    for i in range(3):
        f[i] = f[i].RemoveTermsWithSmallCoefficients(1E-8)
        G[i] = G[i].RemoveTermsWithSmallCoefficients(1E-8)

    u_vertices = np.array([[-u_max, u_max]])
    state_constraints = state_eq_constraints(x)

    load_V_init = False
    V_degree = 2

    if load_V_init:
        pass
    else:
        V_init = find_clf_init(pendulum, V_degree, x, f, G)
        V_init = V_init.RemoveTermsWithSmallCoefficients(1E-6)

    dut = analysis.ControlLyapunov(
        x, f, G, None, u_vertices, state_constraints)
    lambda0_degree = 2
    l_degrees = [4, 4]
    p_degrees = [8]
    V_degree = 4

    # Maximize rho such that V(x)<=rho defines a valid ROA.
    lagrangian_ret = dut.ConstructLagrangianProgram(
        V_init, sym.Polynomial(), int(V_degree / 2) + 1, l_degrees, p_degrees, kappa)
    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result_rho = mp.Solve(lagrangian_ret.prog(), None, solver_options)
    assert (result_rho.is_success())
    V_init = V_init / result_rho.GetSolution(lagrangian_ret.rho)

    positivity_eps = 0.0001
    positivity_d = int(V_degree / 2)
    positivity_eq_lagrangian_degrees = [V_degree - 2]

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.d_converge_tol = 0.
    search_options.bilinear_iterations = 50
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    #search_options.lyap_step_backoff_scale = 0.01
    search_options.lsol_tiny_coeff_tol = 1E-5
    search_options.lyap_tiny_coeff_tol = 1E-7

    state_swingup, control_swingup = SwingUpTrajectoryOptimization(u_max)
    x_swingup = np.empty((3, state_swingup.shape[1]))
    for i in range(state_swingup.shape[1]):
        x_swingup[:, i] = analysis.ToPendulumTrigState(
            state_swingup[0, i], state_swingup[1, i], np.pi)
    x_samples = np.empty((3, 2))
    x_samples[:, 0] = analysis.ToPendulumTrigState(0., 0., np.pi)
    x_samples[:, 1] = analysis.ToPendulumTrigState(2 * np.pi, 0., np.pi)
    #x_samples = x_swingup

    search_result = dut.Search(
        V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees, kappa,
        x_samples, None, True, search_options)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/pendulum/pendulum_trig_clf2.pickle", "wb") as handle:
        pickle.dump({
            "V": clf_cbf_utils.serialize_polynomial(search_result.V),
            "kappa": kappa, "u_max": u_max, "rho": search_options.rho}, handle)


def draw_clf_contour(fig, ax, V, rho, x, draw_heatmap):
    X, Y = np.meshgrid(np.arange(-0.6 * np.pi, 1.5 * np.pi,
                       0.02), np.arange(-10.0, 10, 0.01))

    V_val = V.EvaluateIndeterminates(x, np.vstack((np.sin(X).reshape(
        (1, -1)), (np.cos(X) + 1).reshape((1, -1)), Y.reshape((1, -1))))).reshape(X.shape)
    if draw_heatmap:
        heatmap_handle = ax.pcolormesh(X, Y, V_val)
        fig.colorbar(heatmap_handle, ax=ax)
    else:
        heatmap_handle = None
    contour_handle = ax.contour(X, Y, V_val, [rho])
    contour_handle.collections[0].set_edgecolor('r')
    ax.plot([np.pi], [0], "*", markersize=20, color="r")
    #ax.clabel(contour_handle, [rho])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\dot{\theta}$")
    ax.set_xticks([-0.5*np.pi, 0, 0.5*np.pi, np.pi, 1.5*np.pi],
                  labels=[r"$-0.5\pi$", "0", r"$0.5\pi$", r"$\pi$", r"$1.5\pi$"])
    ax.set_title("V(x) for pendulum")
    return contour_handle, heatmap_handle


def plot_results():
    x = sym.MakeVectorContinuousVariable(3, "x")
    x_set = sym.Variables(x)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/pendulum/pendulum_trig_clf1.pickle", "rb") as input_file:
        load_data = pickle.load(input_file)
        V = clf_cbf_utils.deserialize_polynomial(x_set, load_data["V"])
        u_max = load_data["u_max"]
        kappa = load_data["kappa"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    contour_handle1, heatmap_handle1 = draw_clf_contour(fig1, ax1, V, rho=1, x=x, draw_heatmap=True)
    for fig_format in ["png", "pdf"]:
        fig1.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/pendulum_V1." +
                    fig_format, format=fig_format)

    # Simulate
    pendulum = pydrake.examples.PendulumPlant()
    theta_des = np.pi
    f, G = analysis.TrigPolyDynamics(pendulum, x, theta_des)
    num_trajs = 3
    state_trajs = [None] * num_trajs
    control_trajs = [None] * num_trajs
    time_trajs = [None] * num_trajs
    V_trajs = [None] * num_trajs
    traj_color = [
        [0.2, 0.3, 0.8],
        [0.8, 0.5, 0.6],
        [0.2, 0.7, 0.6]]
    initial_states = [
        np.array([0, 0]),
        #np.array([-0.4*np.pi, 1]),
        np.array([0, 6]),
        np.array([0,-4])]
    for i in range(num_trajs):
        state_trajs[i], control_trajs[i], V_trajs[i], time_trajs[i] = simulate(x, f, G, V, u_max, kappa, initial_states[i], 100, meshcat=None)
    #pdb.set_trace()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    contour_handle2, _ = draw_clf_contour(fig2, ax2, V, rho=1, x=x, draw_heatmap=False)
    for i in range(num_trajs):
        ax2.plot(state_trajs[i][0, :], state_trajs[i][1, :], color=traj_color[i])
    for fig_format in ["pdf", "png"]:
        fig2.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/pendulum_phase."+fig_format, format=fig_format)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    for i in range(num_trajs):
        ax3.plot(time_trajs[i], V_trajs[i][0, :], color=traj_color[i])
    ax3.set_xlabel("time (s)", fontsize=14)
    ax3.set_ylabel("V(x(t))", fontsize=14)
    for fig_format in ["pdf", "png"]:
        fig3.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/pendulum_V_time."+fig_format, format=fig_format)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    for i in range(num_trajs):
        ax4.plot(time_trajs[i], control_trajs[i][0, :], color=traj_color[i])
    ax4.set_xlabel("time (s)", fontsize=14)
    ax4.set_ylabel("u(t)", fontsize=14)
    for fig_format in ["pdf", "png"]:
        fig4.savefig("/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/pendulum_control_time."+fig_format, format=fig_format)


def main():
    u_max = 5
    kappa = 0.1
    #search(u_max, kappa)
    plot_results()


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
