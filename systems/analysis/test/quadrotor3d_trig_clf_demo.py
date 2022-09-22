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
    QuadrotorTrigGeometry,
)
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    SceneGraph,
)



class QuadrotorClfController(LeafSystem):
    def __init__(self, x, f, G, clf, kappa, thrust_max, Vdot_cost_weight):
        LeafSystem.__init__(self)
        assert (x.shape == (13,))
        self.x = x
        assert (f.shape == (13,))
        self.f = f
        assert (G.shape == (13, 4))
        self.G = G
        self.clf = clf
        self.kappa = kappa
        self.thrust_max = thrust_max
        self.Vdot_cost_weight = Vdot_cost_weight
        dVdx = self.clf.Jacobian(self.x)
        self.dVdx_times_f = dVdx.dot(self.f)
        self.dVdx_times_G = dVdx @ self.G
        self.x_input_index = self.DeclareVectorInputPort("x", 13).get_index()
        self.control_output_index = self.DeclareVectorOutputPort(
            "control", 4, self.CalcControl).get_index()
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
        # dVdx * G * u + dVdx * f <= -kappa * V
        prog.AddLinearConstraint(
            dVdx_times_G_val.reshape((1, -1)), np.array([-np.inf]),
            np.array([-self.kappa * V_val - dVdx_times_f_val]), u)

        
        # Add the cost of Vdot = dVdx*G*u + dVdx * f
        prog.AddLinearCost(self.Vdot_cost_weight * dVdx_times_G_val, self.Vdot_cost_weight * dVdx_times_f_val, u)
        gurobi_solver = GurobiSolver()
        result = gurobi_solver.Solve(prog)
        if not result.is_success():
            raise Exception("CLF controller cannot find u")
        u_sol = result.GetSolution(u)
        output.SetFromVector(u_sol)

    def CalcClf(self, context, output):
        x_val = self.x_input_port().Eval(context)
        env = {self.x[i]: x_val[i] for i in range(13)}
        output.SetFromVector(np.array([self.clf.Evaluate(env)]))


def simulate(x, f, G, clf, thrust_max, kappa, initial_state, duration, meshcat, controller="clf"):
    builder = DiagramBuilder()

    quadrotor = builder.AddSystem(analysis.QuadrotorTrigPlant())

    scene_graph = builder.AddSystem(pydrake.geometry.SceneGraph())

    geom = QuadrotorTrigGeometry.AddToBuilder(
        builder, quadrotor.get_output_port(0), None, scene_graph)

    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams(role=Role.kPerception))

    if controller == "lqr":
        K, _ = SynthesizeTrigLqr()
        lqr_gain = builder.AddSystem(controller.Gain(-K))
        adder = builder.AddSystem(primitives.Adder(2, 4))
        thrust_equilibrium = quadrotor.m() * quadrotor.g() / 4
        lqr_constant = builder.AddSystem(
            primitives.ConstantValueSource(np.full((4,), thrust_equilibrium)))
        builder.Connect(quadrotor.get_output_port(0),
                        lqr_gain.get_input_port())
        builder.Connect(lqr_gain.get_output_port(), adder.get_input_port(0))
        builder.Connect(lqr_constant.get_output_port(),
                        adder.get_input_port(1))
        u_saturation = builder.AddSystem(primitives.Saturation(
            np.zeros((4,)), np.full((4,), thrust_max)))
        builder.Connect(adder.get_output_port(), u_saturation.get_input_port())
        builder.Connect(u_saturation.get_output_port(),
                        quadrotor.get_input_port())
    elif controller == "clf":
        clf_controller = builder.AddSystem(
            QuadrotorClfController(x, f, G, clf, kappa, thrust_max, Vdot_cost_weight=1000))

        builder.Connect(clf_controller.control_output_port(),
                        quadrotor.get_input_port())
        builder.Connect(quadrotor.get_output_port(0),
                        clf_controller.x_input_port())

    state_logger = LogVectorOutput(quadrotor.get_output_port(), builder)
    if controller == "clf":
        clf_logger = LogVectorOutput(clf_controller.clf_output_port(), builder)
    control_logger = LogVectorOutput(
        clf_controller.control_output_port(), builder)

    diagram = builder.Build()

    simulator = analysis.Simulator(diagram)

    analysis.ResetIntegratorFromFlags(simulator, "radau3", 0.01)

    x0 = analysis.ToQuadrotorTrigState(initial_state)
    simulator.get_mutable_context().SetContinuousState(x0)
    simulator.AdvanceTo(duration)

    state_data = state_logger.FindLog(simulator.get_context()).data()
    if controller == "clf":
        clf_data = clf_logger.FindLog(simulator.get_context()).data()
    else:
        clf_data = None
    control_data = control_logger.FindLog(simulator.get_context()).data()
    time_data = state_logger.FindLog(simulator.get_context()).sample_times()
    return state_data, control_data, clf_data, time_data


def simulate_demo(meshcat):
    x = sym.MakeVectorContinuousVariable(13, "x")
    x_set = sym.Variables(x)
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_clf/quadrotor3d_trig_clf_sol3.pickle", "rb") as input_file:
        load_data = pickle.load(input_file)
        clf = clf_cbf_utils.deserialize_polynomial(
            x_set, load_data["V"])
        kappa = load_data["kappa"]
        thrust_max = load_data["thrust_max"]
    quadrotor = analysis.QuadrotorTrigPlant()
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    initial_state = np.zeros((12,))
    initial_state[0] = 2
    initial_state[1] = 0 
    initial_state[3] = np.pi * 0.8
    state_data, control_data, clf_data, time_data = simulate(
        x, f, G, clf, thrust_max, kappa, initial_state, 30, meshcat, controller="clf")
    with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_clf/quadrotor3d_trig_clf3_sim8.pickle", "wb") as handle:
        pickle.dump({"state_data": state_data, "control_data": control_data, "clf_data": clf_data, "time_data": time_data}, handle)
    return


def SynthesizeTrigLqr():
    quadrotor = analysis.QuadrotorTrigPlant()
    context = quadrotor.CreateDefaultContext()
    context.SetContinuousState(np.zeros((13,)))
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    quadrotor.get_input_port().FixValue(context, np.ones((4,)) * thrust_equilibrium)

    linearized_quadrotor = primitives.Linearize(quadrotor, context)
    F = np.zeros((1, 13))
    F[0, 0] = 1
    lqr_Q_diag = np.array([1.] * 7 + [10.] * 6)
    lqr_Q = np.diag(lqr_Q_diag)
    K, S = controllers.LinearQuadraticRegulator(
        linearized_quadrotor.A(), linearized_quadrotor.B(), lqr_Q,
        R=10 * np.eye(4), F=F)
    return K, S


def FindClfInit(V_degree, x) -> sym.Polynomial:
    quadrotor = analysis.QuadrotorTrigPlant()
    quadrotor_sym = analysis.QuadrotorTrigPlant_[sym.Expression]()
    K, _ = SynthesizeTrigLqr()
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    x_expr = np.array([sym.Expression(x[i]) for i in range(13)])
    u_lqr = -K @ x_expr + np.ones(4) * thrust_equilibrium
    context_sym = quadrotor_sym.CreateDefaultContext()
    context_sym.SetContinuousState(x)
    quadrotor_sym.get_input_port().FixValue(context_sym, u_lqr)
    dynamics_expr = quadrotor_sym.EvalTimeDerivatives(
        context_sym).CopyToVector()
    dynamics = np.empty((13,), dtype=object)
    for i in range(13):
        dynamics[i] = sym.Polynomial(dynamics_expr[i])
        dynamics[i] = dynamics[i].RemoveTermsWithSmallCoefficients(1E-8)

    positivity_eps = 0.0001
    d = int(V_degree / 2)
    kappa = 0.1
    state_eq_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [
        int(np.ceil((V_degree + 3) / 2)) * 2 - 2]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 0.01)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = derivative_ceq_lagrangian_degrees

    ret = analysis.FindCandidateRegionalLyapunov(
        x, dynamics, None, V_degree, positivity_eps, d, kappa,
        state_eq_constraints, positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees, state_ineq_constraints,
        positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(ret.prog(), None, solver_options)
    assert (result.is_success())
    V_sol = result.GetSolution(ret.V)
    return V_sol


def SearchWTrigDynamics():
    quadrotor = analysis.QuadrotorTrigPlant()
    x = sym.MakeVectorContinuousVariable(13, "x")
    x_set = sym.Variables(x)
    f, G = analysis.TrigPolyDynamics(quadrotor, x)
    thrust_equilibrium = analysis.EquilibriumThrust(quadrotor)
    thrust_max = 3 * thrust_equilibrium
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

    V_degree = 2

    search_init = False
    if search_init:
        V_init = FindClfInit(V_degree, x)
        V_init = V_init.RemoveTermsWithSmallCoefficients(1E-6)
        with open("quadrotor3d_trig_clf_init.pickle", "wb") as handle:
            pickle.dump({"V": clf_cbf_utils.serialize_polynomial(
                V_init)}, handle)
    else:
        with open("/home/hongkaidai/Dropbox/sos_clf_cbf/quadrotor3d_clf/quadrotor3d_trig_clf_init.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, pickle.load(input_file)["V"])

    dut = analysis.ControlLyapunov(
        x, f, G, None, u_vertices, state_constraints)
    lambda0_degree = 2
    l_degrees = [2] * 16
    p_degrees = [4]

    kappa = 0.1
    maximize_init_rho = False
    if maximize_init_rho:
        # Maximize rho such that V(x) <= rho defines a valid ROA.
        d_degree = int(lambda0_degree / 2) + 1
        lagrangian_ret = dut.ConstructLagrangianProgram(
            V_init, sym.Polynomial(), d_degree, l_degrees, p_degrees, kappa)
        solver_options = mp.SolverOptions()
        solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        result = mp.Solve(lagrangian_ret.prog(), None, solver_options)
        assert (result.is_success())
        rho_sol = result.GetSolution(lagrangian_ret.rho)
        print(f"V_init(x) <= {rho_sol}")
        V_init = V_init / rho_sol
        with open("quadrotor3d_trig_clf_max_rho.pickle", "wb") as handle:
            pickle.dump({"V": clf_cbf_utils.serialize_polynomial(
                V_init), "kappa": kappa, "thrust_max": thrust_max},
                handle)
    else:
        with open("quadrotor3d_trig_clf_sol1.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, pickle.load(input_file)["V"])

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.d_converge_tol = 0.
    search_options.bilinear_iterations = 10
    search_options.lyap_step_backoff_scale = 0.015
    search_options.lsol_tiny_coeff_tol = 1E-8
    search_options.lyap_tiny_coeff_tol = 1E-8
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(
        mp.CommonSolverOption.kPrintToConsole, 1)
    state_samples = np.zeros((12, 4))
    state_samples[:, 0] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 1] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 2] = np.array(
        [1, 0, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0.])
    state_samples[:, 3] = np.array(
        [1, 0, 0, np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0.])
    x_samples = np.zeros((13, state_samples.shape[1]))
    for i in range(state_samples.shape[1]):
        x_samples[:, i] = analysis.ToQuadrotorTrigState(state_samples[:, i])

    positivity_eps = 0.0001
    positivity_d = int(V_degree / 2)
    positivity_eq_lagrangian_degrees = [V_degree - 2]
    minimize_max = True
    search_result = dut.Search(
        V_init, lambda0_degree, l_degrees, V_degree, positivity_eps,
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees, kappa,
        x_samples, None, minimize_max, search_options)
    print(
        f"V(x_samples): {search_result.V.EvaluateIndeterminates(x, x_samples).T}")
    with open("quadrotor3d_trig_clf_sol2.pickle", "wb") as handle:
        pickle.dump({"V": clf_cbf_utils.serialize_polynomial(
            search_result.V), "kappa": kappa, "thrust_max": thrust_max}, handle)


def main():
    pydrake.common.configure_logging()
    #SearchWTrigDynamics()
    meshcat = StartMeshcat()
    simulate_demo(meshcat)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
