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
    deriv_eps = 0.1
    state_eq_constraints = np.array([analysis.QuadrotorStateEqConstraint(x)])
    positivity_ceq_lagrangian_degrees = [V_degree - 2]
    derivative_ceq_lagrangian_degrees = [
        int(np.ceil((V_degree + 3) / 2)) * 2 - 2]
    state_ineq_constraints = np.array([sym.Polynomial(x.dot(x) - 0.01)])
    positivity_cin_lagrangian_degrees = [V_degree - 2]
    derivative_cin_lagrangian_degrees = derivative_ceq_lagrangian_degrees

    ret = analysis.FindCandidateRegionalLyapunov(
        x, dynamics, None, V_degree, positivity_eps, d, deriv_eps,
        state_eq_constraints, positivity_ceq_lagrangian_degrees,
        derivative_ceq_lagrangian_degrees, state_ineq_constraints,
        positivity_cin_lagrangian_degrees, derivative_cin_lagrangian_degrees)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(ret.prog(), None, solver_options)
    assert(result.is_success())
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
        with open("quadrotor3d_trig_clf_init.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, pickle.load(input_file)["V"])

    dut = analysis.ControlLyapunov(
        x, f, G, None, u_vertices, state_constraints)
    lambda0_degree = 2
    l_degrees = [2] * 16
    p_degrees = [4]

    deriv_eps = 0.1
    maximize_init_rho = False 
    if maximize_init_rho:
        # Maximize rho such that V(x) <= rho defines a valid ROA.
        d_degree = int(lambda0_degree / 2) + 1
        lagrangian_ret = dut.ConstructLagrangianProgram(
            V_init, sym.Polynomial(), d_degree, l_degrees, p_degrees, deriv_eps)
        solver_options = mp.SolverOptions()
        solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        result = mp.Solve(lagrangian_ret.prog(), None, solver_options)
        assert(result.is_success())
        rho_sol = result.GetSolution(lagrangian_ret.rho)
        print(f"V_init(x) <= {rho_sol}")
        V_init = V_init / rho_sol
        with open("quadrotor3d_trig_clf_max_rho.pickle", "wb") as handle:
            pickle.dump({"V": clf_cbf_utils.serialize_polynomial(
                V_init), "deriv_eps": deriv_eps, "thrust_max": thrust_max},
                handle)
    else:
        with open("quadrotor3d_trig_clf_max_rho.pickle", "rb") as input_file:
            V_init = clf_cbf_utils.deserialize_polynomial(
                x_set, pickle.load(input_file)["V"])

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.d_converge_tol = 0.
    search_options.bilinear_iterations = 1
    search_options.lyap_step_backoff_scale = 0.01
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
        positivity_d, positivity_eq_lagrangian_degrees, p_degrees, deriv_eps,
        x_samples, None, minimize_max, search_options)
    print(
        f"V(x_samples): {search_result.V.EvaluateIndeterminates(x, x_samples).T}")
    with open("quadrotor3d_trig_clf_sol.pickle", "wb") as handle:
        pickle.dump({"V": clf_cbf_utils.serialize_polynomial(
            search_result.V), "deriv_eps": deriv_eps, "thrust_max": thrust_max}, handle)


def main():
    pydrake.common.configure_logging()
    SearchWTrigDynamics()


if __name__ == "__main__":
    main()
