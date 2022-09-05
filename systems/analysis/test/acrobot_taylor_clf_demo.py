import numpy as np
import pickle

import clf_cbf_utils

import acrobot
import pydrake.systems.analysis as analysis
import pydrake.systems.controllers as controllers
import pydrake.examples
import pydrake.symbolic as sym
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver
from pydrake.autodiffutils import (
    InitializeAutoDiff,
    ExtractGradient,
    AutoDiffXd,
)

def synthesize_lqr(params, Q, R):
    xu = np.array([np.pi, 0, 0, 0, 0])
    xu_ad = InitializeAutoDiff(xu)
    M = acrobot.acrobot_mass_matrix(params, xu_ad[:2, 0])
    bias = acrobot.acrobot_dynamic_bias_term(params, xu_ad[:2, 0], xu_ad[2:4, 0])
    M_det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    M_inv = np.array([[M[1, 1] / M_det, -M[1, 0] / M_det], [-M[0, 1] / M_det, M[0, 0] / M_det]])
    xdot_ad = np.empty((4, 1), dtype=object)
    xdot_ad[:2] = xu_ad[2:4]
    xdot_ad[2:] = M_inv @ (np.array([[0], [1]]) * xu_ad[4] - bias.reshape((-1, 1)))
    xdot_grad = ExtractGradient(xdot_ad)
    A = xdot_grad[:, :4]
    B = xdot_grad[:, 4:]
    K, S = controllers.LinearQuadraticRegulator(A, B, Q, R)
    return K, S

class SearchControllerResult:
    def __init__(self):
        self.u = None
        self.lambda_x = None
        self.l_lo = None
        self.l_up = None

def search_controller(x, f, G, V, rho, u_max, kappa, u_degree, f_degree, G_degree):
    """
    −κV−∂V/∂x*(f+Gu) − λ(x)(ρ−V) is sos
    λ(x) is sos
    u_max - u - l_up(x)(ρ−V) is sos
    u + u_max - l_lo(x)(ρ−V) is sos
    l_up(x) is sos, l_lo(x) is sos
    """
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    dVdx = V.Jacobian(x)
    u = prog.NewFreePolynomial(x_set, u_degree)

    Vdot = dVdx.dot(f.squeeze()) + dVdx.dot(G.squeeze() * u)
    V_degree = V.TotalDegree()
    lambda_degree =  int(np.ceil(np.maximum(- 1 + f_degree, G_degree + u_degree - 1) / 2) * 2)
    lambda_x, _ = prog.NewSosPolynomial(x_set, lambda_degree)
    prog.AddSosConstraint(-kappa * V - Vdot - lambda_x * (rho - V))

    l_degree = int(np.ceil(np.maximum(u_degree - V_degree, 0)/ 2)) * 2
    l_up, _ = prog.NewSosPolynomial(x_set, l_degree)
    l_lo, _ = prog.NewSosPolynomial(x_set, l_degree)
    prog.AddSosConstraint(u_max - u - l_up * (rho - V))
    prog.AddSosConstraint(u + u_max - l_lo * (rho - V))

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    ret = SearchControllerResult()
    if result.is_success():
        ret.u= result.GetSolution(u)
        ret.lambda_x= result.GetSolution(lambda_x)
        ret.l_up = result.GetSolution(l_up)
        ret.l_lo = result.GetSolution(l_lo)
    return ret

def search_lyapunov_w_controller(
    x, f, G, V_degree, u_sol, lambda_sol, l_lo_sol, l_up_sol, u_max, kappa,
        x_fix, V_fix):
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    V, V_monomial, V_gram = analysis.NewSosPolynomialPassOrigin(prog, x_set, V_degree)
    # V(x_fix) = V_fix
    A_V_x_fix, var_V_x_fix, b_V_x_fix = V.EvaluateWithAffineCoefficients(x, x_fix)
    prog.AddLinearEqualityConstraint(A_V_x_fix, V_fix - b_V_x_fix, var_V_x_fix)
    dVdx = V.Jacobian(x)
    Vdot = dVdx.dot(f.reshape((-1,)) + G.reshape((-1,)) * u_sol)
    rho = prog.NewContinuousVariables(1)[0]
    prog.AddSosConstraint(-kappa * V - Vdot - lambda_sol * (rho - V))
    prog.AddLinearCost(-rho)
    prog.AddSosConstraint(u_max - u_sol - l_up_sol * (rho - V))
    prog.AddSosConstraint(u_sol + u_max - l_lo_sol * (rho - V))

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    if result.is_success():
        V_sol = result.GetSolution(V)
        rho_sol = result.GetSolution(rho)
        return V_sol, rho_sol
    else:
        return None, None

def search_controller_and_V(x, f, G, u_max, kappa, f_degree, G_degree, V_init, rho_init, V_degree, u_degree, max_iter):
    iter_count = 0
    V = V_init
    rho = rho_init
    while iter_count < max_iter:
        search_controller_result = search_controller(x, f, G, V, rho, u_max, kappa, u_degree, f_degree, G_degree)
        x_fix = np.full((4,), 0.1)
        V_fix = V.Evaluate({x[i]: x_fix[i] for i in range(4)})
        V, rho = search_lyapunov_w_controller(x, f, G, V_degree, search_controller_result.u, search_controller_result.lambda_x, search_controller_result.l_lo, search_controller_result.l_up, u_max, kappa, x_fix, V_fix)
        iter_count += 1
    return V, rho, search_controller_result.u


def search(params, x, f, G, u_max, kappa, f_degree, G_degree):
    x_set = sym.Variables(x)
    V_degree = 2
    u_vertices = np.array([[-u_max, u_max]])

    dut = analysis.ControlLyapunov(x, f, G, None, u_vertices, state_constraints=np.array([]))

    lambda0_degree = 6
    l_degrees = [4, 4]
    p_degrees = []

    load_clf = False
    if load_clf:
        pass
    else:
        K, S = synthesize_lqr(params, Q=np.diag(np.array([0.1, 0.1, 1, 1])), R=np.array([[0.01]]))
        V_init = sym.Polynomial(0.01*x.dot(S @ x))
        rho = 0.0015
        #max_rho_ret = dut.ConstructLagrangianProgram(V_init, sym.Polynomial(x.dot(x)), int(lambda0_degree / 2), l_degrees, p_degrees, kappa)
        #solver = MosekSolver()
        #solver_options = mp.SolverOptions()
        #solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        #max_rho_result = solver.Solve(max_rho_ret.prog(), None, solver_options)
        #rho_sol = max_rho_result.GetSolution(max_rho_ret.rho)
        solver_options = mp.SolverOptions()
        solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
        lagrangian_ret = dut.ConstructLagrangianProgram(V_init, rho, kappa, lambda0_degree, l_degrees, p_degrees, None)
        result_lagrangian = mp.Solve(lagrangian_ret.prog(), None, solver_options)
        lambda0_sol = result_lagrangian.GetSolution(lagrangian_ret.lambda0)
        l_sol = np.array([result_lagrangian.GetSolution(lagrangian_ret.l[i]) for i in range(2)])
        lyapunov_ret = dut.ConstructLyapunovProgram(lambda0_sol, l_sol, V_degree, rho, 0., 1, [], p_degrees, kappa, None)
        result_lyapunov = mp.Solve(lyapunov_ret.prog(), None, solver_options)

        V_sol, rho_sol, u_sol = search_controller_and_V(x, f, G, u_max, kappa, f_degree, G_degree, V_init, rho, V_degree, u_degree=3, max_iter=5)
        pass

    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.rho = rho
    search_options.lyap_step_backoff_scale = 0.03
    #search_options.lsol_tiny_coeff_tol = 1E-4
    #search_options.lyap_tiny_coeff_tol = 1E-6
    positivity_eps = 0.000
    x_star = np.zeros((4,))
    S = np.diag(np.array([10, 10, 10, 10.]))
    r_degree = 0
    ellipsoid_option = analysis.ControlLyapunov.EllipsoidMaximizeOption(sym.Polynomial(), 0, backoff_scale=0.05)

    #x_samples = np.array([[0, 0.0, 0, 0], [0.0, 0, 0, 0]]).T
    #in_roa_samples = None
    search_result = dut.Search(V_init, lambda0_degree, l_degrees, V_degree, positivity_eps, int(V_degree / 2), [], p_degrees, [], kappa, x_star, S, r_degree, search_options, ellipsoid_option)

    pass

def main():
    params = pydrake.examples.AcrobotParams()
    # x is actually x_bar here, namely it is the original state - [pi, 0, 0, 0]
    x = sym.MakeVectorContinuousVariable(4, "x")
    f_degree = 5
    G_degree = 4
    f, G = acrobot.acrobot_taylor_affine_dynamics(params, x, f_degree, G_degree)
    u_max = 10.
    kappa = 0.01
    search(params, x, f, G, u_max, kappa, f_degree, G_degree)


if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
