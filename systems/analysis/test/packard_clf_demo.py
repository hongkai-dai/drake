import numpy as np
import scipy.integrate
import pickle

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d

import clf_cbf_utils

import pydrake.systems.controllers as controllers
import pydrake.symbolic as sym
import pydrake.systems.analysis as analysis
from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.mosek import MosekSolver


matplotlib.rcParams["text.usetex"] = True

def dynamics(x):
    """
    xdot[0] = u
    xdot[1] = -x[0] + 1/6*x[0]**3-u
    """
    f = np.array([sym.Polynomial(), sym.Polynomial(-x[0] + 1./6*x[0]**3)])
    G = np.array([sym.Polynomial(1), sym.Polynomial(-1)])
    return f, G


def synthesize_lqr(Q, R):
    A = np.array([[0, 0], [-1., 0]])
    B = np.array([[1], [-1.]])
    K, S = controllers.LinearQuadraticRegulator(A, B, Q, R)
    return K, S

def draw_clf_3d(V, rho_val, x, contour_color):
    xs = np.arange(-3, 3, 0.01)
    ys = np.arange(-3, 3, 0.01)
    X, Y = np.meshgrid(xs, ys)
    V_val = V.EvaluateIndeterminates(x, np.vstack((X.reshape((1, -1)), Y.reshape((1, -1))))).reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, V_val)
    contour_handle = ax.contour3D(X, Y, V_val, [rho_val])
    contour_handle.collections[0].set_edgecolor(contour_color)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel("V")
    return fig, ax


class SearchControllerResult:
    def __init__(self):
        self.u = None
        self.lambda_x = None
        self.l_lo = None
        self.l_up = None

def search_controller(x, f, G, V, rho, u_max, kappa, u_degree):
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
    lambda_degree =  int(np.ceil(np.maximum(2, u_degree - 1) / 2) * 2)
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

def search_lagrangian_psatz(x, f, G, V, rho, u_max, kappa, lambda0_degree, l_degrees, l_hi_degree):
    """
    Find the lagrangian multiplier for the condition
    (1+λ₀(x))xᵀx(V(x)−ρ) − ∑ᵢlᵢ(x)(V̇(x, uⁱ)+κV) 
         − l_hi(x)(V̇(x, u⁰)+κV)(V̇(x, u¹)+κV) is sos
    λ₀(x), lᵢ(x), l_hi(x) is sos
    """
    prog = mp.MathematicalProgram()
    prog.AddIndeterminates(x)
    x_set = sym.Variables(x)
    lambda0, _ = prog.NewSosPolynomial(x_set, lambda0_degree)
    l = np.empty((2,), dtype=object)
    u_vertices = np.array([[-u_max, u_max]])
    for i in range(2):
        l[i], _ = prog.NewSosPolynomial(x_set, l_degrees[i])
    l_hi, _ = prog.NewSosPolynomial(x_set, l_hi_degree)
    vdot_sos_condition = (1 + lambda0) * sym.Polynomial(x.dot(x)) * (V - rho)
    dVdx = V.Jacobian(x)
    dVdx_times_f = dVdx.dot(f)
    dVdx_times_G = dVdx.dot(G)
    for i in range(2):
        vdot_sos_condition -= l[i] * (dVdx_times_f + dVdx_times_G * u_vertices[0, i] + kappa * V)

    vdot_sos_condition -= l_hi * (dVdx_times_f + dVdx_times_G * u_vertices[0, 0] + kappa * V) * (dVdx_times_f + dVdx_times_G * u_vertices[0, 1] + kappa * V)
    vdot_gram, vdot_monomials = prog.AddSosConstraint(vdot_sos_condition)

    solver_options = mp.SolverOptions()
    solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    result = mp.Solve(prog, None, solver_options)
    if result.is_success():
        lambda0_sol = result.GetSolution(lambda0)
        l_sol = np.array([result.GetSolution(l[i]) for i in range(2)])
        l_hi_sol = result.GetSolution(l_hi)
        return lambda0_sol, l_sol, l_hi_sol
    else:
        return None, None, None

def simulate_clf(x, clf, u_max, kappa, x0, duration):

    f, G = dynamics(x)
    dVdx = clf.Jacobian(x)
    dVdx_times_f = dVdx.dot(f)
    dVdx_times_G = dVdx.dot(G)

    def calc_u(x_val):
        env = {x[0]: x_val[0], x[1]: x_val[1]} 
        V_val = clf.Evaluate(env)
        dVdx_times_f_val = dVdx_times_f.Evaluate(env)
        dVdx_times_G_val = dVdx_times_G.Evaluate(env)
        prog = mp.MathematicalProgram()
        u_var = prog.NewContinuousVariables(1)
        # dVdx*(f+Gu)<= -κV
        prog.AddLinearConstraint(np.array([[dVdx_times_G_val]]), np.array([-np.inf]), np.array([-kappa*V_val - dVdx_times_f_val]), u_var)
        prog.AddBoundingBoxConstraint(-u_max, u_max, u_var[0])
        prog.AddQuadraticCost(np.array([[1]]), np.array([0]), 0, u_var, is_convex=True)

        result = mp.Solve(prog)
        assert (result.is_success())
        u_sol = result.GetSolution(u_var[0])
        return u_sol

    def calc_xdot(x_val, u_val):
        xdot = np.array([u_val, -x_val[0] + 1./6 * x_val[0]**3 - u_val])
        return xdot

    print(f"V(x0)={clf.Evaluate({x[0]: x0[0], x[1]: x0[1]})}")
    simulate_result = scipy.integrate.solve_ivp(lambda t, x_val: calc_xdot(x_val, calc_u(x_val)), [0, duration], x0, max_step=0.5)
    print(simulate_result.y[:, -1])
    return simulate_result

def draw_clf_contour(fig, ax, V, rho_vals, x):
    xs = np.arange(-4, 4, 0.01)
    ys = np.arange(-4, 4, 0.01)
    X, Y = np.meshgrid(xs, ys)
    V_val = V.EvaluateIndeterminates(x, np.vstack((X.reshape((1, -1)), Y.reshape((1, -1))))).reshape(X.shape)
    contour_handle = ax.contour(X, Y, V_val, rho_vals, linewidths=1)
    ax.set_xlabel(r"$x_1$", fontsize=20)
    ax.set_ylabel(r"$x_2$", fontsize=20)
    return contour_handle

def draw_contours(V, x, V_init, rho_init):
    def draw_contours_fun(V, x, V_init, rho_init, rho_vals, edge_colors, contour_labels):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        contour_handle_init = draw_clf_contour(fig, ax, V_init, [rho_init], x)
        contour_handle_init.collections[0].set_edgecolor('g')
        contour_handle = draw_clf_contour(fig, ax, V, rho_vals, x)
        for i in range(len(edge_colors)):
            contour_handle.collections[i].set_edgecolor(edge_colors[i])
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=contour_handle_init.collections[0].get_edgecolor()[0])] + [plt.Rectangle((0, 0), 1, 1, fc=pc.get_edgecolor()[0]) for pc in contour_handle.collections]
        contour_labels = ["CLF_init"] + contour_labels
        plt.legend(proxy, contour_labels, fontsize=12)
        return fig, ax

    fig, ax = draw_contours_fun(V, x, V_init, rho_init, [], [], [])
    for fig_format in ["pdf", "png"]:
        fig.savefig(f"/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/packard/packard_V8_2_contours0."+fig_format, format=fig_format, bbox_inches="tight")
    rho_vals = [0.0002, 0.00025, 0.005, 0.0065, 0.14, 0.4]
    edge_colors = ["slategrey", "tan", "cyan", "plum", "black", "r"]
    contour_labels = ["degree(u)=1", "degree(u)=3", "degree(u)=5", "degree(u)=7", "degree(u)=9", "CLF"]
    for i in range(6):
        fig, ax = draw_contours_fun(V, x, V_init, rho_init, rho_vals[:i] + [rho_vals[-1]], edge_colors[:i] + [edge_colors[-1]], contour_labels[:i] + [contour_labels[-1]])
        for fig_format in ["pdf", "png"]:
            fig.savefig(f"/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/packard/packard_V8_2_contours{i+1}."+fig_format, format=fig_format, bbox_inches="tight")

def simulate_u(x, u, V, kappa, u_max, x0, duration):
    def calc_xdot(x_val):
        u_val = u.Evaluate({x[0]: x_val[0], x[1]: x_val[1]})
        if (u_val > u_max or u_val < -u_max):
            raise Exception(f"u={u_val} out of input limit at time {t}")
        return np.array([u_val, -x_val[0] + 1./6 * x_val[0]**3 - u_val])

    def u_too_large(t, x_val):
        u_val = u.Evaluate({x[0]: x_val[0], x[1]: x_val[1]})
        return u_val - u_max
    u_too_large.terminal = True
    u_too_large.direction = 1.

    def u_too_small(t, x_val):
        u_val = u.Evaluate({x[0]: x_val[0], x[1]: x_val[1]})
        return u_val + u_max
    u_too_small.terminal = True
    u_too_small.direction = -1

    def Vdot_error(t, x_val):
        env = {x[0]: x_val[0], x[1]: x_val[1]}
        V_val = V.Evaluate(env)
        dVdx = V.Jacobian(x)
        dVdx_val = np.array([dVdx[i].Evaluate(env) for i in range(2)])
        xdot = calc_xdot(x_val)
        Vdot = dVdx_val.dot(xdot)
        return Vdot + kappa * V_val
    Vdot_error.terminal = True
    Vdot_error.direction = 1


    simulation_result = scipy.integrate.solve_ivp(
        lambda t, x_val: calc_xdot(x_val), [0, duration], x0,
        events=[u_too_large, u_too_small, Vdot_error], max_step=0.5)
    return simulation_result

def search(u_max, kappa):
    x = sym.MakeVectorContinuousVariable(2, "x")
    f, G = dynamics(x)
    u_vertices = np.array([[-u_max, u_max]])
    dut = analysis.ControlLyapunov(x, f, G, None, u_vertices, np.array([]))
    K, S = synthesize_lqr(Q=np.diag(np.array([1, 1])), R=np.array([[1]]))
    V_init = sym.Polynomial(x.dot(S @ x))
    #V_init = sym.Polynomial(sym.pow(x.dot(x), 2))
    rho_init = 0.4

    V_degree = 8
    lambda0_degree = 6
    l_degrees = [6, 6]
    p_degrees = []
    search_options = analysis.ControlLyapunov.SearchOptions()
    search_options.d_converge_tol = 0.
    search_options.lagrangian_step_solver_options = mp.SolverOptions()
    search_options.lagrangian_step_solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_solver_options = mp.SolverOptions()
    search_options.lyap_step_solver_options.SetOption(mp.CommonSolverOption.kPrintToConsole, 1)
    search_options.lyap_step_backoff_scale = 0.02
    search_options.rho = rho_init
    search_options.bilinear_iterations = 1000
    search_options.lyap_tiny_coeff_tol = 1E-8
    search_options.Vsol_tiny_coeff_tol = 1E-6
    x_star = np.array([0., 0.])
    ellipsoid_option = analysis.ControlLyapunov.EllipsoidMaximizeOption(sym.Polynomial(sym.pow(x.dot(x), int(V_degree / 2)-1)), 0, backoff_scale=0.01)
    search_result = dut.Search(
        V_init, lambda0_degree, l_degrees, V_degree, 0., 1, [], p_degrees, [],
        kappa, x_star, S, r_degree=V_degree - 2, search_options=search_options,
        ellipsoid_option=ellipsoid_option)

    #with open("/home/hongkaidai/Dropbox/sos_clf_cbf/packard/V8_2.pickle", "wb") as handle:
    #    pickle.dump({"V": clf_cbf_utils.serialize_polynomial(search_result.V),
    #        "rho": search_options.rho,
    #        "V_init": clf_cbf_utils.serialize_polynomial(V_init),
    #        "kappa": kappa,
    #        "u_max": u_max,
    #        "lambda0_degree": lambda0_degree,
    #        "l_degrees": l_degrees}, handle)

    fig, ax = draw_clf_3d(V_init, rho_init, x, "r")
    for fig_format in ("pdf", "png"):
        fig.savefig(f"/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/packard/clf_init_3d.{fig_format}", format=fig_format, bbox_inches="tight")
    fig, ax = draw_clf_3d(search_result.V, 0.4, x, "r")
    for fig_format in ("pdf", "png"):
        fig.savefig(f"/home/hongkaidai/Dropbox/talks/pictures/sos_clf_cbf/packard/clf_3d.{fig_format}", format=fig_format, bbox_inches="tight")

    fig, ax, contour_handle = draw_contours(search_result.V, x, V_init, rho_init)

    simulate_result1 = simulate_clf(x, search_result.V, u_max, kappa, np.array([-2.48, 1.5]), 1000)
    simulate_result3 = simulate_clf(x, search_result.V, u_max, kappa, np.array([2, -1.8]), 1000)
    simulate_result4 = simulate_clf(x, search_result.V, u_max, kappa, np.array([-2.52, -1.8]), 1000)
    simulate_result5 = simulate_clf(x, search_result.V, u_max, kappa, np.array([0.2, 7.4]), 1000)

    #lambda0_sol, l_sol, l_hi_sol = search_lagrangian_psatz(x, f, G, search_result.V, search_options.rho + 0.55, u_max, kappa, 16, [16, 16], 8)
    search_controller_result9 = search_controller(x, f, G, search_result.V, search_options.rho + 0.13, u_max, kappa, u_degree=9)
    search_controller(x, f, G, search_result.V, search_options.rho + 0.12, u_max, kappa, u_degree=7)
    search_controller(x, f, G, search_result.V, search_options.rho - 0.041, u_max, kappa, u_degree=5)
    search_controller(x, f, G, search_result.V, search_options.rho - 0.27, u_max, kappa, u_degree=3)
    search_controller(x, f, G, search_result.V, search_options.rho - 0.277, u_max, kappa, u_degree=1)

def main():
    u_max = 0.4
    kappa = 0.5
    search(u_max, kappa)

if __name__ == "__main__":
    with MosekSolver.AcquireLicense():
        main()
