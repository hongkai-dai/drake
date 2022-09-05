import numpy as np
import pydrake.examples.acrobot
import pydrake.symbolic as sym

def acrobot_mass_matrix(params, q):
    assert (q.shape == (2,))
    c2 = np.cos(q[1])
    I1 = params.Ic1() + params.m1() * params.lc1() * params.lc1()
    I2 = params.Ic2() + params.m2() * params.lc2() * params.lc2()
    m2l1lc2 = params.m2() * params.l1() * params.lc2()
    m12 = I2 + m2l1lc2 * c2
    M = np.array([[I1 + I2 + params.m2() * params.l1() * params.l1() + 2 * m2l1lc2 * c2, m12], [m12, I2]])
    return M

def acrobot_dynamic_bias_term(params, q, qdot):
    assert (q.shape == (2,))
    assert (qdot.shape == (2,))
    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s12 = np.sin(q[0] + q[1])
    m2l1lc2 = params.m2() * params.l1() * params.lc2()

    bias = np.array([
        -2 * m2l1lc2 * s2 * qdot[1] * qdot[0] - m2l1lc2 * s2 * qdot[1] ** 2,
        m2l1lc2 * s2 * qdot[0] * qdot[0]])

    bias[0] += params.gravity() * params.m1() * params.lc1() * s1 + params.gravity() * params.m2() * (params.l1() * s1 + params.lc2() * s12)
    bias[1] += params.gravity() * params.m2() * params.lc2() * s12

    bias[0] += params.b1() * qdot[0]
    bias[1] += params.b2() * qdot[1]
    return bias


def acrobot_taylor_affine_dynamics(params, x_bar, f_degree, G_degree):
    # Compute f, G as a polynomial of x_bar = x - [pi, 0, 0, 0]
    assert (x_bar.shape == (4,))
    assert (isinstance(x_bar[0], sym.Variable))
    x = x_bar + np.array([np.pi, 0, 0, 0])
    f = np.empty((4, 1), dtype=object)
    G = np.empty((4, 1), dtype=object)
    f[0, 0] = sym.Polynomial(x[2])
    f[1, 0] = sym.Polynomial(x[3])
    G[0, 0] = sym.Polynomial()
    G[1, 0] = sym.Polynomial()
    # f[2:, :] is M.inv() * -bias
    M = acrobot_mass_matrix(params, x[:2])
    bias = acrobot_dynamic_bias_term(params, x[:2], x[2:])
    M_det = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1]
    M_inv = np.array([[M[1, 1] / M_det, -M[1, 0] / M_det], [-M[0, 1] / M_det, M[0, 0] / M_det]])
    f_tail = M_inv @ (-bias)
    env = {x_bar[i]: 0. for i in range(4)}
    f[2, 0] = sym.Polynomial(sym.TaylorExpand(f_tail[0], env, f_degree))
    f[3, 0] = sym.Polynomial(sym.TaylorExpand(f_tail[1], env, f_degree))
    G_tail = M_inv[:, 1]
    G[2, 0] = sym.Polynomial(sym.TaylorExpand(G_tail[0], env, G_degree))
    G[3, 0] = sym.Polynomial(sym.TaylorExpand(G_tail[1], env, G_degree))
    for i in range(4):
        f[i, 0] = f[i, 0].RemoveTermsWithSmallCoefficients(1E-8)
        G[i, 0] = G[i, 0].RemoveTermsWithSmallCoefficients(1E-8)
    return f, G

