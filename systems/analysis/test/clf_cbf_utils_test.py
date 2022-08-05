import clf_cbf_utils

import pydrake.symbolic as sym

import unittest

class SerializePolynomialTest(unittest.TestCase):
    def setUp(self):
        self.x = [sym.Variable(f"x{i}") for i in range(4)]
        self.x_set = sym.Variables(self.x)

    def polynomial_tester(self, p):
        terms = clf_cbf_utils.serialize_polynomial(p)
        p_reconstruct = clf_cbf_utils.deserialize_polynomial(self.x_set, terms)
        self.assertTrue(p.EqualTo(p_reconstruct))


    def test(self):
        p = sym.Polynomial()
        self.polynomial_tester(p)

        p = sym.Polynomial(1)
        self.polynomial_tester(p)

        p = sym.Polynomial(self.x[0])
        self.polynomial_tester(p)

        p = sym.Polynomial(self.x[0] + 1)
        self.polynomial_tester(p)

        p = sym.Polynomial(self.x[0] * self.x[1] + 3 * self.x[2]**2 + 5)
        self.polynomial_tester(p)

        p = sym.Polynomial(self.x[0] * self.x[2] * 4 + 5 * self.x[1] ** 2 * self.x[0])
        self.polynomial_tester(p)


if __name__ == "__main__":
    unittest.main()
