import numpy as np

import pydrake.symbolic as sym

def serialize_polynomial(p: sym.Polynomial)->list:
    terms = []
    var_to_index = {} 
    indeterminate_count = 0
    for x in p.indeterminates():
        var_to_index[x.get_id()] = indeterminate_count
        indeterminate_count += 1

    for monomial, coeff in p.monomial_to_coefficient_map().items():
        monomial_map = [(
            var_to_index[var.get_id()], degree) for var, degree in
            monomial.get_powers().items()]
        terms.append((monomial_map, coeff.Evaluate({})))
    return terms


def deserialize_polynomial(
        indeterminates: sym.Variables, terms: list)->sym.Polynomial:
    index_to_var = {}
    indeterminate_count = 0
    for x in indeterminates:
        index_to_var[indeterminate_count] = x
        indeterminate_count += 1
    monomial_to_coeff_map = {}
    for (monomial_map, coeff) in terms:
        monomial = sym.Monomial(
            {index_to_var[index]: degree for index, degree in monomial_map})
        monomial_to_coeff_map[monomial] = sym.Expression(coeff)
    return sym.Polynomial(monomial_to_coeff_map)
