#pragma once

#include "drake/common/symbolic.h"

namespace drake {
namespace multibody {
/**
 * The link pose is a polynomial of t_angles. The monomials in this polynomial
 * has the form ∏ tᵢᵐⁱ, where tᵢ is a term in t_angles, and the order mᵢ <= 2.
 * Hence if we compute the monomial basis z for this polynomial, such that the
 * polynomial can be written as zᵀHz, then z should contain all the monomials
 * of form ∏tᵢⁿⁱ, where nᵢ <= 1.
 */
VectorX<symbolic::Monomial> GenerateMonomialBasisWithOrderUpToOne(
    const symbolic::Variables& t_angles);
}  // namespace multibody
}  // namespace drake
