#pragma once

#include "drake/common/symbolic.h"

namespace drake {
namespace multibody {
/**
 * Generate all the monomials of t_angles, such that the order for t_angles(i)
 * is no larger than 1.
 * The link pose is a polynomial of t_angles. The monomials in this polynomial
 * has the form ∏ tᵢᵐⁱ, where tᵢ is a term in t_angles, and the order mᵢ <= 2.
 * Hence if we compute the monomial basis z for this polynomial, such that the
 * polynomial can be written as zᵀHz, then z should contain all the monomials
 * of form ∏tᵢⁿⁱ, where nᵢ <= 1.
 */
VectorX<symbolic::Monomial> GenerateMonomialBasisWithOrderUpToOne(
    const symbolic::Variables& t_angles);

/**
 * Generate all the monomials of t_angles, such that ∃j, t_angles(j)'s order
 * is no larger than 2, and for all i ≠ j, t_angles(i)'s order is no larger
 * than 1.
 */
VectorX<symbolic::Monomial>
GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
    const symbolic::Variables& t_angles);
}  // namespace multibody
}  // namespace drake
