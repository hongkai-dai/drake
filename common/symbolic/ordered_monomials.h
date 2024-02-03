#pragma once

#include <vector>

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic/expression.h"
#include "drake/common/symbolic/monomial.h"
namespace drake {
namespace symbolic {
/**
 This class represent an ordered list of monomials, sorted in the graded reverse
 lexicographic order from small monomials to large monomials. For example, with
 x₁ < x₂ < x₃, the ordered monomials are 1 < x₁ < x₂ < x₃ < x₁² < x₁x₂ < x₁x₃ <
 x₂² < x₂x₃ < x₃² < ....
 */
class OrderedMonomials {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(OrderedMonomials);

  /**
   @params vars All the variables for generating the list of monomials.
   */
  OrderedMonomials(Variables vars);

  /**
   Returns the monomial at a given index in the list of ordered monomials. For
   example, OrderedMonomials(Variables({x1, x2, x3})).At(0) returns monomial 1,
   OrderedMonomials(Variables({x1, x2, x3})).At(5) returns x1*x2.
   */
  Monomial At(int index) const;

  /**
   Returns the first N element of the monomials in the list. For example,
   OrderedMonomials(Variables({x1, x2, x3})).GetFirstN(1) returns a
   single-element vector {1}; OrderedMonomials(Variables({x1, x2,
   x3})).GetFirstN(5) returns a 5-element vector {1, x1, x2, x3, x1*x1}.
   */
  std::vector<Monomial> GetFirstN(int N) const;

 private:
  symbolic::Variables vars_;
};
}  // namespace symbolic
}  // namespace drake
