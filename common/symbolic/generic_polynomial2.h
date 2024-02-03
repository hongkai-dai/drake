#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "drake/common/drake_copyable.h"
#include "drake/common/symbolic/ordered_monomials.h"

namespace drake {
namespace symbolic {

enum class PolynomialBasis {
  kStandard,
  kChebyshevFirst,
  kChebyshevSecond,
};

/**
 The linear transformation from the vector of ordered monomials m(x) to the
 polynomial basis b(x). Namely b(x) = V * m(x), where V is the linear
 transformation matrix. V is a (sparse) square matrix.
 */
class BasisTransform {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(BasisTransform);

  BasisTransform(PolynomialBasis type, int num_vars);

  /**
   Returns the square matrix V that converts the first `num_monomials` monomials
   to the polynomial basis of the same size.
   */
  [[nodiscard]] Eigen::SparseMatrix<double, Eigen::RowMajor> ToMatrix(
      int num_monomials) const;

  [[nodiscard]] PolynomialBasis type() const { return type_; }

  [[nodiscard]] int num_vars() const { return num_vars_; }

 private:
  PolynomialBasis type_;
  int num_vars_;
};

/**
 Represents a generic polynomial supporting different basis.

 A polynomial p(x) can be written as
 p(x) = c * V * m(x)
 where c is a (sparse) vector as the coefficients of the polynomial.
 m(x) is an OrderedMonomials object representing all the monomials.
 V is a (sparse) square matrix that converts the monomials to basis functions.
 */
class GenericPolynomial {
 public:
 private:
};
}  // namespace symbolic
}  // namespace drake
