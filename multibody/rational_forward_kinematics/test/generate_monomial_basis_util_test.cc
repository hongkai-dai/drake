#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"

#include <unordered_set>

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
void CheckGenerateMonomialBasisWithOrderUpToOne(const symbolic::Variables& t) {
  const auto basis = GenerateMonomialBasisWithOrderUpToOne(t);
  const int basis_size_expected = 1 << static_cast<int>(t.size());
  EXPECT_EQ(basis.rows(), basis_size_expected);
  std::unordered_set<symbolic::Monomial> basis_set;
  basis_set.reserve(basis_size_expected);
  for (int i = 0; i < basis.rows(); ++i) {
    for (const symbolic::Variable& ti : t) {
      EXPECT_LE(basis(i).degree(ti), 1);
    }
    basis_set.insert(basis(i));
  }
  EXPECT_EQ(basis_set.size(), basis_size_expected);
}

GTEST_TEST(RationalForwardKinematicsTest,
           GenerateMonomialBasisWithOrderUpToOne) {
  symbolic::Variable t1("t1");
  symbolic::Variable t2("t2");
  symbolic::Variable t3("t3");
  symbolic::Variable t4("t4");

  // CheckGenerateMonomialBasisWithOrderUpToOne(symbolic::Variables({t1}));
  CheckGenerateMonomialBasisWithOrderUpToOne(symbolic::Variables({t1, t2}));
  CheckGenerateMonomialBasisWithOrderUpToOne(symbolic::Variables({t1, t2, t3}));
  CheckGenerateMonomialBasisWithOrderUpToOne(
      symbolic::Variables({t1, t2, t3, t4}));
}
}  // namespace multibody
}  // namespace drake
