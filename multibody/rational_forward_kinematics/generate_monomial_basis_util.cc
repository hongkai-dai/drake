#include "drake/multibody/rational_forward_kinematics/generate_monomial_basis_util.h"

namespace drake {
namespace multibody {
std::vector<symbolic::Monomial> GenerateMonomialBasisWithOrderUpToOneHelper(
    const Eigen::Ref<const VectorX<symbolic::Variable>>& t_angles) {
  if (t_angles.rows() == 0) {
    throw std::runtime_error(
        "GenerateMonomialBasisWithOrderUpToOneHelper: Shouldn't have an empty "
        "input t_angles.");
  }
  if (t_angles.rows() == 1) {
    const symbolic::Monomial monomial_one{};
    return {monomial_one, symbolic::Monomial(t_angles(0), 1)};
  } else {
    std::vector<symbolic::Monomial> monomials =
        GenerateMonomialBasisWithOrderUpToOneHelper(
            t_angles.head(t_angles.rows() - 1));
    const int num_rows = static_cast<int>(monomials.size());
    monomials.reserve(num_rows * 2);
    const symbolic::Monomial t_angles_i(t_angles(t_angles.rows() - 1), 1);
    for (int i = 0; i < num_rows; ++i) {
      monomials.push_back(monomials[i] * t_angles_i);
    }
    return monomials;
  }
}

VectorX<symbolic::Monomial> GenerateMonomialBasisWithOrderUpToOne(
    const symbolic::Variables& t_angles) {
  VectorX<symbolic::Variable> t_angles_vec(t_angles.size());
  int t_angles_count = 0;
  for (const auto& t_angle : t_angles) {
    t_angles_vec[t_angles_count++] = t_angle;
  }
  const std::vector<symbolic::Monomial> monomials_vec =
      GenerateMonomialBasisWithOrderUpToOneHelper(t_angles_vec);
  const VectorX<symbolic::Monomial> monomials =
      Eigen::Map<const VectorX<symbolic::Monomial>>(monomials_vec.data(),
                                                    monomials_vec.size());
  return monomials;
}

VectorX<symbolic::Monomial>
GenerateMonomialBasisOrderAllUpToOneExceptOneUpToTwo(
    const symbolic::Variables& t_angles) {
  VectorX<symbolic::Variable> t_angles_vec(t_angles.size());
  int t_angles_count = 0;
  for (const auto& t_angle : t_angles) {
    t_angles_vec[t_angles_count++] = t_angle;
  }
  std::vector<symbolic::Monomial> monomials_vec =
      GenerateMonomialBasisWithOrderUpToOneHelper(t_angles_vec);
  if (t_angles_vec.rows() > 1) {
    for (int i = 0; i < t_angles_vec.rows(); ++i) {
      // Now generate the monomial with t_angles_vec(i)^2, and all other
      // variables with order up to 1.
      VectorX<symbolic::Variable> t_angles_vec_without_i(t_angles_vec.rows() -
                                                         1);
      if (i >= 1) {
        t_angles_vec_without_i.head(i) = t_angles_vec.head(i);
      }
      if (i < t_angles_vec.rows() - 1) {
        t_angles_vec_without_i.tail(t_angles_vec.rows() - 1 - i) =
            t_angles_vec.tail(t_angles_vec.rows() - 1 - i);
      }
      std::vector<symbolic::Monomial> monomials_vec_ti_order_two =
          GenerateMonomialBasisWithOrderUpToOneHelper(t_angles_vec_without_i);
      symbolic::Monomial ti_square(t_angles_vec(i), 2);
      for (auto& monomial_vec_ti_order_two : monomials_vec_ti_order_two) {
        monomial_vec_ti_order_two *= ti_square;
      }
      monomials_vec.reserve(monomials_vec.size() +
                            monomials_vec_ti_order_two.size());
      monomials_vec.insert(monomials_vec.end(),
                           monomials_vec_ti_order_two.begin(),
                           monomials_vec_ti_order_two.end());
    }
  } else {
    monomials_vec.push_back(symbolic::Monomial(t_angles_vec(0), 2));
  }
  const VectorX<symbolic::Monomial> monomials =
      Eigen::Map<const VectorX<symbolic::Monomial>>(monomials_vec.data(),
                                                    monomials_vec.size());
  return monomials;
}
}  // namespace multibody
}  // namespace drake
