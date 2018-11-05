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
}  // namespace multibody
}  // namespace drake
