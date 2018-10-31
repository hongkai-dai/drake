#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
class ConfigurationSpaceCollisionFreeRegionTester {
 public:
  ConfigurationSpaceCollisionFreeRegionTester(
      const ConfigurationSpaceCollisionFreeRegion& dut)
      : dut_{&dut} {}

  std::vector<symbolic::RationalFunction>
  GenerateLinkOutsideHalfspaceRationalFunction(
      const Eigen::VectorXd& q_star) const {
    return dut_->GenerateLinkOutsideHalfspaceRationalFunction(q_star);
  }

 private:
  const ConfigurationSpaceCollisionFreeRegion* dut_;
};

GTEST_TEST(ConfigurationSpaceCollisionFreeRegionTest,
           GenerateLinkOutsideHalfspacePolynomials) {
  auto iiwa = ConstructIiwaPlant("iiwa14_no_collision.sdf");

  std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope> link5_polytopes;
  link5_polytopes.emplace_back(5, (Eigen::Matrix<double, 3, 4>() << 0.1, 0.1, 0,
                                   -0.1, 0.2, -0.2, 0, 0.3, 0, -0.3, 0.2, 0.1)
                                      .finished());
  link5_polytopes.emplace_back(
      5, (Eigen::Matrix<double, 3, 4>() << -0.1, 0.3, -0.2, -0.3, -0.2, 0.3,
          0.2, 0.2, 0.3, 0.3, 0.1, 0.4)
             .finished());

  ConfigurationSpaceCollisionFreeRegion(*iiwa
}
}  // namespace multibody
}  // namespace drake
