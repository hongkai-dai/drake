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

void ComparePolytopes(
    const ConfigurationSpaceCollisionFreeRegion::Polytope& p1,
    const ConfigurationSpaceCollisionFreeRegion::Polytope& p2) {
  EXPECT_EQ(p1.body_index, p2.body_index);
  EXPECT_TRUE(CompareMatrices(p1.vertices, p2.vertices));
}

GTEST_TEST(ConfigurationSpaceCollisionFreeRegionTest,
           GenerateLinkOutsideHalfspacePolynomials) {
  auto iiwa = ConstructIiwaPlant("iiwa14_no_collision.sdf");
  std::cout << "iiwa num_bodies: " << iiwa->num_bodies() << "\n";

  std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope> link_polytopes;
  link_polytopes.emplace_back(5, (Eigen::Matrix<double, 3, 4>() << 0.1, 0.1, 0,
                                  -0.1, 0.2, -0.2, 0, 0.3, 0, -0.3, 0.2, 0.1)
                                     .finished());
  link_polytopes.emplace_back(
      5, (Eigen::Matrix<double, 3, 4>() << -0.1, 0.3, -0.2, -0.3, -0.2, 0.3,
          0.2, 0.2, 0.3, 0.3, 0.1, 0.4)
             .finished());
  link_polytopes.emplace_back(
      7, (Eigen::Matrix<double, 3, 4>() << -0.1, -0.3, 0.2, 0.3, 0.2, 0.1, 0.2,
          0.2, 0.3, 0.3, 0.1, 0.4)
             .finished());

  std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope> obstacles;
  Eigen::Isometry3d obstacle_pose;
  obstacle_pose.setIdentity();
  obstacle_pose.translation() << 0.4, 0.5, 0.2;
  obstacles.emplace_back(
      0, GenerateBoxVertices(Eigen::Vector3d(0.3, 0.2, 0.4), obstacle_pose));
  obstacle_pose.translation() << 0.2, -0.3, 0.1;
  obstacles.emplace_back(
      0, GenerateBoxVertices(Eigen::Vector3d(0.5, 0.1, 0.4), obstacle_pose));

  ConfigurationSpaceCollisionFreeRegion dut(iiwa->tree(), link_polytopes,
                                            obstacles);

  EXPECT_EQ(dut.link_polytopes().size(), iiwa->num_bodies());
  ComparePolytopes(dut.link_polytopes()[5][0], link_polytopes[0]);
  ComparePolytopes(dut.link_polytopes()[5][1], link_polytopes[1]);
  ComparePolytopes(dut.link_polytopes()[7][0], link_polytopes[2]);

  ConfigurationSpaceCollisionFreeRegionTester tester(dut);
}
}  // namespace multibody
}  // namespace drake
