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

  const ConfigurationSpaceCollisionFreeRegion& dut() const { return *dut_; }

 private:
  const ConfigurationSpaceCollisionFreeRegion* dut_;
};

void ComparePolytopes(
    const ConfigurationSpaceCollisionFreeRegion::Polytope& p1,
    const ConfigurationSpaceCollisionFreeRegion::Polytope& p2) {
  EXPECT_EQ(p1.body_index, p2.body_index);
  EXPECT_TRUE(CompareMatrices(p1.vertices, p2.vertices));
}

void CheckGenerateLinkOutsideHalfspacePolynomials(
    const ConfigurationSpaceCollisionFreeRegionTester& tester,
    const Eigen::VectorXd& q_star, const Eigen::VectorXd& q_val,
    const Eigen::VectorXd& t_val,
    const std::vector<std::vector<std::vector<Eigen::Vector3d>>>&
        a_hyperplane_val) {
  std::vector<double> link_outside_halfspace_rational_expected;

  auto context =
      tester.dut().rational_forward_kinematics().tree().CreateDefaultContext();
  auto mbt_context = dynamic_cast<MultibodyTreeContext<double>*>(context.get());
  mbt_context->get_mutable_positions() = q_val;
  std::vector<Eigen::Isometry3d> X_WB_expected;
  tester.dut().rational_forward_kinematics().tree().CalcAllBodyPosesInWorld(
      *mbt_context, &X_WB_expected);

  for (int i = 1;
       i < tester.dut().rational_forward_kinematics().tree().num_bodies();
       ++i) {
    for (int j = 0;
         j < static_cast<int>(tester.dut().link_polytopes()[i].size()); ++j) {
      const Eigen::Matrix3Xd p_WV_ij =
          X_WB_expected[i].linear() *
              tester.dut().link_polytopes()[i][j].vertices +
          X_WB_expected[i].translation() *
              Eigen::RowVectorXd::Ones(
                  1, tester.dut().link_polytopes()[i][j].vertices.cols());
      for (int k = 0; k < static_cast<int>(tester.dut().obstacles().size());
           ++k) {
        for (int l = 0; l < p_WV_ij.cols(); ++l) {
          link_outside_halfspace_rational_expected.push_back(
              a_hyperplane_val[i][j][k].dot(p_WV_ij.col(l) -
                                            tester.dut().obstacle_center()[k]) -
              1);
        }
      }
    }
  }

  const std::vector<symbolic::RationalFunction>
      link_outside_halfspace_rationals =
          tester.GenerateLinkOutsideHalfspaceRationalFunction(q_star);
  symbolic::Environment env;
  for (int i = 0; i < tester.dut().rational_forward_kinematics().t().rows();
       ++i) {
    env[tester.dut().rational_forward_kinematics().t()(i)] = t_val(i);
  }
  const int num_bodies =
      tester.dut().rational_forward_kinematics().tree().num_bodies();
  for (int i = 1; i < num_bodies; ++i) {
    for (int j = 0; j < static_cast<int>(tester.dut().a_hyperplane()[i].size());
         ++j) {
      for (int k = 0; k < static_cast<int>(tester.dut().obstacles().size());
           ++k) {
        for (int l = 0; l < 3; ++l) {
          env[tester.dut().a_hyperplane()[i][j][k](l)] =
              a_hyperplane_val[i][j][k](l);
        }
      }
    }
  }
  EXPECT_EQ(link_outside_halfspace_rationals.size(),
            link_outside_halfspace_rational_expected.size());
  const double tol{1E-12};
  for (int i = 0; i < static_cast<int>(link_outside_halfspace_rationals.size());
       ++i) {
    EXPECT_NEAR(
        link_outside_halfspace_rationals[i].numerator().Evaluate(env) /
            link_outside_halfspace_rationals[i].denominator().Evaluate(env),
        link_outside_halfspace_rational_expected[i], tol);
  }
}

GTEST_TEST(ConfigurationSpaceCollisionFreeRegionTest,
           GenerateLinkOutsideHalfspaceRationalFunction) {
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
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::VectorXd q_val(7);
  q_val.setZero();
  Eigen::VectorXd t_val(7);
  t_val.setZero();
  std::vector<std::vector<std::vector<Eigen::Vector3d>>> a_hyperplane_val(
      iiwa->num_bodies());
  a_hyperplane_val[5].resize(2);
  a_hyperplane_val[5][0].resize(2);
  a_hyperplane_val[5][0][0] << 0.1, 0.2, 0.4;
  a_hyperplane_val[5][0][1] << 0.3, -0.2, 1.3;
  a_hyperplane_val[5][1].resize(2);
  a_hyperplane_val[5][1][0] << 0.2, 1.3, -0.3;
  a_hyperplane_val[5][1][1] << -0.4, 1.2, 0.1;
  a_hyperplane_val[7].resize(1);
  a_hyperplane_val[7][0].resize(2);
  a_hyperplane_val[7][0][0] << 0.5, 0.3, -1.2;
  a_hyperplane_val[7][0][1] << 0.1, -0.4, 1.5;

  CheckGenerateLinkOutsideHalfspacePolynomials(tester, q_star, q_val, t_val,
                                               a_hyperplane_val);

  q_val << 0.1, 0.3, 0.2, -0.5, 0.2, 0.4, -0.2;
  t_val = ((q_val - q_star) / 2).array().tan().matrix();
  CheckGenerateLinkOutsideHalfspacePolynomials(tester, q_star, q_val, t_val,
                                               a_hyperplane_val);

  q_star << -0.3, 0.2, 1.5, 3.2, 0.3, 0.4, 1.3;
  t_val = ((q_val - q_star) / 2).array().tan().matrix();
  CheckGenerateLinkOutsideHalfspacePolynomials(tester, q_star, q_val, t_val,
                                               a_hyperplane_val);

  const std::vector<symbolic::Polynomial> link_outside_halfspace_polynomials =
      dut.GenerateLinkOutsideHalfspacePolynomials(q_star);
  for (const auto& link_outside_halfspace_polynomial :
       link_outside_halfspace_polynomials) {
    for (int i = 0; i < dut.rational_forward_kinematics().t().rows(); ++i) {
      EXPECT_LE(link_outside_halfspace_polynomial.Degree(
                    dut.rational_forward_kinematics().t()(i)),
                2);
    }
  }
}
}  // namespace multibody
}  // namespace drake
