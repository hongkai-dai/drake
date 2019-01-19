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

void ComparePolytopes(const ConvexPolytope& p1, const ConvexPolytope& p2) {
  EXPECT_EQ(p1.body_index(), p2.body_index());
  EXPECT_TRUE(CompareMatrices(p1.p_BV(), p2.p_BV()));
}

void CheckGenerateLinkOutsideHalfspacePolynomials(
    const ConfigurationSpaceCollisionFreeRegionTester& tester,
    const Eigen::VectorXd& q_star, const Eigen::VectorXd& q_val,
    const Eigen::VectorXd& t_val,
    const std::vector<std::vector<std::vector<Eigen::Vector3d>>>&
        a_hyperplane_val) {
  std::vector<double> link_outside_halfspace_rational_expected;

  auto context =
      tester.dut().rational_forward_kinematics().plant().CreateDefaultContext();
  tester.dut().rational_forward_kinematics().plant().SetPositions(context.get(),
                                                                  q_val);
  std::vector<Eigen::Isometry3d> X_WB_expected;
  const auto& tree = internal::GetInternalTree(
      tester.dut().rational_forward_kinematics().plant());
  tree.CalcAllBodyPosesInWorld(*context, &X_WB_expected);

  for (int i = 1;
       i < tester.dut().rational_forward_kinematics().plant().num_bodies();
       ++i) {
    for (int j = 0;
         j < static_cast<int>(tester.dut().link_polytopes()[i].size()); ++j) {
      const Eigen::Matrix3Xd p_WV_ij =
          X_WB_expected[i].linear() *
              tester.dut().link_polytopes()[i][j].p_BV() +
          X_WB_expected[i].translation() *
              Eigen::RowVectorXd::Ones(
                  1, tester.dut().link_polytopes()[i][j].p_BV().cols());
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
      tester.dut().rational_forward_kinematics().plant().num_bodies();
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

TEST_F(IiwaTest, GenerateLinkOutsideHalfspaceRationalFunction) {
  std::vector<ConvexPolytope> link_polytopes;
  link_polytopes.emplace_back(
      iiwa_link_[4], (Eigen::Matrix<double, 3, 4>() << 0.1, 0.1, 0, -0.1, 0.2,
                      -0.2, 0, 0.3, 0, -0.3, 0.2, 0.1)
                         .finished());
  link_polytopes.emplace_back(
      iiwa_link_[4], (Eigen::Matrix<double, 3, 4>() << -0.1, 0.3, -0.2, -0.3,
                      -0.2, 0.3, 0.2, 0.2, 0.3, 0.3, 0.1, 0.4)
                         .finished());
  link_polytopes.emplace_back(
      iiwa_link_[7], (Eigen::Matrix<double, 3, 4>() << -0.1, -0.3, 0.2, 0.3,
                      0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.1, 0.4)
                         .finished());

  std::vector<ConvexPolytope> obstacles;
  Eigen::Isometry3d obstacle_pose;
  obstacle_pose.setIdentity();
  obstacle_pose.translation() << 0.4, 0.5, 0.2;
  obstacles.emplace_back(
      world_,
      GenerateBoxVertices(Eigen::Vector3d(0.3, 0.2, 0.4), obstacle_pose));
  obstacle_pose.translation() << 0.2, -0.3, 0.1;
  obstacles.emplace_back(
      world_,
      GenerateBoxVertices(Eigen::Vector3d(0.5, 0.1, 0.4), obstacle_pose));

  ConfigurationSpaceCollisionFreeRegion dut(*iiwa_, link_polytopes, obstacles);

  EXPECT_EQ(dut.link_polytopes().size(), iiwa_->num_bodies());
  ComparePolytopes(dut.link_polytopes()[5][0], link_polytopes[0]);
  ComparePolytopes(dut.link_polytopes()[5][1], link_polytopes[1]);
  ComparePolytopes(dut.link_polytopes()[8][0], link_polytopes[2]);

  ConfigurationSpaceCollisionFreeRegionTester tester(dut);
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::VectorXd q_val(7);
  q_val.setZero();
  Eigen::VectorXd t_val(7);
  t_val.setZero();
  std::vector<std::vector<std::vector<Eigen::Vector3d>>> a_hyperplane_val(
      iiwa_->num_bodies());
  a_hyperplane_val[5].resize(2);
  a_hyperplane_val[5][0].resize(2);
  a_hyperplane_val[5][0][0] << 0.1, 0.2, 0.4;
  a_hyperplane_val[5][0][1] << 0.3, -0.2, 1.3;
  a_hyperplane_val[5][1].resize(2);
  a_hyperplane_val[5][1][0] << 0.2, 1.3, -0.3;
  a_hyperplane_val[5][1][1] << -0.4, 1.2, 0.1;
  a_hyperplane_val[8].resize(1);
  a_hyperplane_val[8][0].resize(2);
  a_hyperplane_val[8][0][0] << 0.5, 0.3, -1.2;
  a_hyperplane_val[8][0][1] << 0.1, -0.4, 1.5;

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

void CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope& link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    const Eigen::Ref<const Eigen::VectorXd>& q_val,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<double>>& a_A_val,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side) {
  symbolic::Environment env;
  Vector3<symbolic::Variable> a_A;
  for (int i = 0; i < 3; ++i) {
    a_A(i) = symbolic::Variable("a_A(" + std::to_string(i) + ")");
    env.insert(a_A(i), a_A_val(i));
  }
  const std::vector<symbolic::RationalFunction> rational_fun =
      GenerateLinkOnOneSideOfPlaneRationalFunction(
          rational_forward_kinematics, link_polytope, q_star,
          expressed_body_index, a_A, p_AC, plane_side);
  const Eigen::VectorXd t_val =
      rational_forward_kinematics.ComputeTValue(q_val, q_star);
  for (int i = 0; i < t_val.size(); ++i) {
    env.insert(rational_forward_kinematics.t()(i), t_val(i));
  }

  // Compute link points position in the expressed body.
  auto context = rational_forward_kinematics.plant().CreateDefaultContext();
  rational_forward_kinematics.plant().SetPositions(context.get(), q_val);
  Eigen::Matrix3Xd p_AV_expected(3, link_polytope.p_BV().cols());
  rational_forward_kinematics.plant().CalcPointsPositions(
      *context, rational_forward_kinematics.plant()
                    .get_body(link_polytope.body_index())
                    .body_frame(),
      link_polytope.p_BV(), rational_forward_kinematics.plant()
                                .get_body(expressed_body_index)
                                .body_frame(),
      &p_AV_expected);

  const VectorX<symbolic::Variable> t_on_path =
      rational_forward_kinematics.FindTOnPath(expressed_body_index,
                                              link_polytope.body_index());
  const symbolic::Variables t_on_path_set(t_on_path);

  EXPECT_EQ(rational_fun.size(), link_polytope.p_BV().cols());
  const double tol{1E-12};
  for (int i = 0; i < link_polytope.p_BV().cols(); ++i) {
    EXPECT_TRUE(
        rational_fun[i].numerator().indeterminates().IsSubsetOf(t_on_path_set));
    // Check that rational_fun[i] only contains the right t.
    const double rational_fun_val = rational_fun[i].numerator().Evaluate(env) /
                                    rational_fun[i].denominator().Evaluate(env);
    const double rational_fun_val_expected =
        plane_side == PlaneSide::kPositive
            ? a_A_val.dot(p_AV_expected.col(i) - p_AC) - 1
            : 1 - a_A_val.dot(p_AV_expected.col(i) - p_AC);
    EXPECT_NEAR(rational_fun_val, rational_fun_val_expected, tol);
  }
}

TEST_F(IiwaTest, GenerateLinkOnOneSideOfPlaneRationalFunction1) {
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

  // Arbitrary pose between link polytope and the attached link.
  Eigen::Isometry3d X_6V;
  X_6V.linear() =
      Eigen::AngleAxisd(0.2 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())
          .toRotationMatrix();
  X_6V.translation() << 0.2, -0.1, 0.3;
  const auto p_6V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_6V);
  ConvexPolytope link6_polytope(iiwa_link_[6], p_6V);

  Eigen::VectorXd q(7);
  q.setZero();
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::Vector3d a_A(1.2, -0.4, 3.1);
  const Eigen::Vector3d p_AC(0.5, -2.1, 0.6);
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, q_star, q, world_, a_A, p_AC,
      PlaneSide::kPositive);

  q << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7;
  q_star << -0.25, 0.13, 0.26, 0.65, -0.02, 0.87, 0.42;
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link6_polytope, q_star, q, iiwa_link_[3],
      a_A, p_AC, PlaneSide::kNegative);
}

TEST_F(IiwaTest, GenerateLinkOnOneSideOfPlaneRationalFunction2) {
  const RationalForwardKinematics rational_forward_kinematics(*iiwa_);

  // Arbitrary pose between link polytope and the attached link.
  Eigen::Isometry3d X_3V;
  X_3V.linear() =
      Eigen::AngleAxisd(0.3 * M_PI, Eigen::Vector3d(0.1, 0.4, 0.3).normalized())
          .toRotationMatrix();
  X_3V.translation() << -0.2, -0.1, 0.3;
  const auto p_3V = GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.3), X_3V);
  ConvexPolytope link3_polytope(iiwa_link_[3], p_3V);

  Eigen::VectorXd q(7);
  q.setZero();
  Eigen::VectorXd q_star(7);
  q_star.setZero();
  Eigen::Vector3d a_A(1.2, -0.4, 3.1);
  const Eigen::Vector3d p_AC(0.5, -2.1, 0.6);
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, q_star, q, iiwa_link_[7],
      a_A, p_AC, PlaneSide::kPositive);

  q << 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7;
  q_star << -0.25, 0.13, 0.26, 0.65, -0.02, 0.87, 0.42;
  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, q_star, q, iiwa_link_[5],
      a_A, p_AC, PlaneSide::kNegative);

  CheckGenerateLinkOnOneSideOfPlaneRationalFunction(
      rational_forward_kinematics, link3_polytope, q_star, q, world_, a_A, p_AC,
      PlaneSide::kNegative);
}
}  // namespace multibody
}  // namespace drake
