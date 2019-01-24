#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
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

class IiwaConfigurationSpaceTest : public IiwaTest {
 public:
  IiwaConfigurationSpaceTest() {
    // Arbitrarily add some polytopes to links
    link7_polytopes_.emplace_back(
        iiwa_link_[7], GenerateBoxVertices(Eigen::Vector3d(0.1, 0.1, 0.2),
                                           Eigen::Isometry3d::Identity()));
    Eigen::Isometry3d X_7P;
    X_7P.linear() = Eigen::AngleAxisd(0.2 * M_PI, Eigen::Vector3d::UnitX())
                        .toRotationMatrix();
    X_7P.translation() << 0.1, 0.2, -0.1;
    link7_polytopes_.emplace_back(
        iiwa_link_[7],
        GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.1), X_7P));

    Eigen::Isometry3d X_5P = X_7P;
    X_5P.translation() << -0.2, 0.1, 0;
    link5_polytopes_.emplace_back(
        iiwa_link_[5],
        GenerateBoxVertices(Eigen::Vector3d(0.2, 0.1, 0.2), X_5P));

    Eigen::Isometry3d X_WP = X_5P * Eigen::Translation3d(0.15, -0.1, 0.05);
    obstacles_.emplace_back(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.2, 0.15), X_WP));
    X_WP = X_WP * Eigen::AngleAxisd(-0.1 * M_PI, Eigen::Vector3d::UnitY());
    obstacles_.emplace_back(
        world_, GenerateBoxVertices(Eigen::Vector3d(0.1, 0.25, 0.15), X_WP));
  }

 protected:
  std::vector<ConvexPolytope> link7_polytopes_;
  std::vector<ConvexPolytope> link5_polytopes_;
  std::vector<ConvexPolytope> obstacles_;
};

TEST_F(IiwaConfigurationSpaceTest, TestConstructor) {
  ConfigurationSpaceCollisionFreeRegion dut(
      *iiwa_, {link7_polytopes_[0], link7_polytopes_[1], link5_polytopes_[0]},
      obstacles_,
      {std::make_pair(link7_polytopes_[0].get_id(), obstacles_[0].get_id())});

  const auto& separation_planes = dut.separation_planes();
  EXPECT_EQ(separation_planes.size(), 5);

  auto CheckSeparationPlane = [&](
      const SeparationPlane<symbolic::Variable>& separation_plane,
      ConvexGeometry::Id expected_positive_polytope,
      ConvexGeometry::Id expected_negative_polytope,
      BodyIndex expressed_body_index) {
    EXPECT_EQ(separation_plane.positive_side_polytope->get_id(),
              expected_positive_polytope);
    EXPECT_EQ(separation_plane.negative_side_polytope->get_id(),
              expected_negative_polytope);
    EXPECT_EQ(separation_plane.expressed_link, expressed_body_index);
  };
  CheckSeparationPlane(separation_planes[0], link5_polytopes_[0].get_id(),
                       obstacles_[0].get_id(), iiwa_link_[3]);
  CheckSeparationPlane(separation_planes[1], link7_polytopes_[1].get_id(),
                       obstacles_[0].get_id(), iiwa_link_[4]);
  CheckSeparationPlane(separation_planes[2], link5_polytopes_[0].get_id(),
                       obstacles_[1].get_id(), iiwa_link_[3]);
  CheckSeparationPlane(separation_planes[3], link7_polytopes_[0].get_id(),
                       obstacles_[1].get_id(), iiwa_link_[4]);
  CheckSeparationPlane(separation_planes[4], link7_polytopes_[1].get_id(),
                       obstacles_[1].get_id(), iiwa_link_[4]);
}
}  // namespace multibody
}  // namespace drake
