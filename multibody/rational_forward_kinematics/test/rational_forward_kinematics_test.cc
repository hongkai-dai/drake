#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/symbolic_test_util.h"
#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
namespace {
using symbolic::Polynomial;
using symbolic::RationalFunction;
using symbolic::test::PolyEqualAfterExpansion;

void CheckReplaceCosAndSinWithRationalFunction(
    const symbolic::Expression& e, const VectorX<symbolic::Variable>& cos_delta,
    const VectorX<symbolic::Variable>& sin_delta,
    const VectorX<symbolic::Variable>& t_angle, const symbolic::Variables& t,
    const symbolic::RationalFunction& e_rational_expected) {
  VectorX<symbolic::Variable> cos_sin_delta(cos_delta.rows() +
                                            sin_delta.rows());
  cos_sin_delta << cos_delta, sin_delta;
  const symbolic::Variables cos_sin_delta_variables(cos_sin_delta);
  const Polynomial e_poly(e, cos_sin_delta_variables);
  symbolic::RationalFunction e_rational;
  ReplaceCosAndSinWithRationalFunction(e_poly, cos_delta, sin_delta, t_angle, t,
                                       &e_rational);
  EXPECT_PRED2(PolyEqualAfterExpansion, e_rational.numerator(),
               e_rational_expected.numerator());
  EXPECT_PRED2(PolyEqualAfterExpansion, e_rational.denominator(),
               e_rational_expected.denominator());
}

GTEST_TEST(RationalForwardKinematics, ReplaceCosAndSinWithRationalFunction) {
  VectorX<symbolic::Variable> cos_delta(3);
  VectorX<symbolic::Variable> sin_delta(3);
  VectorX<symbolic::Variable> t_angle(3);
  for (int i = 0; i < 3; ++i) {
    cos_delta(i) =
        symbolic::Variable("cos(delta_q(" + std::to_string(i) + "))");
    sin_delta(i) =
        symbolic::Variable("sin(delta_q(" + std::to_string(i) + "))");
    t_angle(i) = symbolic::Variable("t_angle(" + std::to_string(i) + ")");
  }

  symbolic::Variable a("a");
  symbolic::Variable b("b");

  symbolic::Variables t(t_angle);

  // test cos(delta_q(0))
  CheckReplaceCosAndSinWithRationalFunction(
      cos_delta(0), cos_delta, sin_delta, t_angle, t,
      symbolic::RationalFunction(Polynomial(1 - t_angle(0) * t_angle(0)),
                                 Polynomial(1 + t_angle(0) * t_angle(0))));
  // test sin(delta_q(0))
  CheckReplaceCosAndSinWithRationalFunction(
      sin_delta(0), cos_delta, sin_delta, t_angle, t,
      symbolic::RationalFunction(Polynomial(2 * t_angle(0)),
                                 Polynomial(1 + t_angle(0) * t_angle(0))));
  // test 1.
  CheckReplaceCosAndSinWithRationalFunction(1, cos_delta, sin_delta, t_angle, t,
                                            symbolic::RationalFunction(1));

  // test a + b
  CheckReplaceCosAndSinWithRationalFunction(
      a + b, cos_delta, sin_delta, t_angle, t,
      RationalFunction(Polynomial(a + b, t)));

  // test 1 + cos(delta_q(0))
  CheckReplaceCosAndSinWithRationalFunction(
      1 + cos_delta(0), cos_delta, sin_delta, t_angle, t,
      symbolic::RationalFunction(Polynomial(2),
                                 Polynomial(1 + t_angle(0) * t_angle(0))));

  // test a + b*cos(delta_q(0)) + sin(delta_q(1))
  CheckReplaceCosAndSinWithRationalFunction(
      a + b * cos_delta(0) + sin_delta(1), cos_delta, sin_delta, t_angle, t,
      RationalFunction(
          Polynomial(a * (1 + t_angle(0) * t_angle(0)) *
                             (1 + t_angle(1) * t_angle(1)) +
                         b * (1 - t_angle(0) * t_angle(0)) *
                             (1 + t_angle(1) * t_angle(1)) +
                         2 * t_angle(1) * (1 + t_angle(0) * t_angle(0)),
                     t),
          Polynomial((1 + t_angle(0) * t_angle(0)) *
                     (1 + t_angle(1) * t_angle(1)))));

  // test a + b * cos(delta_q(0) * sin(delta_q(1)) + sin(delta_q(0))
  CheckReplaceCosAndSinWithRationalFunction(
      a + b * cos_delta(0) * sin_delta(1) + sin_delta(0), cos_delta, sin_delta,
      t_angle, t,
      RationalFunction(
          Polynomial(a * (1 + t_angle(0) * t_angle(0)) *
                             (1 + t_angle(1) * t_angle(1)) +
                         b * (1 - t_angle(0) * t_angle(0)) * 2 * t_angle(1) +
                         2 * t_angle(0) * (1 + t_angle(1) * t_angle(1)),
                     t),
          Polynomial((1 + t_angle(0) * t_angle(0)) *
                     (1 + t_angle(1) * t_angle(1)))));

  // test a + b * cos(delta_q(0)) * sin(delta_q(1)) + sin(delta_q(0)) *
  // cos(delta_q(2))
  CheckReplaceCosAndSinWithRationalFunction(
      a + b * cos_delta(0) * sin_delta(1) + sin_delta(0) * cos_delta(2),
      cos_delta, sin_delta, t_angle, t,
      RationalFunction(
          Polynomial(a * (1 + t_angle(0) * t_angle(0)) *
                             (1 + t_angle(1) * t_angle(1)) *
                             (1 + t_angle(2) * t_angle(2)) +
                         b * (1 - t_angle(0) * t_angle(0)) * 2 * t_angle(1) *
                             (1 + t_angle(2) * t_angle(2)) +
                         2 * t_angle(0) * (1 + t_angle(1) * t_angle(1)) *
                             (1 - t_angle(2) * t_angle(2)),
                     t),
          Polynomial((1 + t_angle(0) * t_angle(0)) *
                     (1 + t_angle(1) * t_angle(1)) *
                     (1 + t_angle(2) * t_angle(2)))));
}

void CheckLinkKinematics(
    const RationalForwardKinematics& rational_forward_kinematics,
    const Eigen::Ref<const Eigen::VectorXd>& q_val,
    const Eigen::Ref<const Eigen::VectorXd>& q_star_val,
    const Eigen::Ref<const Eigen::VectorXd>& t_val,
    BodyIndex expressed_body_index) {
  DRAKE_DEMAND(t_val.rows() == rational_forward_kinematics.t().rows());
  auto context = rational_forward_kinematics.plant().CreateDefaultContext();

  rational_forward_kinematics.plant().SetPositions(context.get(), q_val);

  std::vector<Eigen::Isometry3d> X_WB_expected;

  const auto& tree =
      internal::GetInternalTree(rational_forward_kinematics.plant());
  tree.CalcAllBodyPosesInWorld(*context, &X_WB_expected);

  symbolic::Environment env;
  for (int i = 0; i < t_val.rows(); ++i) {
    env.insert(rational_forward_kinematics.t()(i), t_val(i));
  }

  const auto& poses = rational_forward_kinematics.CalcLinkPoses(
      q_star_val, expressed_body_index);

  const double tol{1E-12};
  for (int i = 0; i < rational_forward_kinematics.plant().num_bodies(); ++i) {
    EXPECT_EQ(poses[i].frame_A_index, expressed_body_index);
    Matrix3<double> R_AB_i;
    Vector3<double> p_AB_i;
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        R_AB_i(m, n) = poses[i].R_AB(m, n).numerator().Evaluate(env) /
                       poses[i].R_AB(m, n).denominator().Evaluate(env);
      }
      p_AB_i(m) = poses[i].p_AB(m).numerator().Evaluate(env) /
                  poses[i].p_AB(m).denominator().Evaluate(env);
    }

    const Eigen::Isometry3d X_AB_expected =
        X_WB_expected[expressed_body_index].inverse() * X_WB_expected[i];
    EXPECT_TRUE(CompareMatrices(R_AB_i, X_AB_expected.linear(), tol));
    EXPECT_TRUE(CompareMatrices(p_AB_i, X_AB_expected.translation(), tol));

    // Now check CalcLinkPoseAsMultilinearPolynomial, namely to compute a single
    // link pose.
    const RationalForwardKinematics::Pose<symbolic::Polynomial> pose_poly_i =
        rational_forward_kinematics.CalcLinkPoseAsMultilinearPolynomial(
            q_star_val, BodyIndex(i), expressed_body_index);
    EXPECT_EQ(pose_poly_i.frame_A_index, expressed_body_index);
    for (int m = 0; m < 3; ++m) {
      for (int n = 0; n < 3; ++n) {
        const auto R_AB_i_mn_rational =
            rational_forward_kinematics
                .ConvertMultilinearPolynomialToRationalFunction(
                    pose_poly_i.R_AB(m, n));
        R_AB_i(m, n) = R_AB_i_mn_rational.numerator().Evaluate(env) /
                       R_AB_i_mn_rational.denominator().Evaluate(env);
      }
      const auto p_AB_i_m_rational =
          rational_forward_kinematics
              .ConvertMultilinearPolynomialToRationalFunction(
                  pose_poly_i.p_AB(m));
      p_AB_i(m) = p_AB_i_m_rational.numerator().Evaluate(env) /
                  p_AB_i_m_rational.denominator().Evaluate(env);
    }
    EXPECT_TRUE(CompareMatrices(R_AB_i, X_AB_expected.linear(), tol));
    EXPECT_TRUE(CompareMatrices(p_AB_i, X_AB_expected.translation(), tol));
  }
}

GTEST_TEST(RationalForwardKinematicsTest, CalcLinkPoses) {
  auto iiwa_plant = ConstructIiwaPlant("iiwa14_no_collision.sdf");
  RationalForwardKinematics rational_forward_kinematics(*iiwa_plant);
  EXPECT_EQ(rational_forward_kinematics.t().rows(), 7);

  const BodyIndex world_index = iiwa_plant->world_body().index();
  std::array<BodyIndex, 8> iiwa_link;
  for (int i = 0; i < 8; ++i) {
    iiwa_link[i] =
        iiwa_plant->GetBodyByName("iiwa_link_" + std::to_string(i)).index();
  }

  // Call CalcLinkPosesAsMultilinearPolynomial to make sure the expressed link
  // index is correct in each pose.
  {
    const auto poses =
        rational_forward_kinematics.CalcLinkPosesAsMultilinearPolynomials(
            Eigen::VectorXd::Zero(7), iiwa_link[2]);
    for (const auto& pose : poses) {
      EXPECT_EQ(pose.frame_A_index, iiwa_link[2]);
    }
  }
  // q_val = 0 and q* = 0.
  CheckLinkKinematics(rational_forward_kinematics, Eigen::VectorXd::Zero(7),
                      Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7),
                      world_index);
  // Compute the pose in the iiwa_link[i]'s frame.
  for (int i = 0; i < 8; ++i) {
    CheckLinkKinematics(rational_forward_kinematics, Eigen::VectorXd::Zero(7),
                        Eigen::VectorXd::Zero(7), Eigen::VectorXd::Zero(7),
                        iiwa_link[i]);
  }

  // Non-zero q_val and zero q_star_val.
  Eigen::VectorXd q_val(7);
  // arbitrary value
  q_val << 0.2, 0.3, 0.5, -0.1, 1.2, 2.3, -0.5;
  Eigen::VectorXd t_val = (q_val / 2).array().tan().matrix();
  CheckLinkKinematics(rational_forward_kinematics, q_val,
                      Eigen::VectorXd::Zero(7), t_val, world_index);
  // Compute the pose in the iiwa_link[i]'s frame.
  for (int i = 0; i < 8; ++i) {
    CheckLinkKinematics(rational_forward_kinematics, q_val,
                        Eigen::VectorXd::Zero(7), t_val, iiwa_link[i]);
  }

  // Non-zero q_val and non-zero q_star_val.
  Eigen::VectorXd q_star_val(7);
  q_star_val << 1.2, -0.4, 0.3, -0.5, 0.4, 1, 0.2;
  t_val = ((q_val - q_star_val) / 2).array().tan().matrix();
  CheckLinkKinematics(rational_forward_kinematics, q_val, q_star_val, t_val,
                      world_index);
  // Compute the pose in the iiwa_link[i]'s frame.
  for (int i = 0; i < 8; ++i) {
    CheckLinkKinematics(rational_forward_kinematics, q_val, q_star_val, t_val,
                        iiwa_link[i]);
  }
}

GTEST_TEST(RationalForwardKinematicsTest, CalcLinkPosesForDualArmIiwa) {
  Eigen::Isometry3d X_WL, X_WR;
  X_WL.linear() = Eigen::Matrix3d::Identity();
  X_WL.translation() << 0, 0, 0;
  X_WR.linear() = Eigen::AngleAxisd(0.2 * M_PI, Eigen::Vector3d::UnitZ())
                      .toRotationMatrix();
  X_WR.translation() << 0.2, 0.8, 0.1;
  ModelInstanceIndex left_iiwa_instance, right_iiwa_instance;
  auto iiwa_plant =
      ConstructDualArmIiwaPlant("iiwa14_no_collision.sdf", X_WL, X_WR,
                                &left_iiwa_instance, &right_iiwa_instance);

  RationalForwardKinematics rational_forward_kinematics(*iiwa_plant);
  EXPECT_EQ(rational_forward_kinematics.t().size(), 14);

  Eigen::VectorXd q_star_val(14);
  q_star_val.setZero();
  Eigen::VectorXd q_val(14);
  q_val.setZero();
  Eigen::VectorXd t_val(14);
  t_val.setZero();

  auto set_t_val = [&](
      const Eigen::VectorXd& q_left, const Eigen::VectorXd& q_right,
      const Eigen::VectorXd& q_left_star, const Eigen::VectorXd& q_right_star) {
    auto context = iiwa_plant->CreateDefaultContext();
    iiwa_plant->SetPositions(context.get(), left_iiwa_instance, q_left);
    iiwa_plant->SetPositions(context.get(), right_iiwa_instance, q_right);
    q_val = iiwa_plant->GetPositions(*context);
    iiwa_plant->SetPositions(context.get(), left_iiwa_instance, q_left_star);
    iiwa_plant->SetPositions(context.get(), right_iiwa_instance, q_right_star);
    q_star_val = iiwa_plant->GetPositions(*context);

    t_val = rational_forward_kinematics.ComputeTValue(q_val, q_star_val);
  };

  const BodyIndex world_index = iiwa_plant->world_body().index();
  CheckLinkKinematics(rational_forward_kinematics, q_val, q_star_val, t_val,
                      world_index);

  Eigen::VectorXd q_left(7);
  Eigen::VectorXd q_right(7);
  Eigen::VectorXd q_left_star(7);
  Eigen::VectorXd q_right_star(7);
  q_left_star.setZero();
  q_right_star.setZero();
  q_left << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7;
  q_right << -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7;
  set_t_val(q_left, q_right, q_left_star, q_right_star);
  CheckLinkKinematics(rational_forward_kinematics, q_val, q_star_val, t_val,
                      world_index);

  q_left_star << -0.2, 1.2, 0.3, 0.4, -2.1, 2.2, 2.3;
  q_right_star << 0.1, 0.2, 0.5, 1.1, -0.3, -0.2, 2.1;
  set_t_val(q_left, q_right, q_left_star, q_right_star);
  CheckLinkKinematics(rational_forward_kinematics, q_val, q_star_val, t_val,
                      world_index);
}

}  // namespace
}  // namespace multibody
}  // namespace drake
