#include "drake/examples/irb140/IRB140_analytical_kinematics.h"

#include <gtest/gtest.h>

#include "drake/examples/irb140/test/irb140_common.h"
#include "drake/common/test_utilities/eigen_matrix_compare.h"

using Eigen::Isometry3d;

namespace drake {
namespace examples {
namespace IRB140 {
namespace {

class IRB140Test : public ::testing::Test {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IRB140Test)

  IRB140Test() : analytical_kinematics() {}

  ~IRB140Test() override{}

 protected:
  IRB140AnalyticalKinematics analytical_kinematics;
};

void printPose(const Eigen::Matrix<symbolic::Expression, 4, 4>& pose) {
  std::cout <<"R\n";
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      std::cout << pose(i, j) << std::endl;
    }
  }
  std::cout <<"p\n";
  for (int i = 0; i < 3; ++i) {
    std::cout << pose(i, 3) << std::endl;
  }
}

void TestForwardKinematics(const IRB140AnalyticalKinematics& analytical_kinematics, const Eigen::Matrix<double, 6, 1>& q) {
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();
  cache.initialize(q);
  analytical_kinematics.robot()->doKinematics(cache);

  std::array<Isometry3d, 7> X_WB;  // The pose of body frame `B` in the world frame `W`.
  X_WB[0].linear() = Eigen::Matrix3d::Identity();
  X_WB[0].translation() = Eigen::Vector3d::Zero();
  for (int i = 1; i < 7; ++i) {
    X_WB[i] = analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache, *(analytical_kinematics.robot()->FindBody("link_" + std::to_string(i))));
  }

  // X_PC[i] is the pose of child body frame `C` (body[i+1]) in the parent body
  // frame `P` (body[i])
  std::array<Isometry3d, 6> X_PC;
  X_PC[0] = X_WB[1];
  for (int i = 1; i < 6; ++i) {
    X_PC[i].linear() = X_WB[i].linear().transpose() * X_WB[i + 1].linear();
    X_PC[i].translation() = X_WB[i].linear().transpose() * (X_WB[i+1].translation() - X_WB[i].translation());
  }

  const auto X_01 = analytical_kinematics.X_01(q(0));
  CompareIsometry3d(X_PC[0], X_01, 1E-5);

  const auto X_12 = analytical_kinematics.X_12(q(1));
  CompareIsometry3d(X_PC[1], X_12, 1e-5);

  const auto X_23 = analytical_kinematics.X_23(q(2));
  CompareIsometry3d(X_PC[2], X_23, 1e-5);

  const auto X_13 = analytical_kinematics.X_13(q(1), q(2));
  CompareIsometry3d(X_PC[1] * X_PC[2], X_13, 1e-5);

  const auto X_34 = analytical_kinematics.X_34(q(3));
  CompareIsometry3d(X_PC[3], X_34, 1E-5);

  const auto X_45 = analytical_kinematics.X_45(q(4));
  CompareIsometry3d(X_PC[4], X_45, 1E-5);

  const auto X_56 = analytical_kinematics.X_56(q(5));
  CompareIsometry3d(X_PC[5], X_56, 1E-5);

  const auto X_06 = analytical_kinematics.X_06(q);
  CompareIsometry3d(X_06, X_WB[6], 1E-5);
}

TEST_F(IRB140Test, link_forward_kinematics) {
  const auto X_01_sym = analytical_kinematics.X_01();
  const auto X_12_sym = analytical_kinematics.X_12();
  const auto X_23_sym = analytical_kinematics.X_23();
  const auto X_13_sym = analytical_kinematics.X_13();
  const auto X_34_sym = analytical_kinematics.X_34();
  const auto X_45_sym = analytical_kinematics.X_45();
  const auto X_56_sym = analytical_kinematics.X_56();

  const auto X_16_sym = X_13_sym * X_34_sym * X_45_sym * X_56_sym;
  const auto X_06_sym  = X_01_sym * X_16_sym;
  std::cout <<"X_01\n";
  printPose(X_01_sym);
  const auto X_02_sym = X_01_sym * X_12_sym;
  std::cout <<"X_02\n";
  printPose(X_02_sym);
  const auto X_03_sym = X_02_sym * X_23_sym;
  std::cout <<"X_03\n";
  printPose(X_03_sym);
  const auto X_04_sym = X_03_sym * X_34_sym;
  std::cout << "X_04\n";
  printPose(X_04_sym);
  const auto X_05_sym = X_04_sym * X_45_sym;
  std::cout << "X_05\n";
  printPose(X_05_sym);
  std::cout <<"X_06\n";
  printPose(X_06_sym);

  const int num_joint_sample = 3;
  Eigen::Matrix<double, 6, num_joint_sample> q_sample;
  for (int i = 0; i < 6; ++i) {
    q_sample.row(i) = Eigen::Matrix<double, 1, num_joint_sample>::LinSpaced(analytical_kinematics.robot()->joint_limit_min(i) + 1E-4, analytical_kinematics.robot()->joint_limit_max(i) - 1E-4);
  }
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();

  for (int i0 = 0; i0 < num_joint_sample; ++i0) {
    for (int i1 = 0; i1 < num_joint_sample; ++i1) {
      for (int i2 = 0; i2 < num_joint_sample; ++i2) {
        for (int i3 = 0; i3 < num_joint_sample; ++i3) {
          for (int i4 = 0; i4 < num_joint_sample; ++i4) {
            for (int i5 = 0; i5 < num_joint_sample; ++i5) {
              Eigen::Matrix<double, 6, 1> q;
              q(0) = q_sample(0, i0);
              q(1) = q_sample(1, i1);
              q(2) = q_sample(2, i2);
              q(3) = q_sample(3, i3);
              q(4) = q_sample(4, i4);
              q(5) = q_sample(5, i5);
              TestForwardKinematics(analytical_kinematics, q);
            }
          }
        }
      }
    }
  }
}

void TestInverseKinematics(const IRB140AnalyticalKinematics& analytical_kinematics, const Eigen::Matrix<double, 6, 1>& q, double tol = 1E-5) {
  const Eigen::Matrix<double, 6, 1>
      q_lb = analytical_kinematics.robot()->joint_limit_min;
  const Eigen::Matrix<double, 6, 1>
      q_ub = analytical_kinematics.robot()->joint_limit_max;
  auto cache = analytical_kinematics.robot()->CreateKinematicsCache();
  cache.initialize(q);
  analytical_kinematics.robot()->doKinematics(cache);
  const Isometry3d link6_pose =
      analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache,
                                                              *(analytical_kinematics.robot()->FindBody(
                                                                  "link_6")));

  const auto &q_all = analytical_kinematics.inverse_kinematics(link6_pose);
  EXPECT_GE(q_all.size(), 1);
  if (q_all.size() == 0) {
    std::cout << "q\n" << q << std::endl;
    const auto &X_06 = analytical_kinematics.X_06(q);
    CompareIsometry3d(X_06, link6_pose, tol);
    analytical_kinematics.inverse_kinematics(link6_pose);
  }


  int num_match_posture = 0;

  for (const auto &q_ik : q_all) {
    EXPECT_TRUE((q_ik.array() >= q_lb.array()).all());
    EXPECT_TRUE((q_ik.array() <= q_ub.array()).all());
    if (!(q_ik.array() >= q_lb.array()).all()) {
      std::cout << q_ik - q_lb << std::endl;
    }
    if (!(q_ik.array() <= q_ub.array()).all()) {
      std::cout << q_ub - q_ik << std::endl;
    }
    cache.initialize(q_ik);
    analytical_kinematics.robot()->doKinematics(cache);
    const Isometry3d link6_pose_ik =
        analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache,
                                                                *(analytical_kinematics.robot()->FindBody(
                                                                    "link_6")));
    CompareIsometry3d(link6_pose_ik, link6_pose, tol);
    if (!CompareIsometry3d(link6_pose_ik, link6_pose, tol)) {
      std::cout << "q\n" << q << std::endl;
      std::cout << "q_ik\n" << q_ik << std::endl;
      analytical_kinematics.inverse_kinematics(link6_pose);
    }
    if (std::abs(sin(q(4))) > 1E-5) {
      // Non degenerate case.
      if ((q - q_ik).norm() <= 5E-3) {
        ++num_match_posture;
      }
    } else {
      // Degenerate case.
      if (std::cos(q(4)) > 0) {
        // If cos(q5) = 1, then q4 + q6 should be constant.
        if (std::abs(q_ik(3) + q_ik(5) - q(3) - q(5)) < 1E-3
            && (q_ik.head<3>() - q.head<3>()).norm() < 1E-2
            && std::abs(q_ik(4) - q(4)) < 1E-3) {
          ++num_match_posture;
        }
      } else {
        // If cos(q5) = -1, then q4 - q6 should be constant.
        if (std::abs(q_ik(3) - q_ik(5) - q(3) + q(5)) < 1E-3
            && (q_ik.head<3>() - q.head<3>()).norm() < 1E-2
            && std::abs(q_ik(4) - q(4)) < 1E-3) {
          ++num_match_posture;
        }
      }
    }
  }
  EXPECT_NE(num_match_posture, 0);
  if (num_match_posture == 0) {
    std::cout << "q:\n" << q.transpose() << std::endl << "q_ik:\n";
    for (const auto& q_ik : q_all) {
      std::cout << q_ik.transpose() << std::endl;
    }
    analytical_kinematics.inverse_kinematics(link6_pose);
  }
}

TEST_F(IRB140Test, inverse_kinematics_test) {
  std::vector<Eigen::Isometry3d> link6_pose_all;
  Eigen::Isometry3d link6_pose;
  link6_pose.linear() = Eigen::Matrix3d::Identity();
  link6_pose.translation() = Eigen::Vector3d(0, 0, 0.9);
  link6_pose_all.push_back(link6_pose);
  link6_pose.linear() =  Eigen::Matrix3d::Identity();
  link6_pose.translation() = Eigen::Vector3d(0.45, 0, 0.55);
  link6_pose_all.push_back(link6_pose);

  KinematicsCache<double> cache = analytical_kinematics.robot()->CreateKinematicsCache();
  for (const auto& ee_pose : link6_pose_all) {
    const auto& q_all = analytical_kinematics.inverse_kinematics(ee_pose);
    for (const auto& q : q_all) {
      cache.initialize(q);
      analytical_kinematics.robot()->doKinematics(cache);
      const auto& ee_pose_fk = analytical_kinematics.robot()->CalcBodyPoseInWorldFrame(cache, *(analytical_kinematics.robot()->FindBody("link_6")));
      CompareIsometry3d(ee_pose_fk, ee_pose, 1E-5);
    }
  }
}

TEST_F(IRB140Test, inverse_kinematics_exhaustive_test) {
  std::vector<Eigen::Matrix<double, 6, 1>> q_all;
  const int num_joint_sample = 11;
  Eigen::Matrix<double, 6, num_joint_sample> q_sample;
  for (int i = 0; i < 6; ++i) {
    q_sample.row(i) = Eigen::Matrix<double, 1, num_joint_sample>::LinSpaced(
        analytical_kinematics.robot()->joint_limit_min(i) + 1E-2,
        analytical_kinematics.robot()->joint_limit_max(i) - 1E-2);
  }

  for (int i0 = 0; i0 < num_joint_sample; ++i0) {
    for (int i1 = 0; i1 < num_joint_sample; ++i1) {
      for (int i2 = 0; i2 < num_joint_sample; ++i2) {
        for (int i3 = 0; i3 < num_joint_sample; ++i3) {
          for (int i4 = 0; i4 < num_joint_sample; ++i4) {
            for (int i5 = 0; i5 < num_joint_sample; ++i5) {
              Eigen::Matrix<double, 6, 1> q;
              q(0) = q_sample(0, i0);
              q(1) = q_sample(1, i1);
              q(2) = q_sample(2, i2);
              q(3) = q_sample(3, i3);
              q(4) = q_sample(4, i4);
              q(5) = q_sample(5, i5);

              TestInverseKinematics(analytical_kinematics, q, 3E-5);
            }
          }
        }
      }
    }
  }
}


TEST_F(IRB140Test, inverse_kinematics_corner_test) {
  // Degenerate case q = 0
  Eigen::Matrix<double, 6, 1> q;
  q.setZero();
  TestInverseKinematics(analytical_kinematics, q);

  // Degenerate case, q5 = 0
  q << 0.1, -0.2, -0.6, 0.3, 0, 0.2;
  DRAKE_DEMAND((q.array() >= analytical_kinematics.robot()->joint_limit_min.array()).all());
  DRAKE_DEMAND((q.array() <= analytical_kinematics.robot()->joint_limit_max.array()).all());
  TestInverseKinematics(analytical_kinematics, q, 1E-4);

  q << -0.3, 0.6, 0.2, 1.2, 0, -1.2;
  DRAKE_DEMAND((q.array() >= analytical_kinematics.robot()->joint_limit_min.array()).all());
  DRAKE_DEMAND((q.array() <= analytical_kinematics.robot()->joint_limit_max.array()).all());
  TestInverseKinematics(analytical_kinematics, q, 1E-4);
}

TEST_F(IRB140Test, inverse_kinematics_infeasible_test) {
  Eigen::Isometry3d link6_pose;
  link6_pose.translation() << 1.0, 1.0, -0.2;
  link6_pose.linear() = Eigen::Matrix3d::Identity();
  const auto& q_ik = analytical_kinematics.inverse_kinematics(link6_pose);
  EXPECT_EQ(q_ik.size(), 0);
}
}  // namespace
}  // namespace IRB140
}  // namespace examples
}  // namespace drake
