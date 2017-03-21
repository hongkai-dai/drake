#include "drake/examples/IRB140/IRB140_analytical_kinematics.h"

using Eigen::Isometry3d;
using Eigen::Matrix;
using drake::symbolic::Variable;
using drake::symbolic::Expression;

namespace drake {
namespace examples {
namespace IRB140 {
IRB140AnalyticalKinematics::IRB140AnalyticalKinematics()
    : l0_(0.1095),
      l1_x_(0.07),
      l1_y_(0.2425),
      l2_(0.36),
      l3_(0.2185),
      l4_(0.1615),
      l5_(0.065),
      c_{}, s_{},
      l0_var_("l0"),
      l1_x_var_("l1x"),
      l1_y_var_("l1y"),
      l2_var_("l2"),
      l3_var_("l3"),
      l4_var_("l4"),
      l5_var_("l5") {
  for (int i = 0; i < 6; ++i) {
    c_[i] = symbolic::Variable("c" + std::to_string(i + 1));
    s_[i] = symbolic::Variable("s" + std::to_string(i + 1));
  }
}

Isometry3d IRB140AnalyticalKinematics::X_01(double theta) const {
  Isometry3d X;
  Eigen::Matrix3d R_0J = Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();
  X.linear() = R_0J * Eigen::AngleAxisd(theta, Eigen::Vector3d(0, -1, 0)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(0, 0, l0_);
  return X;
}

Isometry3d IRB140AnalyticalKinematics::X_12(double theta) const {
  Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(l1_x_, -l1_y_, 0);
  return X;
}

Isometry3d IRB140AnalyticalKinematics::X_23(double theta) const {
  Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(0, -l2_, 0);
  return X;
}

Isometry3d IRB140AnalyticalKinematics::X_34(double theta) const {
  Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(l3_, 0, 0);
  return X;
}

Isometry3d IRB140AnalyticalKinematics::X_45(double theta) const {
  Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(0, 0, -1)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(l4_, 0, 0);
  return X;
}

Isometry3d IRB140AnalyticalKinematics::X_56(double theta) const {
  Isometry3d X;
  X.linear() = Eigen::AngleAxisd(theta, Eigen::Vector3d(1, 0, 0)).toRotationMatrix();
  X.translation() = Eigen::Vector3d(l5_, 0, 0);
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_01_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[0], 0, s_[0], 0,
       -s_[0], 0, c_[0], 0,
       0, -1, 0, l0_var_,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_12_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[1], -s_[1], 0, l1_x_var_,
       s_[1], c_[1], 0, -l1_y_var_,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_23_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[2], -s_[2], 0, 0,
       s_[2], c_[2], 0, -l2_var_,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_34_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << 1, 0, 0, l3_var_,
       0, c_[3], -s_[3], 0,
       0, s_[3], c_[3], 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
}

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_45_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << c_[4], s_[4], 0, l4_var_,
       -s_[4], c_[4], 0, 0,
       0, 0, 1, 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
};

Matrix<Expression, 4, 4> IRB140AnalyticalKinematics::X_56_sym() const {
  Matrix<Expression, 4, 4> X;
  // clang-format off
  X << 1, 0, 0, l5_var_,
       0, c_[5], -s_[5], 0,
       0, s_[5], c_[5], 0,
       0, 0, 0, 1;
  // clang-format on
  return X;
};
}
}
}