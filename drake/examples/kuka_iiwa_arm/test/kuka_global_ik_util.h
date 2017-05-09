#pragma once
#include "drake/multibody/rigid_body_tree.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
Eigen::Matrix<double, 3, 8> AddBoxToTree(RigidBodyTreed* tree, const Eigen::Ref<const Eigen::Vector3d>& box_size, const Eigen::Isometry3d& box_pose, const std::string& name);
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake