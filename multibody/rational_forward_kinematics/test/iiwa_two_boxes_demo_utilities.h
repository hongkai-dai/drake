#include <memory>

#include "drake/multibody/rational_forward_kinematics/test/rational_forward_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
struct IiwaTwoBoxesDemo {
  IiwaTwoBoxesDemo();

  void ComputeBoxPosition(const Eigen::Ref<const Eigen::VectorXd>& q_star,
                          Eigen::Vector3d* p_4C0_star,
                          Eigen::Vector3d* p_4C1_star) const;

  std::unique_ptr<MultibodyPlant<double>> plant;
  std::array<BodyIndex, 8> iiwa_link;
  BodyIndex world;
  std::vector<std::shared_ptr<const ConvexPolytope>> link_polytopes;
  std::vector<std::shared_ptr<const ConvexPolytope>> obstacle_boxes;
};
}  // namespace multibody
}  // namespace drake
