#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

namespace drake {
namespace multibody {
ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyTree<double>& tree,
    const std::vector<
        std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>>&
        link_polytopes,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        obstacles)
    : rational_forward_kinematics_{tree},
      link_polytopes_{link_polytopes},
      obstacles_{obstacles},
      obstacle_center_{obstacles_.size()},
      a_hyperplane_(link_polytopes_.size()) {
  const int num_links = tree.num_bodies();
  const int num_obstacles = static_cast<int>(obstacles_.size());
  DRAKE_DEMAND(num_obstacles > 0);
  DRAKE_DEMAND(static_cast<int>(link_polytopes_.size()) == num_links);
  for (int i = 0; i < num_links; ++i) {
    const int num_link_polytopes = static_cast<int>(link_polytopes[i].size());
    a_hyperplane_[i].resize(num_link_polytopes);
    for (int j = 0; j < num_link_polytopes; ++j) {
      a_hyperplane_[i][j].resize(num_obstacles);
      for (int k = 0; k < num_obstacles; ++k) {
        const std::string a_name = "a[" + std::to_string(i) + "][" +
                                   std::to_string(j) + "][" +
                                   std::to_string(k) + "]";
        for (int l = 0; l < 3; ++l) {
          a_hyperplane_[i][j][k](l) =
              symbolic::Variable(a_name + "(" + std::to_string(l) + ")");
        }
      }
    }
  }
  for (int i = 0; i < num_obstacles; ++i) {
    obstacle_center_[i] =
        obstacles_[i].vertices.rowwise().sum() / obstacles_[i].vertices.cols();
  }
}

std::vector<symbolic::Polynomial>
ConfigurationSpaceCollisionFreeRegion::GenerateLinkOutsideHalfspacePolynomials(
    const Eigen::VectorXd& q_star) const {
  const std::vector<RationalForwardKinematics::Pose> link_poses =
      rational_forward_kinematics_.CalcLinkPoses(q_star);
  std::vector<symbolic::Polynomial> collision_free_polynomials;
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        // For each pair of link polytope and obstacle polytope, we need to
        // impose the constraint that all vertices of the link polytope are on
        // the "outer" side of the hyperplane. So each vertex of the link
        // polytope will introduce one polynomial. Likewise, we will impose the
        // constraint that each vertex of the obstacle polytope is in the
        // "inner" side of the hyperplane. This will be some linear constraints
        // on the hyperplane parameter a.
        for (int l = 0; l < link_polytopes_[i][j].vertices.cols(); ++l) {


        }
      }
    }
  }
  return collision_free_polynomials;
}

}  // namespace multibody
}  // namespace drake
