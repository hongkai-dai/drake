#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

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
  std::vector<RationalForwardKinematics::LinkPoints> link_polytope_vertices(
      rational_forward_kinematics_.tree().num_bodies());
  for (int i = 0; i < static_cast<int>(link_polytopes_.size()); ++i) {
    // Horizontally concatenate all the polytope vertices attached to the same
    // link.
    Eigen::Matrix3Xd link_i_vertices(3, 0);
    for (const auto& polytope : link_polytopes_[i]) {
      link_i_vertices.conservativeResize(
          3, link_i_vertices.cols() + polytope.vertices.cols());
      link_i_vertices.rightCols(polytope.vertices.cols()) = polytope.vertices;
    }
    link_polytope_vertices.emplace_back(i, link_i_vertices);
  }
  const std::vector<Matrix3X<RationalFunction>> p_WV =
      rational_forward_kinematics_.CalcLinkPointsPosition(
          q_star, link_polytope_vertices, 0);
  std::vector<symbolic::Polynomial> collision_free_polynomials;
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
    int link_vertex_count = 0;
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      const auto& p_WVj = p_WV[i].block<3, Eigen::Dynamic>(
          0, link_vertex_count, 3, link_polytopes_[i][j].vertices.cols());
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        // For each pair of link polytope and obstacle polytope, we need to
        // impose the constraint that all vertices of the link polytope are on
        // the "outer" side of the hyperplane. So each vertex of the link
        // polytope will introduce one polynomial. Likewise, we will impose the
        // constraint that each vertex of the obstacle polytope is in the
        // "inner" side of the hyperplane. This will be some linear constraints
        // on the hyperplane parameter a.
        for (int l = 0; l < link_polytopes_[i][j].vertices.cols(); ++l) {
          a_hyperplane[i][j][k].dot(p_WVj.col(l) 
        }
      }
      link_vertex_count += link_polytopes_[i][j].vertices.cols();
    }
  }
  return collision_free_polynomials;
}

}  // namespace multibody
}  // namespace drake
