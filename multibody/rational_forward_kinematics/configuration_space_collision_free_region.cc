#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

namespace drake {
namespace multibody {
using symbolic::RationalFunction;

ConfigurationSpaceCollisionFreeRegion::ConfigurationSpaceCollisionFreeRegion(
    const MultibodyTree<double>& tree,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        link_polytopes,
    const std::vector<ConfigurationSpaceCollisionFreeRegion::Polytope>&
        obstacles)
    : rational_forward_kinematics_{tree},
      link_polytopes_{static_cast<size_t>(tree.num_bodies())},
      obstacles_{obstacles},
      obstacle_center_{obstacles_.size()},
      a_hyperplane_(link_polytopes_.size()) {
  const int num_links = tree.num_bodies();
  const int num_obstacles = static_cast<int>(obstacles_.size());
  DRAKE_DEMAND(num_obstacles > 0);
  DRAKE_DEMAND(static_cast<int>(link_polytopes_.size()) == num_links);
  for (const auto& obstacle : obstacles_) {
    DRAKE_ASSERT(obstacle.body_index == 0);
  }
  for (const auto& link_polytope : link_polytopes) {
    DRAKE_ASSERT(link_polytope.body_index != 0);
    link_polytopes_[link_polytope.body_index].push_back(link_polytope);
  }
  for (int i = 1; i < num_links; ++i) {
    const int num_link_polytopes = static_cast<int>(link_polytopes_[i].size());
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

std::vector<symbolic::RationalFunction> ConfigurationSpaceCollisionFreeRegion::
    GenerateLinkOutsideHalfspaceRationalFunction(
        const Eigen::VectorXd& q_star) const {
  const std::vector<RationalForwardKinematics::Pose<symbolic::Polynomial>>
      link_poses_poly =
          rational_forward_kinematics_.CalcLinkPosesAsMultilinearPolynomials(
              q_star, 0);
  std::vector<symbolic::RationalFunction> collision_free_rationals;
  const symbolic::Monomial monomial_one{};
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      const int num_polytope_vertices = link_polytopes_[i][j].vertices.cols();
      Matrix3X<symbolic::Polynomial> p_WV(3, num_polytope_vertices);
      for (int l = 0; l < num_polytope_vertices; ++l) {
        p_WV.col(l) =
            link_poses_poly[i].p_AB +
            link_poses_poly[i].R_AB * link_polytopes_[i][j].vertices.col(l);
      }
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        // For each pair of link polytope and obstacle polytope, we need to
        // impose the constraint that all vertices of the link polytope are on
        // the "outer" side of the hyperplane. So each vertex of the link
        // polytope will introduce one polynomial. Likewise, we will impose the
        // constraint that each vertex of the obstacle polytope is in the
        // "inner" side of the hyperplane. This will be some linear constraints
        // on the hyperplane parameter a.
        // We want to impose the constraint a_hyperplane[i][j]k]ᵀ (p_WV -
        // p_WB_center) >= 1
        Vector3<symbolic::Polynomial> a_poly;
        for (int idx = 0; idx < 3; ++idx) {
          a_poly(idx) = symbolic::Polynomial(
              {{monomial_one, a_hyperplane_[i][j][k](idx)}});
        }
        for (int l = 0; l < link_polytopes_[i][j].vertices.cols(); ++l) {
          const symbolic::Polynomial outside_hyperplane_poly =
              a_poly.dot(p_WV.col(l) - obstacle_center_[k]) - 1;
          const symbolic::Polynomial outside_hyperplane_poly_trimmed =
              outside_hyperplane_poly.RemoveTermsWithSmallCoefficients(1e-12);
          const symbolic::RationalFunction outside_hyperplane_rational =
              rational_forward_kinematics_
                  .ConvertMultilinearPolynomialToRationalFunction(
                      outside_hyperplane_poly_trimmed);
          collision_free_rationals.push_back(outside_hyperplane_rational);
        }
      }
    }
  }
  return collision_free_rationals;
}

std::vector<symbolic::Polynomial>
ConfigurationSpaceCollisionFreeRegion::GenerateLinkOutsideHalfspacePolynomials(
    const Eigen::VectorXd& q_star) const {
  const std::vector<symbolic::RationalFunction> collision_free_rationals =
      GenerateLinkOutsideHalfspaceRationalFunction(q_star);
  std::vector<symbolic::Polynomial> collision_free_polynomials;
  collision_free_polynomials.reserve(collision_free_rationals.size());
  for (const auto& rational : collision_free_rationals) {
    collision_free_polynomials.push_back(rational.numerator());
  }
  return collision_free_polynomials;
}

std::vector<symbolic::Expression> ConfigurationSpaceCollisionFreeRegion::
    GenerateObstacleInsideHalfspaceExpression() const {
  std::vector<symbolic::Expression> exprs;
  for (int i = 1; i < rational_forward_kinematics_.tree().num_bodies(); ++i) {
    for (int j = 0; j < static_cast<int>(link_polytopes_[i].size()); ++j) {
      for (int k = 0; k < static_cast<int>(obstacles_.size()); ++k) {
        for (int l = 0; l < obstacles_[k].vertices.cols(); ++l) {
          exprs.push_back(
              a_hyperplane_[i][j][k].dot(obstacles_[k].vertices.col(l) -
                                         obstacle_center_[k]) -
              1);
        }
      }
    }
  }
  return exprs;
}

}  // namespace multibody
}  // namespace drake
