#pragma once

#include <string>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace multibody {
/**
 * This class tries to find a large convex set in the configuration space, such
 * that this whole convex set is collision free. We assume that the obstacles
 * are unions of polytopes in the workspace, and the robot link poses
 * (position/orientation) can be written as rational functions of some
 * variables. Such robot can have only revolute (or prismatic joint). We also
 * suppose that the each link of the robot is represented as a union of
 * polytopes. We will find the convex collision free set in the configuration
 * space through convex optimization.
 */
class ConfigurationSpaceCollisionFreeRegion {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConfigurationSpaceCollisionFreeRegion)

  struct Polytope {
    Polytope(int m_body_index,
             const Eigen::Ref<const Eigen::Matrix3Xd>& m_vertices)
        : body_index{m_body_index}, vertices{m_vertices} {
      DRAKE_ASSERT(vertices.cols() > 0)
    }
    int body_index;
    Eigen::Matrix3Xd vertices;
  };

  /**
   * Verify the collision free configuration space for the given robot. The
   * geometry of each robot link is represented as a union of polytopes. The
   * obstacles are also a union ob polytopes.
   */
  ConfigurationSpaceCollisionFreeRegion(
      const MultibodyTree<double>& tree,
      const std::vector<Polytope>& link_polytopes,
      const std::vector<Polytope>& obstacles);

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const std::vector<Polytope>& obstacles() const { return obstacles_; }

  const std::vector<std::vector<Polytope>>& link_polytopes() const {
    return link_polytopes_;
  }

  /**
   * To enforce that the robot is collision free, we want to find a hyperplane
   * between each polytope on the link, and each obstacle polytope. The
   * hyperplane is parameterized as {x: aᵀ(x-p) = 1}, where p is a point within
   * the obstacle. All points x within the obstacle polytope should satisfy
   * aᵀ(x-p) ≤ 1, and all points on the link polytope should satisfy aᵀ(x-p)
   * ≥ 1.
   */
  const std::vector<std::vector<std::vector<Vector3<symbolic::Variable>>>>&
  a_hyperplane() const {
    return a_hyperplane_;
  }

  const std::vector<Eigen::Vector3d>& obstacle_center() const {
    return obstacle_center_;
  }

  /**
   * Generate polynomials on the indeterminates t
   * (rational_forward_kinematics().t()), such that if all these polynomials are
   * non-negative, then each polytope on the robot links are outside of a
   * corresponding halfspace. This expression is a polynomial on t.
   * @param q_star The nominal posture. We will verify a convex neighbourhood
   * around this nominal posture as a collision free region in the configuration
   * space.
   */
  std::vector<symbolic::Polynomial> GenerateLinkOutsideHalfspacePolynomials(
      const Eigen::VectorXd& q_star) const;

  /**
   * Generate expressions on a_hyperplane, such that the expression <= 0
   * indicates that the obstacle is inside the hyperplane.
   * Mathematically, we use aᵀ(v - p) <= 1 to denote obstalce inside the
   * hyperplane, where v is the vertex of the obstacle, and p is the obstacle
   * center.
   */
  std::vector<symbolic::Expression> GenerateObstacleInsideHalfspaceExpression()
      const;

 private:
  std::vector<symbolic::RationalFunction>
  GenerateLinkOutsideHalfspaceRationalFunction(
      const Eigen::VectorXd& q_star) const;

  // forward declare the tester class.
  friend class ConfigurationSpaceCollisionFreeRegionTester;

  RationalForwardKinematics rational_forward_kinematics_;
  // link_polytopes_[i] contains all the polytopes fixed to the i'th link
  std::vector<std::vector<Polytope>> link_polytopes_;
  // obstacles_[i] is the i'th polytope, fixed to the world.
  std::vector<Polytope> obstacles_;
  // We need a point p inside the obstacle to parameterize the hyperplane {x:
  // aᵀ(x-p) = 1}. We choose the average of all the vertices as this point, and
  // call it the obstacle center.
  std::vector<Eigen::Vector3d> obstacle_center_;

  // a_hyperplane[i][j][k] is the parameter for the hyper plane between
  // link_polytopes_[i][j] and the obstacles_[k].
  // TODO(hongkai.dai): add the hyperplane for self collision avoidance.
  std::vector<std::vector<std::vector<Vector3<symbolic::Variable>>>>
      a_hyperplane_;
};
}  // namespace multibody
}  // namespace drake
