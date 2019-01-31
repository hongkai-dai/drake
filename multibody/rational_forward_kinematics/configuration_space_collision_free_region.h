#pragma once

#include <string>
#include <unordered_set>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace multibody {
/**
 * A separation plane {x | aᵀ(x-c) = 1} separates two polytopes.
 * c is the center of the negative_side_polytope.
 * All the vertices in the positive_side_polytope satisfies aᵀ(v-c) ≥ 1
 * All the vertices in the negative_side_polytope satisfies aᵀ(v-c) ≤ 1
 */
template <typename T>
struct SeparationPlane {
  SeparationPlane(
      const Vector3<T>& m_a,
      std::shared_ptr<const ConvexPolytope> m_positive_side_polytope,
      std::shared_ptr<const ConvexPolytope> m_negative_side_polytope,
      BodyIndex m_expressed_link)
      : a{m_a},
        positive_side_polytope{m_positive_side_polytope},
        negative_side_polytope{m_negative_side_polytope},
        expressed_link{m_expressed_link} {}
  const Vector3<T> a;
  std::shared_ptr<const ConvexPolytope> positive_side_polytope;
  std::shared_ptr<const ConvexPolytope> negative_side_polytope;
  // The link frame in which a is expressed.
  const BodyIndex expressed_link;
};

/** We need to verify that these polynomials are non-negative
 * Lagrangians l_lower(t) >= 0, l_upper(t) >= 0
 * Link polynomial p(t) - l_lower(t) * (t - t_lower) - l_upper(t)(t_upper_t)
 * >= 0
 */
struct VerificationOption {
  VerificationOption()
      : link_polynomial_type{solvers::MathematicalProgram::
                                 NonnegativePolynomial::kSos},
        lagrangian_type{
            solvers::MathematicalProgram::NonnegativePolynomial::kSos} {}

  solvers::MathematicalProgram::NonnegativePolynomial link_polynomial_type;
  solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
};

/**
 * The rational function representing that a link vertex V is on the desired
 * side of the plane.
 * If the link is on the positive side of the plane, then the rational is
 * aᵀ(x-c)- 1; otherwise it is 1 - aᵀ(x-c).
 */
template <typename T>
struct LinkVertexOnPlaneSideRational {
  LinkVertexOnPlaneSideRational(
      symbolic::RationalFunction m_rational,
      std::shared_ptr<const ConvexPolytope> m_link_polytope,
      BodyIndex m_expressed_body_index,
      const Eigen::Ref<const Eigen::Vector3d>& m_p_BV, const Vector3<T>& m_a_A,
      PlaneSide m_plane_side)
      : rational(std::move(m_rational)),
        link_polytope(m_link_polytope),
        expressed_body_index(m_expressed_body_index),
        p_BV(m_p_BV),
        a_A(m_a_A),
        plane_side(m_plane_side) {}
  const symbolic::RationalFunction rational;
  const std::shared_ptr<const ConvexPolytope> link_polytope;
  const BodyIndex expressed_body_index;
  // The position of the vertex V in link's frame B.
  const Eigen::Vector3d p_BV;
  const Vector3<T> a_A;
  const PlaneSide plane_side;
};

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

  struct GeometryIdPairHash {
    std::size_t operator()(
        const std::pair<ConvexGeometry::Id, ConvexGeometry::Id>& p) const {
      return p.first * 100 + p.second;
      // return std::hash<ConvexGeometry::Id>()(p.first * 100) +
      //       std::hash<ConvexGeometry::Id>()(p.second);
    };
  };

  using FilteredCollisionPairs =
      std::unordered_set<std::pair<ConvexGeometry::Id, ConvexGeometry::Id>,
                         GeometryIdPairHash>;

  /**
   * Verify the collision free configuration space for the given robot. The
   * geometry of each robot link is represented as a union of polytopes. The
   * obstacles are also a union ob polytopes.
   */
  ConfigurationSpaceCollisionFreeRegion(
      const MultibodyPlant<double>& plant,
      const std::vector<std::shared_ptr<const ConvexPolytope>>& link_polytopes,
      const std::vector<std::shared_ptr<const ConvexPolytope>>& obstacles);

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const std::vector<SeparationPlane<symbolic::Variable>>& separation_planes()
      const {
    return separation_planes_;
  }

  /**
   * Generate all the rational functions representing the the link vertices are
   * on the correct side of the planes.
   */
  std::vector<LinkVertexOnPlaneSideRational<symbolic::Variable>>
  GenerateLinkOnOneSideOfPlaneRationals(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs) const;

  const std::unordered_map<std::pair<ConvexGeometry::Id, ConvexGeometry::Id>,
                           const SeparationPlane<symbolic::Variable>*,
                           GeometryIdPairHash>&
  map_polytopes_to_separation_planes() const {
    return map_polytopes_to_separation_planes_;
  }

  /**
   * Construct an optimization program to verify the the box region t_lower <= t
   * <= t_upper is collision free.
   * The program verifies that t_lower <= t <= t_upper implies @p polynomials[i]
   * >= 0 for all i, where rationals[i] is the result of calling
   * GenerateLinkOnOneSideOfPlaneRationals.
   */
  std::unique_ptr<solvers::MathematicalProgram>
  ConstructProgramToVerifyCollisionFreeBox(
      const std::vector<LinkVertexOnPlaneSideRational<symbolic::Variable>>&
          rationals,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const VerificationOption& verification_option = {}) const;

  /**
   * Find the largest box in the configuration space, that we can verify to be
   * collision free. The box is defined as
   * max(ρ * Δt₋, tan((θ_lower - θ*)/2)) <=
   *                     tan((θ - θ*)/2) <=
   *                                  min(ρ * Δt₊, tan((θ_upper - θ*) /  2))
   * where θ_lower, θ_upper are the joint limits.
   * @param q_star The nominal configuration around which we verify the
   * collision free box. This nominal configuration should be collision free.
   * @param filtered_collision_pairs The set of polytope pairs between which the
   * collision check is ignored.
   * @param negative_delta_t Δt₋ in the documentation above.
   * @param positive_delta_t Δt₊ in the documentation above.
   * @param rho_lower_initial The initial guess on the lower bound of ρ. The box
   * defined with rho_lower_initial is collision free.
   * @param rho_upper_initial The initial guess on the upper bound of ρ. The box
   * defined with rho_upper_initial is not collision free.
   * @param rho_tolerance The tolerance on ρ in the binary search.
   */
  double FindLargestBoxThroughBinarySearch(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const FilteredCollisionPairs& filtered_collision_pairs,
      const Eigen::Ref<const Eigen::VectorXd>& negative_delta_t,
      const Eigen::Ref<const Eigen::VectorXd>& positive_delta_t,
      double rho_lower_initial, double rho_upper_initial, double rho_tolerance,
      const VerificationOption& verification_option = {}) const;

 private:
  bool IsLinkPairCollisionIgnored(
      ConvexGeometry::Id id1, ConvexGeometry::Id id2,
      const FilteredCollisionPairs& filtered_collision_pairs) const;

  RationalForwardKinematics rational_forward_kinematics_;
  std::map<BodyIndex, std::vector<std::shared_ptr<const ConvexPolytope>>>
      link_polytopes_;
  // obstacles_[i] is the i'th polytope, fixed to the world.
  std::vector<std::shared_ptr<const ConvexPolytope>> obstacles_;

  std::vector<SeparationPlane<symbolic::Variable>> separation_planes_;

  // In the key, the first ConvexGeometry::Id is for the polytope on the
  // positive side, the second ConvexGeometry::Id is for the one on the negative
  // side.
  std::unordered_map<std::pair<ConvexGeometry::Id, ConvexGeometry::Id>,
                     const SeparationPlane<symbolic::Variable>*,
                     GeometryIdPairHash>
      map_polytopes_to_separation_planes_;
};

/**
 * Generate the rational functions a_A.dot(p_AVi(t) - p_AC) <= 1 (or >= 1)
 * which
 * represents that the link (whose vertex Vi has position p_AVi in frame A) is
 * on the negative (or positive, respectively) side of the hyperplane.
 * @param rational_forward_kinematics The utility class that computes the
 * position of Vi in A's frame as a rational function of t.
 * @param link_polytope The polytopic representation of the link collision
 * geometry, Vi is the i'th vertex of the polytope.
 * @param q_star The nominal configuration.
 * @param expressed_body_index Frame A in the documentation above. The body in
 * which the position is expressed in.
 * @param a_A The normal vector of the plane. This vector is expressed in
 * frame
 * A.
 * @param p_AC The point within the interior of the negative side of the
 * plane.
 * @param plane_side Whether the link is on the positive or the negative side
 * of
 * the plane.
 * @return rational_fun rational_fun[i] should be non-negative to represent
 * that
 * the vertiex i is on the correct side of the plane. rational_fun[i] =
 * a_A.dot(p_AVi(t) - p_AC) - 1 if @p plane_side = kPositive, rational_fun[i]
 * =
 * 1 - a_A.dot(p_AVi(t) - p_AC) if @p plane_side = kNegative.
 */
std::vector<LinkVertexOnPlaneSideRational<symbolic::Variable>>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side);

/**
 * Overloaded GenerateLinkOnOnseSideOfPlaneRationalFunction, except X_AB,
 * i.e.,
 * the pose of the link polytope in the expressed_frame is given.
 * @param X_AB_multilinear The pose of the link polytope frame B in the
 * expressed body frame A. Note that this pose is a multilinear function of
 * sinθ and cosθ.
 */
std::vector<LinkVertexOnPlaneSideRational<symbolic::Variable>>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side);

std::vector<LinkVertexOnPlaneSideRational<symbolic::Polynomial>>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    std::shared_ptr<const ConvexPolytope> link_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<symbolic::Polynomial>>& a_A_poly,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side);

/** Impose the constraint that
 * l_lower(t) >= 0                                                         (1)
 * l_upper(t) >= 0                                                         (2)
 * p(t) - l_lower(t) * (t - t_lower) - l_upper(t) (t_upper - t) >= 0       (3)
 * where p(t) is the numerator of @p polytope_on_one_side_rational
 * @param monomial_basis The basis for the monomial of p(t), l_lower(t),
 * l_upper(t) and the polynomial (3) above.
 */
void AddNonnegativeConstraintForPolytopeOnOneSideOfPlane(
    solvers::MathematicalProgram* prog,
    const symbolic::RationalFunction& polytope_on_one_side_rational,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_minus_t_lower,
    const Eigen::Ref<const VectorX<symbolic::Polynomial>>& t_upper_minus_t,
    const Eigen::Ref<const VectorX<symbolic::Monomial>>& monomial_basis,
    const VerificationOption& verification_option = {});
}  // namespace multibody
}  // namespace drake
