#pragma once

#include <string>

#include "drake/common/drake_copyable.h"
#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"
#include "drake/multibody/rational_forward_kinematics/plane_side.h"
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

  /**
   * Verify the collision free configuration space for the given robot. The
   * geometry of each robot link is represented as a union of polytopes. The
   * obstacles are also a union ob polytopes.
   */
  ConfigurationSpaceCollisionFreeRegion(
      const MultibodyPlant<double>& plant,
      const std::vector<ConvexPolytope>& link_polytopes,
      const std::vector<ConvexPolytope>& obstacles);

  const RationalForwardKinematics& rational_forward_kinematics() const {
    return rational_forward_kinematics_;
  }

  const std::vector<ConvexPolytope>& obstacles() const { return obstacles_; }

  const std::vector<std::vector<ConvexPolytope>>& link_polytopes() const {
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

  struct VerificationOptions {
    VerificationOptions()
        : lagrangian_type{solvers::MathematicalProgram::NonnegativePolynomial::
                              kSos},
          link_polynomial_type{
              solvers::MathematicalProgram::NonnegativePolynomial::kSos} {}

    // Whether the Lagrangian should be SOS, SDSOS or DSOS.
    solvers::MathematicalProgram::NonnegativePolynomial lagrangian_type;
    // Whether the polynomial q(t) - l(t)(ρ - ∑ᵢ wᵢtᵢ²) should be SOS, SDSOS or
    // DSOS. This polynomial >= 0 means that the link is outside the separating
    // hyperplane when the joint values are in the configuration neighbourhood.
    solvers::MathematicalProgram::NonnegativePolynomial link_polynomial_type;
  };

  /**
   * Construct a mathematical program to verify that the region
   * c(t) <= 0 is a collision free region, where tᵢ = tan((θᵢ - θᵢ*)/2).
   * A convex-shaped link is not colliding with another convex-shaped obstacle
   * if there exist a hyperplane aᵀ(x - p)= 1 that separates the link from the
   * obstacle, where p is the obstacle center. Hence we impose the following
   * constraint
   * aᵀ(v - p) <= 1 for each obstacle vertex v
   * q(t) + l(t)ᵀc(t) >= 0 ∀t
   * lᵢ(t) >= 0 ∀t, i
   * where q(t) is a polynomial of t, q(t) = aᵀ(n(t) - p*d(t)) - d(t)
   * where n(t)/d(t) is the position of a link vertex, expressed as a rational
   * function of t.
   * @param q_star The nominal posture of the robot. The collision free
   * neighbourhood is around this nominal posture.
   * @param
   * @param weights w in the documentation above. weights must be non-negative.
   * @param rho The size of the neighbourhood. rho >= 0.
   * @param options The verification options.
   * @param[out] prog The constructed optimization program.
   */
  void ConstructProgramToVerifyFreeRegionAroundPosture(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const Eigen::Ref<const VectorX<symbolic::Polynomial>>& c,
      const Eigen::Ref<const VectorX<symbolic::Monomial>>&
          lagrangian_monomial_basis,
      const Eigen::Ref<const VectorX<symbolic::Monomial>>&
          aggregated_monomial_basis,
      const VerificationOptions& options,
      solvers::MathematicalProgram* prog) const;

  /**
   * Verifies that the ellipsoidal region {t| ∑ᵢ wᵢ*tᵢ²≤ ρ} is collision free,
   * where tᵢ = tan((θᵢ - θᵢ*)/2).
   * A convex-shaped link is not colliding with another convex-shaped obstacle
   * if there exist a hyperplane aᵀ(x - p)= 1 that separates the link from the
   * obstacle, where p is the obstacle center. Hence we impose the following
   * constraint
   * aᵀ(v - p) <= 1 for each obstacle vertex v
   * q(t) + l(t)(ρ-∑ᵢ wᵢtᵢ²) >= 0 ∀t
   * l(t) >= 0 ∀t
   * where q(t) is a polynomial of t, q(t) = aᵀ(n(t) - p*d(t)) - d(t)
   * where n(t)/d(t) is the position of a link vertex, expressed as a rational
   * function of t.
   * @param q_star The nominal posture of the robot. The collision free
   * neighbourhood is around this nominal posture.
   * @param
   * @param weights w in the documentation above. weights must be non-negative.
   * @param rho The size of the neighbourhood. rho >= 0.
   * @param options The verification options.
   * @param[out] prog The constructed optimization program.
   */
  void ConstructProgramToVerifyEllipsoidalFreeRegionAroundPosture(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const Eigen::Ref<const Eigen::VectorXd>& weights, double rho,
      const ConfigurationSpaceCollisionFreeRegion::VerificationOptions& options,
      solvers::MathematicalProgram* prog) const;

  /**
   * Verifies that the box region {t| tᵢ_lower ≤ tᵢ ≤ tᵢ_upper} is collision
   * free, where tᵢ = tan((θᵢ - θᵢ*)/2).
   * @return lagrangian_pairs lagrangian_pairs[k] contains all the pairs
   * (t_upper(j) - t(j), l1j(t)) or (t(j) - t_lower(j), l2j(t)) to verify that
   * GenerateLinkOutsideHalfspacePolynomials(q_star)(k) is nonnegative, where
   * l1j(t) and l2j(t) are lagrangians for t_upper(j) - tj) and t(j) -
   * t_lower(j) respectively.
   */
  std::vector<
      std::vector<std::pair<symbolic::Polynomial, symbolic::Polynomial>>>
  ConstructProgramToVerifyBoxFreeRegionAroundPosture(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      const Eigen::Ref<const Eigen::VectorXd>& t_lower,
      const Eigen::Ref<const Eigen::VectorXd>& t_upper,
      const ConfigurationSpaceCollisionFreeRegion::VerificationOptions& options,
      solvers::MathematicalProgram* prog) const;

 private:
  void AddIndeterminatesAndObstacleInsideHalfspaceToProgram(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      solvers::MathematicalProgram* prog) const;

  std::vector<symbolic::RationalFunction>
  GenerateLinkOutsideHalfspaceRationalFunction(
      const Eigen::VectorXd& q_star) const;

  // forward declare the tester class.
  friend class ConfigurationSpaceCollisionFreeRegionTester;

  RationalForwardKinematics rational_forward_kinematics_;
  // link_polytopes_[i] contains all the polytopes fixed to the i'th link
  std::vector<std::vector<ConvexPolytope>> link_polytopes_;
  // obstacles_[i] is the i'th polytope, fixed to the world.
  std::vector<ConvexPolytope> obstacles_;
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

/**
 * Generate the rational functions a_A.dot(p_AVi(t) - p_AC) <= 1 (or >= 1) which
 * represents that the link (whose vertex Vi has position p_AVi in frame A) is
 * on the negative (or positive, respectively) side of the hyperplane.
 * @param rational_forward_kinematics The utility class that computes the
 * position of Vi in A's frame as a rational function of t.
 * @param link_polytope The polytopic representation of the link collision
 * geometry, Vi is the i'th vertex of the polytope.
 * @param q_star The nominal configuration.
 * @param expressed_body_index Frame A in the documentation above. The body in
 * which the position is expressed in.
 * @param a_A The normal vector of the plane. This vector is expressed in frame
 * A.
 * @param p_AC The point within the interior of the negative side of the plane.
 * @param plane_side Whether the link is on the positive or the negative side of
 * the plane.
 * @return rational_fun rational_fun[i] should be non-negative to represent that
 * the vertiex i is on the correct side of the plane. rational_fun[i] =
 * a_A.dot(p_AVi(t) - p_AC) - 1 if @p plane_side = kPositive, rational_fun[i] =
 * 1 - a_A.dot(p_AVi(t) - p_AC) if @p plane_side = kNegative.
 */
std::vector<symbolic::RationalFunction>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope& link_polytope,
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side);

/**
 * Overloaded GenerateLinkOnOnseSideOfPlaneRationalFunction, except X_AB, i.e.,
 * the pose of the link polytope in the expressed_frame is given.
 * @param X_AB_multilinear The pose of the link polytope frame B in the
 * expressed body frame A. Note that this pose is a multilinear function of
 * sinθ and cosθ.
 */
std::vector<symbolic::RationalFunction>
GenerateLinkOnOneSideOfPlaneRationalFunction(
    const RationalForwardKinematics& rational_forward_kinematics,
    const ConvexPolytope& link_polytope,
    const RationalForwardKinematics::Pose<symbolic::Polynomial>&
        X_AB_multilinear,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& a_A,
    const Eigen::Ref<const Eigen::Vector3d>& p_AC, PlaneSide plane_side);

/** We need to verify that these polynomials are non-negative
 * Lagrangians l_lower(t) >= 0, l_upper(t) >= 0
 * Link polynomial p(t) - l_lower(t) * (t - t_lower) - l_upper(t)(t_upper_t) >=
 * 0
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
