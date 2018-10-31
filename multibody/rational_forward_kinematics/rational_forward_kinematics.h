#pragma once

#include <unordered_map>
#include <vector>

#include "drake/common/symbolic.h"
#include "drake/multibody/multibody_tree/multibody_tree.h"
#include "drake/multibody/multibody_tree/prismatic_mobilizer.h"
#include "drake/multibody/multibody_tree/quaternion_floating_mobilizer.h"
#include "drake/multibody/multibody_tree/revolute_mobilizer.h"
#include "drake/multibody/multibody_tree/space_xyz_mobilizer.h"
#include "drake/multibody/multibody_tree/weld_mobilizer.h"

namespace drake {
namespace multibody {
/**
 * We can represent the pose (position, orientation) of each link, as rational
 * functions, namely n(t) / d(t) where both the numerator n(t) and denominator
 * d(t) are polynomials of t, and t is some variable related to the generalized
 * position.
 *
 * One example is that for a rotation matrix with angle θ and axis a, the
 * rotation matrix can be written as I + sinθ A + (1-cosθ) A², where A is the
 * skew-symmetric matrix from axis a. We can use the half-angle formulat to
 * substitute the trigonometric function sinθ and cosθ as
 * cosθ = cos(θ*+Δθ) = cosθ*cosΔθ - sinθ*sinΔθ
 *      = (1-t²)/(1+t²) cosθ*- 2t/(1+t²) sinθ*     (1)
 * sinθ = sin(θ*+Δθ) = sinθ*cosΔθ - cosθ*sinΔθ
 *      = (1-t²)/(1+t²) sinθ*- 2t/(1+t²) cosθ*     (2)
 * where θ = θ*+Δθ, and t = tan(Δθ/2). θ* is some given angle.
 * With (1) and (2), both sinθ and cosθ are written as a rational function of t.
 * Thus the rotation matrix can be written as rational functions of t.
 */
class RationalForwardKinematics {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(RationalForwardKinematics)

  template <typename T>
  struct Pose {
    Vector3<T> p_AB;
    Matrix3<T> R_AB;
    int frame_A_index;
  };

  struct LinkPoints {
    LinkPoints(int m_link_index,
               const Eigen::Ref<const Eigen::Matrix3Xd>& m_p_BQ)
        : link_index{m_link_index}, p_BQ{m_p_BQ} {}
    int link_index;
    // The position of the points Q in the link frame B.
    Eigen::Matrix3Xd p_BQ;
  };

  explicit RationalForwardKinematics(const MultibodyTree<double>& tree);

  /** Compute the pose of each link as fractional functions of t.
   * We will set up the indeterminates t also.
   * A revolute joint requires a single t, where t = tan(Δθ/2).
   * A prismatic joint requires a single t, where t = Δd, d being the
   * translational motion of the prismatic joint.
   * A free-floating joint requires 12 t, 3 for position, and 9 for the rotation
   * matrix.
   * A gimbal joint requires 9 t, for the rotation matrix.
   * @param q_star The nominal posture around which we compute the link poses.
   * @param expressed_body_index The pose of each link is expressed in this
   * body.
   */
  std::vector<Pose<symbolic::RationalFunction>> CalcLinkPoses(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      int expressed_body_index) const;

  std::vector<Pose<symbolic::Polynomial>> CalcLinkPosesAsMultilinearPolynomials(
      const Eigen::Ref<const Eigen::VectorXd>& q_star,
      int expressed_body_index) const;

  symbolic::RationalFunction ConvertMultilinearPolynomialToRationalFunction(
      const symbolic::Polynomial e) const;

  const MultibodyTree<double>& tree() const { return tree_; }

  const VectorX<symbolic::Variable>& t() const { return t_; }

 private:
  // Compute the pose of the link, connected to its parent link through a
  // revolute joint.
  // We will first compute the link pose as multilinear polynomials, with
  // indeterminates cos_delta and sin_delta, representing cos(Δθ) and sin(Δθ)
  // respectively. We will then replace cos_delta and sin_delta in the link
  // pose with rational functions (1-t^2)/(1+t^2) and 2t/(1+t^2) respectively.
  // The reason why we don't use RationalFunction directly, is that currently
  // our rational function can't find the common factor in the denominator,
  // namely the sum between rational functions p1(x) / (q1(x) * r(x)) + p2(x) /
  // r(x) is computed as (p1(x) * r(x) + p2(x) * q1(x) * r(x)) / (q1(x) * r(x) *
  // r(x)), without handling the common factor r(x) in the denominator.
  template <typename T>
  void CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
      const RevoluteMobilizer<double>* revolute_mobilizer, const Pose<T>& X_AP,
      double theta_star, const symbolic::Variable& cos_delta,
      const symbolic::Variable& sin_delta, Pose<T>* X_AC) const;

  // Compute the pose of the link, connected to its parent link through a
  // weld joint.
  template <typename T>
  void CalcLinkPoseWithWeldJoint(const WeldMobilizer<double>* weld_mobilizer,
                                 const Pose<T>& X_AP, Pose<T>* X_AC) const;

  const MultibodyTree<double>& tree_;
  // The variables used in computing the pose as rational functions. t_ are the
  // indeterminates in the rational functions.
  VectorX<symbolic::Variable> t_;

  // The variables used to represent tan(θ / 2).
  VectorX<symbolic::Variable> t_angles_;

  VectorX<symbolic::Variable> cos_delta_;
  VectorX<symbolic::Variable> sin_delta_;
  std::unordered_map<int, int> map_t_index_to_angle_index_;
  std::unordered_map<int, int> map_angle_index_to_t_index_;
  symbolic::Variables t_variables_;
};

/** If e is a multilinear polynomial of cos_delta and sin_delta, and no
 * cos_delta(i) and sin_delta(i) appear in the same monomial, then we replace
 * cos_delta(i) with (1-t_angles(i)^2)/(1+t_angles(i)^2), and sin_delta(i)
 * with 2t_angles(i)/(1+t_angles(i)^2), and get a rational polynomial of t.
 */
void ReplaceCosAndSinWithRationalFunction(
    const symbolic::Polynomial& e, const VectorX<symbolic::Variable>& cos_delta,
    const VectorX<symbolic::Variable>& sin_delta,
    const VectorX<symbolic::Variable>& t_angles, const symbolic::Variables& t,
    symbolic::RationalFunction* e_rational);
}  // namespace multibody
}  // namespace drake
