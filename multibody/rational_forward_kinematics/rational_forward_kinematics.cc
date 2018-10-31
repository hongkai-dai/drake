#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

#include <set>

namespace drake {
namespace multibody {
using symbolic::Expression;
using symbolic::Polynomial;
using symbolic::RationalFunction;
RationalForwardKinematics::RationalForwardKinematics(
    const MultibodyTree<double>& tree)
    : tree_(tree) {
  int num_t = 0;
  for (BodyIndex body_index(1); body_index < tree_.num_bodies(); ++body_index) {
    const BodyTopology& body_topology =
        tree_.get_topology().get_body(body_index);
    const auto mobilizer =
        &(tree_.get_mobilizer(body_topology.inboard_mobilizer));
    if (dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer) != nullptr) {
      const symbolic::Variable t_angle("t[" + std::to_string(num_t) + "]");
      t_.conservativeResize(t_.rows() + 1);
      t_angles_.conservativeResize(t_angles_.rows() + 1);
      cos_delta_.conservativeResize(cos_delta_.rows() + 1);
      sin_delta_.conservativeResize(sin_delta_.rows() + 1);
      t_(t_.rows() - 1) = t_angle;
      t_angles_(t_angles_.rows() - 1) = t_angle;
      cos_delta_(cos_delta_.rows() - 1) = symbolic::Variable(
          "cos_delta[" + std::to_string(cos_delta_.rows() - 1) + "]");
      sin_delta_(sin_delta_.rows() - 1) = symbolic::Variable(
          "sin_delta[" + std::to_string(cos_delta_.rows() - 1) + "]");
      num_t += 1;
      map_t_index_to_angle_index_.emplace(t_.rows() - 1, t_angles_.rows() - 1);
      map_angle_index_to_t_index_.emplace(t_angles_.rows() - 1, t_.rows() - 1);
    } else if (dynamic_cast<const WeldMobilizer<double>*>(mobilizer) !=
               nullptr) {
    } else if (dynamic_cast<const SpaceXYZMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Gimbal joint has not been handled yet.");
    } else if (dynamic_cast<const PrismaticMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Prismatic joint has not been handled yet.");
    }
  }
  t_variables_ = symbolic::Variables(t_);
}

template <typename Scalar1, typename Scalar2>
void CalcChildPose(const Matrix3<Scalar2>& R_WP, const Vector3<Scalar2>& p_WP,
                   const Mobilizer<double>& mobilizer,
                   const Matrix3<Scalar1>& R_FM, const Vector3<Scalar1>& p_FM,
                   Matrix3<Scalar2>* R_WC, Vector3<Scalar2>* p_WC) {
  // Frame F is the inboard frame (attached to the parent link), and frame
  // M is the outboard frame (attached to the child link).
  const Frame<double>& frame_F = mobilizer.inboard_frame();
  const Frame<double>& frame_M = mobilizer.outboard_frame();
  const Isometry3<double> X_PF = frame_F.GetFixedPoseInBodyFrame();
  const Isometry3<double> X_MC = frame_M.GetFixedPoseInBodyFrame();
  const Matrix3<Scalar2> R_WF = R_WP * X_PF.linear();
  const Vector3<Scalar2> p_WF = R_WP * X_PF.translation() + p_WP;
  const Matrix3<Scalar2> R_WM = R_WF * R_FM;
  const Vector3<Scalar2> p_WM = R_WF * p_FM + p_WF;
  const Matrix3<double> R_MC = X_MC.linear();
  const Vector3<double> p_MC = X_MC.translation();
  *R_WC = R_WM * R_MC;
  *p_WC = R_WM * p_MC + p_WM;
}

template <typename T>
void RationalForwardKinematics::
    CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
        const RevoluteMobilizer<double>* revolute_mobilizer,
        const Pose<T>& X_AP, double theta_star,
        const symbolic::Variable& cos_delta_theta,
        const symbolic::Variable& sin_delta_theta, Pose<T>* X_AC) const {
  const Eigen::Vector3d& axis_F = revolute_mobilizer->revolute_axis();
  // clang-format off
      const Eigen::Matrix3d A_F =
          (Eigen::Matrix3d() << 0, -axis_F(2), axis_F(1),
                                axis_F(2), 0, -axis_F(0),
                                -axis_F(1), axis_F(0), 0).finished();
  // clang-format on
  const symbolic::Variables cos_sin_delta({cos_delta_theta, sin_delta_theta});
  const double cos_theta_star = cos(theta_star);
  const double sin_theta_star = sin(theta_star);
  const Polynomial cos_angle(
      {{symbolic::Monomial(cos_delta_theta, 1), cos_theta_star},
       {symbolic::Monomial(sin_delta_theta, 1), -sin_theta_star}});
  const Polynomial sin_angle(
      {{symbolic::Monomial(cos_delta_theta, 1), sin_theta_star},
       {symbolic::Monomial(sin_delta_theta, 1), cos_theta_star}});
  // Frame F is the inboard frame (attached to the parent link), and frame
  // M is the outboard frame (attached to the child link).
  const Matrix3<symbolic::Polynomial> R_FM = Eigen::Matrix3d::Identity() +
                                             sin_angle * A_F +
                                             (1 - cos_angle) * A_F * A_F;
  const symbolic::Polynomial poly_zero{};
  const Vector3<symbolic::Polynomial> p_FM(poly_zero, poly_zero, poly_zero);
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, *revolute_mobilizer, R_FM, p_FM,
                &(X_AC->R_AB), &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

template <typename T>
void RationalForwardKinematics::CalcLinkPoseWithWeldJoint(
    const WeldMobilizer<double>* weld_mobilizer, const Pose<T>& X_AP,
    Pose<T>* X_AC) const {
  const Isometry3<double> X_FM = weld_mobilizer->get_X_FM();
  const Matrix3<double> R_FM = X_FM.linear();
  const Vector3<double> p_FM = X_FM.translation();
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, *weld_mobilizer, R_FM, p_FM,
                &(X_AC->R_AB), &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

std::vector<RationalForwardKinematics::Pose<Polynomial>>
RationalForwardKinematics::CalcLinkPosesAsMultilinearPolynomials(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    int expressed_body_index) const {
  // TODO(hongkai.dai): support expressed frame not in the world.
  if (expressed_body_index != 0) {
    throw std::runtime_error(
        "Not implemented yet. The expressed frame needs to be the world "
        "frame.");
  }
  std::vector<RationalForwardKinematics::Pose<Polynomial>> poses_poly(
      tree_.num_bodies());
  const Polynomial poly_zero{};
  const Polynomial poly_one{1};
  // clang-format off
  poses_poly[expressed_body_index].R_AB <<
    poly_one, poly_zero, poly_zero,
    poly_zero, poly_one, poly_zero,
    poly_zero, poly_zero, poly_one;
  poses_poly[expressed_body_index].p_AB << poly_zero, poly_zero, poly_zero;
  // clang-format on
  poses_poly[expressed_body_index].frame_A_index = expressed_body_index;
  for (BodyIndex body_index(1); body_index < tree_.num_bodies(); ++body_index) {
    const BodyTopology& body_topology =
        tree_.get_topology().get_body(body_index);
    const auto mobilizer =
        &(tree_.get_mobilizer(body_topology.inboard_mobilizer));
    const BodyIndex parent_index =
        tree_.get_topology().get_body(body_index).parent_body;
    if (dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer) != nullptr) {
      const RevoluteMobilizer<double>* revolute_mobilizer =
          dynamic_cast<const RevoluteMobilizer<double>*>(mobilizer);
      const int q_index = revolute_mobilizer->get_topology().positions_start;
      const int t_angle_index = map_t_index_to_angle_index_.at(q_index);
      CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
          revolute_mobilizer, poses_poly[parent_index], q_star(q_index),
          cos_delta_(t_angle_index), sin_delta_(t_angle_index),
          &(poses_poly[body_index]));
    } else if (dynamic_cast<const PrismaticMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Prismatic joint has not been handled yet.");
    } else if (dynamic_cast<const WeldMobilizer<double>*>(mobilizer) !=
               nullptr) {
      const WeldMobilizer<double>* weld_mobilizer =
          dynamic_cast<const WeldMobilizer<double>*>(mobilizer);
      CalcLinkPoseWithWeldJoint(weld_mobilizer, poses_poly[parent_index],
                                &(poses_poly[body_index]));
    } else if (dynamic_cast<const SpaceXYZMobilizer<double>*>(mobilizer) !=
               nullptr) {
      throw std::runtime_error("Gimbal joint has not been handled yet.");
    } else if (dynamic_cast<const QuaternionFloatingMobilizer<double>*>(
                   mobilizer) != nullptr) {
      throw std::runtime_error("Free floating joint has not been handled yet.");
    } else {
      throw std::runtime_error(
          "RationalForwardKinematics: Can't handle this mobilizer.");
    }
  }
  return poses_poly;
}

RationalFunction
RationalForwardKinematics::ConvertMultilinearPolynomialToRationalFunction(
    const symbolic::Polynomial e) const {
  RationalFunction e_rational;
  ReplaceCosAndSinWithRationalFunction(e, cos_delta_, sin_delta_, t_angles_,
                                       t_variables_, &e_rational);
  return e_rational;
}

std::vector<RationalForwardKinematics::Pose<RationalFunction>>
RationalForwardKinematics::CalcLinkPoses(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    int expressed_body_index) const {
  // We will first compute the link pose as multilinear polynomials, with
  // indeterminates cos_delta and sin_delta, representing cos(Δθ) and
  // sin(Δθ)
  // respectively. We will then replace cos_delta and sin_delta in the link
  // pose with rational functions (1-t^2)/(1+t^2) and 2t/(1+t^2)
  // respectively.
  // The reason why we don't use RationalFunction directly, is that
  // currently
  // our rational function can't find the common factor in the denominator,
  // namely the sum between rational functions p1(x) / (q1(x) * r(x)) +
  // p2(x) /
  // r(x) is computed as (p1(x) * r(x) + p2(x) * q1(x) * r(x)) / (q1(x) *
  // r(x) *
  // r(x)), without handling the common factor r(x) in the denominator.
  const RationalFunction rational_zero(0);
  const RationalFunction rational_one(1);
  std::vector<Pose<RationalFunction>> poses(tree_.num_bodies());
  // We denote the expressed body frame as A.
  poses[expressed_body_index].p_AB << rational_zero, rational_zero,
      rational_zero;
  // clang-format off
  poses[expressed_body_index].R_AB <<
    rational_one, rational_zero, rational_zero,
    rational_zero, rational_one, rational_zero,
    rational_zero, rational_zero, rational_one;
  // clang-format on
  poses[expressed_body_index].frame_A_index = expressed_body_index;
  std::vector<Pose<Polynomial>> poses_poly =
      CalcLinkPosesAsMultilinearPolynomials(q_star, expressed_body_index);
  for (BodyIndex body_index{1}; body_index < tree_.num_bodies(); ++body_index) {
    // Now convert the multilinear polynomial of cos and sin to rational
    // function of t.
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        poses[body_index].R_AB(i, j) =
            ConvertMultilinearPolynomialToRationalFunction(
                poses_poly[body_index].R_AB(i, j));
      }
      poses[body_index].p_AB(i) =
          ConvertMultilinearPolynomialToRationalFunction(
              poses_poly[body_index].p_AB(i));
    }
  }
  return poses;
}

bool CheckPolynomialIndeterminatesAreCosSinDelta(
    const Polynomial& e_poly, const VectorX<symbolic::Variable>& cos_delta,
    const VectorX<symbolic::Variable>& sin_delta) {
  VectorX<symbolic::Variable> cos_sin_delta(cos_delta.rows() +
                                            sin_delta.rows());
  cos_sin_delta << cos_delta, sin_delta;
  const symbolic::Variables cos_sin_delta_variables(cos_sin_delta);
  return e_poly.indeterminates().IsSubsetOf(cos_sin_delta_variables);
}

void ReplaceCosAndSinWithRationalFunction(
    const symbolic::Polynomial& e_poly,
    const VectorX<symbolic::Variable>& cos_delta,
    const VectorX<symbolic::Variable>& sin_delta,
    const VectorX<symbolic::Variable>& t_angle, const symbolic::Variables&,
    symbolic::RationalFunction* e_rational) {
  DRAKE_DEMAND(cos_delta.rows() == sin_delta.rows());
  DRAKE_DEMAND(cos_delta.rows() == t_angle.rows());
  DRAKE_DEMAND(CheckPolynomialIndeterminatesAreCosSinDelta(e_poly, cos_delta,
                                                           sin_delta));
  // First find the angles whose cos or sin appear in the polynomial. This
  // will determine the denominator of the rational function.
  std::set<int> angle_indices;
  for (const auto& pair : e_poly.monomial_to_coefficient_map()) {
    // Also check that this monomial can't contain both cos_delta(i) and
    // sin_delta(i).
    for (int i = 0; i < cos_delta.rows(); ++i) {
      const int angle_degree =
          pair.first.degree(cos_delta(i)) + pair.first.degree(sin_delta(i));
      DRAKE_DEMAND(angle_degree <= 1);
      if (angle_degree == 1) {
        angle_indices.insert(i);
      }
    }
  }
  if (angle_indices.empty()) {
    *e_rational = RationalFunction(e_poly);
    return;
  }
  const symbolic::Monomial monomial_one{};
  symbolic::Polynomial denominator{1};
  for (int angle_index : angle_indices) {
    // denominator *= (1 + t_angle(angle_index)^2)
    const Polynomial one_plus_t_square(
        {{monomial_one, 1}, {symbolic::Monomial(t_angle(angle_index), 2), 1}});
    denominator *= one_plus_t_square;
  }
  symbolic::Polynomial numerator{};
  for (const auto& pair : e_poly.monomial_to_coefficient_map()) {
    // If the monomial contains cos_delta(i), then replace cos_delta(i) with
    // 1 - t_angle(i) * t_angle(i).
    // If the monomial contains sin_delta(i), then replace sin_delta(i) with
    // 2 * t_angle(i).
    // Otherwise, multiplies with 1 + t_angle(i) * t_angle(i)

    // We assume that t pair.second doesn't contain any indeterminates. So
    // pair.second is the coefficient.
    Polynomial numerator_monomial{{{monomial_one, pair.second}}};
    for (int angle_index : angle_indices) {
      if (pair.first.degree(cos_delta(angle_index)) > 0) {
        const Polynomial one_minus_t_square(
            {{monomial_one, 1},
             {symbolic::Monomial{t_angle(angle_index), 2}, -1}});
        numerator_monomial *= one_minus_t_square;
      } else if (pair.first.degree(sin_delta(angle_index)) > 0) {
        const Polynomial two_t(
            {{symbolic::Monomial(t_angle(angle_index), 1), 2}});
        numerator_monomial *= two_t;
      } else {
        const Polynomial one_plus_t_square(
            {{monomial_one, 1},
             {symbolic::Monomial(t_angle(angle_index), 2), 1}});
        numerator_monomial *= one_plus_t_square;
      }
    }
    numerator += numerator_monomial;
  }

  *e_rational = RationalFunction(numerator, denominator);
}
}  // namespace multibody
}  // namespace drake
