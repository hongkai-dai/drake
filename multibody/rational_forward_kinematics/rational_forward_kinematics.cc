#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics.h"

#include <queue>
#include <set>

#include "drake/multibody/rational_forward_kinematics/rational_forward_kinematics_internal.h"

namespace drake {
namespace multibody {
using symbolic::Expression;
using symbolic::Polynomial;
using symbolic::RationalFunction;

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
    const VectorX<symbolic::Polynomial>& one_plus_t_angles_squared,
    const VectorX<symbolic::Polynomial>& two_t_angles,
    const VectorX<symbolic::Polynomial>& one_minus_t_angles_squared,
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
    denominator *= one_plus_t_angles_squared[angle_index];
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
        numerator_monomial *= one_minus_t_angles_squared[angle_index];
      } else if (pair.first.degree(sin_delta(angle_index)) > 0) {
        numerator_monomial *= two_t_angles[angle_index];
      } else {
        numerator_monomial *= one_plus_t_angles_squared[angle_index];
      }
    }
    numerator += numerator_monomial;
  }

  *e_rational = RationalFunction(numerator, denominator);
}

void ReplaceCosAndSinWithRationalFunction(
    const symbolic::Polynomial& e_poly,
    const VectorX<symbolic::Variable>& cos_delta,
    const VectorX<symbolic::Variable>& sin_delta,
    const VectorX<symbolic::Variable>& t_angle, const symbolic::Variables& t,
    symbolic::RationalFunction* e_rational) {
  const symbolic::Monomial monomial_one{};
  VectorX<Polynomial> one_minus_t_square(t_angle.rows());
  VectorX<Polynomial> two_t(t_angle.rows());
  VectorX<Polynomial> one_plus_t_square(t_angle.rows());
  for (int i = 0; i < t_angle.rows(); ++i) {
    one_minus_t_square[i] = Polynomial(
        {{monomial_one, 1}, {symbolic::Monomial(t_angle(i), 2), -1}});
    two_t[i] = Polynomial({{symbolic::Monomial(t_angle(i), 1), 2}});
    one_plus_t_square[i] =
        Polynomial({{monomial_one, 1}, {symbolic::Monomial(t_angle(i), 2), 1}});
  }
  ReplaceCosAndSinWithRationalFunction(e_poly, cos_delta, sin_delta, t_angle, t,
                                       one_plus_t_square, two_t,
                                       one_minus_t_square, e_rational);
}

RationalForwardKinematics::RationalForwardKinematics(
    const MultibodyPlant<double>& plant)
    : plant_(plant) {
  int num_t = 0;
  const auto& tree = internal::GetInternalTree(plant_);
  for (BodyIndex body_index(1); body_index < plant_.num_bodies();
       ++body_index) {
    const auto& body_topology = tree.get_topology().get_body(body_index);
    const auto mobilizer =
        &(tree.get_mobilizer(body_topology.inboard_mobilizer));
    if (dynamic_cast<const internal::RevoluteMobilizer<double>*>(mobilizer) !=
        nullptr) {
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
      map_t_to_mobilizer_.emplace(t_(t_.rows() - 1).get_id(), mobilizer);
      map_mobilizer_to_t_index_.emplace(mobilizer, t_.rows() - 1);
    } else if (dynamic_cast<const internal::WeldMobilizer<double>*>(
                   mobilizer) != nullptr) {
    } else if (dynamic_cast<const internal::SpaceXYZMobilizer<double>*>(
                   mobilizer) != nullptr) {
      throw std::runtime_error("Gimbal joint has not been handled yet.");
    } else if (dynamic_cast<const internal::PrismaticMobilizer<double>*>(
                   mobilizer) != nullptr) {
      throw std::runtime_error("Prismatic joint has not been handled yet.");
    }
  }
  const symbolic::Monomial monomial_one{};
  one_plus_t_angles_squared_.resize(t_angles_.rows());
  two_t_angles_.resize(t_angles_.rows());
  one_minus_t_angles_squared_.resize(t_angles_.rows());
  for (int i = 0; i < t_angles_.rows(); ++i) {
    one_minus_t_angles_squared_(i) = Polynomial(
        {{monomial_one, 1}, {symbolic::Monomial(t_angles_(i), 2), -1}});
    two_t_angles_(i) = Polynomial({{symbolic::Monomial(t_angles_(i), 1), 2}});
    one_plus_t_angles_squared_(i) = Polynomial(
        {{monomial_one, 1}, {symbolic::Monomial(t_angles_(i), 2), 1}});
  }
  t_variables_ = symbolic::Variables(t_);
}

template <typename Scalar1, typename Scalar2>
void CalcChildPose(const Matrix3<Scalar2>& R_WP, const Vector3<Scalar2>& p_WP,
                   const Isometry3<double>& X_PF, const Isometry3<double>& X_MC,
                   const Matrix3<Scalar1>& R_FM, const Vector3<Scalar1>& p_FM,
                   Matrix3<Scalar2>* R_WC, Vector3<Scalar2>* p_WC) {
  // Frame F is the inboard frame (attached to the parent link), and frame
  // M is the outboard frame (attached to the child link).
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
        const Eigen::Ref<const Eigen::Vector3d>& axis_F,
        const Eigen::Isometry3d& X_PF, const Eigen::Isometry3d& X_MC,
        const Pose<T>& X_AP, double theta_star,
        const symbolic::Variable& cos_delta_theta,
        const symbolic::Variable& sin_delta_theta, Pose<T>* X_AC) const {
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
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, X_PF, X_MC, R_FM, p_FM, &(X_AC->R_AB),
                &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

template <typename T>
void RationalForwardKinematics::CalcLinkPoseWithWeldJoint(
    const Eigen::Isometry3d& X_FM, const Eigen::Isometry3d& X_PF,
    const Eigen::Isometry3d& X_MC, const Pose<T>& X_AP, Pose<T>* X_AC) const {
  const Matrix3<double> R_FM = X_FM.linear();
  const Vector3<double> p_FM = X_FM.translation();
  CalcChildPose(X_AP.R_AB, X_AP.p_AB, X_PF, X_MC, R_FM, p_FM, &(X_AC->R_AB),
                &(X_AC->p_AB));
  X_AC->frame_A_index = X_AP.frame_A_index;
}

std::vector<RationalForwardKinematics::Pose<Polynomial>>
RationalForwardKinematics::CalcLinkPosesAsMultilinearPolynomials(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index) const {
  // We need to change the frame in which the link pose is expressed. To do so,
  // we will reshuffle the tree structure in the MultibodyPlant, namely if we
  // express the pose in link A's frame, we will treat A as the root of the
  // reshuffled tree. From link A, we propogate to its child links and so on, so
  // as to compute the pose of all links in A's frame.
  std::vector<RationalForwardKinematics::Pose<Polynomial>> poses_poly(
      plant_.num_bodies());
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
  // In the reshuffled tree, the expressed body is the root. We will compute the
  // pose of each link w.r.t this root link.
  internal::ReshuffledBody reshuffled_expressed_body(expressed_body_index,
                                                     nullptr, nullptr);
  internal::ReshuffleKinematicsTree(plant_, &reshuffled_expressed_body);
  // Now do a breadth-first-search on this reshuffled tree, to compute the pose
  // of each link w.r.t the root.
  std::queue<internal::ReshuffledBody*> bfs_queue;
  bfs_queue.push(&reshuffled_expressed_body);
  while (!bfs_queue.empty()) {
    const internal::ReshuffledBody* reshuffled_body = bfs_queue.front();
    if (reshuffled_body->parent != nullptr) {
      const internal::Mobilizer<double>* mobilizer = reshuffled_body->mobilizer;
      // if reshuffled_body was a child of reshuffled_body->parent in the
      // original tree before reshuffling, then is_order_reversed = false;
      // otherwise it is true.
      // If we denote the frames related to the two adjacent bodies connected
      // by a mobilizer in the original tree as P->F->M->C, then after reversing
      // the order, the new frames should reverse the order, namely
      // P' = C, F' = M, M' = F, C' = P, and hence we know that
      // X_P'F' = X_MC.inverse()
      // X_F'M' = X_FM.inverse()
      // X_M'C' = X_PF.inverse()
      const bool is_order_reversed =
          mobilizer->inboard_body().index() == reshuffled_body->body_index;
      if (dynamic_cast<const internal::RevoluteMobilizer<double>*>(mobilizer) !=
          nullptr) {
        // A revolute joint.
        const internal::RevoluteMobilizer<double>* revolute_mobilizer =
            dynamic_cast<const internal::RevoluteMobilizer<double>*>(mobilizer);
        const int t_index = map_mobilizer_to_t_index_.at(mobilizer);
        const int q_index = revolute_mobilizer->position_start_in_q();
        const int t_angle_index = map_t_index_to_angle_index_.at(t_index);
        Eigen::Vector3d axis_F;
        Eigen::Isometry3d X_PF, X_MC;
        if (!is_order_reversed) {
          axis_F = revolute_mobilizer->revolute_axis();
          const Frame<double>& frame_F = mobilizer->inboard_frame();
          const Frame<double>& frame_M = mobilizer->outboard_frame();
          X_PF = frame_F.GetFixedPoseInBodyFrame();
          X_MC = frame_M.GetFixedPoseInBodyFrame();
        } else {
          // By negating the revolute axis, we know that R(a, θ)⁻¹ = R(-a, θ)
          axis_F = -revolute_mobilizer->revolute_axis();
          X_PF =
              mobilizer->outboard_frame().GetFixedPoseInBodyFrame().inverse();
          X_MC = mobilizer->inboard_frame().GetFixedPoseInBodyFrame().inverse();
        }
        CalcLinkPoseAsMultilinearPolynomialWithRevoluteJoint(
            axis_F, X_PF, X_MC, poses_poly[reshuffled_body->parent->body_index],
            q_star(q_index), cos_delta_(t_angle_index),
            sin_delta_(t_angle_index),
            &(poses_poly[reshuffled_body->body_index]));
      } else if (dynamic_cast<const internal::PrismaticMobilizer<double>*>(
                     mobilizer) != nullptr) {
        throw std::runtime_error(
            "RationalForwardKinematics: prismatic joint is not supported yet.");
      } else if (dynamic_cast<const internal::WeldMobilizer<double>*>(
                     mobilizer) != nullptr) {
        const internal::WeldMobilizer<double>* weld_mobilizer =
            dynamic_cast<const internal::WeldMobilizer<double>*>(mobilizer);
        Eigen::Isometry3d X_FM, X_PF, X_MC;
        if (!is_order_reversed) {
          X_FM = weld_mobilizer->get_X_FM();
          X_PF = mobilizer->inboard_frame().GetFixedPoseInBodyFrame();
          X_MC = mobilizer->outboard_frame().GetFixedPoseInBodyFrame();
        } else {
          X_FM = weld_mobilizer->get_X_FM().inverse();
          X_PF =
              mobilizer->outboard_frame().GetFixedPoseInBodyFrame().inverse();
          X_MC = mobilizer->inboard_frame().GetFixedPoseInBodyFrame().inverse();
        }
        CalcLinkPoseWithWeldJoint(
            X_FM, X_PF, X_MC, poses_poly[reshuffled_body->parent->body_index],
            &(poses_poly[reshuffled_body->body_index]));
      } else if (dynamic_cast<const internal::SpaceXYZMobilizer<double>*>(
                     mobilizer) != nullptr) {
        throw std::runtime_error("Gimbal joint has not been handled yet.");
      } else if (dynamic_cast<
                     const internal::QuaternionFloatingMobilizer<double>*>(
                     mobilizer) != nullptr) {
        throw std::runtime_error(
            "Free floating joint has not been handled yet.");
      } else {
        throw std::runtime_error(
            "RationalForwardKinematics: Can't handle this mobilizer.");
      }
    }
    bfs_queue.pop();
    for (const auto& reshuffled_child : reshuffled_body->children) {
      bfs_queue.push(reshuffled_child.get());
    }
  }
  return poses_poly;
}

RationalFunction
RationalForwardKinematics::ConvertMultilinearPolynomialToRationalFunction(
    const symbolic::Polynomial& e) const {
  RationalFunction e_rational;
  ReplaceCosAndSinWithRationalFunction(
      e, cos_delta_, sin_delta_, t_angles_, t_variables_,
      one_plus_t_angles_squared_, two_t_angles_, one_minus_t_angles_squared_,
      &e_rational);
  return e_rational;
}

std::vector<RationalForwardKinematics::Pose<RationalFunction>>
RationalForwardKinematics::CalcLinkPoses(
    const Eigen::Ref<const Eigen::VectorXd>& q_star,
    BodyIndex expressed_body_index) const {
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
  std::vector<Pose<RationalFunction>> poses(plant_.num_bodies());
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
  for (BodyIndex body_index{1}; body_index < plant_.num_bodies();
       ++body_index) {
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
      poses[body_index].frame_A_index = expressed_body_index;
    }
  }
  return poses;
}
}  // namespace multibody
}  // namespace drake
