#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"

namespace drake {
namespace multibody {
ConvexPolytope::ConvexPolytope(BodyIndex body_index,
                               const Eigen::Ref<const Eigen::Matrix3Xd>& p_BV)
    : ConvexGeometry(ConvexGeometryType::kPolytope, body_index), p_BV_{p_BV} {}

// All vertices should satisfy the constraint aᵀ(vᵢ-c) ≤ 1 ∀ i
void ConvexPolytope::AddInsideHalfspaceConstraint(
    const Eigen::Ref<const Eigen::Vector3d>& p_BC,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& n_B,
    solvers::MathematicalProgram* prog) const {
  const int num_vertices = p_BV_.cols();
  const Eigen::Matrix3Xd p_CV_B =
      p_BV_ - p_BC * Eigen::RowVectorXd::Ones(num_vertices);
  prog->AddLinearConstraint(
      p_CV_B.transpose(),
      Eigen::VectorXd::Constant(num_vertices,
                                -std::numeric_limits<double>::infinity()),
      Eigen::VectorXd::Ones(num_vertices), n_B);
}

void ConvexPolytope::AddPointInsideGeometryConstraint(
    const Eigen::Isometry3d& X_AB,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& p_AQ,
    solvers::MathematicalProgram* prog) const {
  // Add the slack variables representing the weight of each vertex.
  const int num_vertices = p_BV_.cols();
  auto w = prog->NewContinuousVariables(num_vertices);
  prog->AddBoundingBoxConstraint(0, 1, w);
  prog->AddLinearEqualityConstraint((X_AB * p_BV_) * w == p_AQ);
  prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(num_vertices), 1,
                                    w);
}

Cylinder::Cylinder(BodyIndex body_index,
                   const Eigen::Ref<const Eigen::Vector3d>& p_BO,
                   const Eigen::Ref<const Eigen::Vector3d>& a_B, double radius)
    : ConvexGeometry(ConvexGeometryType::kCylinder, body_index),
      p_BO_{p_BO},
      a_B_{a_B},
      radius_{radius},
      a_normalized_B_{a_B_.normalized()} {
  DRAKE_DEMAND(radius_ > 0);
  // First find a unit vector v that is not colinear with a, then set â₁ to be
  // parallel to v - vᵀa_normalized * a_normalized
  const Eigen::Vector3d v = std::abs(a_normalized_B_(0)) < 0.9
                                ? Eigen::Vector3d::UnitX()
                                : Eigen::Vector3d::UnitY();
  a_hat1_B_ = v - v.dot(a_normalized_B_) * a_normalized_B_;
  a_hat1_B_.normalize();
  a_hat2_B_ = a_normalized_B_.cross(a_hat1_B_);
}

void Cylinder::AddInsideHalfspaceConstraint(
    const Eigen::Ref<const Eigen::Vector3d>& p_BC,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& n_B,
    solvers::MathematicalProgram* prog) const {
  // Constraining that all points within the cylinder satisfies nᵀ(x-c) ≤ 1 is
  // equivalent to all points on the rim of the top/bottom circles satisfying
  // nᵀ(x-c) ≤ 1. This is again equivalent to
  // sqrt(nᵀ(I - aaᵀ/(aᵀa))n) ≤ (1 + nᵀ(c - o - a)) / r
  // sqrt(nᵀ(I - aaᵀ/(aᵀa))n) ≤ (1 + nᵀ(c - o + a)) / r
  // Both are Lorentz cone constraints on n

  // (I - aaᵀ/(aᵀa)) = PᵀP, where P = [â₁ᵀ;â₂ᵀ] is a 2 x 3 matrix, â₁, â₂ are
  // the two unit length vectors that are orthotonal to a, and also â₁ ⊥ â₂.
  // A_lorentz1 * n_B + b_lorentz1 = [(1 + nᵀ(c - o - a) / r; â₁ᵀn; â₂ᵀn];
  // A_lorentz2 * n_B + b_lorentz2 = [(1 + nᵀ(c - o + a) / r; â₁ᵀn; â₂ᵀn];
  Eigen::Matrix3d A_lorentz1, A_lorentz2;
  A_lorentz1.row(0) = (p_BC - p_BO_ - a_B_) / radius_;
  A_lorentz1.row(1) = a_hat1_B_.transpose();
  A_lorentz1.row(2) = a_hat2_B_.transpose();
  A_lorentz2 = A_lorentz1;
  A_lorentz2.row(0) = (p_BC - p_BO_ + a_B_) / radius_;
  Eigen::Vector3d b_lorentz1, b_lorentz2;
  b_lorentz1 << 1.0 / radius_, 0, 0;
  b_lorentz2 = b_lorentz1;
  prog->AddLorentzConeConstraint(A_lorentz1, b_lorentz1, n_B);
  prog->AddLorentzConeConstraint(A_lorentz2, b_lorentz2, n_B);
}

void Cylinder::AddPointInsideGeometryConstraint(
    const Eigen::Isometry3d& X_AB,
    const Eigen::Ref<const Vector3<symbolic::Variable>>& p_AQ,
    solvers::MathematicalProgram* prog) const {
  // Define a̅ = a_normalized
  // -|a| <= a̅ᵀ * OQ <= |a|
  // |(I - a̅a̅ᵀ) * OQ| <= r
  const Vector3<symbolic::Expression> p_BQ =
      X_AB.inverse() * p_AQ.cast<symbolic::Expression>();
  const Vector3<symbolic::Expression> p_OQ_B = p_BQ - p_BO_;
  const symbolic::Expression a_dot_OQ = a_normalized_B_.dot(p_OQ_B);
  const double a_norm = a_B_.norm();
  prog->AddLinearConstraint(a_dot_OQ, -a_norm, a_norm);
  Vector3<symbolic::Expression> lorentz_expr;
  lorentz_expr << radius_, a_hat1_B_.dot(p_OQ_B), a_hat2_B_.dot(p_OQ_B);
  prog->AddLorentzConeConstraint(lorentz_expr);
}
}  // namespace multibody
}  // namespace drake
