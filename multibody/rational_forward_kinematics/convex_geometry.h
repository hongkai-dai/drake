#include "drake/solvers/mathematical_program.h"
namespace drake {
namespace multibody {
enum class ConvexGeometryType {
  kPolytope,
  kCylinder,
  kEllipsoid,
};

class ConvexGeometry {
 public:
  ConvexGeometryType type() const { return type_; }

  int body_index() const { return body_index_; }

  virtual ~ConvexGeometry() {}

  /** Add the constraint that the geometry is in the inner side of the halfspace
   * nᵀ(x-c) ≤ 1. Namely nᵀ(x-c) ≤ 1 ∀ x within the convex geometry.
   * Here we assume that n is expressed in the same body frame as the geometry.
   */
  virtual void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& n_B,
      solvers::MathematicalProgram* prog) const = 0;

  /**
   * Adds the constraint that a point Q is within the convex geometry.
   * @param X_AB The pose of the body B (to which the geometry is attached) in
   * a frame A.
   * @param p_AQ The decision variables representing the position of point Q in
   * frame A.
   */
  virtual void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& p_AQ,
      solvers::MathematicalProgram* prog) const = 0;

 protected:
  ConvexGeometry(ConvexGeometryType type, int body_index)
      : type_{type}, body_index_{body_index} {}

 private:
  const ConvexGeometryType type_;
  // The index of the body that this geometry is attached to.
  const int body_index_;
};

class ConvexPolytope : public ConvexGeometry {
 public:
  ConvexPolytope(int body_index,
                 const Eigen::Ref<const Eigen::Matrix3Xd>& vertices);

  const Eigen::Matrix3Xd p_BV() const { return p_BV_; }

  void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& n_B,
      solvers::MathematicalProgram* prog) const override;

  void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& p_AQ,
      solvers::MathematicalProgram* prog) const override;

 private:
  // position of all vertices V in the body frame B.
  const Eigen::Matrix3Xd p_BV_;
};

class Cylinder : public ConvexGeometry {
 public:
  /**
   * @param p_BO The position of cylinder center O in the body frame B.
   * @param a_B The cylinder axis a expressed in the body frame B. The height
   * of the cylinder is 2 * |a_B|
   * @param radius The radius of the cylinder.
   */
  Cylinder(int body_index, const Eigen::Ref<const Eigen::Vector3d>& p_BO,
           const Eigen::Ref<const Eigen::Vector3d>& a_B, double radius);

  void AddInsideHalfspaceConstraint(
      const Eigen::Ref<const Eigen::Vector3d>& p_BC,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& n_B,
      solvers::MathematicalProgram* prog) const override;

  void AddPointInsideGeometryConstraint(
      const Eigen::Isometry3d& X_AB,
      const Eigen::Ref<const Vector3<symbolic::Variable>>& p_AQ,
      solvers::MathematicalProgram* prog) const override;

  const Eigen::Vector3d& p_BO() const { return p_BO_; }

  const Eigen::Vector3d& a_B() const { return a_B_; }

  double radius() const { return radius_; }

 private:
  // The position of the cylinder center O in the body frame B.
  const Eigen::Vector3d p_BO_;
  // The axis (unnormalized) of the cylinder in the body frame B . The height of
  // the cylinder is 2 * |a_B_|.
  const Eigen::Vector3d a_B_;
  // The radius of the cylinder.
  const double radius_;
  // a_B_ / |a_B_|
  const Eigen::Vector3d a_normalized_B_;
  // â₁, â₂ are the two unit length vectors that are orthotonal to a, and also
  // â₁ ⊥ â₂.
  Eigen::Vector3d a_hat1_B_, a_hat2_B_;
};
}  // namespace multibody
}  // namespace drake
