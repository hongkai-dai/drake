#include "drake/multibody/rational_forward_kinematics/convex_geometry.h"

#include <gtest/gtest.h>

namespace drake {
namespace multibody {
template <typename C>
bool CheckSatisfied(const solvers::Binding<C>& constraint,
                    const Eigen::Ref<const Eigen::VectorXd>& x, double tol) {
  return std::dynamic_pointer_cast<solvers::Constraint>(constraint.evaluator())
      ->CheckSatisfied(x, tol);
}

class CylinderTest : public ::testing::Test {
 public:
  CylinderTest()
      : p_BO_{0.2, 0.3, 0.4},
        a_B_{0.1, -1.2, 0.3},
        radius_{1.2},
        a_B_normalized_{a_B_.normalized()} {
    // Find the two unit vectors a_hat1, a_hat2 that are orthogonal to a_B, and
    // also a_hat1 is perpendicular to a_hat2.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(
        Eigen::Matrix3d::Identity() -
        a_B_normalized_ * a_B_normalized_.transpose());
    a_hat1_B_ = es.eigenvectors().col(1);
    a_hat2_B_ = es.eigenvectors().col(2);
    // Check if a_hat1, a_hat2 and a_B are orthogonal to each other.
    EXPECT_NEAR(a_hat1_B_.dot(a_B_), 0, 1E-6);
    EXPECT_NEAR(a_hat2_B_.dot(a_B_), 0, 1E-6);
    EXPECT_NEAR(a_hat2_B_.dot(a_hat1_B_), 0, 1E-6);
  }

  // p_BQ1 contains the extreme points on the top circle. p_BQ2 contains the
  // extreme points on the bottom circle.
  void GenerateCylinderExtremPoints(std::vector<Eigen::Vector3d>* p_BQ1,
                                    std::vector<Eigen::Vector3d>* p_BQ2) const {
    const Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(2, 0, 2 * M_PI);
    for (int i = 0; i < theta.size(); ++i) {
      // point Q1 is on the top circle rim, Q2 is on the bottom circle rim.
      p_BQ1->push_back(p_BO_ + a_B_ +
                       radius_ * (a_hat1_B_ * std::cos(theta(i)) +
                                  a_hat2_B_ * sin(theta(i))));
      p_BQ2->push_back(p_BO_ - a_B_ +
                       radius_ * (a_hat1_B_ * std::cos(theta(i)) +
                                  a_hat2_B_ * sin(theta(i))));
    }
  }

 protected:
  // Center of the cylinder
  const Eigen::Vector3d p_BO_;
  // axis of the cylinder
  const Eigen::Vector3d a_B_;
  // radius of the cylinder
  const double radius_;
  const Eigen::Vector3d a_B_normalized_;
  // a_hat1 and a_hat2 are two unit vectors, that they are perpendicular to a,
  // and also perpendicular to each other.
  Eigen::Vector3d a_hat1_B_, a_hat2_B_;
};

TEST_F(CylinderTest, AddInsideHalfspaceConstraint) {
  const Cylinder cylinder(0, p_BO_, a_B_, radius_);
  solvers::MathematicalProgram prog;
  auto n_B = prog.NewContinuousVariables<3>();
  const Eigen::Vector3d p_BC = p_BO_ + 0.1 * a_B_;
  cylinder.AddInsideHalfspaceConstraint(p_BC, n_B, &prog);
  EXPECT_TRUE(prog.linear_constraints().empty());
  EXPECT_EQ(prog.lorentz_cone_constraints().size(), 2);

  // Now find some halfspace nᵀ(x - c) = 1 that are tangential to the cylinder
  // at the rim of top/bottom circles.
  std::vector<Eigen::Vector3d> p_BQ1, p_BQ2;
  GenerateCylinderExtremPoints(&p_BQ1, &p_BQ2);
  const Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(2, 0, 2 * M_PI);
  for (int i = 0; i < theta.size(); ++i) {
    // point Q1 is on the top circle rim, Q2 is on the bottom circle rim.
    const Eigen::Vector3d p_CQ1_B = p_BQ1[i] - p_BC;
    const Eigen::Vector3d p_CQ2_B = p_BQ2[i] - p_BC;
    const Eigen::Vector3d n1_B = p_CQ1_B / p_CQ1_B.squaredNorm();
    const Eigen::Vector3d n2_B = p_CQ2_B / p_CQ2_B.squaredNorm();
    const double tol{1E-6};
    // 0.99 * n1_B or 0.99 * n2_B should be valid hyperplane normals.
    EXPECT_TRUE(
        CheckSatisfied(prog.lorentz_cone_constraints()[0], n1_B * 0.99, tol));
    EXPECT_TRUE(
        CheckSatisfied(prog.lorentz_cone_constraints()[1], n1_B * 0.99, tol));
    EXPECT_TRUE(
        CheckSatisfied(prog.lorentz_cone_constraints()[0], n2_B * 0.99, tol));
    EXPECT_TRUE(
        CheckSatisfied(prog.lorentz_cone_constraints()[1], n2_B * 0.99, tol));
    // 1.01 * n1_B is not a valid hyperplane normal, the point p_BQ1 is not in
    // the hyperplane with normal 1.01 * n1_B.
    EXPECT_GE(1.01 * n1_B.dot(p_BQ1[i] - p_BC), 1);
    EXPECT_GE(1.01 * n2_B.dot(p_BQ2[i] - p_BC), 1);
    EXPECT_FALSE(
        CheckSatisfied(prog.lorentz_cone_constraints()[0], n1_B * 1.01, tol));
    EXPECT_FALSE(
        CheckSatisfied(prog.lorentz_cone_constraints()[1], n2_B * 1.01, tol));
  }
}

TEST_F(CylinderTest, AddPointInsideGeometryConstraint) {
  std::vector<Eigen::Vector3d> p_BQ1, p_BQ2;
  GenerateCylinderExtremPoints(&p_BQ1, &p_BQ2);

  solvers::MathematicalProgram prog;
  auto p_AQ = prog.NewContinuousVariables<3>();

  Eigen::Isometry3d X_AB;
  X_AB.linear() = Eigen::AngleAxisd(
                      0.2 * M_PI, Eigen::Vector3d(1.2, 0.3, -0.2).normalized())
                      .toRotationMatrix();
  X_AB.translation() << 0.2, 0.3, 0.4;

  const Cylinder cylinder(0, p_BO_, a_B_, radius_);
  cylinder.AddPointInsideGeometryConstraint(X_AB, p_AQ, &prog);

  auto is_inside_cylinder = [&prog](
      const Eigen::Ref<const Eigen::Vector3d>& p_AQ_val, double tol) -> bool {
    return CheckSatisfied(prog.linear_constraints()[0], p_AQ_val, tol) &&
           CheckSatisfied(prog.lorentz_cone_constraints()[0], p_AQ_val, tol);
  };
  const double tol{1E-6};

  // V are the extreme points in the cylinder.
  auto check_inside_cylinder = [&](const std::vector<Eigen::Vector3d>& p_BV,
                                   const Eigen::Vector3d& axis_B) {
    for (const auto& p_BV_val : p_BV) {
      const Eigen::Vector3d p_BQ = 0.99 * p_BV_val + 0.01 * p_BO_;
      EXPECT_TRUE(is_inside_cylinder(X_AB * p_BQ, tol));
      EXPECT_FALSE(is_inside_cylinder(X_AB * (p_BV_val + 0.01 * axis_B), tol));
      EXPECT_FALSE(is_inside_cylinder(
          X_AB * (p_BV_val + 0.01 * (p_BV_val - p_BO_ - axis_B)), tol));
    }
  };
  check_inside_cylinder(p_BQ1, a_B_);
  check_inside_cylinder(p_BQ2, -a_B_);
}
}  // namespace multibody
}  // namespace drake
