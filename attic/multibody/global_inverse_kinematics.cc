#include "drake/multibody/global_inverse_kinematics.h"

#include <limits>
#include <stack>
#include <string>

#include "drake/common/eigen_types.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/joints/drake_joints.h"
#include "drake/solvers/rotation_constraint.h"

using Eigen::Isometry3d;
using Eigen::Vector3d;
using Eigen::Matrix3d;

using std::string;

using drake::symbolic::Expression;
using drake::solvers::VectorDecisionVariable;

namespace drake {
namespace multibody {
GlobalInverseKinematics::GlobalInverseKinematics(
    const RigidBodyTreed& robot,
    const GlobalInverseKinematics::Options& options)
    : robot_(&robot),
      joint_lower_bounds_{
          Eigen::VectorXd::Constant(robot_->get_num_positions(),
                                    -std::numeric_limits<double>::infinity())},
      joint_upper_bounds_{
          Eigen::VectorXd::Constant(robot_->get_num_positions(),
                                    std::numeric_limits<double>::infinity())} {
  const int num_bodies = robot_->get_num_bodies();
  R_WB_.resize(num_bodies);
  p_WBo_.resize(num_bodies);

  solvers::MixedIntegerRotationConstraintGenerator rotation_generator(
      options.approach, options.num_intervals_per_half_axis,
      options.interval_binning);
  // Loop through each body in the robot, to add the constraint that the bodies
  // are welded by joints.
  for (int body_idx = 1; body_idx < num_bodies; ++body_idx) {
    const RigidBody<double>& body = robot_->get_body(body_idx);
    const string body_R_name = body.get_name() + "_R";
    const string body_pos_name = body.get_name() + "_pos";
    p_WBo_[body_idx] = NewContinuousVariables<3>(body_pos_name);
    // If the body is fixed to the world, then fix the decision variables on
    // the body position and orientation.
    if (body.IsRigidlyFixedToWorld()) {
      R_WB_[body_idx] = NewContinuousVariables<3, 3>(body_R_name);
      const Isometry3d X_WB = body.ComputeWorldFixedPose();
      // TODO(hongkai.dai): clean up this for loop using elementwise matrix
      // constraint when it is ready.
      for (int i = 0; i < 3; ++i) {
        AddBoundingBoxConstraint(X_WB.linear().col(i), X_WB.linear().col(i),
                                 R_WB_[body_idx].col(i));
      }
      AddBoundingBoxConstraint(X_WB.translation(), X_WB.translation(),
                               p_WBo_[body_idx]);
    } else {
      R_WB_[body_idx] = solvers::NewRotationMatrixVars(this, body_R_name);

      if (!options.linear_constraint_only) {
        solvers::AddRotationMatrixOrthonormalSocpConstraint(this,
                                                            R_WB_[body_idx]);
      }

      // If the body has a parent, then add the constraint to connect the
      // parent body with this body through a joint.
      if (body.has_parent_body()) {
        const RigidBody<double>* parent_body = body.get_parent();
        const int parent_idx = parent_body->get_body_index();
        const DrakeJoint* joint = &(body.getJoint());
        // Frame `F` is the inboard frame of the joint, rigidly attached to the
        // parent link.
        const auto& X_PF = joint->get_transform_to_parent_body();
        switch (joint->get_num_velocities()) {
          case 0: {
            // Fixed to the parent body.

            // The position can be computed from the parent body pose.
            // p_WBc = p_WBp + R_WBp * p_BpBc
            // where Bc is the child body frame.
            //       Bp is the parent body frame.
            //       W is the world frame.
            AddLinearEqualityConstraint(
                p_WBo_[parent_idx] + R_WB_[parent_idx] * X_PF.translation() -
                    p_WBo_[body_idx],
                Vector3d::Zero());

            // The orientation can be computed from the parent body orientation.
            // R_WBp * R_BpBc = R_WBc
            Matrix3<Expression> orient_invariance =
                R_WB_[parent_idx] * X_PF.linear() - R_WB_[body_idx];
            for (int i = 0; i < 3; ++i) {
              AddLinearEqualityConstraint(orient_invariance.col(i),
                                          Vector3d::Zero());
            }
            break;
          }
          case 1: {
            // Should NOT do this evil dynamic cast here, but currently we do
            // not have a method to tell if a joint is revolute or not.
            if (dynamic_cast<const RevoluteJoint*>(joint) != nullptr) {
              // Adding mixed-integer constraint will add binary variables into
              // the program.
              rotation_generator.AddToProgram(R_WB_[body_idx], this);

              const RevoluteJoint* revolute_joint =
                  dynamic_cast<const RevoluteJoint*>(joint);
              // axis_F is the vector of the rotation axis in the joint
              // inboard/outboard frame.
              const Vector3d axis_F = revolute_joint->joint_axis().head<3>();

              // Add the constraint R_WB * axis_B = R_WP * R_PF * axis_F, where
              // axis_B = axis_F since the rotation axis is invaraiant in the
              // inboard frame F and the outboard frame B.
              AddLinearEqualityConstraint(
                  R_WB_[body_idx] * axis_F -
                      R_WB_[parent_idx] * X_PF.linear() * axis_F,
                  Vector3d::Zero());

              // The position of the rotation axis is the same on both child and
              // parent bodies.
              AddLinearEqualityConstraint(
                  p_WBo_[parent_idx] + R_WB_[parent_idx] * X_PF.translation() -
                      p_WBo_[body_idx],
                  Vector3d::Zero());

              // Now we process the joint limits constraint.
              const double joint_lb = joint->getJointLimitMin()(0);
              const double joint_ub = joint->getJointLimitMax()(0);
              AddJointLimitConstraint(body_idx, joint_lb, joint_ub,
                                      options.linear_constraint_only);
            } else {
              // TODO(hongkai.dai): Add prismatic and helical joint.
              throw std::runtime_error("Unsupported joint type.");
            }
            break;
          }
          case 6: {
            // This is the floating base case, just add the rotation matrix
            // constraint.
            rotation_generator.AddToProgram(R_WB_[body_idx], this);
            break;
          }
          default:
            throw std::runtime_error("Unsupported joint type.");
        }
      }
    }
  }
}

const solvers::MatrixDecisionVariable<3, 3>&
GlobalInverseKinematics::body_rotation_matrix(int body_index) const {
  if (body_index >= robot_->get_num_bodies() || body_index <= 0) {
    throw std::runtime_error("body index out of range.");
  }
  return R_WB_[body_index];
}

const solvers::VectorDecisionVariable<3>&
GlobalInverseKinematics::body_position(int body_index) const {
  if (body_index >= robot_->get_num_bodies() || body_index <= 0) {
    throw std::runtime_error("body index out of range.");
  }
  return p_WBo_[body_index];
}

void GlobalInverseKinematics::ReconstructGeneralizedPositionSolutionForBody(
    int body_idx, const std::vector<int>& body_children,
    double position_error_weight, const std::vector<Eigen::Vector3d>& p_WBo,
    const std::vector<Eigen::Matrix3d>& R_WB, Eigen::Ref<Eigen::VectorXd> q,
    std::vector<Eigen::Matrix3d>* reconstruct_R_WB) const {
  const RigidBody<double>& body = robot_->get_body(body_idx);
  const RigidBody<double>* parent = body.get_parent();
  if (!body.IsRigidlyFixedToWorld()) {
    const Matrix3d R_WC = R_WB[body_idx];
    // R_WP is the rotation matrix of parent frame to the world frame.
    const Matrix3d& R_WP = reconstruct_R_WB->at(parent->get_body_index());
    const DrakeJoint* joint = &(body.getJoint());
    const auto& X_PF = joint->get_transform_to_parent_body();

    int num_positions = joint->get_num_positions();
    // For each different type of joints, use a separate branch to compute
    // the posture for that joint.
    if (joint->is_floating()) {
      // p_WBi is the position of the body frame in the world frame.
      const Vector3d p_WBi = p_WBo[body_idx];
      const math::RotationMatrix<double> normalized_rotmat =
          math::RotationMatrix<double>::ProjectToRotationMatrix(R_WC);

      q.segment<3>(body.get_position_start_index()) = p_WBi;
      if (num_positions == 6) {
        // The position order is x-y-z-roll-pitch-yaw.
        q.segment<3>(body.get_position_start_index() + 3) =
            math::RollPitchYaw<double>(normalized_rotmat).vector();
      } else {
        // The position order is x-y-z-qw-qx-qy-qz, namely translation
        // first, and quaternion second.
        q.segment<4>(body.get_position_start_index() + 3) =
            normalized_rotmat.ToQuaternionAsVector4();
      }
      reconstruct_R_WB->at(body_idx) = normalized_rotmat.matrix();
    } else if (num_positions == 1) {
      const int joint_idx = body.get_position_start_index();
      const double joint_lb = joint_lower_bounds_(joint_idx);
      const double joint_ub = joint_upper_bounds_(joint_idx);
      // Should NOT do this evil dynamic cast here, but currently we do
      // not have a method to tell if a joint is revolute or not.
      if (dynamic_cast<const RevoluteJoint*>(joint) != nullptr) {
        Eigen::Matrix3Xd p_BC(3, body_children.size());
        Eigen::Matrix3Xd p_WC(3, body_children.size());
        for (int i = 0; i < static_cast<int>(body_children.size()); ++i) {
          p_BC.col(i) = robot_->get_body(body_children[i])
                            .getJoint()
                            .get_transform_to_parent_body()
                            .translation();
          p_WC.col(i) = p_WBo[body_children[i]];
        }
        const RevoluteJoint* revolute_joint =
            dynamic_cast<const RevoluteJoint*>(joint);
        // The joint_rotmat is very likely not on SO(3). The reason is
        // that we use a relaxation of the rotation matrix, and thus
        // R_WC might not lie on SO(3) exactly. Here we need to project
        // joint_rotmat to SO(3), with joint axis as the rotation axis, and
        // joint limits as the lower and upper bound on the rotation angle.
        const Eigen::Vector3d p_WB = p_WBo[body_idx];
        const Vector3d rotate_axis = revolute_joint->joint_axis().head<3>();
        const double revolute_joint_angle =
            ReconstructJointAngleForRevoluteJoint(
                R_WP, R_WC, X_PF.linear(), rotate_axis, p_WB, p_WC, p_BC,
                position_error_weight, joint_lb, joint_ub);
        q(body.get_position_start_index()) = revolute_joint_angle;
        reconstruct_R_WB->at(body_idx) =
            R_WP * X_PF.linear() *
            Eigen::AngleAxisd(revolute_joint_angle, rotate_axis)
                .toRotationMatrix();
      } else {
        // TODO(hongkai.dai): add prismatic and helical joints.
        throw std::runtime_error("Unsupported joint type.");
      }
    } else if (num_positions == 0) {
      // Deliberately left empty because the joint is removed by welding the
      // parent body to the child body.
    }
  } else {
    // The reconstructed body orientation is just the world fixed
    // orientation.
    const Isometry3d X_WB = body.ComputeWorldFixedPose();
    reconstruct_R_WB->at(body_idx) = X_WB.linear();
  }
}

void GlobalInverseKinematics::BuildTreeTopology(
    std::vector<std::vector<int>>* body_children) const {
  DRAKE_DEMAND(body_children);
  body_children->resize(robot_->get_num_bodies());
  for (int i = 1; i < robot_->get_num_bodies(); ++i) {
    const int parent_idx = robot_->get_body(i).get_parent()->get_body_index();
    (*body_children)[parent_idx].push_back(i);
  }
}

Eigen::VectorXd GlobalInverseKinematics::ReconstructGeneralizedPositionSolution(
    double position_error_weight, int solution_number) const {
  std::vector<std::vector<int>> body_children;
  BuildTreeTopology(&body_children);
  // Is the robot a single kinematic chain?
  bool is_kinematic_chain = true;
  int chain_leaf_node = 0;
  for (int i = 0; i < static_cast<int>(body_children.size()); ++i) {
    const auto& children = body_children[i];
    if (children.size() > 1) {
      // Check if the children are welded to this body.
      int num_non_welded_children = 0;
      for (int child : children) {
        if (robot_->get_body(child).getJoint().get_num_velocities() > 0) {
          num_non_welded_children++;
        }
        if (num_non_welded_children > 1) {
          is_kinematic_chain = false;
          break;
        }
      }
    }
    if (children.size() == 0) {
      chain_leaf_node = i;
    }
  }
  Eigen::VectorXd q(robot_->get_num_positions());
  // reconstruct_R_WB[i] is the orientation of body i'th body frame expressed in
  // the world frame, computed from the reconstructed posture.
  std::vector<Eigen::Matrix3d> reconstruct_R_WB(robot_->get_num_bodies());
  // is_link_visited[i] is set to true, if the angle of the joint on link i has
  // been reconstructed.
  // The first one is the world frame, thus the orientation is identity.
  reconstruct_R_WB[0].setIdentity();
  std::vector<Eigen::Vector3d> p_WBo(robot_->get_num_bodies());
  std::vector<Eigen::Matrix3d> R_WB(robot_->get_num_bodies());
  p_WBo[0] = Eigen::Vector3d::Zero();
  R_WB[0] = Eigen::Matrix3d::Identity();
  for (int i = 1; i < robot_->get_num_bodies(); ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    p_WBo[i] = solution_number > 0
                   ? GetSuboptimalSolution(p_WBo_[i], solution_number)
                   : GetSolution(p_WBo_[i]);
    R_WB[i] = solution_number > 0
                  ? GetSuboptimalSolution(R_WB_[i], solution_number)
                  : GetSolution(R_WB_[i]);
#pragma GCC diagnostic pop
  }
  // Only do multiple sweep if the robot is a chain.
  const int max_sweep_iterations = is_kinematic_chain ? 3 : 1;
  for (int sweep_count = 0; sweep_count < max_sweep_iterations; ++sweep_count) {
    std::stack<int> unvisited_links;
    unvisited_links.push(body_children[0][0]);
    while (!unvisited_links.empty()) {
      const int body_index = unvisited_links.top();
      unvisited_links.pop();
      ReconstructGeneralizedPositionSolutionForBody(
          body_index, body_children[body_index], position_error_weight, p_WBo,
          R_WB, q, &reconstruct_R_WB);
      for (int children : body_children[body_index]) {
        unvisited_links.push(children);
      }
    }
    if (is_kinematic_chain) {
      // Now update the link position and orientation from the leaf node to the
      // root. We compute the parent link pose using the reconstructed joint
      // positions and the child link pose.
      int node = chain_leaf_node;
      int parent = robot_->get_body(node).get_parent()->get_body_index();
      while (parent != 0) {
        const auto& body = robot_->get_body(node);
        const DrakeJoint& joint = body.getJoint();
        const Eigen::VectorXd q_joint = q.segment(
            body.get_position_start_index(), joint.get_num_positions());
        const Eigen::Isometry3d X_FB = joint.jointTransform(q_joint);
        const Eigen::Isometry3d& X_PF = joint.get_transform_to_parent_body();
        const Eigen::Isometry3d X_PB = X_PF * X_FB;
        Eigen::Isometry3d X_WB;
        X_WB.linear() = R_WB[node];
        X_WB.translation() = p_WBo[node];
        const Eigen::Isometry3d X_WP = X_WB * X_PB.inverse();
        R_WB[parent] = X_WP.linear();
        p_WBo[parent] = X_WP.translation();
        node = parent;
        parent = robot_->get_body(node).get_parent()->get_body_index();
      }
    }
  }
  return q;
}

solvers::Binding<solvers::LinearConstraint>
GlobalInverseKinematics::AddWorldPositionConstraint(
    int body_idx, const Eigen::Vector3d& p_BQ, const Eigen::Vector3d& box_lb_F,
    const Eigen::Vector3d& box_ub_F, const Isometry3d& X_WF) {
  Vector3<Expression> body_pt_pos = p_WBo_[body_idx] + R_WB_[body_idx] * p_BQ;
  Vector3<Expression> body_pt_in_measured_frame =
      X_WF.linear().transpose() * (body_pt_pos - X_WF.translation());
  return AddLinearConstraint(body_pt_in_measured_frame, box_lb_F, box_ub_F);
}

solvers::Binding<solvers::LinearConstraint>
GlobalInverseKinematics::AddWorldOrientationConstraint(
    int body_idx, const Eigen::Quaterniond& desired_orientation,
    double angle_tol) {
  // The rotation matrix error R_e satisfies
  // trace(R_e) = 2 * cos(θ) + 1, where θ is the rotation angle between
  // desired orientation and the current orientation. Thus the constraint is
  // 2 * cos(angle_tol) + 1 <= trace(R_e) <= 3
  Matrix3<Expression> rotation_matrix_err =
      desired_orientation.toRotationMatrix() * R_WB_[body_idx].transpose();
  double lb = angle_tol < M_PI ? 2 * cos(angle_tol) + 1 : -1;
  return AddLinearConstraint(rotation_matrix_err.trace(), lb, 3);
}

void GlobalInverseKinematics::AddPostureCost(
    const Eigen::Ref<const Eigen::VectorXd>& q_desired,
    const Eigen::Ref<const Eigen::VectorXd>& body_position_cost,
    const Eigen::Ref<const Eigen::VectorXd>& body_orientation_cost) {
  const int num_bodies = robot_->get_num_bodies();
  if (body_position_cost.rows() != num_bodies) {
    std::ostringstream oss;
    oss << "body_position_cost should have " << num_bodies << " rows.";
    throw std::runtime_error(oss.str());
  }
  if (body_orientation_cost.rows() != num_bodies) {
    std::ostringstream oss;
    oss << "body_orientation_cost should have " << num_bodies << " rows.";
    throw std::runtime_error(oss.str());
  }
  for (int i = 1; i < num_bodies; ++i) {
    if (body_position_cost(i) < 0) {
      std::ostringstream oss;
      oss << "body_position_cost(" << i << ") is negative.";
      throw std::runtime_error(oss.str());
    }
    if (body_orientation_cost(i) < 0) {
      std::ostringstream oss;
      oss << "body_orientation_cost(" << i << ") is negative.";
      throw std::runtime_error(oss.str());
    }
  }
  auto cache = robot_->CreateKinematicsCache();
  cache.initialize(q_desired);
  robot_->doKinematics(cache);

  // Sum up the orientation error for each body to orient_err_sum.
  Expression orient_err_sum(0);
  // p_WBo_err(i) is the slack variable, representing the position error for
  // the (i+1)'th body, which is the Euclidean distance from the body origin
  // position, to the desired position.
  solvers::VectorXDecisionVariable p_WBo_err =
      NewContinuousVariables(num_bodies - 1, "p_WBo_error");
  for (int i = 1; i < num_bodies; ++i) {
    // body 0 is the world. There is no position or orientation error on the
    // world, so we just skip i = 0 and start from i = 1.
    const auto& X_WB_desired = robot_->CalcFramePoseInWorldFrame(
        cache, robot_->get_body(i), Isometry3d::Identity());
    // Add the constraint p_WBo_err(i-1) >= body_position_cost(i) *
    // |p_WBo(i) - p_WBo_desired(i) |
    Vector4<symbolic::Expression> pos_error_expr;
    pos_error_expr << p_WBo_err(i - 1),
        body_position_cost(i) * (p_WBo_[i] - X_WB_desired.translation());
    AddLorentzConeConstraint(pos_error_expr);

    // The orientation error is on the angle θ between the body orientation and
    // the desired orientation, namely 1 - cos(θ).
    // cos(θ) can be computed as (trace( R_WB_desired * R_WB_[i]ᵀ) - 1) / 2
    // To see how the angle is computed from a rotation matrix, please refer to
    // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/
    orient_err_sum +=
        body_orientation_cost(i) *
        (1 - ((X_WB_desired.linear() * R_WB_[i].transpose()).trace() - 1) / 2);
  }

  // The total cost is the summation of the position error and the orientation
  // error.
  AddCost(p_WBo_err.cast<Expression>().sum() + orient_err_sum);
}

solvers::VectorXDecisionVariable
GlobalInverseKinematics::BodyPointInOneOfRegions(
    int body_index, const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
    const std::vector<Eigen::Matrix3Xd>& region_vertices) {
  const auto& R_WB = body_rotation_matrix(body_index);
  const auto& p_WBo = body_position(body_index);
  const int num_regions = region_vertices.size();
  const string& body_name = robot_->get_body(body_index).get_name();
  solvers::VectorXDecisionVariable z =
      NewBinaryVariables(num_regions, "z_" + body_name);
  std::vector<solvers::VectorXDecisionVariable> w(num_regions);

  // We will write p_WQ in two ways, we first write p_WQ as
  // sum_i (w_i1 * v_i1 + w_i2 * v_i2 + ... + w_in * v_in). As the convex
  // combination of vertices in one of the regions.
  Vector3<symbolic::Expression> p_WQ;
  p_WQ << 0, 0, 0;
  for (int i = 0; i < num_regions; ++i) {
    const int num_vertices_i = region_vertices[i].cols();
    if (num_vertices_i < 3) {
      throw std::runtime_error("Each region should have at least 3 vertices.");
    }
    w[i] = NewContinuousVariables(
        num_vertices_i, "w_" + body_name + "_region_" + std::to_string(i));
    AddLinearConstraint(w[i].cast<symbolic::Expression>().sum() - z(i) == 0);
    AddBoundingBoxConstraint(Eigen::VectorXd::Zero(num_vertices_i),
                             Eigen::VectorXd::Ones(num_vertices_i), w[i]);
    p_WQ += region_vertices[i] * w[i];
  }

  AddLinearConstraint(z.cast<symbolic::Expression>().sum() == 1);

  // p_WQ must match the body pose, as p_WQ = p_WBo + R_WB * p_BQ
  AddLinearEqualityConstraint(p_WBo + R_WB * p_BQ - p_WQ,
                              Eigen::Vector3d::Zero());

  return z;
}

solvers::VectorXDecisionVariable
GlobalInverseKinematics::BodySphereInOneOfPolytopes(
    int body_index, const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
    double radius,
    const std::vector<GlobalInverseKinematics::Polytope3D>& polytopes) {
  DRAKE_DEMAND(radius >= 0);
  const int num_polytopes = static_cast<int>(polytopes.size());
  const auto z = NewBinaryVariables(num_polytopes, "z");
  // z1 + ... + zn = 1
  AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(num_polytopes), 1, z);

  const auto y =
      NewContinuousVariables<3, Eigen::Dynamic>(3, num_polytopes, "y");
  const Vector3<symbolic::Expression> p_WQ =
      p_WBo_[body_index] + R_WB_[body_index] * p_BQ;
  // p_WQ = y.col(0) + ... + y.col(n)
  AddLinearEqualityConstraint(
      p_WQ - y.cast<symbolic::Expression>().rowwise().sum(),
      Eigen::Vector3d::Zero());

  for (int i = 0; i < num_polytopes; ++i) {
    DRAKE_DEMAND(polytopes[i].A.rows() == polytopes[i].b.rows());
    AddLinearConstraint(
        polytopes[i].A * y.col(i) <=
        (polytopes[i].b - polytopes[i].A.rowwise().norm() * radius) * z(i));
  }

  return z;
}

// Approximate a quadratic constraint (which could be formulated as a Lorentz
// cone constraint) xᵀx ≤ c² by
// -c ≤ xᵢ ≤ c
// ± xᵢ ± xⱼ ≤ √2 * c
// ± x₀ ± x₁ ± x₂ ≤ √3 * c
// These linear approximation are obtained as the tangential planes at some
// points on the surface of the sphere xᵀx ≤ c².
void ApproximateBoundedNormByLinearConstraints(
    const Eigen::Ref<const Vector3<symbolic::Expression>>& x, double c,
    solvers::MathematicalProgram* prog) {
  DRAKE_DEMAND(c >= 0);
  // -c ≤ xᵢ ≤ c
  prog->AddLinearConstraint(x, Eigen::Vector3d::Constant(-c),
                            Eigen::Vector3d::Constant(c));
  const double sqrt2_c = std::sqrt(2) * c;
  const double sqrt3_c = std::sqrt(3) * c;
  // ± xᵢ ± xⱼ ≤ √2 * c
  for (int i = 0; i < 3; ++i) {
    for (int j = i + 1; j < 3; ++j) {
      prog->AddLinearConstraint(x(i) + x(j), -sqrt2_c, sqrt2_c);
      prog->AddLinearConstraint(x(i) - x(j), -sqrt2_c, sqrt2_c);
    }
  }
  // ± x₀ ± x₁ ± x₂ ≤ √3 * c
  prog->AddLinearConstraint(x(0) + x(1) + x(2), -sqrt3_c, sqrt3_c);
  prog->AddLinearConstraint(x(0) + x(1) - x(2), -sqrt3_c, sqrt3_c);
  prog->AddLinearConstraint(x(0) - x(1) + x(2), -sqrt3_c, sqrt3_c);
  prog->AddLinearConstraint(x(0) - x(1) - x(2), -sqrt3_c, sqrt3_c);
}

void GlobalInverseKinematics::AddJointLimitConstraint(
    int body_index, double joint_lower_bound, double joint_upper_bound,
    bool linear_constraint_approximation) {
  if (joint_lower_bound > joint_upper_bound) {
    throw std::runtime_error(
        "The joint lower bound should be no larger than the upper bound.");
  }
  const RigidBody<double>& body = robot_->get_body(body_index);
  if (body.has_parent_body()) {
    const RigidBody<double>* parent_body = body.get_parent();
    const int parent_idx = parent_body->get_body_index();
    const DrakeJoint* joint = &(body.getJoint());
    const auto& X_PF = joint->get_transform_to_parent_body();
    switch (joint->get_num_velocities()) {
      case 0: {
        // Fixed to the parent body.
        throw std::runtime_error(
            "Cannot impose joint limits for a fixed joint.");
      }
      case 1: {
        // If the new bound [joint_lower_bound joint_upper_bound] is not tighter
        // than the existing bound, then we ignore it, without adding new
        // constraints.
        bool is_limits_tightened = false;
        int joint_idx = body.get_position_start_index();
        if (joint_lower_bound > joint_lower_bounds_(joint_idx)) {
          joint_lower_bounds_(joint_idx) = joint_lower_bound;
          is_limits_tightened = true;
        }
        if (joint_upper_bound < joint_upper_bounds_(joint_idx)) {
          joint_upper_bounds_(joint_idx) = joint_upper_bound;
          is_limits_tightened = true;
        }
        if (is_limits_tightened) {
          // Should NOT do this evil dynamic cast here, but currently we do
          // not have a method to tell if a joint is revolute or not.
          if (dynamic_cast<const RevoluteJoint*>(joint) != nullptr) {
            const RevoluteJoint* revolute_joint =
                dynamic_cast<const RevoluteJoint*>(joint);
            // axis_F is the vector of the rotation axis in the joint
            // inboard/outboard frame.
            const Vector3d axis_F = revolute_joint->joint_axis().head<3>();

            // Now we process the joint limits constraint.
            const double joint_bound = (joint_upper_bounds_[joint_idx] -
                                        joint_lower_bounds_[joint_idx]) /
                                       2;

            if (joint_bound < M_PI) {
              // We use the fact that if the angle between two unit length
              // vectors u and v is smaller than α, it is equivalent to
              // |u - v| <= 2*sin(α/2)
              // which is a second order cone constraint.

              // If the rotation angle θ satisfies
              // a <= θ <= b
              // This is equivalent to
              // -α <= θ - (a+b)/2 <= α
              // where α = (b-a) / 2, (a+b) / 2 is the joint offset, such that
              // the bounds on β = θ - (a+b)/2 are symmetric.
              // We use the following notation:
              // R_WP     The rotation matrix of parent frame `P` to world
              //          frame `W`.
              // R_WC     The rotation matrix of child frame `C` to world
              //          frame `W`.
              // R_PF     The rotation matrix of joint frame `F` to parent
              //          frame `P`.
              // R(k, β)  The rotation matrix along joint axis k by angle β.
              // The kinematics constraint is
              // R_WP * R_PF * R(k, θ) = R_WC.
              // This is equivalent to
              // R_WP * R_PF * R(k, (a+b)/2) * R(k, β)) = R_WC.
              // So to constrain that -α <= β <= α,
              // we can constrain the angle between the two vectors
              // R_WC * v and R_WP * R_PF * R(k,(a+b)/2) * v is no larger than
              // α, where v is a unit length vector perpendicular to
              // the rotation axis k, in the joint frame.
              // Thus we can constrain that
              // |R_WC*v - R_WP * R_PF * R(k,(a+b)/2)*v | <= 2*sin (α / 2)
              // as we explained above.

              // First generate a vector v_C that is perpendicular to rotation
              // axis, in child frame.
              Vector3d v_C = axis_F.cross(Vector3d(1, 0, 0));
              double v_C_norm = v_C.norm();
              if (v_C_norm < sqrt(2) / 2) {
                // axis_F is almost parallel to [1; 0; 0]. Try another axis
                // [0, 1, 0]
                v_C = axis_F.cross(Vector3d(0, 1, 0));
                v_C_norm = v_C.norm();
              }
              // Normalizes the revolute vector.
              v_C /= v_C_norm;

              // The constraint would be tighter, if we choose many unit
              // length vector `v`, perpendicular to the joint axis, in the
              // joint frame. Here to balance between the size of the
              // optimization problem, and the tightness of the convex
              // relaxation, we just use four vectors in `v`. Notice that
              // v_basis contains the orthonormal basis of the null space
              // null(axis_F).
              std::array<Eigen::Vector3d, 2> v_basis = {
                  {v_C, axis_F.cross(v_C)}};
              v_basis[1] /= v_basis[1].norm();

              std::array<Eigen::Vector3d, 4> v_samples;
              v_samples[0] = v_basis[0];
              v_samples[1] = v_basis[1];
              v_samples[2] = v_basis[0] + v_basis[1];
              v_samples[2] /= v_samples[2].norm();
              v_samples[3] = v_basis[0] - v_basis[1];
              v_samples[3] /= v_samples[3].norm();

              // rotmat_joint_offset is R(k, (a+b)/2) explained above.
              const Matrix3d rotmat_joint_offset =
                  Eigen::AngleAxisd((joint_lower_bounds_[joint_idx] +
                                     joint_upper_bounds_[joint_idx]) /
                                        2,
                                    axis_F)
                      .toRotationMatrix();

              // joint_limit_expr is going to be within the Lorentz cone.
              Eigen::Matrix<Expression, 4, 1> joint_limit_expr;
              const double joint_limit_lorentz_rhs = 2 * sin(joint_bound / 2);
              joint_limit_expr(0) = joint_limit_lorentz_rhs;
              for (const auto& v : v_samples) {
                // joint_limit_expr.tail<3> is
                // R_WC * v - R_WP * R_PF * R(k,(a+b)/2) * v mentioned above.
                joint_limit_expr.tail<3>() =
                    R_WB_[body_index] * v -
                    R_WB_[parent_idx] * X_PF.linear() * rotmat_joint_offset * v;
                if (linear_constraint_approximation) {
                  ApproximateBoundedNormByLinearConstraints(
                      joint_limit_expr.tail<3>(), joint_limit_lorentz_rhs,
                      this);

                } else {
                  AddLorentzConeConstraint(joint_limit_expr);
                }
              }
              if (robot_->get_body(parent_idx).IsRigidlyFixedToWorld()) {
                // If the parent body is rigidly fixed to the world. Then we
                // can impose a tighter constraint. Based on the derivation
                // above, we have
                // R(k, β) = [R_WP * R_PF * R(k, (a+b)/2)]ᵀ * R_WC
                // as a linear expression of the decision variable R_WC
                // (notice that R_WP is constant, since the parent body is
                // rigidly fixed to the world.
                // Any unit length vector `v` that is perpendicular to
                // joint axis `axis_F` in the joint Frame, can be written as
                //   v = V * u, uᵀ * u = 1
                // where V = [v_basis[0] v_basis[1]] containing the basis
                // vectors for the linear space Null(axis_F).
                // On the other hand, we know
                //   vᵀ * R(k, β) * v = cos(β) >= cos(α)
                // due to the joint limits constraint
                //   -α <= β <= α.
                // So we have the condition that
                // uᵀ * u = 1
                //    => uᵀ * Vᵀ * R(k, β) * V * u >= cos(α)
                // Using S-lemma, we know this implication is equivalent to
                // Vᵀ * [R(k, β) + R(k, β)ᵀ]/2 * V - cos(α) * I is p.s.d
                // We let a 2 x 2 matrix
                //   M = Vᵀ * [R(k, β) + R(k, β)ᵀ]/2 * V - cos(α) * I
                // A 2 x 2 matrix M being positive semidefinite (p.s.d) is
                // equivalent to the condition that
                // [M(0, 0), M(1, 1), M(1, 0)] is in the rotated Lorentz cone.
                const Isometry3d X_WP =
                    robot_->get_body(parent_idx).ComputeWorldFixedPose();
                // R_joint_beta is R(k, β) in the documentation.
                Eigen::Matrix<symbolic::Expression, 3, 3> R_joint_beta =
                    (X_WP.linear() * X_PF.linear() * rotmat_joint_offset)
                        .transpose() *
                    R_WB_[body_index];
                const double joint_bound_cos{std::cos(joint_bound)};
                if (!linear_constraint_approximation) {
                  Eigen::Matrix<double, 3, 2> V;
                  V << v_basis[0], v_basis[1];
                  const Eigen::Matrix<symbolic::Expression, 2, 2> M =
                      V.transpose() *
                          (R_joint_beta + R_joint_beta.transpose()) / 2 * V -
                      joint_bound_cos * Eigen::Matrix2d::Identity();
                  AddRotatedLorentzConeConstraint(
                      Vector3<symbolic::Expression>(M(0, 0), M(1, 1), M(1, 0)));
                }

                // From Rodriguez formula, we know that -α <= β <= α implies
                // trace(R(k, β)) = 1 + 2 * cos(β) >= 1 + 2*cos(α)
                // So we can impose the constraint
                // 1+2*cos(α) ≤ trace(R(k, β))
                const symbolic::Expression R_joint_beta_trace{
                    R_joint_beta.trace()};
                AddLinearConstraint(R_joint_beta_trace >=
                                    1 + 2 * joint_bound_cos);
              }
            }
          } else {
            // TODO(hongkai.dai): add prismatic and helical joint.
            throw std::runtime_error("Unsupported joint type.");
          }
        }
        break;
      }
      case 6: {
        break;
      }
      default:
        throw std::runtime_error("Unsupported joint type.");
    }
  } else {
    throw std::runtime_error("The body " + body.get_name() +
                             " does not have a joint.");
  }
}

double GlobalInverseKinematics::ReconstructJointAngleForRevoluteJoint(
    const Eigen::Matrix3d& R_WP, const Eigen::Matrix3d& R_WB,
    const Eigen::Matrix3d& R_PF, const Eigen::Vector3d& a_F,
    const Eigen::Vector3d& p_WB, const Eigen::Matrix3Xd& p_WC,
    const Eigen::Matrix3Xd& p_BC, double beta, double angle_lower,
    double angle_upper) {
  DRAKE_DEMAND(beta >= 0);
  DRAKE_DEMAND(p_WC.cols() == p_BC.cols());
  Eigen::Matrix3d A;
  // clang-format off
  A << 0, -a_F(2), a_F(1),
       a_F(2), 0, -a_F(0),
       -a_F(1), a_F(0), 0;
  // clang-format on
  const Eigen::Matrix3d A_square = A * A;
  const Eigen::Matrix3d M = R_WB.transpose() * R_WP * R_PF;
  const Eigen::Array3Xd p_CB_W =
      (p_WB * Eigen::RowVectorXd::Ones(p_WC.cols()) - p_WC).array();
  const double position_error_y =
      p_WC.cols() == 0
          ? 0
          : -(p_CB_W * (R_WP * R_PF * A * p_BC).array()).colwise().sum().sum();
  const double position_error_x =
      p_WC.cols() == 0 ? 0 : (p_CB_W * (R_WP * R_PF * A_square * p_BC).array())
                                 .colwise()
                                 .sum()
                                 .sum();
  const double theta_y = (M * A).trace() + beta * position_error_y;
  const double theta_x = -(M * A_square).trace() + beta * position_error_x;
  const double theta_zero_gradient = std::atan2(theta_y, theta_x);
  for (int i = -1; i <= 1; ++i) {
    const double theta_zero_gradient_shift = theta_zero_gradient + 2 * M_PI * i;
    if (theta_zero_gradient_shift <= angle_upper &&
        theta_zero_gradient_shift >= angle_lower) {
      return theta_zero_gradient_shift;
    }
  }
  // The cost gradient doesn't vanish within [angle_lower, angle_upper], the
  // minimal occurs at the boundary.
  if (-theta_y * std::sin(angle_lower) - theta_x * std::cos(angle_lower) <=
      -theta_y * std::sin(angle_upper) - theta_x * std::cos(angle_upper)) {
    return angle_lower;
  } else {
    return angle_upper;
  }
}
}  // namespace multibody
}  // namespace drake
