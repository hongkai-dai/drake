#include "drake/multibody/inverse_kinematics/in_collision_constraint.h"

#include "drake/multibody/inverse_kinematics/test/inverse_kinematics_test_utilities.h"

namespace drake {
namespace multibody {
const double kInf = std::numeric_limits<double>::infinity();

AutoDiffVecXd EvalMinimumDistanceAutoDiff(
    const systems::Context<AutoDiffXd>& context,
    const MultibodyPlant<AutoDiffXd>& plant) {
  AutoDiffVecXd y(1);
  y(0).value() = 0;
  y(0).derivatives().resize(
      math::autoDiffToGradientMatrix(plant.GetPositions(context)).cols());
  y(0).derivatives().setZero();
  const auto& query_object =
      plant.get_geometry_query_input_port()
          .Eval<geometry::QueryObject<AutoDiffXd>>(context);

  const std::vector<geometry::SignedDistancePair<double>>
      signed_distance_pairs =
          query_object.ComputeSignedDistancePairwiseClosestPoints();

  double minimum_distance = kInf;
  for (const auto& signed_distance_pair : signed_distance_pairs) {
    const double distance = signed_distance_pair.distance;
    if (distance < minimum_distance) {
      const double sign = distance > 0 ? 1 : -1;

      Vector3<AutoDiffXd> p_WCa, p_WCb;
      const geometry::SceneGraphInspector<AutoDiffXd>& inspector =
          query_object.inspector();
      const geometry::FrameId frame_A_id =
          inspector.GetFrameId(signed_distance_pair.id_A);
      const geometry::FrameId frame_B_id =
          inspector.GetFrameId(signed_distance_pair.id_B);
      plant.CalcPointsPositions(
          context, plant.GetBodyFromFrameId(frame_A_id)->body_frame(),
          (inspector.X_FG(signed_distance_pair.id_A) *
           signed_distance_pair.p_ACa)
              .cast<AutoDiffXd>(),
          plant.world_frame(), &p_WCa);
      plant.CalcPointsPositions(
          context, plant.GetBodyFromFrameId(frame_B_id)->body_frame(),
          (inspector.X_FG(signed_distance_pair.id_B) *
           signed_distance_pair.p_BCb)
              .cast<AutoDiffXd>(),
          plant.world_frame(), &p_WCb);

      const AutoDiffXd distance_autodiff = sign * (p_WCa - p_WCb).norm();

      y(0) = distance_autodiff;
    }
  }
  return y;
}

TEST_F(TwoFreeSpheresTest, Test) {
  InCollisionConstraint constraint(plant_double_, plant_context_double_);

  Eigen::Quaterniond sphere1_quaternion(1, 2, 3, 4);
  Eigen::Quaterniond sphere2_quaternion(0, 2, 3, 4);
  Eigen::Vector3d p_WB1(0, 1, 2);
  Eigen::Vector3d p_WB2(0.02, 1.02, 2.01);

  Eigen::Matrix<double, 14, 1> q;
  q << QuaternionToVectorWxyz(sphere1_quaternion).normalized(), p_WB1,
      QuaternionToVectorWxyz(sphere2_quaternion).normalized(), p_WB2;
}
}  // namespace multibody
}  // namespace drake
