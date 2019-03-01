#pragma once

#include <memory>
#include <string>
#include <gtest/gtest.h>

#include "drake/manipulation/dev/remote_tree_viewer_wrapper.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/rational_forward_kinematics/configuration_space_collision_free_region.h"

namespace drake {
namespace multibody {

std::unique_ptr<MultibodyPlant<double>> ConstructIiwaPlant(
    const std::string& iiwa_sdf_name);

Eigen::Matrix<double, 3, 8> GenerateBoxVertices(const Eigen::Vector3d& size,
                                                const Eigen::Isometry3d& pose);

std::vector<std::shared_ptr<const ConvexPolytope>> GenerateIiwaLinkPolytopes(
    const MultibodyPlant<double>& iiwa);

std::unique_ptr<MultibodyPlant<double>> ConstructDualArmIiwaPlant(
    const std::string& iiwa_sdf_name, const Eigen::Isometry3d& X_WL,
    const Eigen::Isometry3d& X_WR, ModelInstanceIndex* left_iiwa_instance,
    ModelInstanceIndex* right_iiwa_instance);

class IiwaTest : public ::testing::Test {
 public:
  IiwaTest();

 protected:
  std::unique_ptr<MultibodyPlant<double>> iiwa_;
  const internal::MultibodyTree<double>& iiwa_tree_;
  const BodyIndex world_;
  std::array<BodyIndex, 8> iiwa_link_;
  std::array<internal::MobilizerIndex, 8> iiwa_joint_;
};

namespace internal {
class MultibodyPlantPostureSource;
}  // namespace internal

class MultibodyPlantVisualizer {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(MultibodyPlantVisualizer)

  MultibodyPlantVisualizer(
      const MultibodyPlant<double>& plant,
      std::unique_ptr<geometry::SceneGraph<double>> scene_graph);

  void VisualizePosture(const Eigen::Ref<const Eigen::VectorXd>& q);

 private:
  std::unique_ptr<systems::Diagram<double>> diagram_;
  internal::MultibodyPlantPostureSource* posture_source_;
};

void VisualizeBodyPoint(manipulation::dev::RemoteTreeViewerWrapper* viewer,
                        const MultibodyPlant<double>& plant,
                        const systems::Context<double>& context,
                        BodyIndex body_index,
                        const Eigen::Ref<const Eigen::Vector3d>& p_BQ,
                        double radius, const Eigen::Vector4d& color,
                        const std::string& name);

/**
 * @param X_7S The transformation from schunk frame to iiwa link 7.
 * @note the plant is not finalized.
 */
std::unique_ptr<MultibodyPlant<double>> ConstructIiwaWithSchunk(
    const Eigen::Isometry3d& X_7S);
}  // namespace multibody
}  // namespace drake
